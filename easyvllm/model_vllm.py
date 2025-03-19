################################################################################
# Copyright 2025 XingYuSSS
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import atexit
import re
from typing import Annotated, List, Optional, Literal, Union
import concurrent.futures
import threading
import os
import subprocess
import signal
import time
from dataclasses import dataclass, replace

from openai import OpenAI, APIConnectionError
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager, ReasoningParser


@dataclass
class ChatParam:
    temperature: float = None
    top_p: float = None
    presence_penalty: float = None
    frequency_penalty: float = None
    max_completion_tokens: int = None

@dataclass
class ChatExtraParam:
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    prompt_logprobs: Optional[int] = None
    
@dataclass
class GenParam:
    temperature: float = None
    top_p: float = None
    presence_penalty: float = None
    frequency_penalty: float = None
    max_tokens: int = None

@dataclass
class GenExtraParam:
    use_beam_search: bool = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    allowed_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[int] = None


class InferenceModel:
    """Fast inference model implemented with vllm, support large reasoning model (LRM)."""
    def __init__(
        self, 
        model_path: str, 
        device_ids: list[int] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        port: int = 50000,
        max_model_len: int = None,
        show_vllm_log: bool = True,
        openai_timeout: int = 30,
        enable_reasoning: bool = False,
        reasoning_parser: str = 'deepseek_r1',
        chat_template: str = None,
        use_ray: bool = False,
        ray_host_ip: str = None,
    ):
        """Fast inference model implemented with vllm, support large reasoning model (LRM).

        Args:
            model_path (str): path to your model.
            device_ids (list[int], optional): CUDA devide_ids. Set if you need to run model with choiced devices. Note: to accelerate inference, this class will init multiple models, the model num is `len(self.device_ids) // tensor_parallel_size`. Defaults to None.
            tensor_parallel_size (int, optional): Tensor parallel size to init vllm model. Defaults to 1.
            pipeline_parallel_size (int, optional): Pipeline parallel size to init vllm model. Defaults to 1.
            port (int, optional): The port vllm serve use. To avoid crash when init multiple models, the real port of vllm serve is `port + devide_id`. Defaults to 50000.
            max_model_len (int, optional): Max model length to init vllm model. Defaults to None.
            show_vllm_log (bool, optional): Set False to disable vllm output. Defaults to True.
            openai_timeout (int, optional): The timeout used by openai client, if the reasoning content is too large, please set this bigger. Defaults to 30.
            enable_reasoning (bool, optional): Set True to use LRM. Defaults to False.
            reasoning_parser (str, optional): The reasoning parser used to parse reasoning content. Defaults to 'deepseek_r1'.
            chat_template (str, optional): chat template file to init vllm model. Defaults to None.
            use_ray (bool, optional): Set True if using multiple nodes. Please create ray cluster first. Defaults to False.
            ray_host_ip (str, optional): The ray host ip of running ray cluster, required when use_ray. Defaults to None.
        """
        self.model_process = []
        self.sub_pid = set()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.model_path = model_path
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.port = port
        self.max_model_len = max_model_len
        self.show_vllm_log = show_vllm_log
        self.openai_timeout = openai_timeout
        self.chat_template = chat_template
        
        self.enable_reasoning = enable_reasoning
        if enable_reasoning and reasoning_parser is None:
            raise ValueError("enable_reasoning reqiures reasoning_parser")
        self.reasoning_parser = reasoning_parser
        
        self.use_ray = use_ray
        if use_ray and ray_host_ip is None:
            raise ValueError('ray_host_ip is needed when using ray')
        self.ray_host_ip = ray_host_ip
        
        self.tokenizer = None
        self.reasoning_parser_obj = None
        self.more_reasoning_prompt = 'Wait! Wait! Wait!'
        
        if use_ray:
            print('Using Ray, Only init 1 model')
            self.model_num = 1
            self.device_ids = [self.device_ids]
        else:
            self.model_num = len(self.device_ids) // tensor_parallel_size
            self.device_ids = [self.device_ids[i * tensor_parallel_size:(i + 1) * tensor_parallel_size] for i in range(self.model_num)]

            print('device num: ', len(self.device_ids))
            print('total device num: ', torch.cuda.device_count())
            print('model num: ', self.model_num)
            print('device ids: ', self.device_ids)

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.model_num) as executor:
            self.model_openais = list(executor.map(self._load_model_on_gpu, self.device_ids))

        print('*' * 100)
        print(f'Loaded {len(self.model_openais)} models')
        print(f'Used {time.time() - start_time} sec')
        print('*' * 100)
        self.last = 0


    def _read_output(self, process):
        for line in process.stdout:
            if self.show_vllm_log and line != '' :
                print(line, end='')


    def _load_model_on_gpu(self, device_ids):        
        print(f"Loading model in device {device_ids}...")
        cmd = f"export CUDA_VISIBLE_DEVICES={','.join([str(i) for i in device_ids])}\n" if not self.use_ray else f"VLLM_HOST_IP={self.ray_host_ip} VLLM_USE_RAY_SPMD_WORKER=1 VLLM_USE_RAY_COMPILED_DAG=1 "
        cmd += f"vllm serve {self.model_path} "
        cmd += f'--dtype auto --trust-remote-code '
        cmd += f'--port {self.port + device_ids[0]} '
        cmd += f'--api-key abc '
        cmd += f'--max-model-len {self.max_model_len} ' if self.max_model_len else ''
        cmd += f'--gpu-memory-utilization 0.95 '
        cmd += f'--tensor-parallel-size {self.tensor_parallel_size} '
        cmd += f'--pipeline-parallel-size {self.pipeline_parallel_size} '
        cmd += f'--enable-reasoning --reasoning-parser {self.reasoning_parser} ' if self.enable_reasoning else ''
        cmd += f'--chat-template {self.chat_template} ' if self.chat_template else ''
        cmd += f'--distributed-executor-backend ray ' if self.use_ray else ''
        
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.model_process.append(process)
        
        while True:
            out = process.stdout.readline()
            pid_match = re.findall(r'\(VllmWorkerProcess pid=(\d+)\)', out)
            for pid in pid_match:
                self.sub_pid.add(pid)
                
            if self.show_vllm_log and out != '':
                print(out, end='')
                
            if 'Starting vLLM API server' in out:
                print(f'Success loaded model in {device_ids}!')
                time.sleep(5)
                
                client = OpenAI(
                    base_url=f"http://localhost:{self.port + device_ids[0]}/v1",
                    api_key='abc'
                )
                models = client.models.list()
                model = models.data[0].id
                output_thread = threading.Thread(target=self._read_output, args=(process,))
                output_thread.daemon = True
                output_thread.start()
                return client, model

    def _cleanup(self, *args):
        for process in self.model_process:
            try:
                process.terminate()
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        for pid in self.sub_pid:
            print(f'killing {pid}')
            os.system(f"kill -9 {pid}")

    def _signal_handler(self, signum, frame):
        self._cleanup()
        exit(0)
        
    def _except_handler(self, e):
        if isinstance(e, APIConnectionError):
            pass
        else:
            print(f'Error: "{e}", retrying')
    

    def _check_custom_chat(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.enable_reasoning and self.reasoning_parser_obj is None:
            self.reasoning_parser_obj: ReasoningParser = ReasoningParserManager.get_reasoning_parser(self.reasoning_parser)(self.tokenizer)
            if not self.reasoning_parser_obj.think_start_token:
                raise ValueError(f"reasoning parser '{self.reasoning_parser}' should have think_start_token")
            if not self.reasoning_parser_obj.think_end_token:
                raise ValueError(f"reasoning parser '{self.reasoning_parser}' should have think_end_token")
    

    def _chat(self, item, reasoning_max_retry, param: ChatParam, ext_param: ChatExtraParam):
        self.last = (self.last + 1) % self.model_num
        reasoning_retry = 0
        while True:
            try:
                client, model = self.model_openais[self.last]
                completion = client.chat.completions.create(messages=item['message'], model=model, timeout=self.openai_timeout, extra_body=ext_param.__dict__, **param.__dict__)
                response = completion.choices[0].message.content
                if self.enable_reasoning:
                    reasoning = completion.choices[0].message.reasoning_content
                    if response is None or reasoning is None:
                        reasoning_retry += 1
                        if reasoning_retry <= reasoning_max_retry:
                            raise ValueError(f"{'no response' if response is None else 'no reasoning'}: {reasoning or response}, retry {reasoning_retry}")
                        item['response'] = None
                        item['reasoning'] = reasoning or response
                        return item
                break
            except Exception as e:
                self._except_handler(e)
        
        item['response'] = response
        if self.enable_reasoning:
            item['reasoning'] = reasoning
        return item

    def parallel_chat(self, messages_list: list[dict[str, str]], threads: int=20, return_dict: bool=False, reasoning_max_retry: int=10, param: ChatParam=None, ext_param: ChatExtraParam=None):
        """Executes parallel chat inference for a batch of message sequences. Supports both LLM and Large Reasoning Model (LRM).

        - If the model is not an LRM, the function returns a list of generated responses: `list[str]`.
        - If the model is an LRM, the function returns a list of tuples containing both the reasoning process 
        and the final reasoning result: `list[tuple[str, str]]`.
        - When `return_dict=True`, the function returns a list of dictionaries, where each dictionary contains:
            - response: The generated response.
            - reasoning: The reasoning process (only when using an LRM).

        Each message dictionary in `messages_list` must contain:
        - role: Must be one of `"user"`, `"assistant"`, or `"system"` (the `"system"` role is only allowed in the first turn).
        - content: The message text.

        Args:
            messages_list (list[dict[str, str]]): A list of message dictionaries, where each dictionary contains
                - role (str): The role of the speaker ("user", "assistant", or "system").
                - content (str): The content of the message.
            threads (int, optional): Number of threads for parallel execution. Defaults to 20.
            return_dict (bool, optional): Whether to return responses as a list of dictionaries. If `True`, responses are returned as `list[dict[str, str]]`. Defaults to False.
            reasoning_max_retry (int, optional): Maximum number of retries for reasoning. Only relevant when using an LRM. Defaults to 10.
            param (ChatParam, optional): Additional chat parameters for customization. Defaults to None.
            ext_param (ChatExtraParam, optional): Extended chat parameters for further customization. Defaults to None.

        Returns:
            (list[str] | list[tuple[str, str]] | list[dict[str, str]]): 
                - If the model is **not** an LRM, returns `list[str]` (generated responses).
                - If the model **is** an LRM, returns `list[tuple[str, str]]` (reasoning process and final response).
                - If `return_dict=True`, returns `list[dict[str, str]]`, where each dictionary contains:
                    - response (str): The generated response.
                    - reasoning (str, optional): The reasoning process (only available in LRM).
        """
        param = param if param != None else ChatParam()
        ext_param = ext_param if ext_param != None else ChatExtraParam()
        
        futures = []
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for i, msg in enumerate(messages_list):
                futures.append(executor.submit(self._chat, {'message': msg, 'idx': i}, reasoning_max_retry, param, ext_param))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                results.append(future.result())
        results = sorted(results, key=lambda x:x['idx'])
        
        if return_dict:
            return results
        if self.enable_reasoning:
            return [(r['reasoning'], r['response']) for r in results]
        return [r['response'] for r in results]
    

    def _gen(self, item, param: GenParam, ext_param: GenExtraParam, return_completion=False):
        self.last = (self.last + 1) % self.model_num
        while True:
            try:
                client, model = self.model_openais[self.last]
                completion = client.completions.create(prompt=item['prompt'], model=model, timeout=self.openai_timeout, extra_body=ext_param.__dict__, **param.__dict__)
                break
            except Exception as e:
                self._except_handler(e)
        item['response'] = completion.choices[0].text
        if return_completion:
            return item, completion
        return item

    def parallel_generate(self, prompt_list: list[str], threads: int=20, return_dict: bool=False, param: GenParam=None, ext_param: GenExtraParam=None):
        """Generates responses for a batch of prompts in parallel.

        - The function takes a list of prompts and generates corresponding responses for each prompt.
        - If `return_dict=True`, the function returns a list of dictionaries, where each dictionary contains:
            - response: The generated response for the prompt.

        Each item in `prompt_list` is a string containing a prompt for the model to generate a response.

        Args:
            prompt_list (list[str]): A list of prompts, where each string is a prompt to generate a response for.
            threads (int, optional): Number of threads for parallel execution. Defaults to 20.
            return_dict (bool, optional): Whether to return responses as a list of dictionaries. If `True`, responses are returned as `list[dict[str, str]]`. Defaults to False.
            param (GenParam, optional): Additional parameters for customization. Defaults to None.
            ext_param (GenExtraParam, optional): Extended parameters for further customization. Defaults to None.

        Returns:
            (list[str] | list[dict[str, str]]): 
                - Returns `list[str]`, where each string is a generated response corresponding to a prompt.
                - If `return_dict=True`, returns `list[dict[str, str]]`, where each dictionary contains:
                    - response (str): The generated response for the prompt.
        """

        param = param if param != None else GenParam()
        ext_param = ext_param if ext_param != None else GenExtraParam()
        
        futures = []
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for i, prompt in enumerate(prompt_list):
                futures.append(executor.submit(self._gen, {'prompt': prompt, 'idx': i}, param, ext_param))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                results.append(future.result())
        results = sorted(results, key=lambda x:x['idx'])
        
        if return_dict:
            return results
        return [r['response'] for r in results]
    
    def _chat_reasoning_length_ctrl(self, item, reasoning_max_retry, reasoning_max_len, reasoning_min_len, reasoning_scale, cut_by_sentence, param: GenParam, ext_param: GenExtraParam):
        def tlen(s):
            return len(self.tokenizer(s, add_special_tokens=False)['input_ids'])

        if reasoning_max_len and not reasoning_scale:
            reasoning_param = replace(param, max_tokens=reasoning_max_len+1)
        else:
            reasoning_param = param
        
        solution_start_token = self.reasoning_parser_obj.solution_start_token if hasattr(self.reasoning_parser_obj, 'solution_start_token') else ''
        added_think_end_token = self.reasoning_parser_obj.force_think_end_token if hasattr(self.reasoning_parser_obj, 'force_think_end_token') else self.reasoning_parser_obj.think_end_token
        
        reasoning_retry = 0
        while True:
            try:
                if (reasoning_max_len == 0 and not reasoning_scale) or reasoning_scale == 0:
                    prompt = item['prompt']
                    prompt += '' if prompt.endswith(self.reasoning_parser_obj.think_start_token) else self.reasoning_parser_obj.think_start_token
                    prompt += f'\n\n{added_think_end_token}\n\n{solution_start_token}'

                    response_param = replace(param, max_tokens=param.max_tokens - 2)
                    res_item, completion = self._gen({'prompt': prompt}, response_param, ext_param, return_completion=True)

                    if reasoning_retry < reasoning_max_retry:
                        if hasattr(self.reasoning_parser_obj, 'solution_end_token') and self.reasoning_parser_obj.solution_end_token not in res_item['response']:
                            raise ValueError('no solution_end_token')
                        if hasattr(self.reasoning_parser_obj, 'solution_start_token') and self.reasoning_parser_obj.solution_start_token in res_item['response']:
                            raise ValueError('has solution_start_token')
                        if self.reasoning_parser_obj.think_end_token in res_item['response']:
                            raise ValueError('has think_end_token')
                    
                    response = f"{self.reasoning_parser_obj.think_start_token}\n\n{added_think_end_token}\n\n{solution_start_token}{res_item['response']}"

                    reasoning, response = self.reasoning_parser_obj.extract_reasoning_content(response, completion)
                    if response is None:
                        raise ValueError(f'bad reasoning: {reasoning_retry}')
                    break
                
                res_item, completion = self._gen(item, reasoning_param, ext_param, return_completion=True)
                response = res_item['response']
                if self.reasoning_parser_obj.think_end_token not in response:
                    response += self.reasoning_parser_obj.think_end_token

                rean, resp = self.reasoning_parser_obj.extract_reasoning_content(response, completion)
                prompt = item['prompt'] + ('' if item['prompt'].endswith(self.reasoning_parser_obj.think_start_token) else self.reasoning_parser_obj.think_start_token)

                if reasoning_scale > 1:
                    reasoning_min_len = tlen(rean) * reasoning_scale
                if reasoning_scale < 1:
                    rean_token = self.tokenizer(rean, add_special_tokens=False)['input_ids']
                    rean_token = rean_token[:int(len(rean_token)*reasoning_scale)]
                    rean = self.tokenizer.decode(rean_token)

                if cut_by_sentence:
                    rean = rean.rsplit('.', 1)[0] + '.'

                reasoning = rean
                while reasoning_scale > 1 and tlen(reasoning) < reasoning_min_len:
                    prompt += f"{rean}{self.more_reasoning_prompt}"
                    if reasoning_max_len and not reasoning_scale:
                        reasoning_param = replace(param, max_tokens=reasoning_max_len - tlen(reasoning) - tlen(self.more_reasoning_prompt))
                    res_item, completion = self._gen({'prompt': prompt}, reasoning_param, ext_param, return_completion=True)
                    response = res_item['response']
                    rean, resp = self.reasoning_parser_obj.extract_reasoning_content(response, completion)
                    reasoning += f"{self.more_reasoning_prompt}{rean}"
                    if cut_by_sentence:
                        reasoning = reasoning.rsplit('.', 1)[0] + '.'

                prompt += f"{rean}\n\n{added_think_end_token}\n\n"
                
                if param.max_tokens != None:
                    response_param = replace(param, max_tokens=param.max_tokens - tlen(reasoning) - 2)
                res_item, completion = self._gen({'prompt': prompt}, response_param, ext_param, return_completion=True)
                response = res_item['response']
                response = f"{self.reasoning_parser_obj.think_start_token}{reasoning}{added_think_end_token}{response}"
                reasoning, response = self.reasoning_parser_obj.extract_reasoning_content(response, completion)

                if response is None or reasoning is None:
                    raise ValueError(f'bad reasoning: {reasoning_retry}')
                break
            except Exception as e:
                print(f'Error: {e}, retried {reasoning_retry} times')
                reasoning_retry += 1
                if reasoning_retry > reasoning_max_retry:
                    item['response'] = None
                    item['reasoning'] = reasoning or response
                    return item
        
        item['response'] = response
        item['reasoning'] = reasoning
        return item
    
    def _chat_reasoning(self, item, reasoning_max_retry, param: GenParam, ext_param: GenExtraParam):
        reasoning_retry = 0
        while True:
            try:
                item, completion = self._gen(item, param, ext_param, return_completion=True)
                response = item['response']
                reasoning, response = self.reasoning_parser_obj.extract_reasoning_content(response, completion)
                if response is None or reasoning is None:
                    reasoning_retry += 1
                    if reasoning_retry <= reasoning_max_retry:
                        raise ValueError(f'bad reasoning: {reasoning_retry}')
                    item['response'] = None
                    item['reasoning'] = reasoning or response
                    return item
                break
            except Exception as e:
                self._except_handler(e)
        
        item['response'] = response
        item['reasoning'] = reasoning
        return item
  

    def parallel_chat_custom(self, messages_list: list[dict[str, str]], threads: int=20, reasoning_max_retry: int=10, add_reasoning_prompt: bool=False, enable_length_ctrl: bool=False, reasoning_max_len: int=None, reasoning_min_len: int=0, reasoning_scale: float=None, cut_by_sentence: bool = False, return_dict=False, param: GenParam=None, ext_param: GenExtraParam=None):
        """Executes parallel chat inference with additional reasoning customization options. Supports both LLM and Large Reasoning Model (LRM).

        - If the model is not an LRM, the function returns a list of generated responses: `list[str]`.
        - If the model is an LRM, the function returns a list of tuples containing both the reasoning process 
          and the final reasoning result: `list[tuple[str, str]]`.
        - When `return_dict=True`, the function returns a list of dictionaries, where each dictionary contains:
            - response: The generated response.
            - reasoning: The reasoning process (only when using an LRM).

        Each message dictionary in `messages_list` must contain:
        - role: Must be one of `"user"`, `"assistant"`, or `"system"` (the `"system"` role is only allowed in the first turn).
        - content: The message text.

        If `reasoning_scale` is set, `reasoning_max_len` and `reasoning_min_len` are ignored. In this case, 
        the full reasoning content is generated first and then truncated or extended based on the scale value.

        Args:
            messages_list (list[dict[str, str]]): A list of message dictionaries, where each dictionary contains
                - role (str): The role of the speaker ("user", "assistant", or "system").
                - content (str): The content of the message.
            threads (int, optional): Number of threads for parallel execution. Defaults to 20.
            reasoning_max_retry (int, optional): Maximum number of retries for reasoning. Only relevant when using an LRM. Defaults to 10.
            add_reasoning_prompt (bool, optional): Whether to explicitly add a reasoning prompt to guide the model. Defaults to False.
            enable_length_ctrl (bool, optional): Whether to enable reasoning length control. Defaults to False.
            reasoning_max_len (int, optional): Maximum length for the reasoning content. Ignored if `reasoning_scale` is set. Defaults to None.
            reasoning_min_len (int, optional): Minimum length for the reasoning content. Ignored if `reasoning_scale` is set. Defaults to 0.
            reasoning_scale (float, optional): Scaling factor for reasoning content. If set, `reasoning_max_len` and `reasoning_min_len` are ignored, 
                                               and the reasoning process is either truncated or extended accordingly. Defaults to None.
            cut_by_sentence (bool, optional): Whether to truncate reasoning content at sentence boundaries when using `reasoning_scale`. Defaults to False.
            return_dict (bool, optional): Whether to return responses as a list of dictionaries. If `True`, responses are returned as `list[dict[str, str]]`. Defaults to False.
            param (GenParam, optional): Additional chat parameters for customization. Defaults to None.
            ext_param (GenExtraParam, optional): Extended chat parameters for further customization. Defaults to None.

        Returns:
            (list[str] | list[tuple[str, str]] | list[dict[str, str]]): 
                - If the model is **not** an LRM, returns `list[str]` (generated responses).
                - If the model **is** an LRM, returns `list[tuple[str, str]]` (reasoning process and final response).
                - If `return_dict=True`, returns `list[dict[str, str]]`, where each dictionary contains:
                    - response (str): The generated response.
                    - reasoning (str, optional): The reasoning process (only available in LRM).
        """

        self._check_custom_chat()

        param = param if param != None else GenParam()
        ext_param = ext_param if ext_param != None else GenExtraParam()
        prompt_list = self.tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
        
        if not self.enable_reasoning:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                for i, prompt in enumerate(prompt_list):
                    futures.append(executor.submit(self._gen, {'prompt': prompt, 'idx': i}, param, ext_param))
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    results.append(future.result())
            results = sorted(results, key=lambda x:x['idx'])
            if return_dict:
                return results
            return [r['response'] for r in results]
        

        if add_reasoning_prompt:
            prompt_list = [p if p.endswith(self.reasoning_parser_obj.think_start_token) else (p + self.reasoning_parser_obj.think_start_token) for p in prompt_list]

        if enable_length_ctrl:
            if reasoning_max_len is not None and reasoning_max_len < reasoning_min_len:
                raise ValueError(f"reasoning_min_len({reasoning_min_len}) should less than reasoning_max_len({reasoning_max_len})")
            if reasoning_scale:
                print("Setted reasoning_scale, ignore reasoning_max_len and reasoning_min_len")
            
            futures = []
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                for i, prompt in enumerate(prompt_list):
                    futures.append(executor.submit(self._chat_reasoning_length_ctrl, {'prompt': prompt, 'idx': i}, reasoning_max_retry, reasoning_max_len, reasoning_min_len, reasoning_scale, cut_by_sentence, param, ext_param))
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    results.append(future.result())
        else:
            futures = []
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                for i, prompt in enumerate(prompt_list):
                    futures.append(executor.submit(self._chat_reasoning, {'prompt': prompt, 'idx': i}, reasoning_max_retry, param, ext_param))
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    results.append(future.result())
        
        results = sorted(results, key=lambda x:x['idx'])
        
        if return_dict:
            return results
        return [(r['reasoning'], r['response']) for r in results]
    
            
    def parallel_chat_force_reasoning_content(self, messages_list: list[dict[str, str]], reasoning_content: list[str], threads: int=20, reasoning_scale: float=None, cut_by_sentence: bool = False, return_dict: bool=False, param: GenParam=None, ext_param: GenExtraParam=None):
        """Executes parallel chat inference with forced reasoning content. This function only supports Large Reasoning Model (LRM).

        - The function returns a list of tuples containing both the provided reasoning process 
          and the final reasoning result: `list[tuple[str, str]]`.
        - When `return_dict=True`, the function returns a list of dictionaries, where each dictionary contains:
            - response: The generated response.
            - reasoning: The provided reasoning process.

        Each message dictionary in `messages_list` must contain:
        - role: Must be one of `"user"`, `"assistant"`, or `"system"` (the `"system"` role is only allowed in the first turn).
        - content: The message text.

        The `reasoning_content` parameter explicitly defines the reasoning process for each input sequence, ensuring 
        that the model follows the provided reasoning when generating responses.

        If `reasoning_scale` is set, the provided reasoning content is adjusted accordingly—either truncated or extended. 
        When `cut_by_sentence=True`, truncation respects sentence boundaries.

        Args:
            messages_list (list[dict[str, str]]): A list of message dictionaries, where each dictionary contains
                - role (str): The role of the speaker ("user", "assistant", or "system").
                - content (str): The content of the message.
            reasoning_content (list[str]): A list of predefined reasoning processes corresponding to each input sequence.
            threads (int, optional): Number of threads for parallel execution. Defaults to 20.
            reasoning_scale (float, optional): Scaling factor for adjusting reasoning content length. If set, 
                                               the reasoning content is truncated or extended accordingly. Defaults to None.
            cut_by_sentence (bool, optional): Whether to truncate reasoning content at sentence boundaries when using `reasoning_scale`. Defaults to False.
            return_dict (bool, optional): Whether to return responses as a list of dictionaries. If `True`, responses are returned as `list[dict[str, str]]`. Defaults to False.
            param (GenParam, optional): Additional chat parameters for customization. Defaults to None.
            ext_param (GenExtraParam, optional): Extended chat parameters for further customization. Defaults to None.

        Returns:
            (list[tuple[str, str]] | list[dict[str, str]]): 
                - Returns `list[tuple[str, str]]` (reasoning process and final response).
                - If `return_dict=True`, returns `list[dict[str, str]]`, where each dictionary contains:
                    - response (str): The generated response.
                    - reasoning (str): The provided reasoning process.
        """

        
        self._check_custom_chat()
        
        if not self.enable_reasoning:
            raise ValueError("parallel_chat_force_reasoning_content method only support reasoning model!")

        if reasoning_scale is not None:
            if reasoning_scale > 1:
                raise ValueError("Only support to cut off exist reasoning content")
            
            reasoning_content_tokens = self.tokenizer(reasoning_content, add_special_tokens=False)['input_ids']
            reasoning_content_tokens = [r[:int(len(r) * reasoning_scale)] for r in reasoning_content_tokens]
            reasoning_content = [self.tokenizer.decode(r) for r in reasoning_content_tokens]
            if cut_by_sentence: 
                reasoning_content = [r.rsplit('.', 1)[0]+'.' for r in reasoning_content]
        

        param = param if param != None else GenParam()
        ext_param = ext_param if ext_param != None else GenExtraParam()
        
        solution_start_token = '\n\n' + self.reasoning_parser_obj.solution_start_token if hasattr(self.reasoning_parser_obj, 'solution_start_token') else ''
        
        prompt_list = self.tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
        prompt_list = [f'{p}{"" if p.endswith(self.reasoning_parser_obj.think_start_token) else self.reasoning_parser_obj.think_start_token}\n\n{r}\n\n{self.reasoning_parser_obj.think_end_token}{solution_start_token}' for p, r in zip(prompt_list, reasoning_content)]
        
        futures = []
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for i, prompt in enumerate(prompt_list):
                futures.append(executor.submit(self._gen, {'prompt': prompt, 'idx': i}, param, ext_param))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                results.append(future.result())
        results = sorted(results, key=lambda x:x['idx'])
        if return_dict:
            return results
        return [(rean, r['response']) for rean, r in zip(reasoning_content, results)]
        
