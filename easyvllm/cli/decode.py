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

import os
from typing import Annotated, List, Optional, Literal, Union, Tuple

import fire
import torch
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import yaml

from easyvllm import InferenceModel, ChatParam, ChatExtraParam, GenParam, GenExtraParam
from easyvllm.util import read_file, save_file


def decode_query(model: InferenceModel, query_list: Union[list[str], list[list[str]]], system: str=None, threads: int=20, reasoning_max_retry: int=10, param: ChatParam = None, ext_param: ChatExtraParam = None) -> Union[list[str], list[list[str]]]:
    msg_list = [[] for _ in query_list]
    if system:
        msg_list = [[{'role': 'system', 'content': system}] for _ in query_list]

    if type(query_list[0]) == str:
        msg_list = [msg + [{'role': 'user', 'content': d}] for msg, d in zip(msg_list, query_list)]
        responses = model.parallel_chat(msg_list, threads, reasoning_max_retry=reasoning_max_retry, param=param, ext_param=ext_param)
        return responses
    
    response_list = []
    for i in range(len(query_list[0])):
        msg_list = [msg + [{'role': 'user', 'content': d[i]}] for msg, d in zip(msg_list, query_list)]
        responses = model.parallel_chat(msg_list, threads, reasoning_max_retry=reasoning_max_retry, param=param, ext_param=ext_param)
        response_list.append(responses)
        if type(query_list[0]) == str:
            msg_list = [msg + [{'role': 'assistant', 'content': r}] for msg, r in zip(msg_list, responses)]
        else:
            msg_list = [msg + [{'role': 'assistant', 'content': r[1]}] for msg, r in zip(msg_list, responses)]
    response_list = [list(i) for i in zip(*response_list)]
    return response_list


def decode_query_reasoning_ctrl(model: InferenceModel, query_list: Union[list[str], list[list[str]]], system: str=None, threads: int=20, reasoning_max_retry=10, add_reasoning_prompt: bool = False, enable_length_ctrl: bool = False, reasoning_max_len: int=None, reasoning_min_len: int=0, reasoning_scale: float=None, cut_by_sentence=False, param: ChatParam = None, ext_param: ChatExtraParam = None) -> Union[list[str], list[list[str]]]:
    msg_list = [[] for _ in query_list]
    if system:
        msg_list = [[{'role': 'system', 'content': system}] for _ in query_list]

    if type(query_list[0]) == str:
        msg_list = [msg + [{'role': 'user', 'content': d}] for msg, d in zip(msg_list, query_list)]
        responses = model.parallel_chat_custom(msg_list, threads, reasoning_max_retry, add_reasoning_prompt, enable_length_ctrl, reasoning_max_len, reasoning_min_len, reasoning_scale, cut_by_sentence, param=param, ext_param=ext_param)
        return responses
    
    response_list = []
    for i in range(len(query_list[0])):
        msg_list = [msg + [{'role': 'user', 'content': d[i]}] for msg, d in zip(msg_list, query_list)]
        responses = model.parallel_chat_custom(msg_list, threads, reasoning_max_retry, add_reasoning_prompt, enable_length_ctrl, reasoning_max_len, reasoning_min_len, reasoning_scale, cut_by_sentence, param=param, ext_param=ext_param)
        response_list.append(responses)
        msg_list = [msg + [{'role': 'assistant', 'content': r[1]}] for msg, r in zip(msg_list, responses)]
    response_list = [list(i) for i in zip(*response_list)]
    return response_list


def decode_query_force_reasoning_content(model: InferenceModel, query_list: Union[list[str], list[list[str]]], reasoning_content_lines: Union[list[str], list[list[str]]], system: str=None, threads: int=20, reasoning_scale: float=None, cut_by_sentence: bool = False, param: ChatParam = None, ext_param: ChatExtraParam = None) -> Union[list[str], list[list[str]]]:
    msg_list = [[] for _ in query_list]
    if system:
        msg_list = [[{'role': 'system', 'content': system}] for _ in query_list]

    if type(query_list[0]) == str:
        assert type(reasoning_content_lines[0]) == str
        msg_list = [msg + [{'role': 'user', 'content': d}] for msg, d in zip(msg_list, query_list)]
        responses = model.parallel_chat_force_reasoning_content(msg_list, reasoning_content_lines, threads, reasoning_scale, cut_by_sentence, param=param, ext_param=ext_param)
        return responses
    
    response_list = []
    for i in range(len(query_list[0])):
        msg_list = [msg + [{'role': 'user', 'content': d[i]}] for msg, d in zip(msg_list, query_list)]
        reasoning_content = [r[i] for r in reasoning_content_lines]
        responses = model.parallel_chat_force_reasoning_content(msg_list, reasoning_content, threads, reasoning_scale, cut_by_sentence, param=param, ext_param=ext_param)
        response_list.append(responses)
        msg_list = [msg + [{'role': 'assistant', 'content': r}] for msg, r in zip(msg_list, responses)]
    response_list = [list(i) for i in zip(*response_list)]
    return response_list

SUPPROT_DECODE_TYPE = ['query', 'query_reasoning_ctrl', 'query_force_reasoning_content']
    
def decode(
    model_path: str,
    file_path: str,
    decode_type: Literal['query', 'query_reasoning_ctrl', 'query_force_reasoning_content'],
    save_path: str,
    query_keys: Union[str, tuple[str]],
    response_keys: Union[str, tuple[str]] = None,
    reasoning_keys: Union[str, tuple[str]] = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    model_num: int = None,
    port: int = 50000,
    max_model_len: int = None,
    show_vllm_log: bool = True,
    openai_timeout: int = 30,
    threads: int=20,
    enable_reasoning: bool = False,
    reasoning_parser: str = 'deepseek_r1',
    system_prompt_file: str = None,
    chat_template_file: str = None,
    max_new_tokens = 8192,
    device_ids: str = None,
    reasoning_max_retry: int = 10,
    add_reasoning_prompt: bool = False,
    enable_length_ctrl: bool = False,
    reasoning_max_len: int = None,
    reasoning_min_len: int = 0,
    reasoning_scale: float = None,
    cut_by_sentence: bool = False,
    force_reasoning_content_keys: Union[str, tuple[str]] = None,
    overwrite: bool = False,
    use_ray: bool = False,
    ray_host_ip: str = None,
    enforce_eager: bool = False,
    gpu_memory_utilization: float = 0.95,
):
    
    if decode_type not in SUPPROT_DECODE_TYPE:
        raise ValueError(f"unsupport decode_type: '{decode_type}', support types: {SUPPROT_DECODE_TYPE}")
    if decode_type in ['query', 'query_reasoning_ctrl', 'query_force_reasoning_content'] and not query_keys:
        raise ValueError(f"'{decode_type}' method requires query_keys")

    df = read_file(file_path)

    system = None
    if system_prompt_file:
        with open(system_prompt_file) as f:
            system = f.read()

    if device_ids:
        if type(device_ids) != tuple:
            device_ids = [device_ids]
        device_ids = [int(d) for d in device_ids]
        
        if model_num is not None and len(device_ids) < tensor_parallel_size * model_num:
            raise ValueError(f"moel_num({model_num}) * tensor_parallel_size({tensor_parallel_size}) is bigger than len(device_ids)({len(device_ids)})")
        if len(device_ids) < tensor_parallel_size:
            raise ValueError(f"tensor_parallel_size({tensor_parallel_size}) is bigger than len(device_ids)({len(device_ids)})")
        
        if model_num:
            device_ids = device_ids[:tensor_parallel_size * model_num]
    else:
        if not model_num:
            model_num = torch.cuda.device_count() // tensor_parallel_size
        device_ids = list(range(min(torch.cuda.device_count(), model_num * tensor_parallel_size)))
    model = InferenceModel(
        model_path=model_path,
        device_ids=device_ids,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        port=port,
        max_model_len=max_model_len,
        show_vllm_log=show_vllm_log,
        openai_timeout=openai_timeout,
        enable_reasoning=enable_reasoning,
        reasoning_parser=reasoning_parser,
        chat_template=chat_template_file,
        use_ray=use_ray,
        ray_host_ip=ray_host_ip,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    if decode_type in ['query', 'query_reasoning_ctrl', 'query_force_reasoning_content']:
        query_keys = [query_keys] if type(query_keys) != tuple else list(query_keys)
        response_keys = ([response_keys] if type(response_keys) != tuple else list(response_keys)) if response_keys else ['resp_'+q for q in query_keys]
        reasoning_keys = ([reasoning_keys] if type(reasoning_keys) != tuple else list(reasoning_keys)) if reasoning_keys else ['reas_'+q for q in query_keys]

        query_lines = df[query_keys].values.tolist()
        if decode_type == 'query':
            responses = decode_query(model, query_lines, system, threads * model_num, reasoning_max_retry, param=ChatParam(max_completion_tokens=max_new_tokens))
        
        elif decode_type == 'query_reasoning_ctrl':
            if not enable_reasoning:
                raise ValueError("set enable_reasoning to True when using query_reasoning_ctrl")
            responses = decode_query_reasoning_ctrl(model, query_lines, system, threads * model_num, reasoning_max_retry, add_reasoning_prompt, enable_length_ctrl, reasoning_max_len, reasoning_min_len, reasoning_scale, cut_by_sentence, param=GenParam(max_tokens=max_new_tokens))
        
        elif decode_type == 'query_force_reasoning_content':
            if not enable_reasoning:
                raise ValueError("set enable_reasoning to True when using query_force_reasoning_content")
            if force_reasoning_content_keys is None:
                raise ValueError("set force_reasoning_content_keys when using query_force_reasoning_content")
            
            force_reasoning_content_keys = [force_reasoning_content_keys] if type(force_reasoning_content_keys) != tuple else list(force_reasoning_content_keys)
            reasoning_keys = force_reasoning_content_keys
            
            reasoning_content_lines = df[force_reasoning_content_keys].values.tolist()
            responses = decode_query_force_reasoning_content(model, query_lines, reasoning_content_lines, system, threads * model_num, reasoning_scale, cut_by_sentence, param=GenParam(max_tokens=max_new_tokens))
        
        if enable_reasoning:
            if not isinstance(responses[0], list):
                reasonings = [r[0] for r in responses]
                responses = [r[1] for r in responses]
            else:
                reasonings = [[i[0] for i in r] for r in responses]
                responses = [[i[1] for i in r] for r in responses]
            resp_df = pd.DataFrame(responses, columns=response_keys)
            resn_df = pd.DataFrame(reasonings, columns=reasoning_keys)
            
            if overwrite:
                df = df.drop(response_keys + reasoning_keys, axis=1, errors='ignore')
            df = df.join(resp_df).join(resn_df)
        else:
            resp_df = pd.DataFrame(responses, columns=response_keys)
            if overwrite:
                df = df.drop(response_keys, axis=1, errors='ignore')
            df = df.join(resp_df)
        
    
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    save_file(df, save_path)


class TaskConfig(BaseModel):
    file_path: str = Field(...)
    decode_type: Literal['query', 'query_reasoning_ctrl', 'query_force_reasoning_content']
    save_path: str
    query_keys: Union[str, tuple[str]]
    response_keys: Union[str, tuple[str]] = None
    reasoning_keys: Union[str, tuple[str]] = None
    threads: int=20
    system_prompt_file: str = None
    max_new_tokens: int = 8192
    reasoning_max_retry: int = 10
    add_reasoning_prompt: bool = False
    enable_length_ctrl: bool = False
    reasoning_max_len: int = None
    reasoning_min_len: int = 0
    reasoning_scale: float = None
    cut_by_sentence: bool = False
    force_reasoning_content_keys: Union[str, tuple[str]] = None
    overwrite: bool = False


def load_config_list(tasks_yaml_path):
    config_list = []
    with open(tasks_yaml_path, 'r') as file:
        task_config =  yaml.safe_load(file)
    config_list = [TaskConfig(**cfg) for cfg in task_config]
    print('*' * 100)
    print(f'Success loaded {len(config_list)} configs!')
    print('*' * 100)
    return config_list


def decode_multi_task(
    model_path: str,
    tasks_yaml_path: str,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_model_len: int = None,
    model_num: int = None,
    port: int = 50000,
    openai_timeout: int = 30,
    enable_reasoning: bool = False,
    chat_template_file: str = None,
    reasoning_parser: str = 'deepseek_r1',
    show_vllm_log: bool = True,
    device_ids: str = None,
    use_ray: bool = False,
    ray_host_ip: str = None,
    enforce_eager: bool = False,
    gpu_memory_utilization: float = 0.95,
):

    config_list: list[TaskConfig] = load_config_list(tasks_yaml_path)

    if device_ids:
        if type(device_ids) != tuple:
            device_ids = [device_ids]
        device_ids = [int(d) for d in device_ids]
        
        if model_num is not None and len(device_ids) < tensor_parallel_size * model_num:
            raise ValueError(f"moel_num({model_num}) * tensor_parallel_size({tensor_parallel_size}) is bigger than len(device_ids)({len(device_ids)})")
        if len(device_ids) < tensor_parallel_size:
            raise ValueError(f"tensor_parallel_size({tensor_parallel_size}) is bigger than len(device_ids)({len(device_ids)})")
        
        if model_num:
            device_ids = device_ids[:tensor_parallel_size * model_num]
    else:
        if not model_num:
            model_num = torch.cuda.device_count() // tensor_parallel_size
        device_ids = list(range(min(torch.cuda.device_count(), model_num * tensor_parallel_size)))
    
    model = InferenceModel(
        model_path=model_path,
        device_ids=device_ids,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        port=port,
        max_model_len=max_model_len,
        show_vllm_log=show_vllm_log,
        openai_timeout=openai_timeout,
        enable_reasoning=enable_reasoning,
        reasoning_parser=reasoning_parser,
        chat_template=chat_template_file,
        use_ray=use_ray,
        ray_host_ip=ray_host_ip,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    for task_i, config in enumerate(config_list):
        print(f'Start task {task_i}')
        print(f'Task config: {config.model_dump_json(indent=2)}')


        if config.decode_type not in SUPPROT_DECODE_TYPE:
            raise ValueError(f"unsupport decode_type: '{config.decode_type}', support types: {SUPPROT_DECODE_TYPE}")
        if config.decode_type in ['query', 'query_reasoning_ctrl', 'query_force_reasoning_content'] and not config.query_keys:
            raise ValueError(f"'{config.decode_type}' method requires query_keys")

        df = read_file(config.file_path)

        system = None
        if config.system_prompt_file:
            with open(config.system_prompt_file) as f:
                system = f.read()

        if config.decode_type in ['query', 'query_reasoning_ctrl', 'query_force_reasoning_content']:
            query_keys = [config.query_keys] if type(config.query_keys) != tuple else list(config.query_keys)
            response_keys = ([config.response_keys] if type(config.response_keys) != tuple else list(config.response_keys)) if config.response_keys else ['resp_'+q for q in query_keys]
            reasoning_keys = ([config.reasoning_keys] if type(config.reasoning_keys) != tuple else list(config.reasoning_keys)) if config.reasoning_keys else ['reas_'+q for q in query_keys]

            query_lines = df[query_keys].values.tolist()
            if config.decode_type == 'query':
                responses = decode_query(model, query_lines, system, config.threads * model_num, config.reasoning_max_retry, param=ChatParam(max_completion_tokens=config.max_new_tokens))
            
            elif config.decode_type == 'query_reasoning_ctrl':
                if not enable_reasoning:
                    raise ValueError("set enable_reasoning to True when using query_reasoning_ctrl")
                responses = decode_query_reasoning_ctrl(model, query_lines, system, config.threads * model_num, config.reasoning_max_retry, config.add_reasoning_prompt, config.enable_length_ctrl, config.reasoning_max_len, config.reasoning_min_len, config.reasoning_scale, config.cut_by_sentence, param=GenParam(max_tokens=config.max_new_tokens))
            
            elif config.decode_type == 'query_force_reasoning_content':
                if not enable_reasoning:
                    raise ValueError("set enable_reasoning to True when using query_force_reasoning_content")
                if config.force_reasoning_content_keys is None:
                    raise ValueError("set force_reasoning_content_keys when using query_force_reasoning_content")
                
                force_reasoning_content_keys = [config.force_reasoning_content_keys] if type(config.force_reasoning_content_keys) != tuple else list(config.force_reasoning_content_keys)
                reasoning_keys = force_reasoning_content_keys
                
                reasoning_content_lines = df[force_reasoning_content_keys].values.tolist()
                responses = decode_query_force_reasoning_content(model, query_lines, reasoning_content_lines, system, config.threads * model_num, config.reasoning_scale, config.cut_by_sentence, param=GenParam(max_tokens=config.max_new_tokens))
            
            if enable_reasoning:
                if not isinstance(responses[0], list):
                    reasonings = [r[0] for r in responses]
                    responses = [r[1] for r in responses]
                else:
                    reasonings = [[i[0] for i in r] for r in responses]
                    responses = [[i[1] for i in r] for r in responses]
                resp_df = pd.DataFrame(responses, columns=response_keys)
                resn_df = pd.DataFrame(reasonings, columns=reasoning_keys)
                
                if config.overwrite:
                    df = df.drop(response_keys + reasoning_keys, axis=1, errors='ignore')
                df = df.join(resp_df).join(resn_df)
            else:
                resp_df = pd.DataFrame(responses, columns=response_keys)
                if config.overwrite:
                    df = df.drop(response_keys, axis=1, errors='ignore')
                df = df.join(resp_df)
            
        os.makedirs(os.path.split(config.save_path)[0], exist_ok=True)
        save_file(df, config.save_path)

    print(f'Finished {len(config_list)} tasks!!')

