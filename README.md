<h1 align="center">
  easyvllm
</h1>
<h3 align="center">
Easy to use lightweight vllm tool with special support for Large Reasoning Model(LRM).
</h3>

<p align="center">
| <a href="#documentation"><b>Documentation</b></a> | <a href=""><b>Paper</b></a> |
</p>

## Features

We have encapsulated vLLM and utilize the OpenAI client to call `vllm serve`, enabling multiple models to be loaded simultaneously in a multi-GPU environment when working with small models. This approach maximizes batch inference performance. Additionally, we support various LRM inference methods, including:  

- Direct model invocation  
- Inference length control  
- Customizable inference content  

We also support multi-node Ray cluster integration.

## Installation

### Create a new Python environment

To use easyvllm, we support to create a new Python environment.

You can create a new Python environment using `conda`:

```shell
conda create -n easyvllm python=3.11 -y
conda activate easyvllm
```

### Install easyvllm

You can install easyvllm from PyPI:

```shell
pip install easyvllm
```

or you can install in local:

```shell
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd easyvllm
pip install -e .
```

### Add Reasoning Parsers to Vllm

We implemented several reasoning parsers for [vllm](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html#how-to-support-a-new-reasoning-model), including Openthinker and simplescaling. 

To enable reasoning outputs, parsers need to be added in your local vllm package. You can add easily with following steps:

1. Find the vllm package in your Python environment. It should be in the path like `${conda_path}/envs/{your_env_name}/lib/python3.xx/site-packages/vllm`.

2. Find `reasoning_parsers` folder. It should be in `${conda_path}/envs/{your_env_name}/lib/python3.xx/site-packages/vllm/entrypoints/openai/reasoning_parsers`.

3. Add `from easyvllm.parsers import *` in `__init__.py` file. After the modification, the content should be as follows:

```python
# SPDX-License-Identifier: Apache-2.0

from .abs_reasoning_parsers import ReasoningParser, ReasoningParserManager
from .deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from easyvllm.parsers import *

__all__ = [
    "ReasoningParser", "ReasoningParserManager", "DeepSeekR1ReasoningParser"
]
```

## Quick Start

### InferenceModel

Using the encapsulated model is simple. Just import `InferenceModel` and provide the model path along with relevant parameters for initialization:  

```python
from easyvllm import InferenceModel  

model = InferenceModel(model_path="your_model_path")
```

If you are initializing a LRM, please set `enable_reasoning` flag and `reasoning_parser`:

```python
from easyvllm import InferenceModel  

model = InferenceModel(model_path="your_LRM_path", enable_reasoning=True, reasoning_parser='reasoning_parser_name')
```

We support following reasoning parsers now:

- `deepseek_r1`
- `openthinker`
- `simplescaling`

The `deepseek_r1` parser is implemented by vllm, along with its support for parsing `QWQ` inference output. The rest of the parser is in `./easyvllm/parsers` and needs to be [added to the vllm arser path](#add-reasoning-parsers-to-vllm) to be used.

Once initialized, you can use the following methods for inference:

- `parallel_chat`: Dialogue generation (supports LRM)
- `parallel_generate`: Direct text generation (supports LRM)
- `parallel_chat_custom`: Customizable dialogue generation (supports LRM)
- `parallel_chat_force_reasoning_content`: Control reasoning content (for LRM)

These methods return a list of generated content in order. For LRM, they return a list of `(inference content, inference result)` tuples.

### CLI

We provide a CLI interface for users to conveniently invoke the model for inference.  

- decode

You can use the `easyvllm decode` command for inference with the following basic format:  

```shell
easyvllm decode --model_path your_model_path --file_path input.json --decode_type query --save_path output.json --query_keys prompt
```  

This will read queries from `input.json` and save results to `output.json`, the key of query in `input.json` is `prompt`.

For reasoning length control, enable `query_reasoning_ctrl` and set `reasoning_max_len`:  

```shell
easyvllm decode --model_path your_model_path --file_path input.json --decode_type query_reasoning_ctrl --save_path output.json --query_keys prompt --enable_reasoning True --enable_length_ctrl True --reasoning_max_len 500
```  

For control reasoning content, specifying `force_reasoning_content_keys`:  

```shell
easyvllm decode --model_path your_model_path --file_path input.json --decode_type query_force_reasoning_content --save_path output.json --query_keys prompt --enable_reasoning True --force_reasoning_content_keys reasoning
```  

For more details on available parameters, refer to the [decode parameters](#decode).

- decode multask

We provide a **multi-task decoding** feature via the `easyvllm decode multask` command, allowing users to execute multiple inference tasks in parallel by specifying a YAML configuration file.

To run multiple inference tasks, use the following command:

```bash
easyvllm decode multask --model_path your_model_path --tasks_yaml_path tasks_config.yaml
```

Here is a simple example of `tasks_config.yaml` file:

```yaml
- file_path: "input1.json"  
  decode_type: "query"  
  save_path: "output1.json"  
  query_keys: "input_text"  
  response_keys: "generated_text"  
  threads: 20  
  max_new_tokens: 8192  
  enable_reasoning: true  
  enable_length_ctrl: false  
  reasoning_max_len: 500   

- file_path: "input2.json"  
  decode_type: "query_reasoning_ctrl"  
  save_path: "output2.json"  
  query_keys: "input_text"  
  response_keys: "generated_text"  
  reasoning_keys: "reasoning_steps"  
  threads: 20  
  enable_reasoning: true  
  reasoning_max_retry: 10  
  force_reasoning_content_keys: "reasoning_text"  
  overwrite: true  
```

For more details on available parameters, refer to the [decode multask parameters](#decode-multask) and [task yaml configuration](#task-yaml-configuration)


## Documentation

### InferenceModel

#### __init__

- Required Parameters
  - `model_path (str)`: Path to the model.

- Optional Parameters  
  - `device_ids (list[int], optional)`: List of CUDA device IDs. If specified, the model runs on selected devices. The number of models initialized is determined by `len(device_ids) // tensor_parallel_size`. Defaults to `None`.  
  - `tensor_parallel_size (int, optional)`: Tensor parallelism size for initializing the vLLM model. Defaults to `1`.  
  - `pipeline_parallel_size (int, optional)`: Pipeline parallelism size for initializing the vLLM model. Defaults to `1`.  
  - `port (int, optional)`: Base port for the vLLM service. To avoid conflicts when initializing multiple models, the actual port used is `port + device_id`. Defaults to `50000`.  
  - `max_model_len (int, optional)`: Maximum model input length. Defaults to `None`.  
  - `show_vllm_log (bool, optional)`: Whether to display vLLM logs. Set to `False` to disable log output. Defaults to `True`.  
  - `openai_timeout (int, optional)`: Timeout value (in seconds) for OpenAI client operations. Increase if the reasoning content is too large. Defaults to `30`.  
  - `enable_reasoning (bool, optional)`: Enable large reasoning model (LRM) support. Defaults to `False`.  
  - `reasoning_parser (str, optional)`: Parser used for processing reasoning content. Defaults to `'deepseek_r1'`.  
  - `chat_template (str, optional)`: Path to a chat template file for initializing the vLLM model. Defaults to `None`.  
  - `use_ray (bool, optional)`: Enable distributed inference across multiple nodes using **Ray**. Requires a Ray cluster to be created first. Defaults to `False`.  
  - `ray_host_ip (str, optional)`: Host IP address of the running Ray cluster. Required when `use_ray=True`. Defaults to `None`.  


#### parallel_chat

- Required Parameters
  - `messages_list (list[dict[str, str]])`: A list of message dictionaries, where each dictionary contains:
    - `role (str)`: The role of the speaker, must be one of `"user"`, `"assistant"`, or `"system"` (the `"system"` role is only allowed in the first turn).
    - `content (str)`: The content of the message.

- Optional Parameters
  - `threads (int, default=20)`: Number of threads for parallel execution.
  - `return_dict (bool, default=False)`: Whether to return responses as a list of dictionaries. If `True`, responses are returned as `list[dict[str, str]]`.
  - `reasoning_max_retry (int, default=10)`: Maximum number of retries for reasoning. Only relevant when using an LRM.
  - `param (ChatParam, default=None)`: Additional chat parameters for customization.
  - `ext_param (ChatExtraParam, default=None)`: Extended chat parameters for further customization.

- Returns
  - `list[str]`: If the model is **not** an LRM, returns a list of generated responses.
  - `list[tuple[str, str]]`: If the model **is** an LRM, returns a list of tuples containing both the reasoning process and the final response.
  - `list[dict[str, str]]`: If `return_dict=True`, returns a list of dictionaries, where each dictionary contains:
    - `response (str)`: The generated response.
    - `reasoning (str, optional)`: The reasoning process (only available in LRM mode).

#### parallel_generate

- Required Parameters
  - `prompt_list (list[str])`: A list of prompts, where each string is a prompt to generate a response for.

- Optional Parameters
  - `threads (int, optional, default=20)`: Number of threads for parallel execution.
  - `return_dict (bool, optional, default=False)`: Whether to return responses as a list of dictionaries. If `True`, responses are returned as `list[dict[str, str]]`, where each dictionary contains:
    - `response (str)`: The generated response for the prompt.
  - `param (GenParam, optional, default=None)`: Additional parameters for customization.
  - `ext_param (GenExtraParam, optional, default=None)`: Extended parameters for further customization.

- Returns
  - `list[str]`: A list of generated responses corresponding to each prompt in `prompt_list`.
  - `list[dict[str, str]]`: If `return_dict=True`, returns a list of dictionaries, where each dictionary contains:
    - `response (str)`: The generated response for the prompt.


#### parallel_chat_custom

- Required Parameters
  - `messages_list (list[dict[str, str]])`: A list of message dictionaries, where each dictionary contains:
    - `role (str)`: The role of the speaker, must be one of `"user"`, `"assistant"`, or `"system"` (the `"system"` role is only allowed in the first turn).
    - `content (str)`: The content of the message.

- Optional Parameters
  - `threads (int, default=20)`: Number of threads for parallel execution.
  - `reasoning_max_retry (int, default=10)`: Maximum number of retries for reasoning. Only relevant when using an LRM.
  - `add_reasoning_prompt (bool, default=False)`: Whether to explicitly add a reasoning prompt to guide the model.
  - `enable_length_ctrl (bool, default=False)`: Whether to enable reasoning length control.
  - `reasoning_max_len (int, default=None)`: Maximum length for the reasoning content. Ignored if `reasoning_scale` is set.
  - `reasoning_min_len (int, default=0)`: Minimum length for the reasoning content. Ignored if `reasoning_scale` is set.
  - `reasoning_scale (float, default=None)`: Scaling factor for reasoning content. If set, `reasoning_max_len` and `reasoning_min_len` are ignored, and the reasoning process is either truncated or extended accordingly.
  - `cut_by_sentence (bool, default=False)`: Whether to truncate reasoning content at sentence boundaries when using `reasoning_scale`.
  - `return_dict (bool, default=False)`: Whether to return responses as a list of dictionaries. If `True`, responses are returned as `list[dict[str, str]]`.
  - `param (GenParam, default=None)`: Additional chat parameters for customization.
  - `ext_param (GenExtraParam, default=None)`: Extended chat parameters for further customization.

- Returns
  - `list[str]`: If the model is **not** an LRM, returns a list of generated responses.
  - `list[tuple[str, str]]`: If the model **is** an LRM, returns a list of tuples containing both the reasoning process and the final response.
  - `list[dict[str, str]]`: If `return_dict=True`, returns a list of dictionaries, where each dictionary contains:
    - `response (str)`: The generated response.
    - `reasoning (str, optional)`: The reasoning process (only available in LRM mode).

#### parallel_chat_force_reasoning_content

- Required Parameters
  - `messages_list (list[dict[str, str]])`: A list of message dictionaries, where each dictionary contains:
    - `role (str)`: The role of the speaker, must be one of `"user"`, `"assistant"`, or `"system"` (the `"system"` role is only allowed in the first turn).
    - `content (str)`: The content of the message.
  - `reasoning_content (list[str])`: A list of predefined reasoning processes corresponding to each input sequence. This ensures that the model follows the provided reasoning when generating responses.

- Optional Parameters
  - `threads (int, default=20)`: Number of threads for parallel execution.
  - `reasoning_scale (float, default=None)`: Scaling factor for adjusting reasoning content length. If set, the reasoning content is truncated or extended accordingly.
  - `cut_by_sentence (bool, default=False)`: Whether to truncate reasoning content at sentence boundaries when using `reasoning_scale`.
  - `return_dict (bool, default=False)`: Whether to return responses as a list of dictionaries. If `True`, responses are returned as `list[dict[str, str]]`.
  - `param (GenParam, default=None)`: Additional chat parameters for customization.
  - `ext_param (GenExtraParam, default=None)`: Extended chat parameters for further customization.

- Returns
  - `list[tuple[str, str]]`: Returns a list of tuples, each containing:
    - `reasoning (str)`: The provided reasoning process.
    - `response (str)`: The generated response.
  - `list[dict[str, str]]`: If `return_dict=True`, returns a list of dictionaries, where each dictionary contains:
    - `response (str)`: The generated response.
    - `reasoning (str)`: The provided reasoning process.

### CLI

#### decode

- Required Parameters
  - `--model_path`: Path to the model  
  - `--file_path`: Path to the input data file (support json, jsonl, csv and xlsx)
  - `--decode_type`: Decoding type, available options:  
    - `query`: Standard query  
    - `query_reasoning_ctrl`: Control reasoning length  
    - `query_force_reasoning_content`: Control reasoning content  
  - `--save_path`: Path to save the output results  
  - `--query_keys`: Specify query fields (comma-separated)  

- Optional Parameters  
  - `--response_keys`: Specify response save fields (comma-separated)  
  - `--reasoning_keys`: Reasoning save fields (for reasoning mode, comma-separated)  
  - `--tensor_parallel_size`: Tensor parallelism size for the model (default: 1)  
  - `--model_num`: Number of models loaded simultaneously
  - `--port`: Server listening port (default: 50000)
  - `--max_model_len`: max_model_len of vllm model  
  - `--show_vllm_log`: Whether to display vLLM logs (default: enabled)  
  - `--openai_timeout`: Timeout for OpenAI client (default: 30 seconds)  
  - `--threads`: Number of parallel threads (default: 20)  
  - `--enable_reasoning`: Enable reasoning mode  
  - `--reasoning_parser`: Reasoning parser name (default: `deepseek_r1`)  
  - `--system_prompt_file`: Specify system prompt file  
  - `--chat_template_file`: Specify chat template file  
  - `--max_new_tokens`: Maximum number of new tokens to generate (default: 8192)  
  - `--device_ids`: Specify GPU devices (comma-separated for multiple devices)  
  - `--reasoning_max_retry`: Maximum number of retries for reasoning (default: 10)  
  - `--add_reasoning_prompt`: Whether to add a reasoning prompt  
  - `--enable_length_ctrl`: Enable reasoning length control  
  - `--reasoning_max_len`: Maximum reasoning length  
  - `--reasoning_min_len`: Minimum reasoning length (default: 0)  
  - `--reasoning_scale`: Scaling factor for reasoning length  
  - `--cut_by_sentence`: Whether to split reasoning by sentence  
  - `--force_reasoning_content_keys`: Fields for enforcing reasoning content (comma-separated)  
  - `--overwrite`: Whether to overwrite existing fields in input file  

> Note: Set multiple `query_keys` for multi-round generation. If `response_keys`, `reasoning_keys` or/and `force_reasoning_content_keys` specified, they must have same length with `query_keys`. `force_reasoning_content_keys` must be specified when set `decode_type` to `query_force_reasoning_content`.

#### decode multask

- Required Parameters
  - `--model_path`: Path to the model  
  - `--tasks_yaml_path`: Path to a YAML configuration file specifying multiple inference tasks  
- Optional Parameters  
  - `--tensor_parallel_size`: Tensor parallelism size for the model (default: 1)  
  - `--max_model_len`: Maximum input length  
  - `--model_num`: Number of models loaded simultaneously  
  - `--port`: Server listening port (default: 50000)  
  - `--openai_timeout`: Timeout for OpenAI client (default: 30 seconds)  
  - `--enable_reasoning`: Enable reasoning mode  
  - `--chat_template_file`: Specify chat template file  
  - `--reasoning_parser`: Reasoning parser (default: `deepseek_r1`)  
  - `--show_vllm_log`: Whether to display vLLM logs (default: enabled)  
  - `--device_ids`: Specify GPU devices (comma-separated for multiple devices)  

#### Task YAML Configuration

- Required Parameters
  - `file_path`: Path to the input file  
  - `decode_type`: Decoding mode (`query`, `query_reasoning_ctrl`, `query_force_reasoning_content`)  
  - `save_path`: Path to save the output file  
  - `query_keys`: Key(s) in the input file used as queries  
- Optional Parameters  
  - `response_keys`: Key(s) in the output file for generated responses  
  - `reasoning_keys`: Key(s) storing intermediate reasoning steps  
  - `threads`: Number of threads for processing (default: 20)  
  - `system_prompt_file`: Specify system prompt file  
  - `max_new_tokens`: Maximum number of new tokens to generate (default: 8192)  
  - `reasoning_max_retry`: Maximum number of retries for reasoning (default: 10)  
  - `add_reasoning_prompt`: Whether to add a reasoning prompt  
  - `enable_length_ctrl`: Enable length control for responses (default: false)  
  - `reasoning_max_len`: Maximum length for reasoning content  
  - `reasoning_min_len`: Minimum reasoning length (default: 0)  
  - `reasoning_scale`: Scaling factor for reasoning length  
  - `cut_by_sentence`: Whether to split content by sentence (default: false)  
  - `overwrite`: Whether to overwrite existing output files (default: true)    
  - `force_reasoning_content_keys`: Key(s) to enforce reasoning content generation  
  - `overwrite`: Whether to overwrite existing fields in input file  

