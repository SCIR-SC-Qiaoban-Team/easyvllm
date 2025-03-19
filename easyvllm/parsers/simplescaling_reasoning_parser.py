# SPDX-License-Identifier: Apache-2.0

import re
from typing import Optional, Sequence, Tuple, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.entrypoints.openai.reasoning_parsers.abs_reasoning_parsers import (
    ReasoningParser, ReasoningParserManager)
from vllm.logger import init_logger

logger = init_logger(__name__)


@ReasoningParserManager.register_module("simplescaling")
class SimplescalingReasoningParser(ReasoningParser):
    """
    Reasoning parser for simplescaling model.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_start_token = "think\n"
        self.think_end_token = "answer\n"
        self.force_think_end_token = "Final Answer:\n"

        self.reasoning_regex = re.compile(
            rf"{self.think_start_token}(.*?){self.think_end_token}", re.DOTALL)
        self.force_reasoning_regex = re.compile(
            rf"{self.think_start_token}(.*?){self.force_think_end_token}", re.DOTALL)
        # print([self.force_reasoning_regex.pattern])

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")


    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content
        """
        raise NotImplementedError()
        
        
    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> Tuple[Optional[str], Optional[str]]:

        if self.think_end_token not in model_output and self.force_think_end_token not in model_output:
            return model_output, None
        else:
            # Add a start token if it's missing to keep compatibility.
            if self.think_start_token not in model_output:
                model_output = f"{self.think_start_token}{model_output}"
            # Use a regex to find the reasoning content
            if self.force_think_end_token in model_output:
                reasoning_content = self.force_reasoning_regex.findall(model_output)[0]
                end_index = len(
                    f"{self.think_start_token}{reasoning_content}{self.force_think_end_token}"
                )
            else:
                reasoning_content = self.reasoning_regex.findall(model_output)[0]
                end_index = len(
                    f"{self.think_start_token}{reasoning_content}{self.think_end_token}"
                )

            final_output = model_output[end_index:]

            if len(final_output) == 0:
                return reasoning_content, None

            return reasoning_content, final_output
