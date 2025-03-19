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


@ReasoningParserManager.register_module("openthinker")
class OpenThinkerReasoningParser(ReasoningParser):
    """
    Reasoning parser for openthinker model.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_start_token = "<|begin_of_thought|>"
        self.think_end_token = "<|end_of_thought|>"

        self.solution_start_token = "<|begin_of_solution|>"
        self.solution_end_token = "<|end_of_solution|>"

        self.reasoning_regex = re.compile(
            rf"{re.escape(self.think_start_token)}(.*?){re.escape(self.think_end_token)}", re.DOTALL)
        
        self.solution_regex = re.compile(
            rf"{re.escape(self.solution_start_token)}(.*?){re.escape(self.solution_end_token)}", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        # self.think_start_token_id = self.vocab.get(self.think_start_token)
        # self.think_end_token_id = self.vocab.get(self.think_end_token)
        # if (self.think_start_token_id is None
        #         or self.think_end_token_id is None):
        #     raise RuntimeError(
        #         "openthinker reasoning parser could not locate think start/end "
        #         "tokens in the tokenizer!")

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


        if self.think_end_token not in model_output:
            return model_output, None
        else:
            # Add a start token if it's missing to keep compatibility.
            if self.think_start_token not in model_output:
                model_output = f"{self.think_start_token}{model_output}"
            # Use a regex to find the reasoning content
            reasoning_content = self.reasoning_regex.findall(model_output)[0]
            # print([model_output], [reasoning_content])

            end_index = len(
                f"{self.think_start_token}{reasoning_content}{self.think_end_token}"
            )
            final_output = model_output[end_index:]
            if self.solution_start_token in final_output and self.solution_end_token in final_output:
                final_output = self.solution_regex.findall(final_output)[0]
            # print([model_output], [final_output])

            if len(final_output) == 0:
                return reasoning_content, None

            return reasoning_content, final_output
