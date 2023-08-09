# flake8: noqa: E501
import os

from .validated_response_reask_1 import VALIDATED_OUTPUT as VALIDATED_OUTPUT_REASK_1
from .validated_response_reask_2 import VALIDATED_OUTPUT as VALIDATED_OUTPUT_REASK_2
from .validated_response_reask_3 import VALIDATED_OUTPUT as VALIDATED_OUTPUT_REASK_3
from .with_msg_history import Movie as WITH_MSG_HISTORY

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
reader = (
    lambda filename: open(os.path.join(DATA_DIR, filename)).read().replace("\r", "")
)

COMPILED_PROMPT = reader("compiled_prompt.txt")
COMPILED_PROMPT_REASK_1 = reader("compiled_prompt_reask_1.txt")
COMPILED_PROMPT_REASK_2 = reader("compiled_prompt_reask_2.txt")

LLM_OUTPUT = reader("llm_output.txt")
LLM_OUTPUT_REASK_1 = reader("llm_output_reask_1.txt")
LLM_OUTPUT_REASK_2 = reader("llm_output_reask_2.txt")

RAIL_SPEC_WITH_REASK = reader("reask.rail")

MSG_HISTORY_LLM_OUTPUT = reader("msg_history_llm_output.txt")

__all__ = [
    "COMPILED_PROMPT",
    "COMPILED_PROMPT_REASK_1",
    "COMPILED_PROMPT_REASK_2",
    "LLM_OUTPUT",
    "LLM_OUTPUT_REASK_1",
    "LLM_OUTPUT_REASK_2",
    "RAIL_SPEC_WITH_REASK",
    "VALIDATED_OUTPUT_REASK_1",
    "VALIDATED_OUTPUT_REASK_2",
    "VALIDATED_OUTPUT_REASK_3",
    "WITH_MSG_HISTORY",
    "MSG_HISTORY_LLM_OUTPUT"
]
