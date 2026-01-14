# ruff: noqa: E501
import os

from .msg_validated_output_reask import MSG_VALIDATED_OUTPUT_REASK
from .parsing_reask import PersonalDetails
from .parsing_reask import compiled_prompt as PARSING_COMPILED_PROMPT
from .parsing_reask import compiled_reask as PARSING_COMPILED_REASK
from .parsing_reask import document as PARSING_DOCUMENT
from .parsing_reask import expected_llm_output as PARSING_EXPECTED_LLM_OUTPUT
from .parsing_reask import expected_output as PARSING_EXPECTED_OUTPUT
from .parsing_reask import prompt as PARSING_INITIAL_PROMPT
from .parsing_reask import unparseable_llm_response as PARSING_UNPARSEABLE_LLM_OUTPUT
from .validated_response_reask import VALIDATED_OUTPUT_1 as VALIDATED_OUTPUT_REASK_1
from .validated_response_reask import VALIDATED_OUTPUT_2 as VALIDATED_OUTPUT_REASK_2
from .validated_response_reask import VALIDATED_OUTPUT_3 as VALIDATED_OUTPUT_REASK_3
from .validated_response_reask import ListOfPeople
from .validated_response_reask import prompt as VALIDATED_RESPONSE_REASK_PROMPT
from .with_msg_history import Movie as WITH_MSG_HISTORY

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
reader = lambda filename: open(os.path.join(DATA_DIR, filename)).read().replace("\r", "")

COMPILED_PROMPT = reader("compiled_prompt.txt")
COMPILED_PROMPT_CHAT = reader("compiled_prompt_chat.txt")
COMPILED_INSTRUCTIONS_CHAT = reader("compiled_instructions_chat.txt")
COMPILED_PROMPT_REASK_1 = reader("compiled_prompt_reask_1.txt")
COMPILED_PROMPT_FULL_REASK_1 = reader("compiled_prompt_full_reask_1.txt")
COMPILED_INSTRUCTIONS_REASK_1 = reader("compiled_instructions_reask_1.txt")
COMPILED_PROMPT_REASK_2 = reader("compiled_prompt_reask_2.txt")
COMPILED_PROMPT_FULL_REASK_2 = reader("compiled_prompt_full_reask_2.txt")
COMPILED_INSTRUCTIONS_REASK_2 = reader("compiled_instructions_reask_2.txt")
COMPILED_PROMPT_ENUM = reader("compiled_prompt_enum.txt")
COMPILED_PROMPT_ENUM_2 = reader("compiled_prompt_enum_2.txt")

LLM_OUTPUT = reader("llm_output.txt")
LLM_OUTPUT_REASK_1 = reader("llm_output_reask_1.txt")
LLM_OUTPUT_FULL_REASK_1 = reader("llm_output_full_reask_1.txt")
LLM_OUTPUT_REASK_2 = reader("llm_output_reask_2.txt")
LLM_OUTPUT_FULL_REASK_2 = reader("llm_output_full_reask_2.txt")
LLM_OUTPUT_ENUM = reader("llm_output_enum.txt")
LLM_OUTPUT_ENUM_2 = reader("llm_output_enum_2.txt")

RAIL_SPEC_WITH_REASK = reader("reask.rail")

MSG_HISTORY_LLM_OUTPUT_INCORRECT = reader("msg_history_llm_output_incorrect.txt")
MSG_HISTORY_LLM_OUTPUT_CORRECT = reader("msg_history_llm_output_correct.txt")
MSG_COMPILED_PROMPT_REASK = reader("msg_compiled_prompt_reask.txt")
MSG_COMPILED_INSTRUCTIONS_REASK = reader("msg_compiled_instructions_reask.txt")

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
    "MSG_HISTORY_LLM_OUTPUT_INCORRECT",
    "MSG_HISTORY_LLM_OUTPUT_CORRECT",
    "MSG_COMPILED_PROMPT_REASK",
    "MSG_COMPILED_INSTRUCTIONS_REASK",
    "MSG_VALIDATED_OUTPUT_REASK",
    "MSG_HISTORY_LLM_OUTPUT",
    "VALIDATED_RESPONSE_REASK_PROMPT",
    "ListOfPeople",
    "PersonalDetails",
    "PARSING_INITIAL_PROMPT",
    "PARSING_DOCUMENT",
    "PARSING_EXPECTED_LLM_OUTPUT",
    "PARSING_UNPARSEABLE_LLM_OUTPUT",
    "PARSING_COMPILED_PROMPT",
    "PARSING_COMPILED_REASK",
    "PARSING_EXPECTED_OUTPUT",
]
