# ruff: noqa: E501
import os

from .msg_validated_output_reask import MSG_VALIDATED_OUTPUT_REASK
from .validated_output_reask import VALIDATED_OUTPUT_REASK

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
reader = lambda filename: open(os.path.join(DATA_DIR, filename)).read().replace("\r", "")

COMPILED_INSTRUCTIONS = reader("compiled_instructions.txt")
COMPILED_PROMPT = reader("compiled_prompt.txt")
COMPILED_LIST_PROMPT = reader("compiled_list_prompt.txt")
LLM_OUTPUT = reader("llm_output.txt")
LIST_LLM_OUTPUT = reader("llm_list_output.txt")
RAIL_SPEC_FOR_STRING = reader("string.rail")
RAIL_SPEC_FOR_LIST = reader("list.rail")

COMPILED_PROMPT_REASK = reader("compiled_prompt_reask.txt")
RAIL_SPEC_FOR_STRING_REASK = reader("string_reask.rail")
LLM_OUTPUT_REASK = reader("llm_output_reask.txt")

RAIL_SPEC_FOR_MSG_HISTORY = reader("message_history.rail")
MSG_LLM_OUTPUT_INCORRECT = reader("message_history_output.txt")
MSG_LLM_OUTPUT_CORRECT = reader("message_history_reask_output.txt")
MOVIE_MSG_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Can you give me your favorite movie?"},
]
MSG_COMPILED_PROMPT_REASK = reader("msg_compiled_prompt_reask.txt")
MSG_COMPILED_INSTRUCTIONS_REASK = reader("msg_compiled_instructions_reask.txt")
PARSE_COMPILED_PROMPT_REASK = reader("parse_compiled_prompt_reask.txt")

__all__ = [
    "COMPILED_PROMPT",
    "RAIL_SPEC_FOR_LIST",
    "LIST_LLM_OUTPUT",
    "LLM_OUTPUT",
    "RAIL_SPEC_FOR_STRING",
    "COMPILED_PROMPT_REASK",
    "RAIL_SPEC_FOR_STRING_REASK",
    "LLM_OUTPUT_REASK",
    "VALIDATED_OUTPUT_REASK",
    "RAIL_SPEC_FOR_MSG_HISTORY",
    "MSG_LLM_OUTPUT_INCORRECT",
    "MSG_LLM_OUTPUT_CORRECT",
    "MOVIE_MSG_HISTORY",
    "MSG_COMPILED_PROMPT_REASK",
    "MSG_COMPILED_INSTRUCTIONS_REASK",
    "MSG_VALIDATED_OUTPUT_REASK",
]
