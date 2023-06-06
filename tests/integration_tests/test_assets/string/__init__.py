import os

from tests.integration_tests.test_assets.string.validated_output_reask import VALIDATED_OUTPUT_REASK

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
reader = (
    lambda filename: open(os.path.join(DATA_DIR, filename)).read().replace("\r", "")
)

COMPILED_PROMPT = reader("compiled_prompt.txt")
LLM_OUTPUT = reader("llm_output.txt")
RAIL_SPEC_FOR_STRING = reader("string.rail")

COMPILED_PROMPT_REASK = reader("compiled_prompt_reask.txt")
RAIL_SPEC_FOR_STRING_REASK = reader("string_reask.rail")
LLM_OUTPUT_REASK = reader("llm_output_reask.txt")

__all__ = [
    "COMPILED_PROMPT",
    "LLM_OUTPUT",
    "RAIL_SPEC_FOR_STRING",
    "COMPILED_PROMPT_REASK",
    "RAIL_SPEC_FOR_STRING_REASK",
    "LLM_OUTPUT_REASK",
    "VALIDATED_OUTPUT_REASK",
]
