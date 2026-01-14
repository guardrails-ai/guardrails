# ruff: noqa: E501
import os

from .validator_parallelism_reask_1 import VALIDATOR_PARALLELISM_REASK_1  # noqa: F401
from .validator_parallelism_reask_2 import VALIDATOR_PARALLELISM_REASK_2  # noqa: F401

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
reader = lambda filename: open(os.path.join(DATA_DIR, filename)).read().replace("\r", "")

COMPILED_PROMPT_1_WITHOUT_INSTRUCTIONS = reader("compiled_prompt_1.txt")
COMPILED_PROMPT_1_PYDANTIC_2_WITHOUT_INSTRUCTIONS = reader("compiled_prompt_1_pydantic_2.txt")
COMPILED_PROMPT_2_WITHOUT_INSTRUCTIONS = reader("compiled_prompt_2.txt")
COMPILED_INSTRUCTIONS = reader("compiled_instructions.txt")

LLM_OUTPUT_1_FAIL_GUARDRAILS_VALIDATION = reader("llm_output_1_fail_guardrails_validation.txt")
LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION = reader(
    "llm_output_2_succeed_gd_but_fail_pydantic_validation.txt"
)
LLM_OUTPUT_3_SUCCEED_GUARDRAILS_AND_PYDANTIC = reader("llm_output_3_succeed_gd_and_pydantic.txt")

RAIL_SPEC_WITH_VALIDATOR_PARALLELISM = reader("validator_parallelism.rail")
VALIDATOR_PARALLELISM_PROMPT_1 = reader("validator_parallelism_prompt_1.txt")
VALIDATOR_PARALLELISM_RESPONSE_1 = reader("validator_parallelism_1.txt")

VALIDATOR_PARALLELISM_PROMPT_2 = reader("validator_parallelism_prompt_2.txt")
VALIDATOR_PARALLELISM_RESPONSE_2 = reader("validator_parallelism_2.txt")

VALIDATOR_PARALLELISM_PROMPT_3 = reader("validator_parallelism_prompt_3.txt")
VALIDATOR_PARALLELISM_RESPONSE_3 = reader("validator_parallelism_3.txt")

__all__ = [
    "COMPILED_PROMPT_1_WITHOUT_INSTRUCTIONS",
    "COMPILED_PROMPT_2_WITHOUT_INSTRUCTIONS",
    "COMPILED_INSTRUCTIONS",
    "LLM_OUTPUT_1_FAIL_GUARDRAILS_VALIDATION",
    "LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION",
    "LLM_OUTPUT_3_SUCCEED_GUARDRAILS_AND_PYDANTIC",
]
