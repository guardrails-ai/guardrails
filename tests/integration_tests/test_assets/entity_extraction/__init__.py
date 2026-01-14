# ruff: noqa: E501
import os

from .optional_prompts import (
    OPTIONAL_INSTRUCTIONS_CHAT_MODEL,
    OPTIONAL_MSG_HISTORY,
    OPTIONAL_PROMPT_CHAT_MODEL,
    OPTIONAL_PROMPT_COMPLETION_MODEL,
)
from .pydantic_models import (
    INSTRUCTIONS_CHAT_MODEL,
    PROMPT,
    PROMPT_CHAT_MODEL,
    ContractDetailsFilter,
    ContractDetailsFix,
    ContractDetailsNoop,
    ContractDetailsReask,
    ContractDetailsRefrain,
)
from .validated_output_filter import VALIDATED_OUTPUT_FILTER
from .validated_output_fix import VALIDATED_OUTPUT_FIX
from .validated_output_noop import VALIDATED_OUTPUT_NOOP
from .validated_output_reask_1 import VALIDATED_OUTPUT_REASK_1
from .validated_output_reask_2 import VALIDATED_OUTPUT_REASK_2
from .validated_output_refrain import VALIDATED_OUTPUT_REFRAIN
from .validated_output_skeleton_reask_1 import VALIDATED_OUTPUT_SKELETON_REASK_1
from .validated_output_skeleton_reask_2 import VALIDATED_OUTPUT_SKELETON_REASK_2

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
reader = lambda filename: open(os.path.join(DATA_DIR, filename)).read().replace("\r", "")

# Compiled prompts
COMPILED_PROMPT = reader("compiled_prompt.txt")
NON_OPENAI_COMPILED_PROMPT = reader("non_openai_compiled_prompt.txt")
COMPILED_PROMPT_WITHOUT_INSTRUCTIONS = reader("compiled_prompt_without_instructions.txt")
COMPILED_PROMPT_REASK = reader("compiled_prompt_reask.txt")
NON_OPENAI_COMPILED_PROMPT_REASK = reader("non_openai_compiled_prompt_reask.txt")
COMPILED_PROMPT_REASK_WITHOUT_INSTRUCTIONS = reader(
    "compiled_prompt_reask_without_instructions.txt"
)
COMPILED_PROMPT_FULL_REASK = reader("compiled_prompt_full_reask.txt")
COMPILED_INSTRUCTIONS = reader("compiled_instructions.txt")
COMPILED_INSTRUCTIONS_REASK = reader("compiled_instructions_reask.txt")
COMPILED_PROMPT_SKELETON_REASK_1 = reader("compiled_prompt_skeleton_reask_1.txt")
COMPILED_PROMPT_SKELETON_REASK_2 = reader("compiled_prompt_skeleton_reask_2.txt")
COMPILED_MSG_HISTORY = [
    {"role": "system", "content": COMPILED_INSTRUCTIONS},
    {"role": "user", "content": COMPILED_PROMPT_WITHOUT_INSTRUCTIONS},
]

# LLM outputs
LLM_OUTPUT = reader("llm_output.txt")
LLM_OUTPUT_REASK = reader("llm_output_reask.txt")
LLM_OUTPUT_FULL_REASK = reader("llm_output_full_reask.txt")
LLM_OUTPUT_SKELETON_REASK_1 = reader("llm_output_skeleton_reask_1.txt")
LLM_OUTPUT_SKELETON_REASK_2 = reader("llm_output_skeleton_reask_2.txt")

# Rail specs
RAIL_SPEC_WITH_FILTER = reader("filter.rail")
RAIL_SPEC_WITH_FIX = reader("fix.rail")
RAIL_SPEC_WITH_NOOP = reader("noop.rail")
RAIL_SPEC_WITH_REASK = reader("reask.rail")
RAIL_SPEC_WITH_SKELETON_REASK = reader("skeleton_reask.rail")
RAIL_SPEC_WITH_REFRAIN = reader("refrain.rail")
RAIL_SPEC_WITH_FIX_CHAT_MODEL = reader("fix_chat_model.rail")

# Rail specs without prompts and instructions
RAIL_SPEC_WITH_REASK_NO_PROMPT = reader("reask_without_prompt.rail")

# Pydantic models
PYDANTIC_RAIL_WITH_FILTER = ContractDetailsFilter
PYDANTIC_RAIL_WITH_FIX = ContractDetailsFix
PYDANTIC_RAIL_WITH_NOOP = ContractDetailsNoop
PYDANTIC_RAIL_WITH_REASK = ContractDetailsReask
PYDANTIC_RAIL_WITH_REFRAIN = ContractDetailsRefrain
PYDANTIC_PROMPT = PROMPT
PYDANTIC_PROMPT_CHAT_MODEL = PROMPT_CHAT_MODEL
PYDANTIC_INSTRUCTIONS_CHAT_MODEL = INSTRUCTIONS_CHAT_MODEL


__all__ = [
    "COMPILED_PROMPT",
    "NON_OPENAI_COMPILED_PROMPT",
    "COMPILED_PROMPT_WITHOUT_INSTRUCTIONS",
    "COMPILED_PROMPT_REASK",
    "NON_OPENAI_COMPILED_PROMPT_REASK",
    "COMPILED_PROMPT_REASK_WITHOUT_INSTRUCTIONS",
    "COMPILED_INSTRUCTIONS",
    "COMPILED_INSTRUCTIONS_REASK",
    "COMPILED_MSG_HISTORY",
    "COMPILED_MSG_HISTORY_PROMPT",
    "LLM_OUTPUT",
    "LLM_OUTPUT_REASK",
    "PYDANTIC_INSTRUCTIONS",
    "PYDANTIC_PROMPT",
    "RAIL_SPEC_WITH_FILTER",
    "RAIL_SPEC_WITH_FIX",
    "RAIL_SPEC_WITH_FIX_CHAT_MODEL",
    "RAIL_SPEC_WITH_NOOP",
    "RAIL_SPEC_WITH_REASK",
    "RAIL_SPEC_WITH_REFRAIN",
    "VALIDATED_OUTPUT_FILTER",
    "VALIDATED_OUTPUT_FIX",
    "VALIDATED_OUTPUT_NOOP",
    "VALIDATED_OUTPUT_REASK_1",
    "VALIDATED_OUTPUT_REASK_2",
    "VALIDATED_OUTPUT_REFRAIN",
    "VALIDATED_OUTPUT_SKELETON_REASK_1",
    "VALIDATED_OUTPUT_SKELETON_REASK_2",
    "OPTIONAL_PROMPT_COMPLETION_MODEL",
    "OPTIONAL_PROMPT_CHAT_MODEL",
    "OPTIONAL_INSTRUCTIONS_CHAT_MODEL",
    "OPTIONAL_MSG_HISTORY",
]
