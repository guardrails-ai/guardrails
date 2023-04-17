# flake8: noqa: E501
import os

from .validated_output_filter import VALIDATED_OUTPUT_FILTER
from .validated_output_fix import VALIDATED_OUTPUT_FIX
from .validated_output_noop import VALIDATED_OUTPUT_NOOP
from .validated_output_reask_1 import VALIDATED_OUTPUT_REASK_1
from .validated_output_reask_2 import VALIDATED_OUTPUT_REASK_2
from .validated_output_refrain import VALIDATED_OUTPUT_REFRAIN

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
reader = (
    lambda filename: open(os.path.join(DATA_DIR, filename)).read().replace("\r", "")
)

COMPILED_PROMPT = reader("compiled_prompt.txt")
COMPILED_PROMPT_REASK = reader("compiled_prompt_reask.txt")
COMPILED_PROMPT_WITHOUT_INSTRUCTIONS = reader("compiled_prompt_without_instructions.txt")
COMPILED_INSTRUCTIONS = reader("compiled_instructions.txt")

LLM_OUTPUT = reader("llm_output.txt")
LLM_OUTPUT_REASK = reader("llm_output_reask.txt")

RAIL_SPEC_WITH_FILTER = reader("filter.rail")
RAIL_SPEC_WITH_FIX = reader("fix.rail")
RAIL_SPEC_WITH_NOOP = reader("noop.rail")
RAIL_SPEC_WITH_REASK = reader("reask.rail")
RAIL_SPEC_WITH_REFRAIN = reader("refrain.rail")
RAIL_SPEC_WITH_FIX_CHAT_MODEL = reader("fix_chat_model.rail")

__all__ = [
    "COMPILED_PROMPT",
    "COMPILED_PROMPT_REASK",
    "COMPILED_PROMPT_WITHOUT_INSTRUCTIONS",
    "COMPILED_INSTRUCTIONS",
    "LLM_OUTPUT",
    "LLM_OUTPUT_REASK",
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
]
