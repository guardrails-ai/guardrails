# flake8: noqa: E501
import os

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
reader = (
    lambda filename: open(os.path.join(DATA_DIR, filename)).read().replace("\r", "")
)

COMPILED_PROMPT_WITHOUT_INSTRUCTIONS = reader("compiled_prompt.txt")
COMPILED_INSTRUCTIONS = reader("compiled_instructions.txt")

LLM_OUTPUT = reader("llm_output.txt")

__all__ = [
    "COMPILED_PROMPT_WITHOUT_INSTRUCTIONS",
    "COMPILED_INSTRUCTIONS",
    "LLM_OUTPUT",
]
