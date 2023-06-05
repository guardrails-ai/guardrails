import os

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
reader = (
    lambda filename: open(os.path.join(DATA_DIR, filename)).read().replace("\r", "")
)

COMPILED_PROMPT = reader("compiled_prompt.txt")
LLM_OUTPUT = reader("llm_output.txt")
RAIL_SPEC_FOR_STRING = reader("string.rail")

__all__ = [
    "COMPILED_PROMPT",
    "RAIL_SPEC_FOR_STRING"
]
