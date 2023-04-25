# flake8: noqa: E501
import os

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
reader = (
    lambda filename: open(os.path.join(DATA_DIR, filename)).read().replace("\r", "")
)

COMPILED_PROMPT = reader("compiled_prompt.txt")

LLM_OUTPUT = reader("llm_output.txt")

__all__ = [
    "COMPILED_PROMPT",
    "LLM_OUTPUT",
]
