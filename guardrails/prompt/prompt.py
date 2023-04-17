"""The LLM prompt."""
from .base_prompt import BasePrompt


class Prompt(BasePrompt):
    """Prompt class.

    The prompt is passed to the LLM as primary instructions.
    """
