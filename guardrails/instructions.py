"""Instructions to the LLM, to be passed in the prompt."""
from .base_prompt import BasePrompt


class Instructions(BasePrompt):
    """Instructions class.

    The instructions are passed to the LLM as secondary input.
    Different model may use these differently. For example, chat models may receive
    instructions in the system-prompt.
    """
