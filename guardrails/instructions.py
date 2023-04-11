"""Instructions to the LLM, to be passed in the prompt."""
from .base_prompt import BasePrompt


class Instructions(BasePrompt):
    """Instructions class.

    The instructions are passed to the LLM as secondary input.
    Different model may use these differently. For example, chat models may receive
    instructions in the system-prompt.
    """

    # TODO: Don't allow inserting text at run time, that is specific to Prompt?

    def __repr__(self) -> str:
        # Truncate the prompt to 50 characters and add ellipsis if it's longer.
        truncated_instructions = self.source[:50]
        if len(self.source) > 50:
            truncated_instructions += "..."
        return f"Instructions({truncated_instructions})"
