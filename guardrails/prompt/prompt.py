"""The LLM prompt."""
from string import Formatter

from .base_prompt import BasePrompt


class Prompt(BasePrompt):
    """Prompt class.

    The prompt is passed to the LLM as primary instructions.
    """

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Prompt) and self.source == __value.source

    def format(self, **kwargs):
        """Format the prompt using the given keyword arguments."""
        # Only use the keyword arguments that are present in the prompt.
        vars = [x[1] for x in Formatter().parse(self.source) if x[1] is not None]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in vars}

        # Return another instance of the class with the formatted prompt.
        return Prompt(self.source.format(**filtered_kwargs))
