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
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.variable_names}

        # Return another instance of the class with the formatted prompt.
        # If the convention of double escaping prompt params changes, send filtered_kwarfs to super.format instead
        formatted_source = super().format()
        return Prompt(formatted_source.format(**filtered_kwargs), format_instructions_start=self.format_instructions_start)

    def _to_request(self) -> str:
        return self.source
