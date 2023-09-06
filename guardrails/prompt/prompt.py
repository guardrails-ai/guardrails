"""The LLM prompt."""
import warnings
from string import Formatter, Template

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
        if len(filtered_kwargs) == 0:
            warnings.warn(
                "Prompt does not have any variables, "
                "if you are migrating follow the new variable convention "
                "documented here: https://docs.getguardrails.ai/0-2-migration/"
            )

        # Return another instance of the class with the formatted prompt.
        formatted_prompt = Template(self.source).safe_substitute(**filtered_kwargs)
        return Prompt(formatted_prompt)
