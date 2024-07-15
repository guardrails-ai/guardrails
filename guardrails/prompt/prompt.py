"""The LLM prompt."""

from string import Template

from guardrails.utils.templating_utils import get_template_variables

from .base_prompt import BasePrompt


class Prompt(BasePrompt):
    """Prompt class.

    The prompt is passed to the LLM as primary instructions.
    """

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Prompt) and self.source == __value.source

    def format(self, **kwargs) -> "Prompt":
        """Format the prompt using the given keyword arguments."""
        # Only use the keyword arguments that are present in the prompt.
        vars = get_template_variables(self.source)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in vars}

        # Return another instance of the class with the formatted prompt.
        formatted_prompt = Template(self.source).safe_substitute(**filtered_kwargs)
        return Prompt(formatted_prompt)
