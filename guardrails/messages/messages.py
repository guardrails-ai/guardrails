"""The LLM messages."""

from string import Template

from guardrails.utils.parsing_utils import get_template_variables


from .base_messages import BaseMessages

class Messages(BaseMessages):
    """Messages class.

    The messages is passed to the LLM as primary interface for input
    """

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Messages) and self.source == __value.source

    def format(self, **kwargs):
        """Format the messages using the given keyword arguments."""

        # Return another instance of the class with the formatted prompt.
        formatted_messages = [] 
        for msg in self.source:
            # Only use the keyword arguments that are present in the prompt.
            vars = get_template_variables(msg["content"])
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in vars}
            msg["content"] = Template(msg["content"]).safe_substitute(**filtered_kwargs)
            formatted_messages.append(msg)

        return Messages(formatted_messages)
