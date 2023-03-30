import re
from typing import Optional

from guardrails.utils.constants import constants


class Prompt:
    def __init__(self, source: str, output_schema: Optional[str] = None):
        output_schema = output_schema or {}

        # Get variable names in the source string (surronded by 2 curly braces)
        self.variable_names = re.findall(r"{{(.*?)}}", source)

        format_instructions_start_idx = self.get_format_instructions_idx(source)

        # Substitute constants in the prompt.
        source = self.substitute_constants(source)
        # Format instructions contain info for how to format LLM output.
        self.format_instructions = source[format_instructions_start_idx:]

        self.source = source.format(output_schema=output_schema)

    def __repr__(self) -> str:
        # Truncate the prompt to 50 characters and add ellipsis if it's longer.
        truncated_prompt = self.source[:50]
        if len(self.source) > 50:
            truncated_prompt += "..."
        return f"Prompt({truncated_prompt})"

    def substitute_constants(self, text):
        """Substitute constants in the prompt."""
        # Substitute constants by reading the constants file.
        # Regex to extract all occurrences of @<constant_name>
        matches = re.findall(r"@(\w+)", text)

        # Substitute all occurrences of @<constant_name> with the value of the constant.
        for match in matches:
            text = text.replace(f"@{match}", constants[match])

        return text

    def __str__(self) -> str:
        return self.source

    def get_prompt_variables(self):
        return self.variable_names

    def format(self, **kwargs):
        """Format the prompt using the given keyword arguments."""
        return self.source.format(**kwargs)

    def make_vars_optional(self):
        """Make all variables in the prompt optional."""
        for var in self.variable_names:
            self.source = self.source.replace(f"{{{var}}}", f"{{{var}:}}")

    def get_format_instructions_idx(self, text: str) -> Optional[int]:
        """Get the index of the first format instruction in the prompt.

        It checks to see where the first instance of any constant is in the text.
        Everything from then on is considered to be a format instruction.

        Returns:
            The index of the first format instruction in the prompt.
        """
        # TODO(shreya): Optionally add support for special character demarcation.

        # Regex to extract first occurrence of @<constant_name>
        match = re.search(r"@(\w+)", text)
        if match is None:
            return 0

        # Subtract 2*len(variable_names) to account for curly braces.
        start_idx = match.start() - 2 * len(self.variable_names)
        return start_idx
