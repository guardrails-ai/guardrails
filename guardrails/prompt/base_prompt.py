"""Class for representing a prompt entry."""
import re
from string import Formatter
from typing import Optional

from guardrails.utils.constants import constants


class BasePrompt:
    """Base class for representing an LLM prompt."""

    def __init__(self, source: str, output_schema: Optional[str] = None):
        self.format_instructions_start = self.get_format_instructions_idx(source)

        # Substitute constants in the prompt.
        source = self.substitute_constants(source)

        # If an output schema is provided, substitute it in the prompt.
        if output_schema:
            self.source = source.format(output_schema=output_schema)
        else:
            self.source = source

    def __repr__(self) -> str:
        # Truncate the prompt to 50 characters and add ellipsis if it's longer.
        truncated_prompt = self.source[:50]
        if len(self.source) > 50:
            truncated_prompt += "..."
        return f"Prompt({truncated_prompt})"

    def __str__(self) -> str:
        return self.source

    @property
    def variable_names(self):
        return [x[1] for x in Formatter().parse(self.source) if x[1] is not None]

    @property
    def format_instructions(self):
        return self.source[self.format_instructions_start :]

    def substitute_constants(self, text):
        """Substitute constants in the prompt."""
        # Substitute constants by reading the constants file.
        # Regex to extract all occurrences of @<constant_name>
        matches = re.findall(r"@(\w+)", text)

        # Substitute all occurrences of @<constant_name> with the value of the constant.
        for match in matches:
            if match in constants:
                text = text.replace(f"@{match}", constants[match])

        return text

    def get_prompt_variables(self):
        return self.variable_names

    def format(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

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

        matches = re.finditer(r"@(\w+)", text)

        earliest_match_idx = None
        earliest_match = None

        # Find the earliest match where the match belongs to a constant.
        for match in matches:
            if match.group(1) in constants:
                if earliest_match_idx is None or earliest_match_idx > match.start():
                    earliest_match_idx = match.start()
                    earliest_match = match

        if earliest_match_idx is None:
            return 0

        return earliest_match.start()
