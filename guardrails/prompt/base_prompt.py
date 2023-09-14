"""Class for representing a prompt entry."""
import re
import warnings
from string import Formatter, Template
from typing import Optional

import regex

from guardrails.namespace_template import NamespaceTemplate
from guardrails.utils.constants import constants


class BasePrompt:
    """Base class for representing an LLM prompt."""

    def __init__(self, source: str, output_schema: Optional[str] = None):
        self.format_instructions_start = self.get_format_instructions_idx(source)

        # Substitute constants in the prompt.
        source = self.substitute_constants(source)

        # If an output schema is provided, substitute it in the prompt.
        if output_schema:
            self.source = Template(source).safe_substitute(output_schema=output_schema)
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
        return [x[1] for x in Formatter().parse(self.escape()) if x[1] is not None]

    @property
    def format_instructions(self):
        return self.source[self.format_instructions_start :]

    def substitute_constants(self, text):
        """Substitute constants in the prompt."""
        # Substitute constants by reading the constants file.
        # Regex to extract all occurrences of ${gr.<constant_name>}
        if self.uses_old_constant_schema(text):
            warnings.warn(
                "It appears that you are using an old schema for gaurdrails variables, "
                "follow the new namespaced convention "
                "documented here: https://docs.getguardrails.ai/0-2-migration/"
            )

        matches = re.findall(r"\${gr\.(\w+)}", text)

        # Substitute all occurrences of ${gr.<constant_name>}
        #   with the value of the constant.
        for match in matches:
            template = NamespaceTemplate(text)
            mapping = {f"gr.{match}": constants[match]}
            text = template.safe_substitute(**mapping)

        return text

    def uses_old_constant_schema(self, text) -> bool:
        matches = re.findall(r"@(\w+)", text)
        if len(matches) == 0:
            return False
        else:
            return True

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

        # Regex to extract first occurrence of ${gr.<constant_name>}

        matches = re.finditer(r"\${gr\.(\w+)}", text)

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

    def escape(self) -> str:
        start_replaced = regex.sub(r"(?<!\$){", "{{", self.source)
        # This variable length negative lookbehind is why we need `regex` over `re`
        return regex.sub(r"(?<!\${.*)}", "}}", start_replaced)
