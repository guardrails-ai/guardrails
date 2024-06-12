"""Class for representing a prompt entry."""

import re
from string import Template
from typing import Optional

import regex
from warnings import warn
from guardrails.classes.templating.namespace_template import NamespaceTemplate
from guardrails.utils.constants import constants
from guardrails.utils.templating_utils import get_template_variables


class BaseMessages:
    """Base class for representing an LLM messages."""
    #TODO dont know if this should be source 
    def __init__(self, source: list[dict], output_schema: Optional[str] = None, xml_output_schema: Optional[str] = None):
        self._source = source


        # TODO this probably should not be by reference
        for msg in self._source:
            try: 
                msg["content"] = self.substitute_constants(msg["content"])
            except KeyError:
                pass

        source = self._source

        self.source = source
        # If an output schema is provided, substitute it in the prompt.
        if output_schema:
            self.source.append({"role":"system", "content": output_schema})
        # If an xml output schema is provided, substitute it in the prompt.
        if xml_output_schema:
            self.source.append({"role":"system", "content": xml_output_schema})


    def __repr__(self) -> str:
        # Truncate the prompt to 50 characters and add ellipsis if it's longer.
        truncated_prompt = self.source[:50]
        if len(self.source) > 50:
            truncated_prompt += "..."
        return f"Messages({truncated_prompt})"

    def __str__(self) -> str:
        return self.source

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        try:
            result = self.source[self.current]
        except IndexError:
            raise StopIteration
        self.current += 1
        return result
    
    @property
    def variable_names(self):
        variables = []
        for msg in self.source:
            variables.extend(get_template_variables(msg["content"]))
        return variables

    @property
    def format_instructions(self):
        return self.source[self.format_instructions_start :]

    def substitute_constants(self, text):
        """Substitute constants in the prompt."""
        # Substitute constants by reading the constants file.
        # Regex to extract all occurrences of ${gr.<constant_name>}
        matches = re.findall(r"\${gr\.(\w+)}", text)

        # Substitute all occurrences of ${gr.<constant_name>}
        #   with the value of the constant.
        json_constants = [m for m in matches if "json_" in m]

        if len(json_constants) > 0:
            first_const: str = json_constants[0]
            warn(
                Template(
                    "Prompt Primitives are moving! "
                    "To keep the same behaviour, "
                    "switch from `json` constants to `xml` constants. "
                    "Example: ${gr.${first_const}} -> ${gr.${xml_const}}",
                ).safe_substitute(
                    first_const=first_const,
                    xml_const=first_const.replace("json_", "xml_"),
                ),
                FutureWarning,
            )

        for match in matches:
            template = NamespaceTemplate(text)
            mapping = {f"gr.{match}": constants[match]}
            text = template.safe_substitute(**mapping)
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

        if earliest_match is None:
            return None
        return earliest_match.start()

    def escape(self) -> str:
        """Escape single curly braces into double curly braces."""
        start_replaced = regex.sub(r"(?<!\$){", "{{", self.source)
        # This variable length negative lookbehind is why we need `regex` over `re`
        return regex.sub(r"(?<!\${.*)}", "}}", start_replaced)

    def _to_request(self) -> str:
        return self.source
