"""Class for representing a messages entry."""

import re
from string import Template
from typing import Optional

import regex

from warnings import warn
from guardrails.classes.templating.namespace_template import NamespaceTemplate
from guardrails.utils.constants import constants
from guardrails.utils.templating_utils import get_template_variables

class Messages:
    def format(
        self,
        **kwargs,   
    ):
        """Format the messages using the given keyword arguments."""
        formatted_messages = []
        for message in self.messages:
            # Only use the keyword arguments that are present in the message.
            vars = get_template_variables(message.content)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in vars}

            # Return another instance of the class with the formatted message.
            formatted_message = Template(message.content).safe_substitute(**filtered_kwargs)
            formatted_messages.append(formatted_message)
        return Messages(formatted_messages)