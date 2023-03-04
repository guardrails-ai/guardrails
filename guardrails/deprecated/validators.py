import re
from typing import Optional, Union, List

from pyparsing import CaselessKeyword, Regex, Token

from guardrails.prompt_repo import Prompt


class Registry(object):
    """A registry for rules."""
    def __init__(self):
        self._registry = {}

    def register(self, rule, name=None):
        if name is None:
            name = rule.__name__
        self._registry[name] = rule

    def get_rules(self):
        return self._registry.values()


registry = Registry()


class Validator:
    """Base class for all rules."""

    # @staticmethod
    def validate(self, text: str):
        raise NotImplementedError("Validator not implemented.")

    # @staticmethod
    def debug(self, text: str):
        raise NotImplementedError("Debugging not implemented.")

    @staticmethod
    def affordance(self, *args, **kwargs):
        raise NotImplementedError("Affordance not implemented.")


class FormValidator(Validator):
    """Base class for all form validators."""

    @property
    def grammar(self):
        raise NotImplementedError("Grammar not implemented.")

    @property
    def grammar_as_text(self):
        raise NotImplementedError("Grammar as text not implemented.")

    def validate(self, text: str):
        return self.grammar.parse_string(text)

    def debug(self, text: str, placeholder: Token):
        exception = None

        # Figure out where the text should start from.
        # If the placeholder is a CaselessKeyword, then the text should start from the first character.
        placeholder_match = list(placeholder.scan_string(text, max_matches=1))
        if len(placeholder_match) == 0:
            raise ValueError(f"Placeholder {placeholder} not found in text {text}.")
        placeholder_start = placeholder_match[0][1]

        relevant_text = text[placeholder_start:]

        try:
            combined_grammar = placeholder + self.grammar
            combined_grammar.parse_string(relevant_text)
        except Exception as e:
            exception = e

        placeholder_name = placeholder.name[1:-1]
        prediction_without_placeholder = exception.line.replace(placeholder_name, "")

        return Prompt(f"""You previously responded with "{prediction_without_placeholder}" after "{placeholder_name}", which is incorrect. The correct output should be of the format: {self.grammar_as_text} after "{placeholder_name}".""")  # noqa: E501


def register_validator(rule_name: str):
    def wrapper(rule):
        # If the registry already contains a rule with the same name, raise an error.
        if rule_name in registry._registry:
            raise ValueError(f"Rule with name {rule_name} already exists.")
        # Register the rule in the registry.
        registry.register(rule, rule_name)
        return rule
    return wrapper


def get_rule(rule_name: str):
    """Get a rule from the registry."""

    if rule_name in registry._registry:
        return registry._registry[rule_name]
    
    raise ValueError(f"Rule with name {rule_name} does not exist.")


@register_validator("string")
class StringValidator(FormValidator):

    # @staticmethod
    @property
    def grammar(self) -> Regex:
        return Regex(".+", flags=re.IGNORECASE)

    @property
    def grammar_as_text(self) -> str:
        return "a string"


@register_validator("verbose_string_validator")
class VerboseStringValidator(FormValidator):
    # @staticmethod
    @property
    def grammar(self, str_len: Optional[int] = 100) -> Regex:
        return Regex(f".{{{str_len},}}", flags=re.IGNORECASE)
        # return Regex(".{100,}", flags=re.IGNORECASE)

    @property
    def grammar_as_text(self) -> str:
        # return "len(string) > 100"
        return "a string greater than 100 characters in length"

    @property
    def as_definition(self) -> str:
        return "len(string) > 100"


@register_validator("succinct_string_validator")
class SuccinctStringValidator(FormValidator):
    # @staticmethod
    @property
    def grammar(self, str_len: Optional[int] = 100) -> Regex:
        return Regex(f".{{0,{str_len}}}", flags=re.IGNORECASE)
        # return Regex(".{0,100}", flags=re.IGNORECASE)

    @property
    def grammar_as_text(self) -> str:
        # return "len(string) < 100"
        return "a string less than 100 characters in length"

    @property
    def as_definition(self) -> str:
        return "len(string) < 100"



@register_validator("float_validator")
class FloatValidator(FormValidator):

    @property
    def grammar(self) -> Regex:
        # Float or 'None'
        return Regex(r"[-+]?\d*\.\d+|\d+|None", flags=re.IGNORECASE)

    @property
    def grammar_as_text(self) -> str:
        return "a float"


@register_validator("url_validator")
class UrlValidator(FormValidator):

    # @staticmethod
    @property
    def grammar(self) -> Regex:
        # URL or 'None'
        return Regex(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|None", flags=re.IGNORECASE)
    @property
    def grammar_as_text(self) -> str:
        return "a URL"


@register_validator("real_url_validator")
class ReachableUrlValidator(Validator):
    def validate(self, url) -> Regex:
        import requests
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
            else:
                raise Exception(f"URL returned status code {response.status_code}")
        except Exception as e:
            raise Exception(f"URL could not be reached: {e}")

    @staticmethod
    def debug(url: str, exception: Exception) -> str:
        return Prompt(f"""The URL {url} is unreachable and returned the following error: {exception}. Please enter a valid URL.""")  # noqa: E501


@register_validator("list_validator")
class ListValidator(FormValidator):

    def __init__(
            self,
            min_length: Optional[int] = None,
            max_length: Optional[int] = None,
            element_validators: Optional[Union[Validator, List[Validator]]] = None
            ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.element_validators = element_validators
        super().__init__()

    @property
    def grammar(self) -> Regex:
        # Parse a list of the form [a, b, c, ...]
        return Regex(r"\[.*\]", flags=re.IGNORECASE)

    @property
    def grammar_as_text(self) -> str:
        return "a list"
