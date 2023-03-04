from typing import List, Optional, Union, Callable

import guardrails.deprecated.validators as validators


class DataType:

    def __init__(
        self,
        name: Optional[str] = None,
        validator: Optional[validators.Validator] = None,
        prompt_template: Optional[str] = None
    ):
        self._name = name
        self._validator = validator
        self._prompt_template = prompt_template

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def validator(self) -> str:
        return self._validator

    @validator.setter
    def validator(self, validator: str):
        self._validator = validator

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, prompt_template: str):
        self._prompt_template = prompt_template


class String(DataType):
    def __init__(
        self,
        name: Optional[str] = None,
        validator: Optional[validators.Validator] = validators.StringValidator,
        prompt_template: Optional[str] = None
    ):
        super().__init__(name, validator, prompt_template)


class URL(DataType):
    pass


class Float(DataType):
    def __init__(self, name: Optional[str] = None, validator: Optional[validators.Validator] = validators.FloatValidator, prompt_template: Optional[str] = None):
        super().__init__(name, validator, prompt_template)


class CodeSnippet(DataType):
    raise NotImplementedError


class Email(DataType):
    raise NotImplementedError


class Date(DataType):
    raise NotImplementedError


class Time(DataType):
    raise NotImplementedError


class Percentage(DataType):
    raise NotImplementedError
