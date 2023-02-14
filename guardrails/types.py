from typing import List, Optional, Union
from dataclasses import dataclass


@dataclass
class BaseType:
    name: Optional[str] = None
    validator: Optional[str] = None
    prompt_template: Optional[str] = None


class String(BaseType):
    pass


class Float(BaseType):
    pass


class CodeSnippet(BaseType):
    pass


class URL(BaseType):
    pass


class Email(BaseType):
    pass


class Date(BaseType):
    pass


class Time(BaseType):
    pass


class Percentage(BaseType):
    pass


