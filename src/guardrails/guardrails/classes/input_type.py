from typing import TypeVar

from langchain_core.messages import BaseMessage

InputType = TypeVar("InputType", str, BaseMessage)
