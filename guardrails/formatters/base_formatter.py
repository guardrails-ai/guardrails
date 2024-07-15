from abc import ABC, abstractmethod

from guardrails.llm_providers import (
    ArbitraryCallable,
    PromptCallableBase,
)


class BaseFormatter(ABC):
    """A Formatter takes an LLM Callable and wraps the method into an abstract
    callable.

    Used to perform manipulations of the input or the output, like JSON
    constrained- decoding.
    """

    @abstractmethod
    def wrap_callable(self, llm_callable: PromptCallableBase) -> ArbitraryCallable: ...


class PassthroughFormatter(BaseFormatter):
    def wrap_callable(self, llm_callable: PromptCallableBase):
        return llm_callable  # Noop
