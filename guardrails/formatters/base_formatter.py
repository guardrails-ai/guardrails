from abc import ABC, abstractmethod

from guardrails.llm_providers import (
    ArbitraryCallable,
    PromptCallableBase,
)


class BaseFormatter(ABC):
    @abstractmethod
    def wrap_callable(self, llm_callable: PromptCallableBase) -> ArbitraryCallable: ...


class PassthroughFormatter(BaseFormatter):
    def wrap_callable(self, llm_callable: PromptCallableBase):
        return llm_callable  # Noop
