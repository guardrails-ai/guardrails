from typing import Any
from guardrails_api_client import Guard as GuardStruct
from pydantic import ConfigDict
from guardrails.classes.generic import Stack


class MockGuardStruct(GuardStruct):
    # Pydantic Config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = "mock-guard-id"
    name: str = "mock-guard"
    description: str = "mock guard description"
    history: Stack[Any] = Stack()

    def to_guard(self, *args):
        return self

    def parse(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class MockGuardClient:
    def get_guards(self):
        return [MockGuardStruct()]

    def create_guard(self, guard: MockGuardStruct):
        return MockGuardStruct()
