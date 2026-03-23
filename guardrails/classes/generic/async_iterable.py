from __future__ import annotations
from typing import Generic, TypeVar

from pydantic import BaseModel, Field, model_serializer

T = TypeVar("T")


class SerializeableAsyncIterable(BaseModel, Generic[T]):
    content: list[T] = Field(default_factory=lambda: [])
    _index: int = 0

    @model_serializer(mode="plain")
    def serialize_model(self) -> list[T]:
        return self.content

    async def __anext__(self) -> T:
        if self._index >= len(self.content):
            raise StopAsyncIteration
        value = self.content[self._index]
        self._index += 1
        return value

    def __aiter__(self) -> SerializeableAsyncIterable[T]:
        self._index = 0
        return self
