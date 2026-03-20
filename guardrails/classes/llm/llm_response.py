import asyncio
from itertools import tee
from typing import Any, Dict, Iterator, Optional, AsyncIterator, Iterable, Tuple
from typing_extensions import deprecated

from pydantic import Field, field_serializer, field_validator

from guardrails.classes.generic.arbitrary_model import ArbitraryModel


# TODO: Move this somewhere that makes sense
def async_to_sync(awaitable):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(awaitable)


async def iterable_to_async_iter(sync_iterable: Iterable) -> AsyncIterator:
    for i in sync_iterable:
        yield i


async def serialize_aiter(
    async_iter: AsyncIterator,
) -> Tuple[Optional[list[str]], AsyncIterator]:
    _iterable = []
    iter_output: list[str] = []
    async for so in async_iter:
        _iterable.append(so)
        iter_output.append(str(so))

    return iter_output, iterable_to_async_iter(_iterable)


# TODO: We might be able to delete this
class LLMResponse(ArbitraryModel):
    """Standard information collection from LLM responses to feed the
    validation loop."""

    # Pydantic Config
    model_config = {
        "validate_by_alias": True,
        "validate_by_name": True,
        "arbitrary_types_allowed": True,
    }

    prompt_token_count: Optional[int] = Field(
        default=None,
        alias="promptTokenCount",
        description="The number of tokens in the prompt.",
    )
    response_token_count: Optional[int] = Field(
        default=None,
        alias="responseTokenCount",
        description="The number of tokens in the response.",
    )
    output: str = Field(default="", description="The output from the LLM.")
    stream_output: Optional[Iterator] = Field(
        default=None,
        alias="streamOutput",
        description="A stream of output from the LLM.",
    )
    async_stream_output: Optional[AsyncIterator] = Field(
        default=None,
        alias="asyncStreamOutput",
        description="An async stream of output from the LLM.",
    )

    @field_serializer("stream_output")
    def serialize_stream_output(
        self, stream_output: Iterator | None
    ) -> list[str] | None:
        if stream_output:
            copy_1, copy_2 = tee(stream_output)
            self.stream_output = copy_1
            ser_stream_output = [str(so) for so in copy_2]
            return ser_stream_output
        return None

    @field_validator("stream_output", mode="before")
    @classmethod
    def deserialize_stream_output(cls, stream_output: Any | None) -> Iterator | None:
        if isinstance(stream_output, Iterator):
            return stream_output
        if stream_output:
            try:
                return iter(stream_output)
            except TypeError:
                return None
        return None

    @field_serializer("async_stream_output")
    def serialize_async_stream_output(
        self, async_stream_output: AsyncIterator | None
    ) -> list[str] | None:
        # Legacy serialization logic from previous to_interface implementation
        # We probably need a wrapper class for these.
        if async_stream_output:  # and not hasattr(async_stream_output, "__aiter__"):
            awaited_stream_output, _async_stream_output = async_to_sync(
                serialize_aiter(async_stream_output)
            )

            self.async_stream_output = _async_stream_output
            return awaited_stream_output
        return None

    @field_validator("async_stream_output", mode="before")
    @classmethod
    def deserialize_async_stream_output(
        cls, async_stream_output: Any | None
    ) -> AsyncIterator | None:
        if isinstance(async_stream_output, AsyncIterator):
            return async_stream_output
        if async_stream_output and isinstance(async_stream_output, Iterable):

            async def async_iter():
                for aso in async_stream_output:
                    yield aso

            return async_iter()
        return None

    @deprecated("Use LLMResponse.model_dump() instead.")
    def to_interface(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    @deprecated("Use LLMResponse.model_dump() instead.")
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    @classmethod
    @deprecated("Use LLMResponse.model_validate() instead.")
    def from_interface(cls, i_llm_response: Any) -> "LLMResponse":
        return cls.model_validate(i_llm_response)

    @classmethod
    @deprecated("Use LLMResponse.model_validate() instead.")
    def from_dict(cls, obj: Any) -> "LLMResponse":
        return cls.model_validate(obj)
