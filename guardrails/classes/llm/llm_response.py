import asyncio
from typing import Any, Dict, Iterable, Optional, AsyncIterable

from guardrails_api_client import LLMResponse as ILLMResponse
from pydantic.config import ConfigDict


# TODO: Move this somewhere that makes sense
def async_to_sync(awaitable):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(awaitable)


# TODO: We might be able to delete this
class LLMResponse(ILLMResponse):
    """Standard information collection from LLM responses to feed the
    validation loop.

    Attributes:
        output (str): The output from the LLM.
        stream_output (Optional[Iterable]): A stream of output from the LLM.
            Default None.
        async_stream_output (Optional[AsyncIterable]): An async stream of output
            from the LLM.  Default None.
        prompt_token_count (Optional[int]): The number of tokens in the prompt.
            Default None.
        response_token_count (Optional[int]): The number of tokens in the response.
            Default None.
    """

    # Pydantic Config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    output: str
    stream_output: Optional[Iterable] = None
    async_stream_output: Optional[AsyncIterable] = None

    def to_interface(self) -> ILLMResponse:
        stream_output = None
        if self.stream_output:
            stream_output = [str(so) for so in self.stream_output]

        async_stream_output = None
        if self.async_stream_output:
            async_stream_output = [str(async_to_sync(so)) for so in self.stream_output]  # type: ignore - we just established it isn't None

        return ILLMResponse(
            prompt_token_count=self.prompt_token_count,  # type: ignore - pyright doesn't understand aliases
            response_token_count=self.response_token_count,  # type: ignore - pyright doesn't understand aliases
            output=self.output,
            stream_output=stream_output,  # type: ignore - pyright doesn't understand aliases
            async_stream_output=async_stream_output,  # type: ignore - pyright doesn't understand aliases
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.to_interface().to_dict()

    @classmethod
    def from_interface(cls, i_llm_response: ILLMResponse) -> "LLMResponse":
        stream_output = None
        if i_llm_response.stream_output:
            stream_output = [so for so in i_llm_response.stream_output]

        async_stream_output = None
        if i_llm_response.async_stream_output:

            async def async_iter():
                for aso in i_llm_response.async_stream_output:  # type: ignore - just verified it isn't None...
                    yield aso

            async_stream_output = async_iter()

        return cls(
            prompt_token_count=i_llm_response.prompt_token_count,
            response_token_count=i_llm_response.response_token_count,
            output=i_llm_response.output,
            stream_output=stream_output,
            async_stream_output=async_stream_output,
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "LLMResponse":
        i_llm_response = super().from_dict(obj) or ILLMResponse(output="")

        return cls.from_interface(i_llm_response)
