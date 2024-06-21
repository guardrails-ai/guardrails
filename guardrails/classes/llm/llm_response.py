from typing import Any, Dict, Iterable, Optional, AsyncIterable

from guardrails_api_client import LLMResponse as ILLMResponse
from pydantic.config import ConfigDict


# TODO: We might be able to delete this
class LLMResponse(ILLMResponse):
    # Pydantic Config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    output: str
    stream_output: Optional[Iterable] = None
    async_stream_output: Optional[AsyncIterable] = None

    def to_interface(self) -> ILLMResponse:
        return ILLMResponse(
            prompt_token_count=self.prompt_token_count,
            response_token_count=self.response_token_count,
            output=self.output,
            stream_output=[str(so) for so in self.stream_output],
            async_stream_output=[str(aso) for aso in self.async_stream_output],
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.to_interface().to_dict()

    @classmethod
    def from_interface(cls, i_llm_response: ILLMResponse) -> "LLMResponse":
        return cls(
            prompt_token_count=i_llm_response.prompt_token_count,
            response_token_count=i_llm_response.response_token_count,
            output=i_llm_response.output,
            stream_output=[so for so in i_llm_response.stream_output],
            async_stream_output=[aso for aso in i_llm_response.async_stream_output],
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "LLMResponse":
        i_llm_response = super().from_dict(obj)

        return cls.from_interface(i_llm_response)
