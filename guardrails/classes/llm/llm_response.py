from typing import Iterable, Optional

from guardrails_api_client import LLMResponse as ILLMResponse


# TODO: We might be able to delete this
class LLMResponse(ILLMResponse):
    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    output: str
    stream_output: Optional[Iterable] = None
