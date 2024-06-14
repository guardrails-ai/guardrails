from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import Field

from guardrails_api_client import CallInputs as ICallInputs
from guardrails.classes.history.inputs import Inputs
from guardrails.classes.generic.arbitrary_model import ArbitraryModel


class CallInputs(Inputs, ICallInputs, ArbitraryModel):
    llm_api: Optional[Callable[[Any], Awaitable[Any]]] = Field(
        description="The LLM function provided by the user"
        "during Guard.__call__ or Guard.parse.",
        default=None,
    )
    prompt: Optional[str] = Field(
        description="The prompt string as provided by the user.", default=None
    )
    instructions: Optional[str] = Field(
        description="The instructions string as provided by the user.", default=None
    )
    args: List[Any] = Field(
        description="Additional arguments for the LLM as provided by the user.",
        default_factory=list,
    )
    kwargs: Dict[str, Any] = Field(
        description="Additional keyword-arguments for the LLM as provided by the user.",
        default_factory=dict,
    )
