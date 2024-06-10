from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import Field

from guardrails.classes.history.inputs import Inputs


class CallInputs(Inputs):
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
