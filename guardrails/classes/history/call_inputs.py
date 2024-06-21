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

    def to_interface(self) -> ICallInputs:
        rest = super().to_interface()
        return ICallInputs(
            **rest,
            args=self.args,
            kwargs=self.kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.to_interface().to_dict()

    @classmethod
    def from_interface(cls, i_call_inputs: ICallInputs) -> "CallInputs":
        inputs = Inputs.from_interface(i_call_inputs)
        return cls(
            **inputs,
            args=i_call_inputs.args,
            kwargs=i_call_inputs.kwargs,
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]):
        i_call_inputs = super().from_dict(obj)

        return cls.from_interface(i_call_inputs)
