from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import Field

from guardrails_api_client import CallInputs as ICallInputs
from guardrails.classes.history.inputs import Inputs
from guardrails.classes.generic.arbitrary_model import ArbitraryModel


class CallInputs(Inputs, ICallInputs, ArbitraryModel):
    """CallInputs represent the input data that is passed into the Guard from
    the user. Inherits from Inputs with the below overrides and additional
    attributes.

    Attributes:
        llm_api (Optional[Callable[[Any], Awaitable[Any]]]): The LLM function
            provided by the user during Guard.__call__ or Guard.parse.
        messages (Optional[dict[str, str]]): The messages as provided by the user.
        args (List[Any]): Additional arguments for the LLM as provided by the user.
            Default [].
        kwargs (Dict[str, Any]): Additional keyword-arguments for
            the LLM as provided by the user. Default {}.
    """

    llm_api: Optional[Callable[[Any], Awaitable[Any]]] = Field(
        description="The LLM function provided by the user"
        "during Guard.__call__ or Guard.parse.",
        default=None,
    )
    messages: Optional[list[dict[str, str]]] = Field(
        description="The messages as provided by the user.", default=None
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
        inputs = super().to_interface().to_dict() or {}
        inputs["args"] = self.args
        # TODO: Better way to prevent creds from being logged,
        #   if they're passed in as kwargs to the LLM
        redacted_kwargs = {}
        for k, v in self.kwargs.items():
            if ("key" in k.lower() or "token" in k.lower()) and isinstance(v, str):
                redaction_length = len(v) - 4
                stars = "*" * redaction_length
                redacted_kwargs[k] = f"{stars}{v[-4:]}"
            else:
                redacted_kwargs[k] = v
        inputs["kwargs"] = redacted_kwargs
        return ICallInputs(**inputs)

    def to_dict(self) -> Dict[str, Any]:
        return self.to_interface().to_dict()

    @classmethod
    def from_interface(cls, i_call_inputs: ICallInputs) -> "CallInputs":
        return cls(
            llm_api=None,
            llm_output=i_call_inputs.llm_output,
            messages=i_call_inputs.messages,  # type: ignore
            prompt_params=i_call_inputs.prompt_params,
            num_reasks=i_call_inputs.num_reasks,
            metadata=i_call_inputs.metadata,
            full_schema_reask=(i_call_inputs.full_schema_reask is True),
            stream=(i_call_inputs.stream is True),
            args=(i_call_inputs.args or []),
            kwargs=(i_call_inputs.kwargs or {}),
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]):
        i_call_inputs = ICallInputs.from_dict(obj) or ICallInputs()

        return cls.from_interface(i_call_inputs)
