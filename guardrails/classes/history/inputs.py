from __future__ import annotations
from typing_extensions import deprecated
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_serializer, field_validator

from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.classes.llm.prompt_callable import PromptCallableBase
from guardrails.prompt.prompt import Prompt
from guardrails.prompt.messages import Messages
from guardrails.prompt.instructions import Instructions


class Inputs(ArbitraryModel):
    """Inputs represent the input data that is passed into the validation
    loop."""

    llm_api: Optional[PromptCallableBase] = Field(
        description="The constructed class for calling the LLM.", default=None
    )
    llm_output: Optional[str] = Field(
        description="The string output from an external LLM call"
        "provided by the user via Guard.parse.",
        default=None,
    )
    messages: Optional[
        Union[List[Dict[str, Union[str, Prompt, Instructions]]], Messages]
    ] = Field(
        description="The message history provided by the user for chat model calls.",
        default=None,
    )
    prompt_params: Optional[Dict] = Field(
        description="The parameters provided by the user"
        "that will be formatted into the final LLM prompt.",
        default=None,
    )
    num_reasks: Optional[int] = Field(
        description="The total number of reasks allowed; user provided or defaulted.",
        default=None,
    )
    metadata: Optional[Dict[str, Any]] = Field(
        description="The metadata provided by the user to be used during validation.",
        default=None,
    )
    full_schema_reask: Optional[bool] = Field(
        description="Whether to perform reasks across the entire schema"
        "or at the field level.",
        default=None,
    )
    stream: Optional[bool] = Field(
        description="Whether to use streaming.",
        default=False,
    )

    @field_serializer("llm_api")
    def serialize_llm_api(self, llm_api: PromptCallableBase | None) -> str | None:
        if llm_api:
            return str(llm_api)
        return None

    @field_validator("llm_api", mode="before")
    @classmethod
    def deserialize_llm_api(cls, llm_api: Any) -> PromptCallableBase | None:
        if isinstance(llm_api, PromptCallableBase):
            return llm_api
        # Note: We can potentially identify the correct
        #  PrompCallable Class and reconstruct it,
        # but the previous implementation always just returned None.
        return None

    @field_serializer("messages")
    def serialize_messages(
        self, messages: list[dict[str, str | Prompt | Instructions]] | Messages | None
    ) -> list[dict[str, Any]] | None:
        # Legacy serialization logic from previous to_interface implementation
        # TODO: Just make Prompt, Instructions, and Messages pydantic models
        if messages:
            serialized_messages = []
            for msg in messages:
                ser_msg = {**msg}
                content = ser_msg.get("content")
                if content:
                    ser_msg["content"] = (
                        content.source if isinstance(content, Prompt) else content
                    )
                serialized_messages.append(ser_msg)
            return serialized_messages
        return None

    @deprecated("Use Inputs.model_dump() instead.")
    def to_interface(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    @deprecated("Use Inputs.model_dump() instead.")
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    @classmethod
    @deprecated("Use Inputs.model_validate() instead.")
    def from_interface(cls, i_inputs: Any) -> "Inputs":
        return cls.model_validate(i_inputs)

    @classmethod
    @deprecated("Use Inputs.model_validate() instead.")
    def from_dict(cls, obj: Any) -> "Inputs":
        return cls.model_validate(obj)
