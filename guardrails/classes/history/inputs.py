from typing import Any, Dict, List, Optional

from pydantic import Field

from guardrails_api_client import Inputs as IInputs
from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.llm_providers import PromptCallableBase
from guardrails.prompt.instructions import Instructions
from guardrails.prompt.prompt import Prompt


class Inputs(IInputs, ArbitraryModel):
    """Inputs represent the input data that is passed into the validation loop.

    Attributes:
        llm_api (Optional[PromptCallableBase]): The constructed class
            for calling the LLM.
        llm_output (Optional[str]): The string output from an
            external LLM call provided by the user via Guard.parse.
        instructions (Optional[Instructions]): The constructed
            Instructions class for chat model calls.
        prompt (Optional[Prompt]): The constructed Prompt class.
        msg_history (Optional[List[Dict]]): The message history
            provided by the user for chat model calls.
        prompt_params (Optional[Dict]): The parameters provided
            by the user that will be formatted into the final LLM prompt.
        num_reasks (Optional[int]): The total number of reasks allowed;
            user provided or defaulted.
        metadata (Optional[Dict[str, Any]]): The metadata provided
            by the user to be used during validation.
        full_schema_reask (Optional[bool]): Whether reasks we
            performed across the entire schema or at the field level.
        stream (Optional[bool]): Whether or not streaming was used.
    """

    llm_api: Optional[PromptCallableBase] = Field(
        description="The constructed class for calling the LLM.", default=None
    )
    llm_output: Optional[str] = Field(
        description="The string output from an external LLM call"
        "provided by the user via Guard.parse.",
        default=None,
    )
    instructions: Optional[Instructions] = Field(
        description="The constructed Instructions class for chat model calls.",
        default=None,
    )
    prompt: Optional[Prompt] = Field(
        description="The constructed Prompt class.", default=None
    )
    msg_history: Optional[List[Dict]] = Field(
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

    def to_interface(self) -> IInputs:
        serialized_msg_history = None
        if self.msg_history:
            serialized_msg_history = []
            for msg in self.msg_history:
                ser_msg = {**msg}
                content = ser_msg.get("content")
                if content:
                    ser_msg["content"] = (
                        content.source if isinstance(content, Prompt) else content
                    )
                serialized_msg_history.append(ser_msg)

        instructions = (
            self.instructions.source
            if isinstance(self.instructions, Instructions)
            else self.instructions
        )

        prompt = self.prompt.source if isinstance(self.prompt, Prompt) else self.prompt

        return IInputs(
            llm_api=str(self.llm_api) if self.llm_api else None,  # type: ignore - pyright doesn't understand aliases
            llm_output=self.llm_output,  # type: ignore - pyright doesn't understand aliases
            instructions=instructions,
            prompt=prompt,
            msg_history=serialized_msg_history,  # type: ignore - pyright doesn't understand aliases
            prompt_params=self.prompt_params,  # type: ignore - pyright doesn't understand aliases
            num_reasks=self.num_reasks,  # type: ignore - pyright doesn't understand aliases
            metadata=self.metadata,
            full_schema_reask=self.full_schema_reask,  # type: ignore - pyright doesn't understand aliases
            stream=self.stream,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.to_interface().to_dict()

    @classmethod
    def from_interface(cls, i_inputs: IInputs) -> "Inputs":
        deserialized_msg_history = None
        if i_inputs.msg_history:
            deserialized_msg_history = []
            for msg in i_inputs.msg_history:
                ser_msg = {**msg}
                content = ser_msg.get("content")
                if content:
                    ser_msg["content"] = Prompt(content)
                deserialized_msg_history.append(ser_msg)

        instructions = (
            Instructions(i_inputs.instructions) if i_inputs.instructions else None
        )

        prompt = Prompt(i_inputs.prompt) if i_inputs.prompt else None
        num_reasks = (
            int(i_inputs.num_reasks) if i_inputs.num_reasks is not None else None
        )
        return cls(
            llm_api=None,
            llm_output=i_inputs.llm_output,
            instructions=instructions,
            prompt=prompt,
            msg_history=deserialized_msg_history,
            prompt_params=i_inputs.prompt_params,
            num_reasks=num_reasks,
            metadata=i_inputs.metadata,
            full_schema_reask=(i_inputs.full_schema_reask is True),
            stream=(i_inputs.stream is True),
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Inputs":
        i_inputs = IInputs.from_dict(obj) or IInputs()
        return cls.from_interface(i_inputs)
