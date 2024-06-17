from typing import Any, Dict, List, Optional

from pydantic import Field

from guardrails_api_client import Inputs as IInputs
from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.llm_providers import PromptCallableBase
from guardrails.prompt.instructions import Instructions
from guardrails.prompt.prompt import Prompt


class Inputs(IInputs, ArbitraryModel):
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
    num_reasks: int = Field(
        description="The total number of reasks allowed; user provided or defaulted.",
        default=None,
    )
    metadata: Optional[Dict[str, Any]] = Field(
        description="The metadata provided by the user to be used during validation.",
        default=None,
    )
    full_schema_reask: bool = Field(
        description="Whether to perform reasks across the entire schema"
        "or at the field level.",
        default=None,
    )
    stream: Optional[bool] = Field(
        description="Whether to use streaming.",
        default=False,
    )
