import json
from copy import deepcopy
from typing import Dict, Optional, TypeVar, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

from guardrails.errors import ValidationError
from guardrails.functional.guard import Guard as FGuard

T = TypeVar("T", str, BaseMessage)


class Guard(FGuard, Runnable):
    def invoke(self, input: T, config: Optional[RunnableConfig] = None) -> T:
        output = BaseMessage(content="", type="")
        str_input = None
        input_is_chat_message = False
        if isinstance(input, BaseMessage):
            input_is_chat_message = True
            str_input = str(input.content)
            output = deepcopy(input)
        else:
            str_input = str(input)

        response = self.validate(str_input)

        validated_output = response.validated_output
        if not validated_output:
            raise ValidationError(
                (
                    "The response from the LLM failed validation!"
                    "See `guard.history` for more details."
                )
            )

        if isinstance(validated_output, Dict):
            validated_output = json.dumps(validated_output)

        if input_is_chat_message:
            output.content = validated_output
            return cast(T, output)
        return cast(T, validated_output)
