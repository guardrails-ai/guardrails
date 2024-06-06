import json
from copy import deepcopy
from typing import Dict, Optional, cast
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from guardrails.classes.input_type import InputType
from guardrails.errors import ValidationError
from guardrails.guard import Guard


class GuardRunnable(Runnable):
    guard: Guard

    def __init__(self, guard: Guard):
        self.name = guard.name
        self.guard = guard

    def invoke(
        self, input: InputType, config: Optional[RunnableConfig] = None
    ) -> InputType:
        output = BaseMessage(content="", type="")
        str_input = None
        input_is_chat_message = False
        if isinstance(input, BaseMessage):
            input_is_chat_message = True
            str_input = str(input.content)
            output = deepcopy(input)
        else:
            str_input = str(input)

        response = self.guard.validate(str_input)

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
            return cast(InputType, output)
        return cast(InputType, validated_output)
