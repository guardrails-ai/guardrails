from copy import deepcopy
from typing import Optional, cast
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from guardrails.classes.input_type import InputType
from guardrails.errors import ValidationError
from guardrails.validator_base import FailResult, Validator


class ValidatorRunnable(Runnable):
    validator: Validator

    def __init__(self, validator: Validator):
        self.name = validator.rail_alias
        self.validator = validator

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

        response = self.validator.validate(str_input, self.validator._metadata)

        if isinstance(response, FailResult):
            raise ValidationError(
                (
                    "The response from the LLM failed validation!"
                    f" {response.error_message}"
                )
            )

        if input_is_chat_message:
            output.content = str_input
            return cast(InputType, output)
        return cast(InputType, str_input)
