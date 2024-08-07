from copy import deepcopy
from typing import Dict, Optional, cast
import json
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from guardrails.classes.input_type import InputType
from guardrails.classes.output_type import OT


class BaseRunnable(Runnable):
    name: str

    def invoke(
        self, input: InputType, config: Optional[RunnableConfig] = None
    ) -> InputType:
        return self._call_with_config(
            self._process_input,
            input,
            config,
            run_type="parser",
        )

    def _process_input(self, input: InputType) -> InputType:
        str_input = str(input.content) if isinstance(input, BaseMessage) else str(input)

        validated_output = self._validate(str_input)

        if isinstance(validated_output, Dict):
            validated_output = json.dumps(validated_output)

        if isinstance(input, BaseMessage):
            output = deepcopy(input)
            output.content = validated_output
            return cast(InputType, output)

        return cast(InputType, validated_output)

    def _validate(self, input: str) -> OT:
        raise NotImplementedError
