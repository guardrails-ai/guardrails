from guardrails.integrations.langchain.base_runnable import BaseRunnable
from guardrails.guard import Guard
from guardrails.errors import ValidationError
from guardrails.classes.output_type import OT


class GuardRunnable(BaseRunnable):
    guard: Guard

    def __init__(self, guard: Guard):
        self.name = guard.name
        self.guard = guard

    def _validate(self, input: str) -> OT:
        response = self.guard.validate(input)
        validated_output = response.validated_output
        if not validated_output:
            raise ValidationError(
                (
                    "The response from the LLM failed validation!"
                    "See `guard.history` for more details."
                )
            )
        return validated_output
