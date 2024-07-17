from guardrails.actions.reask import FieldReAsk
from guardrails.classes.validation.validation_result import FailResult

VALIDATED_OUTPUT_REASK = FieldReAsk(
    incorrect_value="Tomato Cheese Pizza",
    fail_results=[
        FailResult(
            error_message="must be exactly two words",
            fix_value="Tomato Cheese",
        )
    ],
)
