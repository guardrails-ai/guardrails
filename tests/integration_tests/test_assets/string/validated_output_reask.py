from guardrails.utils.reask_utils import FieldReAsk
from guardrails.validators import FailResult

VALIDATED_OUTPUT_REASK = FieldReAsk(
    incorrect_value="Tomato Cheese Pizza",
    fail_results=[
        FailResult(
            error_message="must be exactly two words",
            fix_value="Tomato Cheese",
        )
    ],
)
