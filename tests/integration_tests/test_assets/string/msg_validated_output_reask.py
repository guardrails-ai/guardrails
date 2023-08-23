from guardrails.utils.reask_utils import FieldReAsk
from guardrails.validators import FailResult

MSG_VALIDATED_OUTPUT_REASK = FieldReAsk(
    incorrect_value="The Matrix Reloaded",
    fail_results=[
        FailResult(
            error_message="must be exactly two words",
            fix_value="The Matrix",
        )
    ],
)
