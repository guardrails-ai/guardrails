from guardrails.utils.reask_utils import SkeletonReAsk
from guardrails.validators import FailResult

MSG_VALIDATED_OUTPUT_REASK = SkeletonReAsk(
    incorrect_value={"name": "Inception", "director": "Christopher Nolan"},
    fail_results=[
        FailResult(
            outcome="fail",
            metadata=None,
            error_message="JSON does not match schema",
            fix_value=None,
        )
    ],
)
