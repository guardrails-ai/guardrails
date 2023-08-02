from guardrails.utils.reask_utils import FieldReAsk
from guardrails.validators import FailResult

VALIDATOR_PARALLELISM_REASK_2 = FieldReAsk(
    incorrect_value="hi theremynameispete",
    fail_results=[
        FailResult(
            outcome="fail",
            metadata=None,
            error_message="Value has length greater than 10. "
            "Please return a shorter output, "
            "that is shorter than 10 characters.",
            fix_value="hi theremy",
        )
    ],
)
