from guardrails.actions.reask import FieldReAsk
from guardrails.classes.validation.validation_result import FailResult

VALIDATOR_PARALLELISM_REASK_1 = FieldReAsk(
    incorrect_value="Hello a you\nand me",
    fail_results=[
        FailResult(
            outcome="fail",
            error_message="must be exactly two words",
            fix_value="Hello a",
        ),
        FailResult(
            outcome="fail",
            error_message="Value Hello a you\nand me is not lower case.",
            fix_value="hello a you\nand me",
        ),
        FailResult(
            outcome="fail",
            error_message="Value has length greater than 10. Please return a shorter output, that is shorter than 10 characters.",  # noqa: E501
            fix_value="Hello a yo",
        ),
    ],
)
