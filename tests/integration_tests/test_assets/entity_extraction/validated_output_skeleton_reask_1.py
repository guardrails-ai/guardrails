# flake8: noqa: E501
from guardrails.utils.reask_utils import SkeletonReAsk

VALIDATED_OUTPUT_SKELETON_REASK_1 = SkeletonReAsk(
    incorrect_value={
        "fees": [
            {"name": "annual membership fee", "value": 0.0},
            {"name": "my chase plan fee", "value": 1.72},
            {"name": "balance transfers", "value": 5.0},
            {"name": "cash advances", "value": 5.0},
            {"name": "foreign transactions", "value": 3.0},
            {"name": "late payment", "value": 0.0},
            {"name": "over-the-credit-limit", "value": 0.0},
            {"name": "return payment", "value": 0.0},
            {"name": "return check", "value": 0.0},
        ],
        "interest_rates": {
            "purchase": {
                "annual_percentage_rate": 0.0,
                "variation_explanation": "This APR will vary with the market based on the Prime Rate.",
            },
            "balance_transfer": {
                "annual_percentage_rate": 0.0,
                "variation_explanation": "This APR will vary with the market based on the Prime Rate.",
            },
            "cash_advance": {
                "annual_percentage_rate": 29.49,
                "variation_explanation": "This APR will vary with the market based on the Prime Rate.",
            },
            "penalty": {
                "annual_percentage_rate": 0.0,
                "variation_explanation": "Up to 29.99%. This APR will vary with the market based on the Prime Rate.",
                "when_applies": "We may apply the Penalty APR to your account if you: fail to make a Minimum Payment by the date and time that it is due; or make a payment to us that is returned unpaid.",
                "how_long_apr_applies": "If we apply the Penalty APR for either of these reasons, the Penalty APR could potentially remain in effect indefinitely.",
            },
        },
    },
    error_message="JSON does not match schema",
    fix_value=None,
)
