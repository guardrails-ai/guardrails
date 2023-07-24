# flake8: noqa: E501
from guardrails.utils.reask_utils import FieldReAsk

VALIDATED_OUTPUT_REASK_1 = {
    "fees": [
        {
            "index": 1,
            "name": "annual membership",
            "explanation": "Annual Membership Fee",
            "value": 0,
        },
        {
            "index": 2,
            "name": FieldReAsk(
                incorrect_value="my chase plan",
                error_message="must be exactly two words",
                fix_value="my chase",
                path=["fees", 1, "name"],
            ),
            "explanation": "My Chase Plan Fee (fixed finance charge)",
            "value": 1.72,
        },
        {
            "index": 3,
            "name": "balance transfers",
            "explanation": "Balance Transfers Intro fee of either $5 or 3% of the amount of each transfer, whichever is greater, on transfers made within 60 days of account opening. After that: Either $5 or 5% of the amount of each transfer.",
            "value": 5,
        },
        {
            "index": 4,
            "name": "cash advances",
            "explanation": "Cash Advances Either $10 or 5% of the amount of each transaction, whichever is greater.",
            "value": 5,
        },
        {
            "index": 5,
            "name": "foreign transactions",
            "explanation": "Foreign Transactions 3% of the amount of each transaction in U.S. dollars.",
            "value": 3,
        },
        {
            "index": 6,
            "name": "late payment",
            "explanation": "Late Payment Up to $40.",
            "value": 0,
        },
        {
            "index": 7,
            "name": FieldReAsk(
                incorrect_value="over-the-credit-limit",
                error_message="must be exactly two words",
                fix_value="over-the-credit-limit",
                path=["fees", 6, "name"],
            ),
            "explanation": "Over-the-Credit-Limit None",
            "value": 0,
        },
        {
            "index": 8,
            "name": "return payment",
            "explanation": "Return Payment Up to $40.",
            "value": 0,
        },
        {
            "index": 9,
            "name": "return check",
            "explanation": "Return Check None",
            "value": 0,
        },
    ],
    "interest_rates": {
        "purchase": {
            "apr": 0,
            "explanation": "Purchase Annual Percentage Rate (APR) 0% Intro APR for the first 18 months that your Account is open. After that, 19.49%. This APR will vary with the market based on the Prime Rate.",
        },
        "my_chase_loan": {
            "apr": 19.49,
            "explanation": "My Chase Loan SM APR 19.49%. This APR will vary with the market based on the Prime Rate.",
        },
        "balance_transfer": {
            "apr": 0,
            "explanation": "Balance Transfer APR 0% Intro APR for the first 18 months that your Account is open. After that, 19.49%. This APR will vary with the market based on the Prime Rate.",
        },
        "cash_advance": {
            "apr": 29.49,
            "explanation": "Cash Advance APR 29.49%. This APR will vary with the market based on the Prime Rate.",
        },
        "penalty": {
            "apr": 29.99,
            "explanation": "Up to 29.99%. This APR will vary with the market based on the Prime Rate.",
        },
        "maximum_apr": 29.99,
    },
}
