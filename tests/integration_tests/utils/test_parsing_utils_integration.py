import json
import pytest

from guardrails.utils.parsing_utils import coerce_types


with open(
    "tests/integration_tests/test_assets/json_schemas/choice_case.json", "r"
) as choice_case_json_file:
    choice_case_json_schema = json.loads(choice_case_json_file.read())

with open(
    "tests/integration_tests/test_assets/json_schemas/choice_case_openapi.json", "r"
) as choice_case_openapi_file:
    choice_case_openapi_schema = json.loads(choice_case_openapi_file.read())

with open(
    "tests/integration_tests/test_assets/json_schemas/credit_card_agreement.json", "r"
) as credit_card_agreement_file:
    credit_card_agreement_schema = json.loads(credit_card_agreement_file.read())


string_schema = {"type": "string"}
integer_schema = {"type": "integer"}
float_schema = {"type": "number"}


@pytest.mark.parametrize(
    "schema,given,expected",
    [
        (string_schema, 3.1, "3.1"),
        (
            integer_schema,
            "3.1",
            "3.1",  # doesn't work on float strings
        ),
        (integer_schema, "3", 3),
        (integer_schema, 3.1, 3),
        (float_schema, "3", 3.0),
        (
            choice_case_json_schema,
            {
                "action": {
                    "chosen_action": "flight",
                    "flight_direction": "north",
                    "distance": 3.1,
                }
            },
            {
                "action": {
                    "chosen_action": "flight",
                    "flight_direction": "north",
                    "distance": 3,
                }
            },
        ),
        (
            choice_case_openapi_schema,
            {
                "action": {
                    "chosen_action": "flight",
                    "flight_direction": "north",
                    "distance": "3",
                }
            },
            {
                "action": {
                    "chosen_action": "flight",
                    "flight_direction": "north",
                    "distance": 3,
                }
            },
        ),
        (
            credit_card_agreement_schema,
            {
                "fees": [
                    {
                        "index": "5",
                        "name": "Foreign Transactions",
                        "explanation": "3% of the amount of each transaction in U.S. dollars.",  # noqa
                        "value": "0",
                    },
                    {
                        "index": 6.0,
                        "name": "Penalty Fees - Late Payment",
                        "explanation": "Up to $40.",
                        "value": 40,
                    },
                ],
                "interest_rates": {
                    "any_key": 123,
                    "doesnt_matter": "this object is a wildcard",
                },
            },
            {
                "fees": [
                    {
                        "index": 5,
                        "name": "Foreign Transactions",
                        "explanation": "3% of the amount of each transaction in U.S. dollars.",  # noqa
                        "value": 0.0,
                    },
                    {
                        "index": 6,
                        "name": "Penalty Fees - Late Payment",
                        "explanation": "Up to $40.",
                        "value": 40.0,
                    },
                ],
                "interest_rates": {
                    "any_key": "123",
                    "doesnt_matter": "this object is a wildcard",
                },
            },
        ),
    ],
)
def test_coerce_types(schema, given, expected):
    coerced_payload = coerce_types(given, schema)
    assert coerced_payload == expected
