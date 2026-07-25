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


class TestCoercePropertyFalsyValues:
    """Regression tests: coerce_property must process falsy values (0, False,
    empty string) rather than skipping them."""

    def test_zero_integer_is_coerced(self):
        """Integer 0 in payload should still be coerced to target type."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "string"},
            },
        }
        payload = {"count": 0}
        result = coerce_types(payload, schema)
        assert result == {"count": "0"}

    def test_false_boolean_is_coerced(self):
        """Boolean False in payload should still be coerced to target type."""
        schema = {
            "type": "object",
            "properties": {
                "flag": {"type": "string"},
            },
        }
        payload = {"flag": False}
        result = coerce_types(payload, schema)
        assert result == {"flag": "False"}

    def test_empty_string_is_coerced(self):
        """Empty string in payload should still be coerced to target type."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "integer"},
            },
        }
        # int("") raises ValueError, coerce() returns original value
        payload = {"name": ""}
        result = coerce_types(payload, schema)
        assert result == {"name": ""}

    def test_zero_float_is_coerced(self):
        """Float 0.0 in payload should still be coerced to target type."""
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "string"},
            },
        }
        payload = {"score": 0.0}
        result = coerce_types(payload, schema)
        assert result == {"score": "0.0"}

    def test_additional_properties_falsy_values(self):
        """Falsy values in additional properties should also be coerced."""
        schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": {"type": "string"},
        }
        payload = {"count": 0, "flag": False}
        result = coerce_types(payload, schema)
        assert result == {"count": "0", "flag": "False"}

    def test_none_is_still_skipped(self):
        """None values (missing keys) should still be skipped."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "integer"},
                "age": {"type": "string"},
            },
        }
        payload = {"name": "42"}
        result = coerce_types(payload, schema)
        assert result == {"name": 42}
