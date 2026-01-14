import json
from typing import Any
import pytest

from guardrails.schema.parser import (
    get_all_paths,
    get_value_from_path,
    write_value_to_path,
)


reader_object = {
    "a": 1,
    "b": {"b2": {"b3": 2}},
    "c": [{"c2": 31}, {"c2": 32}, {"c2": 33}],
    "d": {"d2": [1, 2, 3]},
}


@pytest.mark.parametrize(
    "path,expected_value",
    [
        ("$.a", 1),
        ("$.b.b2", {"b3": 2}),
        ("$.b.b2.b3", 2),
        ("$.c.2", {"c2": 33}),
        ("$.c.1.c2", 32),
        ("$.d.d2", [1, 2, 3]),
        ("$.d.d2.0", 1),
        ("$.d.d2.4", None),
    ],
)
def test_get_value_from_path(path: str, expected_value: Any):
    actual_value = get_value_from_path(reader_object, path)
    assert actual_value == expected_value


@pytest.mark.parametrize(
    "existing_value,path,write_value,expected_value",
    [
        (  # Writes new value top-level path
            {},
            "$.a",
            1,
            {"a": 1},
        ),
        (  # Writes new value at nested path
            {},
            "$.b.b2",
            {"b3": 2},
            {"b": {"b2": {"b3": 2}}},
        ),
        (  # Writes updated value at nested path
            {"b": {"b2": {"b3": 1}}},
            "$.b.b2.b3",
            2,
            {"b": {"b2": {"b3": 2}}},
        ),
        (  # Writes new value into empty array
            {},
            "$.c.2",
            3,
            {"c": [None, None, 3]},
        ),
        (  # Writes new value into existing array of object
            {"c": [{"c2": 31}]},
            "$.c.1.c2",
            32,
            {"c": [{"c2": 31}, {"c2": 32}]},
        ),
        (  # Writes updated value into existing array
            {"d": {"d2": [0, 2, 3]}},
            "$.d.d2.0",
            1,
            {"d": {"d2": [1, 2, 3]}},
        ),
    ],
)
def test_write_value_to_path(existing_value: Any, path: str, write_value: Any, expected_value: Any):
    actual_value = write_value_to_path(existing_value, path, write_value)
    assert actual_value == expected_value


with open(
    "tests/integration_tests/test_assets/json_schemas/choice_case_openapi.json", "r"
) as choice_case_openapi_file:
    choice_case_openapi_schema = json.loads(choice_case_openapi_file.read())

with open(
    "tests/integration_tests/test_assets/json_schemas/choice_case.json", "r"
) as choice_case_file:
    choice_case_schema = json.loads(choice_case_file.read())

with open(
    "tests/integration_tests/test_assets/json_schemas/credit_card_agreement.json", "r"
) as credit_card_agreement_file:
    credit_card_agreement_schema = json.loads(credit_card_agreement_file.read())

with open("tests/integration_tests/test_assets/json_schemas/string.json", "r") as string_file:
    string_schema = json.loads(string_file.read())


@pytest.mark.parametrize(
    "schema,expected_keys",
    [
        (
            choice_case_openapi_schema,
            set(
                [
                    "$",
                    "$.action",
                    "$.action.chosen_action",
                    "$.action.weapon",
                    "$.action.flight_direction",
                    "$.action.distance",
                ]
            ),
        ),
        (
            choice_case_schema,
            set(
                [
                    "$",
                    "$.action",
                    "$.action.chosen_action",
                    "$.action.weapon",
                    "$.action.flight_direction",
                    "$.action.distance",
                ]
            ),
        ),
        (
            credit_card_agreement_schema,
            set(
                [
                    "$",
                    "$.fees",
                    "$.fees.index",
                    "$.fees.name",
                    "$.fees.explanation",
                    "$.fees.value",
                    "$.interest_rates",
                    "$.interest_rates.*",
                ]
            ),
        ),
        (string_schema, set(["$"])),
    ],
)
def test_get_all_paths(schema, expected_keys):
    actual_keys = get_all_paths(schema)
    assert actual_keys == expected_keys
