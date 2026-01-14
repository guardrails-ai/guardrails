import json
import pytest

from guardrails.utils.parsing_utils import (
    get_code_block,
    has_code_block,
    prune_extra_keys,
)

json_code_block = """
```json
{
    "a": 1
}
```
"""

anonymous_code_block = """
```
{
    "a": 1
}
```
"""

no_code_block = """
{
    "a": 1
}
"""

js_code_block = """
```js
{
    "a": 1
}
```
"""


not_even_json = "This isn't even json..."


@pytest.mark.parametrize(
    "llm_ouput,expected_output",
    [
        (json_code_block, (True, 1, 24)),
        (anonymous_code_block, (True, 1, 20)),
        (js_code_block, (True, 1, 22)),
        (no_code_block, (False, None, None)),
        (not_even_json, (False, None, None)),
    ],
)
def test_has_code_block(llm_ouput, expected_output):
    actual_output = has_code_block(llm_ouput)

    assert actual_output == expected_output


json_code = """{
    "a": 1
}"""
js_code = """js
{
    "a": 1
}"""


@pytest.mark.parametrize(
    "llm_ouput,expected_output,code_type",
    [
        (json_code_block, json_code, "json"),
        (anonymous_code_block, json_code, ""),
        (js_code_block, js_code, ""),
    ],
)
def test_get_code_block(llm_ouput, expected_output, code_type):
    has, start, end = has_code_block(llm_ouput)
    actual_output = get_code_block(llm_ouput, start, end, code_type)

    assert actual_output == expected_output


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
    "schema,payload,pruned_payload",
    [
        (
            choice_case_openapi_schema,
            {
                "action": {
                    "chosen_action": "fight",
                    "weapon": "crossbow",
                    "ammo": "fire bolts",
                },
                "reason": "Peregrin Took is a brave hobbit",
            },
            {"action": {"chosen_action": "fight", "weapon": "crossbow"}},
        ),
        (
            choice_case_schema,
            {
                "action": {
                    "chosen_action": "flight",
                    "flight_direction": "north",
                    "distance": 3,
                    "unit": "miles",
                },
                "reason": "Fly you fools!",
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
                        "index": 5,
                        "name": "Foreign Transactions",
                        "explanation": "3% of the amount of each transaction in U.S. dollars.",  # noqa
                        "value": 0,
                        "extra": "some value",
                    },
                    {
                        "index": 6,
                        "name": "Penalty Fees - Late Payment",
                        "explanation": "Up to $40.",
                        "value": 40,
                        "different_extra": "some other value",
                    },
                ],
                "interest_rates": {
                    "any_key": "doesn't matter",
                    "because": "this object is a wildcard",
                },
            },
            {
                "fees": [
                    {
                        "index": 5,
                        "name": "Foreign Transactions",
                        "explanation": "3% of the amount of each transaction in U.S. dollars.",  # noqa
                        "value": 0,
                    },
                    {
                        "index": 6,
                        "name": "Penalty Fees - Late Payment",
                        "explanation": "Up to $40.",
                        "value": 40,
                    },
                ],
                "interest_rates": {
                    "any_key": "doesn't matter",
                    "because": "this object is a wildcard",
                },
            },
        ),
        (
            string_schema,
            "Some string...",
            "Some string...",
        ),
    ],
)
def test_prune_extra_keys(schema, payload, pruned_payload):
    actual = prune_extra_keys(payload, schema)
    assert actual == pruned_payload
