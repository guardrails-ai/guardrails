import json
import jsonref
import re

import pytest

from guardrails.schema.generator import generate_example, gen_string
from guardrails.schema.validator import validate_payload, SchemaValidationError


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


@pytest.mark.parametrize(
    "schema",
    [
        (choice_case_json_schema),
        (choice_case_openapi_schema),
        (credit_card_agreement_schema),
        ({"type": "string", "format": "email"}),
        ({"type": "string", "format": "not a real format"}),
    ],
)
def test_generate_example(schema):
    dereferenced_schema = jsonref.replace_refs(schema)
    sample = generate_example(dereferenced_schema)
    try:
        validate_payload(sample, schema)
    except SchemaValidationError as sve:
        print(sve)
        print(json.dumps(sve.fields, indent=2))
        pytest.fail(reason="Schema validation failed!")
    except Exception as e:
        print(e)
        pytest.fail(reason="validate_payload raised an unexpected exception!")


@pytest.mark.parametrize(
    "schema,property_name,pattern",
    [
        (
            {"type": "string", "format": "email"},
            None,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
        ),
        ({"type": "string", "format": "not a real format"}, None, ".*"),
        (
            {"type": "string"},
            "email",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
        ),
    ],
)
def test_gen_string(schema, property_name, pattern):
    dereferenced_schema = jsonref.replace_refs(schema)
    sample = gen_string(dereferenced_schema, property_name=property_name)
    try:
        validate_payload(sample, schema)
    except Exception as e:
        pytest.fail(reason="validate_payload raises an exception!", msg=str(e))

    assert re.fullmatch(pattern, sample)
