import copy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from guardrails import Guard, settings
from guardrails.utils.parsing_utils import coerce_types


@pytest.mark.parametrize(
    "schema_type,value",
    [
        ("string", "hello"),
        ("string", ""),
        ("integer", 42),
        ("integer", 0),
        ("number", 3.5),
        ("number", 3),
        ("boolean", True),
        ("boolean", False),
        ("object", {"answer": "hello"}),
        ("array", ["hello"]),
    ],
)
@pytest.mark.parametrize("type_form", ["singleton", "nullable", "null_first"])
def test_type_array_preserves_valid_values(schema_type, value, type_form):
    allowed_types = {
        "singleton": [schema_type],
        "nullable": [schema_type, "null"],
        "null_first": ["null", schema_type],
    }[type_form]
    schema = {"type": allowed_types}
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(value)

    actual = coerce_types(copy.deepcopy(value), schema)

    assert actual == value
    assert type(actual) is type(value)
    validator.validate(actual)


@pytest.mark.parametrize("allowed_types", [["null"], ["null", "string"]])
def test_type_array_preserves_null_before_filtering(allowed_types):
    # Guard.parse's separate null-filtering behavior is outside this regression.
    assert coerce_types(None, {"type": allowed_types}) is None


@pytest.mark.parametrize(
    "allowed_types,value",
    [
        (["string", "integer"], "42"),
        (["integer", "string"], "42"),
        (["integer", "boolean"], True),
        (["number", "integer"], 3),
    ],
)
def test_type_array_does_not_choose_a_different_allowed_type(allowed_types, value):
    actual = coerce_types(value, {"type": allowed_types})

    assert actual == value
    assert type(actual) is type(value)


@pytest.mark.parametrize(
    "schema,value,expected",
    [
        (
            {"type": ["object", "null"], "properties": {"n": {"type": "integer"}}},
            {"n": "42"},
            {"n": 42},
        ),
        (
            {"type": ["array", "null"], "items": {"type": "integer"}},
            ["42"],
            [42],
        ),
    ],
)
def test_type_array_still_coerces_nested_scalar_types(schema, value, expected):
    actual = coerce_types(value, schema)

    assert actual == expected
    Draft202012Validator(schema).validate(actual)


@pytest.mark.parametrize("allowed_types", [["integer"], ["integer", "null"]])
def test_type_array_keeps_nonmatching_values_for_schema_validation(allowed_types):
    schema = {"type": allowed_types}

    actual = coerce_types("42", schema)

    assert actual == "42"
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(actual)


@pytest.mark.parametrize("allowed_types", [["string"], ["string", "null"]])
def test_parse_preserves_valid_field_with_type_array(allowed_types, monkeypatch):
    monkeypatch.setattr(settings, "disable_tracing", True)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": allowed_types}},
        "required": ["answer"],
    }
    guard = Guard(output_schema=schema, use_server=False)
    guard.configure(allow_metrics_collection=False)

    result = guard.parse(json.dumps({"answer": "hello"}), num_reasks=0)

    assert result.validation_passed
    assert result.validated_output == {"answer": "hello"}
