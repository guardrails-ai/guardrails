import json

import pytest
from guardrails.schema.validator import SchemaValidationError, validate_payload

with open(
    "tests/integration_tests/test_assets/json_schemas/choice_case.json", "r"
) as choice_case_json_file:
    schema = json.loads(choice_case_json_file.read())


class TestValidatePayload:
    def test_happy_path(self):
        payload = {"action": {"chosen_action": "fight", "weapon": "crossbow"}}

        validate_payload(payload, schema)

    def test_extra_properties_allowed(self):
        payload = {
            "action": {"chosen_action": "fight", "weapon": "crossbow"},
            "reason": "Peregrin Took is a brave hobbit",
        }

        validate_payload(payload, schema)

    def test_failure_invalid_discriminator_value(self):
        payload = {"action": {"chosen_action": "dance", "type": "jig"}}

        with pytest.raises(Exception) as excinfo:
            validate_payload(payload, schema)

        assert isinstance(excinfo.value, SchemaValidationError) is True
        schema_error: SchemaValidationError = excinfo.value
        assert (
            str(schema_error) == "The provided payload is not compliant with the provided schema!"
        )
        assert schema_error.fields == {
            "$.action.chosen_action": ["'dance' is not one of ['fight', 'flight']"]
        }

    def test_failure_invalid_type(self):
        payload = {
            "action": {
                "chosen_action": "flight",
                "flight_direction": "north",
                "distance": "2",
            }
        }

        with pytest.raises(Exception) as excinfo:
            validate_payload(payload, schema)

        assert isinstance(excinfo.value, SchemaValidationError) is True
        schema_error: SchemaValidationError = excinfo.value
        assert (
            str(schema_error) == "The provided payload is not compliant with the provided schema!"
        )
        # Type coercion is not automatic!
        assert schema_error.fields == {"$.action.distance": ["'2' is not of type 'integer'"]}

    # NOTE: Technically the same as an invalid type
    def test_failure_invalid_structure(self):
        payload = {
            "action": [
                {
                    "chosen_action": "flight",
                    "flight_direction": "north",
                    "distance": "2",
                }
            ]
        }

        with pytest.raises(Exception) as excinfo:
            validate_payload(payload, schema)

        assert isinstance(excinfo.value, SchemaValidationError) is True
        schema_error: SchemaValidationError = excinfo.value
        assert (
            str(schema_error) == "The provided payload is not compliant with the provided schema!"
        )
        assert schema_error.fields == {
            "$.action": [
                "[{'chosen_action': 'flight', 'flight_direction': 'north', 'distance': '2'}] is not of type 'object'"  # noqa
            ]
        }

    def test_failure_missing_required_properties(self):
        payload = {"action": {"chosen_action": "flight"}}

        with pytest.raises(Exception) as excinfo:
            validate_payload(payload, schema)

        assert isinstance(excinfo.value, SchemaValidationError) is True
        schema_error: SchemaValidationError = excinfo.value
        assert (
            str(schema_error) == "The provided payload is not compliant with the provided schema!"
        )

        assert schema_error.fields == {"$.action": ["'distance' is a required property"]}

    def test_subschema_validation(self):
        # Missing required properites, but that's allowed with validate_subschema
        payload = {"action": {"chosen_action": "flight"}}

        validate_payload(payload, schema, validate_subschema=True)
