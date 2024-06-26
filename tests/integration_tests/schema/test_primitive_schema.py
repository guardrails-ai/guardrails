import json

from guardrails_api_client.models.validator_reference import ValidatorReference
from guardrails.classes.schema.processed_schema import ProcessedSchema
from guardrails.schema.primitive_schema import primitive_to_schema
from guardrails.classes.output_type import OutputTypes
from guardrails.validator_base import OnFailAction
from tests.integration_tests.test_assets.validators import ValidChoices, ValidLength


class TestPrimitiveSchema:
    # Did this one first because it's what I was most concerned about
    def test_choice_case_happy_path(self):
        with open(
            "tests/integration_tests/test_assets/json_schemas/string.json", "r"
        ) as choice_case_json_file:
            expected_schema = json.loads(choice_case_json_file.read())

        choice_validator = ValidChoices(choices=["north", "south", "east", "west"])
        length_validator = ValidLength(4, 5, "filter")

        processed_schema: ProcessedSchema = primitive_to_schema(
            validators=[choice_validator, length_validator],
            description="Some string...",
        )

        assert processed_schema.json_schema == expected_schema
        assert processed_schema.output_type == OutputTypes.STRING
        assert processed_schema.output_type == "str"
        assert processed_schema.validators == [
            ValidatorReference(
                id="valid-choices",
                on="$",
                on_fail=OnFailAction.NOOP,
                kwargs={"choices": ["north", "south", "east", "west"]},
            ),
            ValidatorReference(
                id="length",
                on="$",
                on_fail=OnFailAction.FILTER,
                kwargs={"min": 4, "max": 5},
            ),
        ]
        assert len(processed_schema.validator_map) == 1
        assert processed_schema.validator_map.get("$") == [
            choice_validator,
            length_validator,
        ]
