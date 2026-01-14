import json

from guardrails.classes.validation.validator_reference import ValidatorReference
from guardrails.classes.schema.processed_schema import ProcessedSchema
from guardrails.schema.pydantic_schema import pydantic_model_to_schema
from guardrails.classes.output_type import OutputTypes
from guardrails.validator_base import OnFailAction
from tests.integration_tests.test_assets.pydantic_models.fight_or_flight import (
    FightOrFlight,
)
from tests.integration_tests.test_assets.validators.valid_choices import ValidChoices


class TestPydanticSchema:
    # Did this one first because it's what I was most concerned about
    def test_choice_case_happy_path(self):
        with open(
            "tests/integration_tests/test_assets/json_schemas/choice_case_openapi.json",
            "r",
        ) as choice_case_json_file:
            expected_schema = json.loads(choice_case_json_file.read())

        processed_schema: ProcessedSchema = pydantic_model_to_schema(FightOrFlight)

        assert processed_schema.json_schema == expected_schema
        assert processed_schema.output_type == OutputTypes.DICT
        assert processed_schema.output_type == "dict"
        assert processed_schema.validators == [
            ValidatorReference(
                id="valid-choices",
                on="$.action.weapon",
                on_fail=OnFailAction.REASK,
                kwargs={"choices": ["crossbow", "machine gun"]},
            ),
            ValidatorReference(
                id="valid-choices",
                on="$.action.flight_direction",
                on_fail=OnFailAction.EXCEPTION,
                kwargs={"choices": ["north", "south", "east", "west"]},
            ),
            ValidatorReference(
                id="valid-choices",
                on="$.action.distance",
                on_fail=OnFailAction.EXCEPTION,
                kwargs={"choices": [1, 2, 3, 4]},
            ),
        ]
        assert len(processed_schema.validator_map) == 3
        assert processed_schema.validator_map.get("$.action.distance") == [
            ValidChoices(choices=[1, 2, 3, 4], on_fail=OnFailAction.EXCEPTION)
        ]
        assert processed_schema.validator_map.get("$.action.flight_direction") == [
            ValidChoices(
                choices=["north", "south", "east", "west"],
                on_fail=OnFailAction.EXCEPTION,
            )
        ]
        assert processed_schema.validator_map.get("$.action.weapon") == [
            ValidChoices(choices=["crossbow", "machine gun"], on_fail=OnFailAction.REASK)
        ]
