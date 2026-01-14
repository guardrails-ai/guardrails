import json
import pytest

from xml.etree.ElementTree import canonicalize

from guardrails.classes.validation.validator_reference import ValidatorReference
from guardrails.classes.schema.processed_schema import ProcessedSchema
from guardrails.schema.rail_schema import (
    rail_file_to_schema,
    json_schema_to_rail_output,
)
from guardrails.classes.output_type import OutputTypes
from guardrails.validator_base import OnFailAction
from tests.integration_tests.test_assets.validators import (
    ValidChoices,
    LowerCase,
    OneLine,
    TwoWords,
)


### JSON Schemas ###
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

with open("tests/integration_tests/test_assets/json_schemas/string.json", "r") as string_file:
    string_schema = json.loads(string_file.read())


class TestRailToJsonSchema:
    # Did this one first because it's what I was most concerned about
    def test_choice_case_happy_path(self):
        from tests.integration_tests.test_assets.validators.valid_choices import (
            ValidChoices,
        )

        processed_schema: ProcessedSchema = rail_file_to_schema(
            "tests/integration_tests/test_assets/rail_specs/choice_case.rail"
        )

        assert processed_schema.json_schema == choice_case_json_schema
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


### ReConstructed RAIL Specs for Prompting ###
case_choice_rail = """
<output>
  <choice name="action" discriminator="chosen_action" required="true">
    <case name="fight">
      <string name="weapon" format="valid-choices: ['crossbow', 'machine gun']" required="true" />
    </case>
    <case name="flight">
      <string name="flight_direction" format="valid-choices: ['north', 'south', 'east', 'west']" required="false" />
      <integer name="distance" format="valid-choices: [1, 2, 3, 4]" required="true" />
    </case>
  </choice>
</output>
""".strip()  # noqa

# flight_direction.required is true here because Pydantic compiles optional properties
#   as a Union of the actual type and null but still marks it as required...
case_choice_openapi_rail = """
<output>
  <choice name="action" discriminator="chosen_action" required="true">
    <case name="fight">
      <string name="weapon" format="valid-choices: ['crossbow', 'machine gun']" required="true" />
    </case>
    <case name="flight">
      <string name="flight_direction" format="valid-choices: ['north', 'south', 'east', 'west']" required="true" />
      <integer name="distance" format="valid-choices: [1, 2, 3, 4]" required="true" />
    </case>
  </choice>
</output>
""".strip()  # noqa

credit_card_agreement_rail = """
<output>
  <list name="fees" description="What fees and charges are associated with my account?" required="true">
    <object required="true" >
      <integer name="index" format="1-indexed" required="true" />
      <string name="name" format="lower-case; two-words" required="true" />
      <string name="explanation" format="one-line" required="true" />
      <float name="value" format="percentage" required="true" />
    </object>
  </list>
  <object name="interest_rates" description="What are the interest rates offered by the bank on savings and checking accounts, loans, and credit products?" required="true" />
</output>
""".strip()  # noqa

string_schema_rail = """
<output type="string" description="Some string..." format="lower-case; two-words" />
""".strip()  # noqa

### Validator Maps ###
case_choice_validator_map = {
    "$.action.weapon": [ValidChoices(["crossbow", "machine gun"], OnFailAction.REASK)],
    "$.action.flight_direction": [
        ValidChoices(["north", "south", "east", "west"], OnFailAction.EXCEPTION)
    ],
    "$.action.distance": [ValidChoices([1, 2, 3, 4], OnFailAction.EXCEPTION)],
}

credit_card_agreement_validator_map = {
    "$.fees.name": [LowerCase(), TwoWords()],
    "$.fees.explanation": [OneLine()],
}


@pytest.mark.parametrize(
    "json_schema,validator_map,rail_output",
    [
        (choice_case_json_schema, case_choice_validator_map, case_choice_rail),
        (
            choice_case_openapi_schema,
            case_choice_validator_map,
            case_choice_openapi_rail,
        ),
        (
            credit_card_agreement_schema,
            credit_card_agreement_validator_map,
            credit_card_agreement_rail,
        ),
    ],
)
def test_json_schema_to_rail_output(json_schema, validator_map, rail_output):
    actual_rail_output = json_schema_to_rail_output(json_schema, validator_map)
    actual_rail_xml = canonicalize(actual_rail_output)
    expected_rail_xml = canonicalize(rail_output)
    assert actual_rail_xml == expected_rail_xml
