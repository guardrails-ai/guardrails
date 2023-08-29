# flake8: noqa: E501
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from guardrails.utils.reask_utils import FieldReAsk
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

prompt = """Generate data for possible users in accordance with the specification below.

${gr.xml_prefix_prompt}

${output_schema}

${gr.complete_json_suffix_v2}"""


@register_validator(name="zip_code_must_be_numeric", data_type="string")
class ZipCodeMustBeNumeric(Validator):
    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        if not value.isnumeric():
            return FailResult(error_message="Zip code must be numeric.")
        return PassResult()


@register_validator(name="age_must_be_between_0_and_150", data_type="integer")
class AgeMustBeBetween0And150(Validator):
    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        if not 0 <= value <= 150:
            return FailResult(error_message="Age must be between 0 and 150.")
        return PassResult()


@register_validator(name="zip_code_in_california", data_type="string")
class ZipCodeInCalifornia(Validator):
    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        if not value.startswith("9"):
            return FailResult(
                error_message="Zip code must be in California, and start with 9."
            )
        if value == "90210":
            return FailResult(error_message="Zip code must not be Beverly Hills.")
        return PassResult()


class Person(BaseModel):
    """Information about a person.

    Args:
        name (str): The name of the person.
        age (int): The age of the person.
        zip_code (str): The zip code of the person.
    """

    name: str
    age: int = Field(..., validators=[AgeMustBeBetween0And150(on_fail="reask")])
    zip_code: str = Field(
        ...,
        validators=[
            ZipCodeMustBeNumeric(on_fail="reask"),
            ZipCodeInCalifornia(on_fail="reask"),
        ],
    )


class ListOfPeople(BaseModel):
    """A list of people.

    Args:
        people (List[Person]): A list of people.
    """

    people: List[Person]


VALIDATED_OUTPUT_1 = {
    "people": [
        {
            "name": "John Doe",
            "age": 28,
            "zip_code": FieldReAsk(
                incorrect_value="90210",
                fail_results=[
                    FailResult(
                        error_message="Zip code must not be Beverly Hills.",
                        fix_value=None,
                    )
                ],
                path=["people", 0, "zip_code"],
            ),
        },
        {"name": "Jane Doe", "age": 32, "zip_code": "94103"},
        {"name": "James Smith", "age": 40, "zip_code": "92101"},
    ]
}


VALIDATED_OUTPUT_2 = {
    "people": [
        {
            "name": "John Doe",
            "age": 28,
            "zip_code": FieldReAsk(
                incorrect_value="None",
                fail_results=[
                    FailResult(
                        error_message="Zip code must be numeric.",
                        fix_value=None,
                    ),
                    FailResult(
                        error_message="Zip code must be in California, and start with 9.",
                        fix_value=None,
                    ),
                ],
                path=["people", 0, "zip_code"],
            ),
        },
        {"name": "Jane Doe", "age": 32, "zip_code": "94103"},
        {"name": "James Smith", "age": 40, "zip_code": "92101"},
    ]
}


VALIDATED_OUTPUT_3 = {
    "people": [
        {"name": "John Doe", "age": 28, "zip_code": None},
        {"name": "Jane Doe", "age": 32, "zip_code": "94103"},
        {"name": "James Smith", "age": 40, "zip_code": "92101"},
    ]
}
