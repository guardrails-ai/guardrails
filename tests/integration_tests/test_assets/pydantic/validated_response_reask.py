# flake8: noqa: E501
from pydantic import BaseModel, validator

from guardrails.utils.reask_utils import FieldReAsk
from guardrails.validators import FailResult

prompt = """Generate data for possible users in accordance with the specification below.

@xml_prefix_prompt

{output_schema}

@complete_json_suffix_v2"""


class Person(BaseModel):
    """Information about a person.

    Args:
        name (str): The name of the person.
        age (int): The age of the person.
        zip_code (str): The zip code of the person.
    """

    name: str
    age: int
    zip_code: str

    @validator("zip_code")
    def zip_code_must_be_numeric(cls, v):
        if not v.isnumeric():
            raise ValueError("Zip code must be numeric.")
        return v

    @validator("age")
    def age_must_be_between_0_and_150(cls, v):
        if not 0 <= v <= 150:
            raise ValueError("Age must be between 0 and 150.")
        return v

    @validator("zip_code")
    def zip_code_in_california(cls, v):
        if not v.startswith("9"):
            raise ValueError("Zip code must be in California, and start with 9.")
        if v == "90210":
            raise ValueError("Zip code must not be Beverly Hills.")
        return v


class ListOfPeople(BaseModel):
    """A list of people.

    Args:
        people (list[Person]): A list of people.
    """

    people: list[Person]


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
                path=["people", 0],
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
                    )
                ],
                path=["people", 0],
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
