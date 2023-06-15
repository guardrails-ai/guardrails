# flake8: noqa: E501
from pydantic import BaseModel, validator

from guardrails.utils.pydantic_utils import register_pydantic
from guardrails.utils.reask_utils import FieldReAsk


@register_pydantic
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


VALIDATED_OUTPUT = {
    "people": [
        {
            "name": "John Doe",
            "age": 28,
            "zip_code": FieldReAsk(
                incorrect_value="None",
                error_message="Zip code must be numeric.",
                fix_value=None,
                path=["people", 0],
            ),
        },
        Person(name="Jane Doe", age=32, zip_code="94103"),
        Person(name="James Smith", age=40, zip_code="92101"),
    ]
}
