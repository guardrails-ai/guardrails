import json
from datetime import date, time
from typing import List, Literal, Union

import pytest
from pydantic import BaseModel, Field, field_validator, model_validator

import guardrails as gd
from guardrails import Validator, register_validator
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.types import OnFailAction
from guardrails.classes.validation.validation_result import (
    FailResult,
    PassResult,
    ValidationResult,
)
from tests.integration_tests.test_assets.validators import ValidLength, TwoWords

from .test_assets import python_rail, string


@register_validator(name="is-valid-director", data_type="string")
class IsValidDirector(Validator):
    def validate(self, value, metadata) -> ValidationResult:
        valid_names = [
            "Christopher Nolan",
            "Steven Spielberg",
            "Martin Scorsese",
            "Quentin Tarantino",
            "James Cameron",
        ]
        if value not in valid_names:
            return FailResult(
                error_message=f"Value {value} is not a valid director name. "
                f"Valid choices are {valid_names}.",
            )
        return PassResult()


class BoxOfficeRevenue(BaseModel):
    revenue_type: Literal["box_office"]
    gross: float
    opening_weekend: float

    # Field-level validation using Pydantic (not Guardrails)
    @field_validator("gross")
    def validate_gross(cls, gross):
        if gross <= 0:
            raise ValueError("Gross revenue must be a positive value")
        return gross


class StreamingRevenue(BaseModel):
    revenue_type: Literal["streaming"]
    subscriptions: int
    subscription_fee: float


class Details(BaseModel):
    release_date: date
    duration: time
    budget: float
    is_sequel: bool = Field(default=False)
    website: str = Field(
        json_schema_extra={
            "validators": [ValidLength(min=9, max=100, on_fail=OnFailAction.REASK)]
        }
    )

    # Root-level validation using Pydantic (Not in Guardrails)
    @model_validator(mode="before")
    def validate_budget_and_gross(cls, values):
        budget = values.get("budget")
        revenue = values.get("revenue")
        if revenue["revenue_type"] == "box_office":
            gross = revenue["gross"]
            if budget >= gross:
                raise ValueError("Budget must be less than gross revenue")
        return values

    contact_email: str
    revenue: Union[BoxOfficeRevenue, StreamingRevenue] = Field(
        ..., discriminator="revenue_type"
    )


class Movie(BaseModel):
    rank: int
    title: str
    details: Details


class Director(BaseModel):
    name: str = Field(validators=[IsValidDirector()])
    movies: List[Movie]


def test_python_rail(mocker):
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.LiteLLMCallable._invoke_llm"
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=python_rail.LLM_OUTPUT_1_FAIL_GUARDRAILS_VALIDATION,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    guard = gd.Guard.for_pydantic(
        output_class=Director,
        messages=[
            {
                "role": "system",
                "content": "\nYou are a helpful assistant"
                " only capable of communicating"
                " with valid JSON, and no other"
                " text.\n${gr.xml_suffix_prompt_examples}",
            },
            {
                "role": "user",
                "content": "Provide detailed information"
                " about the top 5 grossing movies from"
                " ${director} including release date, duration, budget, whether "
                "it's a sequel, website, and contact email.\n"
                "${gr.xml_suffix_without_examples}",
            },
        ],
    )

    # Guardrails runs validation and fixes the first failing output through reasking
    final_output = guard(
        model="gpt-3.5-turbo",
        prompt_params={"director": "Christopher Nolan"},
        num_reasks=2,
        full_schema_reask=False,
    )

    # Assertions are made on the guard state object.
    expected_gd_output = json.loads(
        python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
    )
    assert final_output.validated_output == expected_gd_output

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    assert (
        call.compiled_messages[1]["content"]
        == python_rail.COMPILED_PROMPT_1_PYDANTIC_2_WITHOUT_INSTRUCTIONS
    )

    assert (
        call.iterations.first.raw_output
        == python_rail.LLM_OUTPUT_1_FAIL_GUARDRAILS_VALIDATION
    )

    assert call.iterations.last.inputs.messages[1]["content"] == gd.Prompt(
        python_rail.COMPILED_PROMPT_2_WITHOUT_INSTRUCTIONS
    )
    # Same as above
    assert (
        call.reask_messages[0][1]["content"]
        == python_rail.COMPILED_PROMPT_2_WITHOUT_INSTRUCTIONS
    )
    assert (
        call.raw_outputs.last
        == python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
    )

    with pytest.raises(ValueError):
        Director.model_validate_json(
            python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
        )
    Director.model_validate_json(
        python_rail.LLM_OUTPUT_3_SUCCEED_GUARDRAILS_AND_PYDANTIC
    )


def test_python_string(mocker):
    """Test single string (non-JSON) generation via pydantic with re-asking."""
    # mocker.patch("guardrails.llm_providers.LiteLLMCallable", new=MockLiteLLMCallable)
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.LiteLLMCallable._invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=string.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=string.LLM_OUTPUT_REASK,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]
    validators = [TwoWords(on_fail=OnFailAction.REASK)]
    description = "Name for the pizza"
    instructions = """
You are a helpful assistant, and you are helping me come up with a name for a pizza.

${gr.complete_string_suffix}
"""

    prompt = """
Given the following ingredients, what would you call this pizza?

${ingredients}
"""

    guard = gd.Guard.for_string(
        validators,
        string_description=description,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
    )
    final_output = guard(
        model="gpt-3.5-turbo",
        prompt_params={"ingredients": "tomato, cheese, sour cream"},
        num_reasks=1,
        max_tokens=100,
    )

    assert final_output.validated_output == string.LLM_OUTPUT_REASK

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    # For orginal prompt and output
    assert call.compiled_messages[0]["content"] == string.COMPILED_INSTRUCTIONS
    assert call.compiled_messages[1]["content"] == string.COMPILED_PROMPT
    assert call.iterations.first.raw_output == string.LLM_OUTPUT
    assert call.iterations.first.validation_response == string.VALIDATED_OUTPUT_REASK

    # For re-asked prompt and output
    assert (
        call.iterations.last.inputs.messages[1]["content"]
        == string.COMPILED_PROMPT_REASK
    )
    # Same as above
    assert call.reask_messages.last[-1]["content"] == string.COMPILED_PROMPT_REASK
    assert call.raw_outputs.last == string.LLM_OUTPUT_REASK
    assert call.guarded_output == string.LLM_OUTPUT_REASK
