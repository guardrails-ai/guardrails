import json
from datetime import date, time
from typing import List, Literal, Union

import pytest
from pydantic import BaseModel, Field

import guardrails as gd
from guardrails.utils.openai_utils import (
    get_static_openai_chat_create_func,
    get_static_openai_create_func,
)
from guardrails.utils.pydantic_utils import PYDANTIC_VERSION, add_validator
from guardrails.validators import (
    FailResult,
    PassResult,
    TwoWords,
    ValidationResult,
    Validator,
    ValidLength,
    register_validator,
)

from .mock_llm_outputs import MockOpenAICallable, MockOpenAIChatCallable
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


def test_python_rail(mocker):
    mocker.patch(
        "guardrails.llm_providers.OpenAIChatCallable",
        new=MockOpenAIChatCallable,
    )

    class BoxOfficeRevenue(BaseModel):
        revenue_type: Literal["box_office"]
        gross: float
        opening_weekend: float

        # Field-level validation using Pydantic (not Guardrails)
        if PYDANTIC_VERSION.startswith("1"):
            from pydantic import validator

            decorator = validator("gross")
        else:
            from pydantic import field_validator

            decorator = field_validator("gross")

        @decorator
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

        # Root-level validation using Pydantic (Not in Guardrails)
        if PYDANTIC_VERSION.startswith("1"):
            website: str = Field(
                validators=[ValidLength(min=9, max=100, on_fail="reask")]
            )
            from pydantic import root_validator

            @root_validator
            def validate_budget_and_gross(cls, values):
                budget = values.get("budget")
                revenue = values.get("revenue")
                if isinstance(revenue, BoxOfficeRevenue):
                    gross = revenue.gross
                    if budget >= gross:
                        raise ValueError("Budget must be less than gross revenue")
                return values

        else:
            website: str = Field(
                json_schema_extra={
                    "validators": [ValidLength(min=9, max=100, on_fail="reask")]
                }
            )
            from pydantic import model_validator

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

    guard = gd.Guard.from_pydantic(
        output_class=Director,
        prompt=(
            "Provide detailed information about the top 5 grossing movies from"
            " ${director} including release date, duration, budget, whether "
            "it's a sequel, website, and contact email.\n"
            "${gr.json_suffix_without_examples}"
        ),
        instructions="\nYou are a helpful assistant only capable of communicating"
        " with valid JSON, and no other text.\n${gr.json_suffix_prompt_examples}",
    )

    # Guardrails runs validation and fixes the first failing output through reasking
    final_output = guard(
        get_static_openai_chat_create_func(),
        prompt_params={"director": "Christopher Nolan"},
        num_reasks=2,
        full_schema_reask=False,
    )

    # Assertions are made on the guard state object.
    expected_gd_output = json.loads(
        python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
    )
    assert final_output.validated_output == expected_gd_output

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    if PYDANTIC_VERSION.startswith("1"):
        assert guard_history[0].prompt == gd.Prompt(
            python_rail.COMPILED_PROMPT_1_WITHOUT_INSTRUCTIONS
        )
    else:
        assert guard_history[0].prompt == gd.Prompt(
            python_rail.COMPILED_PROMPT_1_PYDANTIC_2_WITHOUT_INSTRUCTIONS
        )

    assert (
        guard_history[0].output == python_rail.LLM_OUTPUT_1_FAIL_GUARDRAILS_VALIDATION
    )

    assert guard_history[1].prompt == gd.Prompt(
        python_rail.COMPILED_PROMPT_2_WITHOUT_INSTRUCTIONS
    )
    assert (
        guard_history[1].output
        == python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
    )

    if PYDANTIC_VERSION.startswith("1"):
        with pytest.raises(ValueError):
            Director.parse_raw(
                python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
            )

        # The user can take corrective action based on the failed validation.
        # Either manipulating the output themselves, taking corrective action
        # in their application, or upstreaming their validations into Guardrails.

        # The fixed output should pass validation using Pydantic
        Director.parse_raw(python_rail.LLM_OUTPUT_3_SUCCEED_GUARDRAILS_AND_PYDANTIC)
    else:
        with pytest.raises(ValueError):
            Director.model_validate_json(
                python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
            )
        Director.model_validate_json(
            python_rail.LLM_OUTPUT_3_SUCCEED_GUARDRAILS_AND_PYDANTIC
        )


@pytest.mark.skipif(not PYDANTIC_VERSION.startswith("1"), reason="Pydantic 1.x only")
def test_python_rail_add_validator(mocker):
    from pydantic import root_validator, validator

    mocker.patch(
        "guardrails.llm_providers.OpenAIChatCallable",
        new=MockOpenAIChatCallable,
    )

    class BoxOfficeRevenue(BaseModel):
        revenue_type: Literal["box_office"]
        gross: float
        opening_weekend: float

        # Field-level validation using Pydantic (not Guardrails)
        @validator("gross")
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
        website: str
        contact_email: str
        revenue: Union[BoxOfficeRevenue, StreamingRevenue] = Field(
            ..., discriminator="revenue_type"
        )

        # Register guardrails validators
        _website_validator = add_validator(
            "website", fn=ValidLength(min=9, max=100, on_fail="reask")
        )

        # Root-level validation using Pydantic (Not in Guardrails)
        @root_validator
        def validate_budget_and_gross(cls, values):
            budget = values.get("budget")
            revenue = values.get("revenue")
            if isinstance(revenue, BoxOfficeRevenue):
                gross = revenue.gross
                if budget >= gross:
                    raise ValueError("Budget must be less than gross revenue")
            return values

    class Movie(BaseModel):
        rank: int
        title: str
        details: Details

    class Director(BaseModel):
        name: str
        movies: List[Movie]

        # Add guardrails validators
        _name_validator = add_validator("name", fn=IsValidDirector())

    guard = gd.Guard.from_pydantic(
        output_class=Director,
        prompt=(
            "Provide detailed information about the top 5 grossing movies from"
            " ${director} including release date, duration, budget, whether "
            "it's a sequel, website, and contact email.\n"
            "${gr.json_suffix_without_examples}"
        ),
        instructions="\nYou are a helpful assistant only capable of communicating"
        " with valid JSON, and no other text.\n${gr.json_suffix_prompt_examples}",
    )

    # Guardrails runs validation and fixes the first failing output through reasking
    final_output = guard(
        get_static_openai_chat_create_func(),
        prompt_params={"director": "Christopher Nolan"},
        num_reasks=2,
        full_schema_reask=False,
    )

    # Assertions are made on the guard state object.
    expected_gd_output = json.loads(
        python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
    )
    assert final_output.validated_output == expected_gd_output

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    assert guard_history[0].prompt == gd.Prompt(
        python_rail.COMPILED_PROMPT_1_WITHOUT_INSTRUCTIONS
    )
    assert (
        guard_history[0].output == python_rail.LLM_OUTPUT_1_FAIL_GUARDRAILS_VALIDATION
    )

    assert guard_history[1].prompt == gd.Prompt(
        python_rail.COMPILED_PROMPT_2_WITHOUT_INSTRUCTIONS
    )
    assert (
        guard_history[1].output
        == python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
    )

    with pytest.raises(ValueError):
        Director.parse_raw(
            python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
        )

    # The user can take corrective action based on the failed validation.
    # Either manipulating the output themselves, taking corrective action
    # in their application, or upstreaming their validations into Guardrails.

    # The fixed output should pass validation using Pydantic
    Director.parse_raw(python_rail.LLM_OUTPUT_3_SUCCEED_GUARDRAILS_AND_PYDANTIC)


def test_python_string(mocker):
    """Test single string (non-JSON) generation via pydantic with re-asking."""
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    validators = [TwoWords(on_fail="reask")]
    description = "Name for the pizza"
    instructions = """
You are a helpful assistant, and you are helping me come up with a name for a pizza.

${gr.complete_string_suffix}
"""

    prompt = """
Given the following ingredients, what would you call this pizza?

${ingredients}
"""

    guard = gd.Guard.from_string(
        validators, description, prompt=prompt, instructions=instructions
    )
    final_output = guard(
        llm_api=get_static_openai_create_func(),
        prompt_params={"ingredients": "tomato, cheese, sour cream"},
        num_reasks=1,
        max_tokens=100,
    )

    assert final_output.validated_output == string.LLM_OUTPUT_REASK

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    # For orginal prompt and output
    assert guard_history[0].instructions == gd.Instructions(
        string.COMPILED_INSTRUCTIONS
    )
    assert guard_history[0].prompt == gd.Prompt(string.COMPILED_PROMPT)
    assert guard_history[0].output == string.LLM_OUTPUT
    assert guard_history[0].validated_output == string.VALIDATED_OUTPUT_REASK

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(string.COMPILED_PROMPT_REASK)
    assert guard_history[1].output == string.LLM_OUTPUT_REASK
    assert guard_history[1].validated_output == string.LLM_OUTPUT_REASK
