import json
from datetime import date, time
from typing import List, Optional

import openai
import pytest
from pydantic import Field as PydanticField
from pydantic import root_validator, validator

import guardrails as gd
from guardrails.datatypes import Field, GuardModel
from guardrails.validators import (
    EventDetail,
    Validator,
    ValidChoices,
    ValidLength,
    register_validator,
)

from .mock_llm_outputs import openai_chat_completion_create
from .test_assets import python_rail


@register_validator(name="is-valid-director", data_type="string")
class IsValidDirector(Validator):
    def validate(self, key, value, schema) -> dict:
        valid_names = [
            "Christopher Nolan",
            "Steven Spielberg",
            "Martin Scorsese",
            "Quentin Tarantino",
            "James Cameron",
        ]
        if value not in valid_names:
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is not a valid director name. "
                f"Valid choices are {valid_names}.",
                None,
            )
        return schema


def test_python_rail(mocker):
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_chat_completion_create
    )

    class BoxOfficeRevenue(GuardModel):
        gross: float
        opening_weekend: float

        # Field-level validation using Pydantic (not Guardrails)
        @validator("gross")
        def validate_gross(cls, gross):
            if gross <= 0:
                raise ValueError("Gross revenue must be a positive value")
            return gross

    class StreamingRevenue(GuardModel):
        subscriptions: int
        subscription_fee: float

    class Details(GuardModel):
        release_date: date
        duration: time
        budget: float
        is_sequel: bool = PydanticField(default=False)
        website: str = Field(
            gd_validators=[ValidLength(min=9, max=100, on_fail="reask")]
        )
        contact_email: str
        revenue_type: str = Field(
            gd_validators=[ValidChoices(choices=["box_office", "streaming"])]
        )
        box_office: Optional[BoxOfficeRevenue] = Field(when="revenue_type==box_office")
        streaming: Optional[StreamingRevenue] = Field(when="revenue_type==streaming")

        # Root-level validation using Pydantic (Not in Guardrails)
        @root_validator
        def validate_budget_and_gross(cls, values):
            budget = values.get("budget")
            revenue_type = values.get("revenue_type")
            box_office_revenue = values.get("box_office")
            if revenue_type == "box_office" and box_office_revenue:
                gross = box_office_revenue.gross
                if budget >= gross:
                    raise ValueError("Budget must be less than gross revenue")
            return values

    class Movie(GuardModel):
        rank: int
        title: str
        details: Details

    class Director(GuardModel):
        name: str = Field(gd_validators=[IsValidDirector()])
        movies: List[Movie]

    guard = gd.Guard.from_pydantic(
        output_class=Director,
        prompt=(
            "Provide detailed information about the top 5 grossing movies from"
            " {{director}} including release date, duration, budget, whether "
            "it's a sequel, website, and contact email."
        ),
        instructions="You are a helpful assistant only capable of communicating"
        " with valid JSON, and no other text.\n\n@json_suffix_prompt_examples",
    )

    # Guardrails runs validation and fixes the first failing output through reasking
    _, final_output = guard(
        openai.Completion.create,
        engine="text-davinci-003",
        prompt_params={"director": "Christopher Nolan"},
        max_tokens=512,
        temperature=0.5,
        num_reasks=2,
    )

    # Assertions are made on the guard state object.
    expected_gd_output = json.loads(
        python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION
    )
    assert final_output == expected_gd_output

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
