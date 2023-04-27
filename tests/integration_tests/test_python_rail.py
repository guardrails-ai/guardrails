import json
import openai
from pydantic import EmailStr

from typing import Optional
from guardrails.datatypes import GuardModel, Field
from datetime import date, time
import guardrails as gd

from .mock_llm_outputs import openai_chat_completion_create
from .test_assets import python_rail


def test_python_rail(mocker):
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_chat_completion_create
    )

    class BoxOfficeRevenue(GuardModel):
        gross: float
        opening_weekend: float

    class StreamingRevenue(GuardModel):
        subscriptions: int
        subscription_fee: float

    class Details(GuardModel):
        release_date: date
        duration: time
        budget: float
        is_sequel: bool
        website: str
        contact_email: EmailStr
        revenue_type: str
        box_office_revenue: Optional[BoxOfficeRevenue] = Field(gd_if='revenue_type==box_office')
        streaming_revenue: Optional[StreamingRevenue] = Field(gd_if='revenue_type==streaming')

    class Movie(GuardModel):
        rank: int
        title: str
        details: Details

    class Director(GuardModel):
        name: str
        movies: list[Movie]

    guard = gd.Guard.from_class(
        output_class=Director,
        prompt=(
            "Provide detailed information about the top 5 grossing movies from {{director}} "
            "including release date, duration, budget, whether it's a sequel, website, and contact email."
        ),
        instructions="You are a helpful assistant only capable of communicating with valid JSON, "
                     "and no other text.\n\n@json_suffix_prompt_examples"
    )

    _, final_output = guard(
        openai.Completion.create,
        engine="text-davinci-003",
        prompt_params={"director": "Christopher Nolan"},
        max_tokens=512,
        temperature=0.5,
        num_reasks=2,
    )

    # Assertions are made on the guard state object.
    expected_llm_output_dict = json.loads(python_rail.LLM_OUTPUT)
    assert final_output == expected_llm_output_dict
