import json

import openai

from pydantic import BaseModel
from typing import List
import guardrails as gd

from .mock_llm_outputs import openai_completion_create
from .test_assets import python_rail


def test_python_rail_with_reask(mocker):

    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    class Movie(BaseModel):
        rank: int
        title: str

    class Movies(BaseModel):
        movies: List[Movie]

    guard = gd.Guard.from_class(
        output_class=Movies,
        prompt="What are the top 5 grossing movies from {{director}}?"
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

