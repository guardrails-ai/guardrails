import openai

from pydantic import BaseModel
from typing import List
import guardrails as gd

from .mock_llm_outputs import openai_completion_create


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

    expected_output_data = [
        {"rank": 1, "title": "Inception"},
        {"rank": 2, "title": "The Dark Knight"},
        {"rank": 3, "title": "The Dark Knight Rises"},
        {"rank": 4, "title": "Interstellar"},
        {"rank": 5, "title": "Dunkirk"}
    ]

    expected_output = """
What are the top 5 grossing movies from Christopher Nolan?

Given below is XML that describes the information to extract from this document and the tags to extract it into.

<output>
    <list name="Movies">
        <object name="Movie">
            <integer name="rank"/>
            <string name="title"/>
        </object>
    </list>
</output>
""".strip()

    assert final_output == expected_output

