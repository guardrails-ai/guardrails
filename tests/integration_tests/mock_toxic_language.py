# noqa: EW291, E501
from typing import List


class MockPipeline:
    """Mock class for transformers.pipeline.

    Does nothing.
    """

    def __init__(self, *args, **kwargs):
        pass


TOXIC_PARAGRAPH = """
I hate how pathetic and needy you are. You're so annoying.
I wish you would just go away and never come back.
You're a waste of space and I hate you. I hope you die."""

NON_TOXIC_PARAGRAPH = """
I love how smart and funny you are. You're so amazing.
I wish you would stay here forever and never leave.
You're a gift to humanity and I love you. I hope you live forever."""

PARAGRAPH_WITH_TOXIC_SENTENCES = """
Generative AI is a field of AI that aims to create machines
capable of generating new ideas and content. It is a subset of machine
learning, which is a subset of artificial intelligence.
Tommy Daniels is a demon and a weirdo who makes shitty movies;I hate him so much.
GenAI can sometimes generate harmful content."""

EXPECTED_PARAGRAPH_WITH_TOXIC_SENTENCES = """
Generative AI is a field of AI that aims to create machines
capable of generating new ideas and content.
It is a subset of machine
learning, which is a subset of artificial intelligence.
GenAI can sometimes generate harmful content."""

TOXIC_SENTENCES = [
    "Tommy Daniels is a demon and a weirdo who makes shitty movies;I hate him so much.",
    "I hate how pathetic and needy you are. You're so annoying.",
    "I wish you would just go away and never come back.",
    "You're a waste of space and I hate you. I hope you die.",
]

NON_TOXIC_SENTENCES = [
    "Generative AI is a field of AI that aims to create machines capable of generating new ideas and content.",  # noqa: E501
    "It is a subset of machine learning, which is a subset of artificial intelligence.",
    "GenAI can sometimes generate harmful content.",
    "I love how smart and funny you are. You're so amazing.",
    "I wish you would stay here forever and never leave.",
    "You're a gift to humanity and I love you. I hope you live forever.",
]


def mock_get_toxicity(self, value: str) -> List[str]:
    """Mocks the get_toxicity function."""

    if value == TOXIC_PARAGRAPH:
        return ["toxicity", "insult"]
    elif value == NON_TOXIC_PARAGRAPH:
        return []
    elif value in TOXIC_SENTENCES:
        return ["toxicity", "insult"]
    elif value in NON_TOXIC_SENTENCES:
        return []
    else:
        return []
