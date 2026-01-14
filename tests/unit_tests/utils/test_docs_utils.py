# from unittest.mock import Mock, patch

import pytest

from guardrails.utils.docs_utils import (  # sentence_split,
    TextSplitter,
    get_chunks_from_text,
)


class MockTokenizer:
    def encode(self, text):
        return [token_id for token_id in range(1, len(text) + 1)]

    def decode(self, tokens):
        return " ".join([str(token_id) for token_id in tokens])


class MockPromptTemplate:
    def get_prompt_variables(self):
        return ["var1", "var2"]

    def format(self, **kwargs):
        return " ".join([f"{key}:{value}" for key, value in kwargs.items()])


@pytest.fixture
def mock_tokenizer(monkeypatch):
    mock = MockTokenizer()
    monkeypatch.setattr("tiktoken.get_encoding", lambda _: mock)
    return mock


@pytest.fixture
def mock_prompt_template():
    return MockPromptTemplate()


def test_text_splitter_split(mock_tokenizer):
    text_splitter = TextSplitter()
    text = "This is a test text."
    chunks = text_splitter.split(text, tokens_per_chunk=10, token_overlap=5, buffer=2)

    assert len(chunks) == 7
    assert chunks[0] == "1 2 3 4 5 6 7 8"
    assert chunks[1] == "4 5 6 7 8 9 10 11"
    assert chunks[2] == "7 8 9 10 11 12 13 14"
    assert chunks[3] == "10 11 12 13 14 15 16 17"


def test_prompt_template_token_length(mock_tokenizer, mock_prompt_template):
    text_splitter = TextSplitter()
    length = text_splitter.prompt_template_token_length(mock_prompt_template)
    assert length == 11  # Assuming the encoded tokens count is 11


def test_text_splitter_callable(mock_tokenizer):
    text_splitter = TextSplitter()
    text = "This is a test text."
    chunks = text_splitter(text, tokens_per_chunk=10, token_overlap=5, buffer=2)

    assert len(chunks) == 7
    assert chunks[0] == "1 2 3 4 5 6 7 8"
    assert chunks[1] == "4 5 6 7 8 9 10 11"
    assert chunks[2] == "7 8 9 10 11 12 13 14"
    assert chunks[3] == "10 11 12 13 14 15 16 17"


class MockNLTK:
    @staticmethod
    def sent_tokenize(text):
        return ["sentence1", "sentence2", "sentence3"]

    @staticmethod
    def word_tokenize(text):
        return ["word1", "word2", "word3"]


def test_get_chunks_from_text_char():
    text = "This is a test."
    chunks = get_chunks_from_text(text, chunk_strategy="char", chunk_size=4, chunk_overlap=1)
    assert len(chunks) == 5
    assert chunks[0] == "T h i s"
    assert chunks[1] == "s   i s"
    assert chunks[2] == "s   a  "
    assert chunks[3] == "  t e s"
    assert chunks[4] == "s t ."


def test_get_chunks_from_text_invalid_strategy():
    with pytest.raises(ValueError):
        get_chunks_from_text(
            "Invalid strategy.",
            chunk_strategy="invalid_strategy",
            chunk_size=1,
            chunk_overlap=0,
        )
