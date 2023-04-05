import os

import pytest

from guardrails.embedding import OpenAIEmbedding


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None, reason="openai api key not set"
)
class TestOpenAIEmbedding:
    def test_embedding_texts(self):
        e = OpenAIEmbedding()
        result = e.embed(["foo", "bar"])
        assert len(result) == 2
        assert len(result[0]) == 1536

    def test_embedding_query(self):
        e = OpenAIEmbedding()
        result = e.embed_query("foo")
        assert len(result) == 1536
