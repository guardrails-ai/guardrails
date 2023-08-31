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

@pytest.fixture
def openai_embeddings_instance():
    # You can customize this fixture creation based on your actual class initialization
    return OpenAIEmbedding("text-embedding-ada-002")  # Initialize with a model name

def test_output_dim_for_text_embedding_ada_002(openai_embeddings_instance):
    assert openai_embeddings_instance.output_dim == 1536

def test_output_dim_for_ada_model(openai_embeddings_instance):
    openai_embeddings_instance._model = "some-ada-model"
    assert openai_embeddings_instance.output_dim == 1024

def test_output_dim_for_babbage_model(openai_embeddings_instance):
    openai_embeddings_instance._model = "some-babbage-model"
    assert openai_embeddings_instance.output_dim == 2048

def test_output_dim_for_curie_model(openai_embeddings_instance):
    openai_embeddings_instance._model = "some-curie-model"
    assert openai_embeddings_instance.output_dim == 4096

def test_output_dim_for_davinci_model(openai_embeddings_instance):
    openai_embeddings_instance._model = "some-davinci-model"
    assert openai_embeddings_instance.output_dim == 12288

def test_output_dim_for_unknown_model(openai_embeddings_instance):
    openai_embeddings_instance._model = "unknown-model"
    with pytest.raises(ValueError):
        openai_embeddings_instance.output_dim