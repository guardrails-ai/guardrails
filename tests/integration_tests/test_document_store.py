import os

import pytest

from guardrails.document_store import EphemeralDocumentStore
from guardrails.embedding import OpenAIEmbedding
from guardrails.vectordb import Faiss


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") in [None, "mocked"],
    reason="openai api key not set",
)
class TestEphemeralDocumentStore:
    def test_similarity_search(self):
        sentences = {
            "who is the best player in the world": "Kevin Durant",
            "who is the current mvp": "Steph Curry",
            "who is on mount rushmore of nba": "Lebron James, Michael Jordan",
        }
        db = Faiss.new_flat_l2_index(1536, OpenAIEmbedding())
        store = EphemeralDocumentStore(db)
        _ = [store.add_text(text, {"ctx": addn_ctx}) for text, addn_ctx in sentences.items()]
        pages = store.search("mvp", 1)
        assert len(pages) == 1
        assert pages[0].text == "who is the current mvp"
        assert pages[0].metadata["ctx"] == "Steph Curry"

    def test_batched_add(self):
        db = Faiss.new_flat_l2_index(1536, OpenAIEmbedding())
        store = EphemeralDocumentStore(db)
        new_doc_ids = store.add_texts({"foo": {"ctx": "bar"}, "pipe": {"ctx": "baz"}})
        assert len(new_doc_ids) == 2

    def test_persistence(self):
        if os.path.exists("test.db"):
            os.remove("test.db")
        if os.path.exists("test.index"):
            os.remove("test.index")

        db = Faiss.new_flat_l2_index(1536, OpenAIEmbedding(), "test.index")
        store = EphemeralDocumentStore(db, "test.db")
        doc_id = store.add_text("foo", {"ctx": "bar"})
        store.add_text("foo", {"ctx": "bar"})
        assert doc_id is not None
        store.flush()

        db2 = Faiss.load("test.index", OpenAIEmbedding())
        store2 = EphemeralDocumentStore(db2, "test.db")
        pages = store2.search("foo")
        assert len(pages) == 1
