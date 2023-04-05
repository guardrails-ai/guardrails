from abc import ABC, abstractmethod
from typing import List, Optional

from guardrails.llm_providers import openai_embedding_wrapper


class EmbeddingBase(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        ...


class OpenAIEmbedding(EmbeddingBase):
    def __init__(self, model: Optional[str] = "text-embedding-ada-002"):
        self._model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        return openai_embedding_wrapper(texts, self._model)

    def embed_query(self, query: str) -> List[float]:
        resp = openai_embedding_wrapper([query], self._model)
        return resp[0]
