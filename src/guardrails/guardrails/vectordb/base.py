from abc import ABC, abstractmethod
from typing import Any, List, Optional

from guardrails.embedding import EmbeddingBase


# TODO Parameterize the init with the distance algorithm to use: cosine, L2, etc.
class VectorDBBase(ABC):
    """Base class for vector databases."""

    def __init__(self, embedder: EmbeddingBase, path: Optional[str] = None) -> None:
        """Creates a new VectorDBBase.

        Args:
            embedder: EmbeddingBase instance to use for embedding the text.
            path: Path to store or load the vector database.
        """
        self._embedder = embedder
        self._path = path

    @abstractmethod
    def add_vectors(self, vectors: List[List[float]]) -> None:
        """Adds a list of vectors to the store.

        Args:
            vectors: List of vectors to add.
        Returns:
            None
        """
        ...

    @abstractmethod
    def similarity_search_vector(self, vector: List[float], k: int) -> List[int]:
        """Searches for vectors which are similar to the given vector.

        Args:
            vector: Vector to search for.
            k: Number of similar vectors to return.
        """
        ...

    @abstractmethod
    def similarity_search_vector_with_threshold(
        self, vector: List[float], k: int, threshold: float
    ) -> List[int]:
        """Searches for vectors which are similar to the given vector.

        Args:
            vector: Vector to search for.
            k: Number of similar vectors to return.
            threshold: Minimum similarity threshold to return.
        """
        ...

    def similarity_search(self, text: str, k: int) -> List[int]:
        """Searches for vectors which are similar to the given text.
        Args:
            text: Text to search for.
            k: Number of similar vectors to return.

        Returns:
            List[int] List of indexes of the similar vectors."""
        vector = self._embedder.embed_query(text)
        return self.similarity_search_vector(vector, k)

    def similarity_search_with_threshold(
        self, text: str, k: int, threshold: float
    ) -> List[int]:
        vector = self._embedder.embed_query(text)
        return self.similarity_search_vector_with_threshold(vector, k, threshold)

    def add_texts(self, texts: List[str], ids: Optional[List[Any]] = None) -> None:
        """Adds a list of texts to the store.

        Args:
            texts: List of texts to add.
            ids: List of ids to associate with the texts.
        """
        vectors = self._embedder.embed(texts)
        self.add_vectors(vectors)

    @abstractmethod
    def save(self, path: Optional[str] = None):
        """Saves the vector database to the given path."""
        ...

    @classmethod
    def load(cls, path: str):
        """Loads the vector database from the given path."""
        ...

    @abstractmethod
    def last_index(self) -> int:
        """Returns the last index of the vector database."""
        ...
