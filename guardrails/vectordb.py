from abc import ABC, abstractmethod
from typing import Any, List

from guardrails.embedding import EmbeddingBase


# TODO Parameterize the init with the distance algorithm to use: cosine, L2, etc.
class VectorDBBase(ABC):
    """Base class for vector databases."""

    def __init__(self, embedder: EmbeddingBase, path: str = None) -> None:
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

    def add_texts(self, texts: List[str], ids: List[Any] = None) -> None:
        """Adds a list of texts to the store.

        Args:
            texts: List of texts to add.
            ids: List of ids to associate with the texts.
        """
        vectors = self._embedder.embed(texts)
        self.add_vectors(vectors)

    @abstractmethod
    def save(self, path: str = None):
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


class Faiss(VectorDBBase):
    import faiss

    def __init__(
        self, index: faiss.Index, embedder: EmbeddingBase, path: str = None
    ) -> None:
        super().__init__(embedder, path)
        self._index = index

    @classmethod
    def new_flat_l2_index(
        cls, vector_dim: int, embedder: EmbeddingBase, path: str = None
    ):
        from faiss import IndexFlatL2

        return cls(IndexFlatL2(vector_dim), embedder, path)

    @classmethod
    def new_flat_ip_index(
        cls, vector_dim: int, embedder: EmbeddingBase, path: str = None
    ):
        from faiss import IndexFlatIP

        return cls(IndexFlatIP(vector_dim), embedder, path)

    @classmethod
    def new_flat_l2_index_from_embedding(
        cls, embedding: List[List[float]], embedder: EmbeddingBase, path: str = None
    ):
        from faiss import IndexFlatL2

        store = cls(IndexFlatL2(len(embedding[0])), embedder, path)
        store.add_vectors(embedding)
        return store

    @classmethod
    def load(cls, path: str, embedder: EmbeddingBase):
        import faiss

        index = faiss.read_index(path)
        return cls(index, embedder, path)

    def save(self, path: str = None):
        import faiss

        write_path = path if path else self._path
        faiss.write_index(self._index, write_path)

    def similarity_search_vector(self, vector: List[float], k: int) -> List[int]:
        import numpy as np

        _, scores = self._index.search(np.array([vector]), k)
        return scores[0].tolist()

    def similarity_search_vector_with_threshold(
        self, vector: List[float], k: int, threshold: float
    ) -> List[int]:
        import numpy as np

        # Call faiss range search and get all the vectors with a score >= threshold
        _, dist, indexes = self._index.range_search(np.array([vector]), threshold)

        if len(indexes) == 0:
            return []

        sorted_indices = np.argsort(dist)
        sorted_indexes = indexes[sorted_indices]
        return sorted_indexes.tolist()[:k]

    def add_vectors(self, vectors: List[List[float]]) -> None:
        import numpy as np

        self._index.add(np.array(vectors))

    def last_index(self) -> int:
        return self._index.ntotal
