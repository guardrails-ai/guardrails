from typing import List, Optional

from guardrails.embedding import EmbeddingBase
from guardrails.vectordb.base import VectorDBBase

try:
    import faiss
    from faiss import Index

except ImportError:
    pass

faiss_error = (
    "`faiss` is required for using vectordb.faiss."
    "Install it with `poetry add faiss-cpu`."
)


class Faiss(VectorDBBase):
    def __init__(
        self, index: "Index", embedder: EmbeddingBase, path: Optional[str] = None
    ) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError(faiss_error)

        super().__init__(embedder, path)
        self._index = index

    @classmethod
    def new_flat_l2_index(
        cls, vector_dim: int, embedder: EmbeddingBase, path: Optional[str] = None
    ):
        try:
            import faiss
        except ImportError:
            raise ImportError(faiss_error)
        return cls(faiss.IndexFlatL2(vector_dim), embedder, path)

    @classmethod
    def new_flat_ip_index(
        cls, vector_dim: int, embedder: EmbeddingBase, path: Optional[str] = None
    ):
        if faiss is None:
            raise ImportError(faiss_error)
        return cls(faiss.IndexFlatIP(vector_dim), embedder, path)

    @classmethod
    def new_flat_l2_index_from_embedding(
        cls,
        embedding: List[List[float]],
        embedder: EmbeddingBase,
        path: Optional[str] = None,
    ):
        if faiss is None:
            raise ImportError(faiss_error)
        store = cls(faiss.IndexFlatL2(len(embedding[0])), embedder, path)
        store.add_vectors(embedding)
        return store

    @classmethod
    def load(cls, path: str, embedder: EmbeddingBase):
        if faiss is None:
            raise ImportError(faiss_error)

        index = faiss.read_index(path)
        return cls(index, embedder, path)

    def save(self, path: Optional[str] = None):
        write_path = path if path else self._path
        faiss.write_index(self._index, write_path)

    def similarity_search_vector(self, vector: List[float], k: int) -> List[int]:
        import numpy as np

        # FIXME is this correct usage of `search`?
        #  Arguments missing for parameters "k", "distances", "labels"
        _, scores = self._index.search(np.array([vector]), k)  # type: ignore
        return scores[0].tolist()

    def similarity_search_vector_with_threshold(
        self, vector: List[float], k: int, threshold: float
    ) -> List[int]:
        import numpy as np

        # Call faiss range search and get all the vectors with a score >= threshold
        # FIXME is this correct usage of `range_search`?
        #  Arguments missing for parameters "radius", "result"
        _, dist, indexes = self._index.range_search(np.array([vector]), threshold)  # type: ignore

        if len(indexes) == 0:
            return []

        sorted_indices = np.argsort(dist)
        sorted_indexes = indexes[sorted_indices]
        return sorted_indexes.tolist()[:k]

    def add_vectors(self, vectors: List[List[float]]) -> None:
        import numpy as np

        # FIXME is this correct usage of `add`?
        #  Arguments missing for parameters "x"
        self._index.add(np.array(vectors))  # type: ignore

    def last_index(self) -> int:
        return self._index.ntotal
