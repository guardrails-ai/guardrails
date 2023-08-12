from typing import List

from guardrails.embedding import EmbeddingBase
from guardrails.vectordb.base import VectorDBBase

try:
    import chromadb
except ImportError:
    chromadb = None


class ChromaDB(VectorDBBase):
    def __init__(
        self, embedder: EmbeddingBase, path: str=None,) -> None:
        if chromadb is None:
            raise ImportError("`chromadb` is required for using vectordb.ChromaDB."
            "Install it with `pip install chroma-db`")
        super().__init__(embedder, path)
        if path is not None:
            # In-memory chroma with saving and loading to disk.
            self._client = chromadb.PersistentClient(path=path)
        else:
            # In-memory chroma.
            self._client = chromadb.Client()
    
    
    @classmethod
    def load(cls, path: str, embedder: EmbeddingBase):
        if chromadb is None:
            raise ImportError("`chromadb` is required for using vectordb.ChromaDB."
                "Install it with `pip install chroma-db`")
        return cls(embedder, path)
    
    
    @property
    def get_collection_names(self) -> List[str]:
        return self._client.get_collections()
    
    
    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.get_collection_names
    
    
    def get_num_items(self, collection_name: str) -> int:
        if self.has_collection(collection_name):
            return self._client.get_collection(collection_name).count()
        return 0
    

    def add_vectors(self, collection_name: str, vectors: List[List[float]], ids: List[str], metadata: Optional[List[Dict[str, str]]]=None) -> None:
        collection = self._client.get_or_create_collection(collection_name)
        collection.add(
            embeddings=vectors,
            metadatas=metadata,
            ids=ids,
        )
    

    def similarity_search_vector(self, collection_name: str, vector: List[float], k: int)-> List[str]:
        collection = self._client.get_or_create_collection(collection_name)
        results = collection.query(query_embeddings=[vector],n_results=k)
        return results['ids'][0]
    

    def similarity_search_vector_with_threshold(self, collection_name: str, vector: List[float], k : int , threshold: float)-> List[str]:
        collection = self._client.get_or_create_collection(collection_name)
        results = collection.query(query_embeddings=[vector], n_results=k,)
        if not len(results['ids']):
            return []
        import numpy as np
        return np.array(results['ids'])[np.array(results['distances']) < threshold].tolist()


    @classmethod
    def new_flat_l2_index(
        cls, vector_dim: int, embedder: EmbeddingBase, path: str = None
    ):
        if faiss is None:
            raise ImportError(
                "`faiss` is required for using vectordb.faiss."
                "Install it with `pip install faiss-cpu`."
            )
        return cls(faiss.IndexFlatL2(vector_dim), embedder, path)

    @classmethod
    def new_flat_ip_index(
        cls, vector_dim: int, embedder: EmbeddingBase, path: str = None
    ):
        if faiss is None:
            raise ImportError(
                "`faiss` is required for using vectordb.faiss."
                "Install it with `pip install faiss-cpu`."
            )
        return cls(faiss.IndexFlatIP(vector_dim), embedder, path)

    @classmethod
    def new_flat_l2_index_from_embedding(
        cls, embedding: List[List[float]], embedder: EmbeddingBase, path: str = None
    ):
        if faiss is None:
            raise ImportError(
                "`faiss` is required for using vectordb.faiss."
                "Install it with `pip install faiss-cpu`."
            )
        store = cls(faiss.IndexFlatL2(len(embedding[0])), embedder, path)
        store.add_vectors(embedding)
        return store

    @classmethod
    def load(cls, path: str, embedder: EmbeddingBase):
        if faiss is None:
            raise ImportError(
                "`faiss` is required for using vectordb.faiss."
                "Install it with `pip install faiss-cpu`."
            )

        index = faiss.read_index(path)
        return cls(index, embedder, path)

    def save(self, path: str = None):
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
