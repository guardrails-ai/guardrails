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

