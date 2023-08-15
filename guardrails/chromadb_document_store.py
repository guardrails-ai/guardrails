import chromadb
from typing import Optional, List, Dict
from guardrails.document_store import Document, DocumentStoreBase
from guardrails.embedding import EmbeddingBase
from nltk.tokenize import sent_tokenize
from sklearn.metrics import accuracy_score, auc, roc_curve
from tqdm.notebook import tqdm

try:
    import sqlalchemy
    import sqlalchemy.orm as orm
except ImportError:
    sqlalchemy = None
    orm = None

THRESHOLD = 0.86
CHUNK_STRATEGY = "sentence"
CHUNK_SIZE = 5
CHUNK_OVERLAP = 2
DISTANCE_METRIC = "cosine"
EMBEDDING_MODEL = "bge-small"

CONFIG = {
    'threshold': THRESHOLD,
    'chunk_strategy': CHUNK_STRATEGY,
    'chunk_size': CHUNK_SIZE,
    'chunk_overlap': CHUNK_OVERLAP,
    'embedding_model': EMBEDDING_MODEL,
    'distance_metric': DISTANCE_METRIC,
}


class ChromaDbDocumentStore(DocumentStoreBase):

    def __init__(self, path: Optional[str] = None):
        if sqlalchemy is None:
            raise ImportError(
                "SQLAlchemy is required for EphemeralDocumentStore"
                "Please install it using `pip install SqlAlchemy`"
            )
        chroma_client = chromadb.PersistentClient(path=path)
        config_str = f"model_{EMBEDDING_MODEL}_dist_{DISTANCE_METRIC}_chunking_{CHUNK_STRATEGY}_size_{CHUNK_SIZE}_overlap_{CHUNK_OVERLAP}"
        self.db_collection = chroma_client.get_or_create_collection(config_str, metadata=CONFIG)

    
    def add_document(self, raw_texts):
        return self.add_sources_to_vector_db(raw_texts)
    
    def get_chunks_from_text(self, text: str, chunk_strategy: str, chunk_size: int, chunk_overlap: int ) -> List[str]:
        if chunk_strategy == "sentence":
            atomic_units = sent_tokenize(text)
        else:
            raise ValueError(f"Invalid chunk strategy: {chunk_strategy}. Valid choices are 'sentences'.")

        chunks = []
        for i in range(0, len(atomic_units), chunk_size - chunk_overlap):
            chunk = " ".join(atomic_units[i : i + chunk_size])
            chunks.append(chunk)
        return chunks

    def add_sources_to_vector_db(self,raw_texts: Dict[str, str]):
        for idx, text in tqdm(raw_texts.items()):
            chunks = self.get_chunks_from_text(text, CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP)
            self.db_collection.add(
                documents=chunks,
                metadatas=[{"doc_id": idx, "chunk_id": chunk_id} for chunk_id in range(len(chunks))],
                ids=[f"{idx}-{chunk_id}" for chunk_id in range(len(chunks))]
            )
    
    def similarity_search_vector_with_metadata_filters(self, vector: List[float], k: int, metadata_filters: Dict[str, str]) -> List[int]:
        doc_id = row.name # This is the index of the row.
        text = row["response"]

        query_output = self.db_collection.query(
            query_texts=[text]
            n_results=k,
            where={"doc_id": doc_id}
        )
