import hashlib
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import Field

from guardrails.vectordb import VectorDBBase

try:
    from sqlalchemy.exc import IntegrityError
except ImportError:
    pass


@dataclass
class Document:
    """Document holds text and metadata of a document.

    Examples of documents are PDFs, Word documents, etc. A collection of
    related text in an NLP application can be thought of a document as
    well.
    """

    id: str
    pages: Dict[int, str]
    metadata: Dict[Any, Any] = Field(default_factory=dict)


# PageCoordinates is a datastructure that points to the location
# of a page in a document.
PageCoordinates = namedtuple("PageCoordinates", ["doc_id", "page_num"])


@dataclass
class Page:
    """Page holds text and metadata of a page in a document.

    It also containts the coordinates of the page in the document.
    """

    cordinates: PageCoordinates
    text: str
    metadata: Dict[Any, Any]


class DocumentStoreBase(ABC):
    """Abstract class for a store that can store text, and metadata from
    documents.

    The store can be queried by text for similar documents.
    """

    def __init__(self, vector_db: VectorDBBase, path: Optional[str] = None): ...

    @abstractmethod
    def add_document(self, document: Document) -> None:
        """Adds a document to the store.

        Args:
            document: Document object to be added

        Returns:
            None if the document was added successfully
        """
        ...

    @abstractmethod
    def search(self, query: str, k: int = 4) -> List[Page]:
        """Searches for pages which contain the text similar to the query.

        Args:
            query: Text to search for.
            k: Number of similar pages to return.

        Returns:
            List[Pages] List of pages which contains similar texts
        """
        ...

    @abstractmethod
    def add_text(self, text: str, meta: Dict[Any, Any]) -> str:
        """Adds a text to the store.
        Args:
            text: Text to add.
            meta: Metadata to associate with the text.

        Returns:
            The id of the text.
        """
        ...

    @abstractmethod
    def add_texts(self, texts: Dict[str, Dict[Any, Any]]) -> List[str]:
        """Adds a list of texts to the store.
        Args:
            texts: List of texts to add, and their associalted metadata.
            example:
            ``` json
            [{"I am feeling good", {"sentiment": "postive"}}]
            ```

        Returns:
            List of ids of the texts."""
        ...

    @abstractmethod
    def flush():
        """Flushes the store to disk."""
        ...


try:
    import sqlalchemy
    from sqlalchemy.orm import Mapped, Session, declarative_base, mapped_column

    class RealEphemeralDocumentStore(DocumentStoreBase):
        """EphemeralDocumentStore is a document store that stores the documents
        on local disk and use a ephemeral vector store like Faiss."""

        def __init__(self, vector_db: VectorDBBase, path: Optional[str] = None):
            """Creates a new EphemeralDocumentStore.

            Args:
                vector_db: VectorDBBase instance to use for storing the vectors.
                path: Path to the database file store metadata.
            """
            self._vector_db = vector_db
            self._storage = RealSQLMetadataStore(path=path)

        def add_document(self, document: Document):
            # Add the document, in case the document is already there it
            # would raise an exception and we assume the document and
            # vectors are present.
            try:
                self._storage.add_docs(
                    [document], vdb_last_index=self._vector_db.last_index()
                )
            except IntegrityError:
                return
            self._vector_db.add_texts(list(document.pages.values()))

        def add_text(self, text: str, meta: Dict[Any, Any]) -> str:
            hash = hashlib.md5()
            hash.update(text.encode("utf-8"))
            hash.update(str(meta).encode("utf-8"))
            id = hash.hexdigest()

            doc = Document(id, {0: text}, meta)
            self.add_document(doc)
            return doc.id

        def add_texts(self, texts: Dict[str, Dict[Any, Any]]) -> List[str]:
            doc_ids = []
            for text, meta in texts.items():
                doc_id = self.add_text(text, meta)
                doc_ids.append(doc_id)
            return doc_ids

        def search(self, query: str, k: int = 4) -> List[Page]:
            vector_db_indexes = self._vector_db.similarity_search(query, k)
            filtered_ids = list(filter(lambda x: x != -1, vector_db_indexes))
            return self._storage.get_pages_for_for_indexes(filtered_ids)

        def search_with_threshold(
            self, query: str, threshold: float, k: int = 4
        ) -> List[Page]:
            vector_db_indexes = self._vector_db.similarity_search_with_threshold(
                query, k, threshold
            )
            filtered_ids = list(filter(lambda x: x != -1, vector_db_indexes))
            return self._storage.get_pages_for_for_indexes(filtered_ids)

        def flush(self, path: Optional[str] = None):
            self._vector_db.save(path)

    Base = declarative_base()

    class RealSqlDocument(Base):
        __tablename__ = "documents"

        id: Mapped[int] = mapped_column(primary_key=True)  # type: ignore
        page_num: Mapped[int] = mapped_column(sqlalchemy.Integer, primary_key=True)  # type: ignore
        text: Mapped[str] = mapped_column(sqlalchemy.String)  # type: ignore
        meta: Mapped[dict] = mapped_column(sqlalchemy.PickleType)  # type: ignore
        vector_index: Mapped[int] = mapped_column(sqlalchemy.Integer)  # type: ignore

    class RealSQLMetadataStore:
        def __init__(self, path: Optional[str] = None):
            conn = f"sqlite:///{path}" if path is not None else "sqlite://"
            self._engine = sqlalchemy.create_engine(conn)  # type: ignore
            RealSqlDocument.metadata.create_all(self._engine, checkfirst=True)

        def add_docs(self, docs: List[Document], vdb_last_index: int):
            vector_id = vdb_last_index
            with Session(self._engine) as session:
                for doc in docs:
                    for page_num, text in doc.pages.items():
                        session.add(
                            RealSqlDocument(
                                id=doc.id,
                                page_num=page_num,
                                text=text,
                                meta=doc.metadata,
                                vector_index=vector_id,
                            )
                        )
                        vector_id += 1

                session.commit()

        def get_pages_for_for_indexes(self, indexes: List[int]) -> List[Page]:
            pages: List[Page] = []
            with Session(self._engine) as session:
                for index in indexes:
                    query = sqlalchemy.select(RealSqlDocument).where(
                        RealSqlDocument.vector_index == index
                    )
                    sql_docs = session.execute(query)
                    sql_doc = sql_docs.first()
                    if sql_doc is None:
                        continue
                    sql_doc = sql_doc[0]
                    pages.append(
                        Page(
                            PageCoordinates(sql_doc.id, sql_doc.page_num),
                            sql_doc.text,
                            sql_doc.meta,
                        )
                    )

            return pages

    EphemeralDocumentStore = RealEphemeralDocumentStore
    SQLDocument = RealSqlDocument
    SQLMetadataStore = RealSQLMetadataStore
except ImportError:

    class FallbackEphemeralDocumentStore:
        def __init__(self, *args, **kwargs):
            # Why don't we just raise this when the import
            # error occurs instead of at runtime?
            raise ImportError(
                "SQLAlchemy is required for EphemeralDocumentStore"
                "Please install it using `poetry add SqlAlchemy`"
            )

    class FallbackSQLDocument:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SQLAlchemy is required for SQLDocument"
                "Please install it using `poetry add SqlAlchemy`"
            )

    class FallbackSQLMetadataStore:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SQLAlchemy is required for SQLMetadataStore"
                "Please install it using `poetry add SqlAlchemy`"
            )

    EphemeralDocumentStore = FallbackEphemeralDocumentStore
    SQLDocument = FallbackSQLDocument
    SQLMetadataStore = FallbackSQLMetadataStore
