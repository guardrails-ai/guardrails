import hashlib
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from guardrails.vectordb import VectorDBBase

try:
    import sqlalchemy
    import sqlalchemy.orm as orm
except ImportError:
    sqlalchemy = None
    orm = None


@dataclass
class Document:
    """Document holds text and metadata of a document.

    Examples of documents are PDFs, Word documents, etc. A collection of
    related text in an NLP application can be thought of a document as
    well.
    """

    id: str
    pages: Dict[int, str]
    metadata: Dict[Any, Any] = None


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

    @abstractmethod
    def add_document(self, document: Document):
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
            example: [{"I am feeling good", {"sentiment": "postive"}}]

        Returns:
            List of ids of the texts."""
        ...

    @abstractmethod
    def flush():
        """Flushes the store to disk."""
        ...


class EphemeralDocumentStore(DocumentStoreBase):
    """EphemeralDocumentStore is a document store that stores the documents on
    local disk and use a ephemeral vector store like Faiss."""

    def __init__(self, vector_db: VectorDBBase, path: Optional[str] = None):
        """Creates a new EphemeralDocumentStore.

        Args:
            vector_db: VectorDBBase instance to use for storing the vectors.
            path: Path to the database file store metadata.
        """
        if sqlalchemy is None:
            raise ImportError(
                "SQLAlchemy is required for EphemeralDocumentStore"
                "Please install it using `pip install SqlAlchemy`"
            )
        self._vector_db = vector_db
        self._storage = SQLMetadataStore(path=path)

    def add_document(self, document: Document):
        # Add the document, in case the document is already there it
        # would raise an exception and we assume the document and
        # vectors are present.
        try:
            self._storage.add_docs(
                [document], vdb_last_index=self._vector_db.last_index()
            )
        except sqlalchemy.exc.IntegrityError:
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
        filtered_ids = filter(lambda x: x != -1, vector_db_indexes)
        return self._storage.get_pages_for_for_indexes(filtered_ids)

    def search_with_threshold(
        self, query: str, threshold: float, k: int = 4
    ) -> List[Page]:
        vector_db_indexes = self._vector_db.similarity_search_with_threshold(
            query, k, threshold
        )
        filtered_ids = filter(lambda x: x != -1, vector_db_indexes)
        return self._storage.get_pages_for_for_indexes(filtered_ids)

    def flush(self, path: Optional[str] = None):
        self._vector_db.save(path)


if orm is not None:
    Base = orm.declarative_base()

    class SqlDocument(Base):
        __tablename__ = "documents"

        id: orm.Mapped[int] = orm.mapped_column(primary_key=True)
        page_num: orm.Mapped[int] = orm.mapped_column(
            sqlalchemy.Integer, primary_key=True
        )
        text: orm.Mapped[str] = orm.mapped_column(sqlalchemy.String)
        meta: orm.Mapped[dict] = orm.mapped_column(sqlalchemy.PickleType)
        vector_index: orm.Mapped[int] = orm.mapped_column(sqlalchemy.Integer)

else:

    class SqlDocument:
        pass


class SQLMetadataStore:
    def __init__(self, path: Optional[str] = None):
        conn = f"sqlite:///{path}" if path is not None else "sqlite://"
        self._engine = sqlalchemy.create_engine(conn)
        SqlDocument.metadata.create_all(self._engine, checkfirst=True)

    def add_docs(self, docs: List[Document], vdb_last_index: int):
        vector_id = vdb_last_index
        with orm.Session(self._engine) as session:
            for doc in docs:
                for page_num, text in doc.pages.items():
                    session.add(
                        SqlDocument(
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
        with orm.Session(self._engine) as session:
            for index in indexes:
                query = sqlalchemy.select(SqlDocument).where(
                    SqlDocument.vector_index == index
                )
                sql_docs = session.execute(query)
                sql_doc = sql_docs.first()[0]
                pages.append(
                    Page(
                        PageCoordinates(sql_doc.id, sql_doc.page_num),
                        sql_doc.text,
                        sql_doc.meta,
                    )
                )

        return pages
