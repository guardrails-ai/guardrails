from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

try:
    import sqlalchemy
    from sqlalchemy import text

    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False


class SQLDriver(ABC):
    """
    Abstract class for SQL drivers. The expose common functionality
    for validating SQL queries.
    """

    @abstractmethod
    def validate_sql(self, query: str) -> List[str]:
        ...


class SimpleSqlDriver(SQLDriver):
    """
    Simple SQL driver which uses sqlvalidator to validate SQL queries.
    Does not understands dialects and is not connected to a database.
    """

    def validate_sql(self, query: str) -> List[str]:
        import sqlvalidator

        sql_query = sqlvalidator.parse(query)
        if not sql_query.is_valid():
            return sql_query.errors
        return sql_query.errors


class SqlAlchemyDriver(SQLDriver):
    """
    SQL driver which uses sqlalchemy to validate SQL queries.
    It can setup the database schema and check if the queries are valid
    by connecting to the database.
    """

    def __init__(self, schema_file: Optional[str], conn: Optional[str]) -> None:
        if not _HAS_SQLALCHEMY:
            raise ImportError(
                """The functionality requires sqlalchemy to be installed.
                              Please install it using `pip install SqlAlchemy`"""
            )

        if schema_file is not None and conn is None:
            raise ValidationError(
                """schema_file should accompany a sql connection string for
           guardrails to apply it to a database backend.
           Use sqlite for ex: sqlite://"""
            )

        if conn is not None:
            try:
                self._engine = sqlalchemy.create_engine(conn)
                self._conn = self._engine.connect()
            except Exception as ex:
                raise ValueError(ex)

        if schema_file is not None:
            schema = Path(schema_file).read_text()
            if conn.startswith("sqlite"):
                self._conn.connection.executescript(schema)
            else:
                from sqlalchemy import text

                self._conn.execute(text(schema))

    def validate_sql(self, query: str) -> List[str]:
        exceptions: List[str] = []
        try:
            self._conn.execute(text(query))
        except Exception as ex:
            exceptions.append(str(ex))
        return exceptions


def create_sql_driver(schema_file: Optional[str], conn: Optional[str]) -> SQLDriver:
    if schema_file is None and conn is None:
        return SimpleSqlDriver()
    return SqlAlchemyDriver(schema_file=schema_file, conn=conn)
