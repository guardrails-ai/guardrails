from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError


class SQLDriver(ABC):
    @abstractmethod
    def validate_sql(self, query: str) -> List[str]:
        ...


class SimpleSqlDriver(SQLDriver):
    def validate_sql(self, query: str) -> List[str]:
        import sqlvalidator

        sql_query = sqlvalidator.parse(query)
        if not sql_query.is_valid():
            return sql_query.errors
        return sql_query.errors


class SqlAlchemyDriver(SQLDriver):
    def __init__(self, schema_file: Optional[str], conn: Optional[str]) -> None:
        import sqlalchemy as db
        from sqlalchemy import text

        if schema_file is not None and conn is None:
            raise ValidationError(
                """schema_file should accompany a sql connection string for
           guardrails to apply it to a database backend.
           Use sqlite for ex: sqlite://"""
            )

        if conn is not None:
            try:
                self._engine = db.create_engine(conn)
                self._conn = self._engine.connect()
            except Exception as ex:
                raise ValueError(ex)

        if schema_file is not None:
            schema = Path(schema_file).read_text()
            self._conn.execute(text(schema))

    def validate_sql(self, query: str) -> List[str]:
        from sqlalchemy import text

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
