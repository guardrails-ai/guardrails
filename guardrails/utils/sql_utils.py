from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

try:
    import sqlalchemy
    from sqlalchemy import text

    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False


class SQLDriver(ABC):
    """Abstract class for SQL drivers.

    The expose common functionality for validating SQL queries.
    """

    @abstractmethod
    def validate_sql(self, query: str) -> List[str]: ...

    @abstractmethod
    def get_schema(self) -> str: ...


class SimpleSqlDriver(SQLDriver):
    """Simple SQL driver which uses sqlvalidator to validate SQL queries.

    Does not understands dialects and is not connected to a database.
    """

    def validate_sql(self, query: str) -> List[str]:
        import sqlvalidator

        sql_query = sqlvalidator.parse(query)
        if not sql_query.is_valid():
            return sql_query.errors
        return sql_query.errors

    def get_schema(self) -> str:
        raise NotImplementedError


class SqlAlchemyDriver(SQLDriver):
    """SQL driver which uses sqlalchemy to validate SQL queries.

    It can setup the database schema and check if the queries are valid
    by connecting to the database.
    """

    def __init__(self, schema_file: Optional[str], conn: Optional[str]) -> None:
        if not _HAS_SQLALCHEMY:
            raise ImportError(
                """The functionality requires sqlalchemy to be installed.
                              Please install it using `poetry add SqlAlchemy`"""
            )

        if schema_file is not None and conn is None:
            raise RuntimeError(
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
            if conn is not None and conn.startswith("sqlite"):
                self._conn.connection.executescript(schema)  # type: ignore
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

    def get_schema(self) -> str:
        # Get table schema using sqlalchemy.inspect
        insp = sqlalchemy.inspect(self._conn)

        schema = {}
        for table in insp.get_table_names():
            schema[table] = {}
            for column in insp.get_columns(table):
                schema[table][column["name"]] = {"type": column["type"]}

            # Get foreign keys
            for fk in insp.get_foreign_keys(table):
                schema[table][fk["constrained_columns"][0]]["foreign_key"] = {
                    "table": fk["referred_table"],
                    "column": fk["referred_columns"][0],
                }

        # Create a nicely formatted schema from the dictionary
        formatted_schema = []
        for table, columns in schema.items():
            formatted_schema.append(f"Table: {table}")
            for column, column_info in columns.items():
                formatted_schema.append(f"    Column: {column}")
                for info, value in column_info.items():
                    formatted_schema.append(f"        {info}: {value}")

        return "\n".join(formatted_schema)


def create_sql_driver(
    schema_file: Optional[str] = None, conn: Optional[str] = None
) -> SQLDriver:
    if schema_file is None and conn is None:
        return SimpleSqlDriver()
    return SqlAlchemyDriver(schema_file=schema_file, conn=conn)
