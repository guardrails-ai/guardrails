from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from typing import List, Optional

from pydantic import ValidationError

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
    def validate_sql(self, query: str) -> List[str]:
        ...

    @abstractmethod
    def get_schema(self) -> str:
        ...


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

    def _get_k_rows(self, table_name: str, k: int = 1000) -> pd.DataFrame:
        """Get the first k rows of a table."""
        query = f"SELECT * FROM {table_name} LIMIT {k}"
        return pd.read_sql(query, self._conn)

    def validate_sql(self, query: str) -> List[str]:
        exceptions: List[str] = []
        try:
            self._conn.execute(text(query))
        except Exception as ex:
            exceptions.append(str(ex))
        return exceptions

    def get_schema(self, lower: bool = False) -> str:
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
        # LAUREL
        lower = True
        do_lower = lambda x: x.lower() if lower else x
        for table, columns in schema.items():
            # table_str = f"CREATE TABLE {do_lower(table)}"

            table_str = f"CREATE TABLE {do_lower(table)} (\n"
            column_str = ",\n".join(
                [
                    f"    {do_lower(column)} {column_info['type']}"
                    for column, column_info in columns.items()
                ]
            )
            table_str += column_str + "\n)"
            max_examples = 3
            df = self._get_k_rows(table)
            # Sort by all columns
            df = df.sort_values(by=df.columns.tolist(), ignore_index=True)
            # Extract a list of unique values for string columns only
            cols_to_distinct_values = {
                col: sorted(list(map(str, set(df[col].unique()))))[:max_examples]
                for col in df.columns
                if pd.api.types.is_string_dtype(df[col].dtype)
                or len(set(df[col].unique())) <= max_examples
            }
            cols_str = "\n".join(
                f"{col}: {', '.join(values)}"
                for col, values in cols_to_distinct_values.items()
            )
            table_str += f"\n/*\nExample Values for String Columns\n{cols_str}\n*/"

            formatted_schema.append(table_str)

            # formatted_schema.append(f"Table: {do_lower(table)}")
            # for column, column_info in columns.items():
            #     formatted_schema.append(f"    Column: {do_lower(column)}")
            #     for info, value in column_info.items():
            #         formatted_schema.append(f"        {do_lower(info)}: {value}")

        return "\n".join(formatted_schema)


def create_sql_driver(
    schema_file: Optional[str] = None, conn: Optional[str] = None
) -> SQLDriver:
    if schema_file is None and conn is None:
        return SimpleSqlDriver()
    return SqlAlchemyDriver(schema_file=schema_file, conn=conn)
