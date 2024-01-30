from typing import Any, Callable, Dict, Optional

from guardrails.utils.sql_utils import SQLDriver, create_sql_driver
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="bug-free-sql", data_type=["string"])
class BugFreeSQL(Validator):
    """Validates that there are no SQL syntactic bugs in the generated code.

    This is a very minimal implementation that uses the Pypi `sqlvalidator` package
    to check if the SQL query is valid. You can implement a custom SQL validator
    that uses a database connection to check if the query is valid.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `bug-free-sql`                    |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |
    """

    def __init__(
        self,
        conn: Optional[str] = None,
        schema_file: Optional[str] = None,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail, conn=conn, schema_file=schema_file)
        self._driver: SQLDriver = create_sql_driver(schema_file=schema_file, conn=conn)

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        errors = self._driver.validate_sql(value)
        if len(errors) > 0:
            return FailResult(
                error_message=". ".join(errors),
            )

        return PassResult()
