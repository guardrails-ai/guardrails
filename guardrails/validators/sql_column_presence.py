from typing import Any, Callable, Dict, List, Optional
from warnings import warn

from guardrails.validator_base import (
    VALIDATOR_IMPORT_WARNING,
    VALIDATOR_NAMING,
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="sql-column-presence", data_type="string")
class SqlColumnPresence(Validator):
    """Validates that all columns in the SQL query are present in the schema.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `sql-column-presence`             |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        cols: The list of valid columns.
    """

    def __init__(self, cols: List[str], on_fail: Optional[Callable] = None):
        class_name = self.__class__.__name__
        if class_name not in VALIDATOR_NAMING:
            warn(
                f"""Validator {class_name} is deprecated and
                will be removed after version 0.5.x.
                """,
                FutureWarning,
            )
        else:
            warn(
                VALIDATOR_IMPORT_WARNING.format(
                    validator_name=class_name,
                    hub_validator_name=VALIDATOR_NAMING[class_name][0],
                    hub_validator_url=VALIDATOR_NAMING[class_name][1],
                ),
                FutureWarning,
            )
        super().__init__(on_fail=on_fail, cols=cols)
        self._cols = set(cols)

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        from sqlglot import exp, parse

        expressions = parse(value)
        cols = set()
        for expression in expressions:
            if expression is None:
                continue
            for col in expression.find_all(exp.Column):
                cols.add(col.alias_or_name)

        diff = cols.difference(self._cols)
        if len(diff) > 0:
            return FailResult(
                error_message=f"Columns [{', '.join(diff)}] "
                f"not in [{', '.join(self._cols)}]",
            )

        return PassResult()
