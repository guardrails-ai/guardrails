# This file contains the validator for the exclude-sql-predicates guardrail

from typing import Any, Callable, Dict, List, Optional

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="exclude-sql-predicates", data_type="string")
class ExcludeSqlPredicates(Validator):
    """Validates that the SQL query does not contain certain predicates.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `exclude-sql-predicates`          |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        predicates: The list of predicates to avoid.
    """

    def __init__(self, predicates: List[str], on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, predicates=predicates)
        self._predicates = set(predicates)

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        from sqlglot import exp, parse

        expressions = parse(value)
        for expression in expressions:
            if expression is None:
                continue
            for pred in self._predicates:
                try:
                    getattr(exp, pred)
                except AttributeError:
                    raise ValueError(f"Predicate {pred} does not exist")
                if len(list(expression.find_all(getattr(exp, pred)))):
                    return FailResult(
                        error_message=f"SQL query contains predicate {pred}",
                        fix_value="",
                    )

        return PassResult()
