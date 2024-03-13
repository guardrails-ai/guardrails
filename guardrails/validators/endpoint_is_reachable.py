from typing import Any, Callable, Dict, Optional
from warnings import warn

from guardrails.logger import logger
from guardrails.validator_base import (
    VALIDATOR_IMPORT_WARNING,
    VALIDATOR_NAMING,
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="is-reachable", data_type=["string"])
class EndpointIsReachable(Validator):
    """Validates that a value is a reachable URL.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `is-reachable`                    |
    | Supported data types          | `string`,                         |
    | Programmatic fix              | None                              |
    """

    def __init__(self, on_fail: Optional[Callable] = None):
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
        super().__init__(on_fail=on_fail)

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} is a valid URL...")

        import requests

        # Check that the URL exists and can be reached
        try:
            response = requests.get(value)
            if response.status_code != 200:
                return FailResult(
                    error_message=f"URL {value} returned "
                    f"status code {response.status_code}",
                )
        except requests.exceptions.ConnectionError:
            return FailResult(
                error_message=f"URL {value} could not be reached",
            )
        except requests.exceptions.InvalidSchema:
            return FailResult(
                error_message=f"URL {value} does not specify "
                f"a valid connection adapter",
            )
        except requests.exceptions.MissingSchema:
            return FailResult(
                error_message=f"URL {value} does not contain " f"a http schema",
            )

        return PassResult()
