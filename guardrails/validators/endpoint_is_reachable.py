from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional

from guardrails.logger import logger
from guardrails.validator_base import (
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

    Args:
        host_whitelist: The list of allowed hosts.
    """

    def __init__(self, domain_whitelist: List[Any], on_fail: Optional[Callable] = None):
        super().__init__(
            on_fail=on_fail,
            domain_whitelist=domain_whitelist,
        )
        self._domain_whitelist = domain_whitelist

    def _is_whitelisted(self, url):
        from tldextract import extract

        extracted = extract(url)
        domain = "{}.{}".format(extracted.domain, extracted.suffix)
        
        return domain in self._domain_whitelist

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} is in choices {self._domain_whitelist}...")

        if value not in self._domain_whitelist:
            return FailResult(
                error_message=f"Host of {value} is not whitelisted in {self._domain_whitelist}.",
            )

        logger.debug(f"Requesting {value} for a reachability validation...")
        import requests

        # Check that the URL exists and can be reached
        try:
            response = requests.head(value)
            if response.status_code != HTTPStatus.OK:
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
