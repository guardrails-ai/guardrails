from typing import Any, Dict

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
    """

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} is a valid URL...")

        import requests
        from urllib.parse import urlparse
        from tldextract import extract

        # Check that the URL exists and can be reached
        try:
            url = urlparse(value)
            extracted = extract(value)

            # Check that the URL has a scheme and network location
            if not url.scheme or not extracted.domain or not extracted.suffix:
                return FailResult(
                    error_message=f"URL {value} is not valid.",
                )

            # Reconstruct the URL with consideration for 'www' subdomain
            subdomain = f"{extracted.subdomain}." if "www" in extracted.subdomain else ""
            sanitized_url = f"{url.scheme}://{subdomain}{extracted.domain}.{extracted.suffix}/"

            response = requests.get(sanitized_url)
            if response.status_code != 200:
                return FailResult(
                    error_message=f"URL {sanitized_url} returned "
                    f"status code {response.status_code}",
                )
        except requests.exceptions.ConnectionError:
            return FailResult(
                error_message=f"URL {sanitized_url} could not be reached",
            )
        except requests.exceptions.InvalidSchema:
            return FailResult(
                error_message=f"URL {sanitized_url} does not specify "
                f"a valid connection adapter",
            )
        except requests.exceptions.MissingSchema:
            return FailResult(
                error_message=f"URL {sanitized_url} does not contain " f"a http schema",
            )
        except ValueError:
            return FailResult(
                error_message=f"URL {value} is not valid.",
            )

        return PassResult()
