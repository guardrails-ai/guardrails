from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="url-reachability", data_type=["string"])
class URLReachability(Validator):
    """Validates that a URL is reachable via HTTP and optionally checks TLS.

    Sends an HTTP HEAD request to the URL and checks whether the response
    status code falls within an acceptable range.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `url-reachability`                |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        timeout: HTTP request timeout in seconds.
        check_tls: Whether to verify TLS certificates.
        accept_status: HTTP status codes considered acceptable.
            Defaults to 200-399.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        check_tls: bool = True,
        accept_status: Optional[List[int]] = None,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(
            on_fail=on_fail,
            timeout=timeout,
            check_tls=check_tls,
            accept_status=accept_status,
        )
        self._timeout = timeout
        self._check_tls = check_tls
        self._accept_status = accept_status or list(range(200, 400))

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Checking URL reachability: {value}")

        url = value if "://" in value else "https://" + value

        try:
            parsed = urlparse(url)
        except ValueError:
            return FailResult(
                error_message=f"Cannot parse URL: {value}"
            )

        if not parsed.scheme or not parsed.netloc:
            return FailResult(
                error_message=f"Invalid URL (missing scheme or host): {value}"
            )

        try:
            response = httpx.head(
                url,
                timeout=self._timeout,
                verify=self._check_tls,
                follow_redirects=True,
            )
        except httpx.ConnectError:
            return FailResult(
                error_message=f"URL not reachable (connection failed): {value}"
            )
        except httpx.TimeoutException:
            return FailResult(
                error_message=f"URL not reachable (timeout after {self._timeout}s): {value}"
            )
        except httpx.HTTPError as exc:
            if "SSL" in str(exc) or "TLS" in str(exc) or "certificate" in str(exc).lower():
                return FailResult(
                    error_message=f"TLS certificate error for: {value}"
                )
            return FailResult(
                error_message=f"HTTP error for {value}: {exc}"
            )

        if response.status_code not in self._accept_status:
            return FailResult(
                error_message=(
                    f"URL returned status {response.status_code} "
                    f"(accepted: {self._accept_status[0]}-{self._accept_status[-1]}): {value}"
                )
            )

        return PassResult()
