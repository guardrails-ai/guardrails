from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

VALID_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int",
    "io", "co", "us", "uk", "de", "fr", "jp", "cn", "ru", "br", "in",
    "au", "ca", "it", "es", "nl", "se", "no", "fi", "dk", "pl", "cz",
    "at", "ch", "be", "ie", "pt", "kr", "tw", "hk", "sg", "nz", "za",
    "mx", "ar", "cl", "info", "biz", "name", "pro", "museum", "coop",
    "aero", "asia", "cat", "jobs", "mobi", "tel", "travel",
    "ai", "app", "dev", "cloud", "tech", "online", "store", "site",
    "xyz", "top", "tk", "ml", "ga", "cf", "gq",
}


def _is_ip_address(host: str) -> bool:
    parts = host.split(".")
    if len(parts) == 4:
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except ValueError:
            pass
    return ":" in host


@register_validator(name="url-format", data_type=["string"])
class URLFormatValidator(Validator):
    """Enhanced URL format validation.

    Validates URL structure beyond basic parsing: checks scheme whitelist,
    rejects bare IP addresses, and verifies domain structural quality.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `url-format`                      |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        require_https: Reject non-HTTPS URLs.
        allowed_schemes: Accepted URL schemes. Defaults to ["http", "https"].
        allow_ip: Whether to accept IP addresses as the host.
    """

    def __init__(
        self,
        require_https: bool = False,
        allowed_schemes: Optional[List[str]] = None,
        allow_ip: bool = False,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(
            on_fail=on_fail,
            require_https=require_https,
            allowed_schemes=allowed_schemes,
            allow_ip=allow_ip,
        )
        self._require_https = require_https
        self._allowed_schemes = allowed_schemes or ["http", "https"]
        self._allow_ip = allow_ip

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating URL format: {value}")

        try:
            parsed = urlparse(value)
        except ValueError:
            return FailResult(error_message=f"URL cannot be parsed: {value}")

        if not parsed.scheme or not parsed.netloc:
            return FailResult(
                error_message=f"URL missing scheme or host: {value}"
            )

        if parsed.scheme not in self._allowed_schemes:
            return FailResult(
                error_message=f"Scheme '{parsed.scheme}' not allowed. "
                f"Accepted: {self._allowed_schemes}"
            )

        if self._require_https and parsed.scheme != "https":
            return FailResult(
                error_message=f"HTTPS required, got '{parsed.scheme}'"
            )

        hostname = parsed.hostname or ""

        if not self._allow_ip and _is_ip_address(hostname):
            return FailResult(
                error_message=f"IP addresses not allowed as host: {hostname}"
            )

        if ".." in hostname:
            return FailResult(
                error_message=f"Invalid domain (consecutive dots): {hostname}"
            )

        if not _is_ip_address(hostname):
            parts = hostname.split(".")
            if len(parts) < 2:
                return FailResult(
                    error_message=f"Domain must have at least two parts: {hostname}"
                )

            tld = parts[-1]
            if len(tld) < 2 or len(tld) > 63:
                return FailResult(
                    error_message=f"Invalid TLD length ({len(tld)}): {hostname}"
                )

            if len(hostname) > 253:
                return FailResult(
                    error_message=f"Domain too long ({len(hostname)} chars): {hostname}"
                )

        return PassResult()
