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


def _extract_domain(value: str) -> Optional[str]:
    if "://" not in value:
        value = "https://" + value
    try:
        return urlparse(value).hostname
    except ValueError:
        return None


def _domain_matches(domain: str, pattern: str, match_subdomains: bool) -> bool:
    if domain == pattern:
        return True
    if match_subdomains and domain.endswith("." + pattern):
        return True
    return False


@register_validator(name="domain-blocklist", data_type=["string"])
class DomainBlocklist(Validator):
    """Domain blocklist and allowlist filtering.

    Checks whether the domain in a URL appears in a blocked list or is
    absent from an allowed list, depending on the chosen mode.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `domain-blocklist`                |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        blocked: Domains to reject.
        allowed: Domains to accept (used in allowlist mode).
        mode: "blocklist" rejects domains in `blocked`;
              "allowlist" rejects domains NOT in `allowed`.
        match_subdomains: Whether sub.evil.com matches evil.com.
    """

    def __init__(
        self,
        blocked: Optional[List[str]] = None,
        allowed: Optional[List[str]] = None,
        mode: str = "blocklist",
        match_subdomains: bool = True,
        on_fail: Optional[Callable] = None,
    ):
        assert mode in ("blocklist", "allowlist"), (
            'mode must be "blocklist" or "allowlist"'
        )
        super().__init__(
            on_fail=on_fail,
            blocked=blocked,
            allowed=allowed,
            mode=mode,
            match_subdomains=match_subdomains,
        )
        self._blocked = [d.lower() for d in (blocked or [])]
        self._allowed = [d.lower() for d in (allowed or [])]
        self._mode = mode
        self._match_subdomains = match_subdomains

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Checking domain blocklist for: {value}")

        domain = _extract_domain(value)
        if domain is None:
            return FailResult(error_message=f"Cannot extract domain from: {value}")

        domain = domain.lower()

        if self._mode == "blocklist":
            for pattern in self._blocked:
                if _domain_matches(domain, pattern, self._match_subdomains):
                    return FailResult(
                        error_message=f"Domain '{domain}' is blocked"
                    )

        elif self._mode == "allowlist":
            matched = any(
                _domain_matches(domain, pattern, self._match_subdomains)
                for pattern in self._allowed
            )
            if not matched:
                return FailResult(
                    error_message=f"Domain '{domain}' is not in the allowlist"
                )

        return PassResult()
