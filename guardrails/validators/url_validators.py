import ipaddress
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="is-valid-url", data_type=["string"])
class IsValidURL(Validator):
    """Validates that a value is a well-formed URL.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `is-valid-url`                    |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        require_https: If True, only https scheme is accepted (default False).
        allowed_schemes: List of allowed schemes (default ["http", "https"]).
    """

    def __init__(
        self,
        require_https: bool = False,
        allowed_schemes: Optional[List[str]] = None,
        on_fail: Optional[Any] = None,
    ):
        super().__init__(
            on_fail=on_fail,
            require_https=require_https,
            allowed_schemes=allowed_schemes,
        )
        self._require_https = require_https
        self._allowed_schemes = allowed_schemes or ["http", "https"]

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if not isinstance(value, str) or not value.strip():
            return FailResult(
                error_message=f"Value is not a valid URL: {value}.",
            )

        try:
            result = urlparse(value)
            if not result.scheme or not result.netloc:
                return FailResult(
                    error_message=f"URL {value} is not valid: missing scheme or network location.",
                )

            if self._require_https and result.scheme != "https":
                return FailResult(
                    error_message=f"URL {value} must use https scheme.",
                )

            if result.scheme not in self._allowed_schemes:
                return FailResult(
                    error_message=f"URL {value} has scheme '{result.scheme}', "
                    f"which is not in allowed schemes: {self._allowed_schemes}.",
                )

            domain = result.netloc.split(":")[0].lower()
            domain_pattern = re.compile(
                r"^([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
            )
            # Check if netloc is an IP address - that's fine too
            is_ip = False
            try:
                ipaddress.ip_address(domain)
                is_ip = True
            except ValueError:
                pass

            if not is_ip and not domain_pattern.match(domain):
                return FailResult(
                    error_message=f"URL {value} has an invalid domain: {domain}.",
                )

        except Exception:
            return FailResult(
                error_message=f"URL {value} is not valid.",
            )

        return PassResult()


@register_validator(name="is-valid-email", data_type=["string"])
class IsValidEmail(Validator):
    """Validates that a value is a well-formed email address.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `is-valid-email`                  |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |
    """

    EMAIL_REGEX = re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?"
        r"(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$"
    )

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if not isinstance(value, str) or not value.strip():
            return FailResult(
                error_message=f"Value is not a valid email address: {value}.",
            )

        if not self.EMAIL_REGEX.match(value.strip()):
            return FailResult(
                error_message=f"Value {value} is not a valid email address.",
            )

        return PassResult()


@register_validator(name="is-valid-domain", data_type=["string"])
class IsValidDomain(Validator):
    """Validates that a value is a valid domain name.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `is-valid-domain`                 |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        require_tld: If True, a TLD is required (default True).
    """

    DOMAIN_REGEX = re.compile(
        r"^([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
        r"[a-zA-Z]{2,}$"
    )

    def __init__(
        self,
        require_tld: bool = True,
        on_fail: Optional[Any] = None,
    ):
        super().__init__(
            on_fail=on_fail,
            require_tld=require_tld,
        )
        self._require_tld = require_tld

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if not isinstance(value, str) or not value.strip():
            return FailResult(
                error_message=f"Value is not a valid domain name: {value}.",
            )

        domain = value.strip().lower()

        if self._require_tld:
            if not self.DOMAIN_REGEX.match(domain):
                return FailResult(
                    error_message=f"Domain {value} is not a valid domain name.",
                )
        else:
            simple_pattern = re.compile(
                r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
                r"(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
            )
            if not simple_pattern.match(domain):
                return FailResult(
                    error_message=f"Domain {value} is not a valid domain name.",
                )

        # Check domain part length constraints
        labels = domain.split(".")
        for label in labels:
            if len(label) > 63:
                return FailResult(
                    error_message=f"Domain {value} has a label exceeding 63 characters.",
                )
            if label.startswith("-") or label.endswith("-"):
                return FailResult(
                    error_message=f"Domain {value} has a label with leading or trailing hyphen.",
                )

        if len(domain) > 253:
            return FailResult(
                error_message=f"Domain {value} exceeds maximum length of 253 characters.",
            )

        return PassResult()


@register_validator(name="is-valid-ip-address", data_type=["string"])
class IsValidIPAddress(Validator):
    """Validates that a value is a valid IPv4 or IPv6 address.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `is-valid-ip-address`             |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        ip_version: If 4, only IPv4. If 6, only IPv6. If None, both (default None).
    """

    def __init__(
        self,
        ip_version: Optional[int] = None,
        on_fail: Optional[Any] = None,
    ):
        super().__init__(
            on_fail=on_fail,
            ip_version=ip_version,
        )
        self._ip_version = ip_version

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if not isinstance(value, str) or not value.strip():
            return FailResult(
                error_message=f"Value is not a valid IP address: {value}.",
            )

        try:
            addr = ipaddress.ip_address(value.strip())
        except ValueError:
            return FailResult(
                error_message=f"Value {value} is not a valid IP address.",
            )

        if self._ip_version == 4 and not isinstance(addr, ipaddress.IPv4Address):
            return FailResult(
                error_message=f"Value {value} is not a valid IPv4 address.",
            )
        if self._ip_version == 6 and not isinstance(addr, ipaddress.IPv6Address):
            return FailResult(
                error_message=f"Value {value} is not a valid IPv6 address.",
            )

        return PassResult()


@register_validator(name="url-categorization", data_type=["string"])
class URLCategorization(Validator):
    """Categorizes a URL as safe/phishing/malware based on domain heuristics.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `url-categorization`              |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        safe_domains: List of known safe domains.
        malicious_domains: List of known malicious domains.
        phishing_domains: List of known phishing domains.
        threshold: Risk score threshold above which validation fails (default 50).
    """

    SUSPICIOUS_TLDS = {
        ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
        ".club", ".work", ".download", ".review", ".date",
        ".trade", ".men", ".win", ".bid", ".loan",
    }

    SUSPICIOUS_KEYWORDS = [
        "secure", "login", "verify", "account", "update",
        "confirm", "bank", "paypal", "signin", "auth",
    ]

    def __init__(
        self,
        safe_domains: Optional[List[str]] = None,
        malicious_domains: Optional[List[str]] = None,
        phishing_domains: Optional[List[str]] = None,
        threshold: int = 50,
        on_fail: Optional[Any] = None,
    ):
        super().__init__(
            on_fail=on_fail,
            safe_domains=safe_domains,
            malicious_domains=malicious_domains,
            phishing_domains=phishing_domains,
            threshold=threshold,
        )
        self._safe_domains = set(d.lower() for d in (safe_domains or []))
        self._malicious_domains = set(d.lower() for d in (malicious_domains or []))
        self._phishing_domains = set(d.lower() for d in (phishing_domains or []))
        self._threshold = threshold

    def _score_url(self, domain: str, full_url: str) -> int:
        score = 0

        domain_lower = domain.lower()
        url_lower = full_url.lower()

        if domain_lower in self._malicious_domains:
            score += 100
        if domain_lower in self._phishing_domains:
            score += 80
        if domain_lower in self._safe_domains:
            return 0

        for tld in self.SUSPICIOUS_TLDS:
            if domain_lower.endswith(tld):
                score += 30
                break

        for keyword in self.SUSPICIOUS_KEYWORDS:
            if keyword in url_lower:
                score += 15

        labels = domain_lower.split(".")
        if len(labels) > 3:
            score += 10

        if "-" in labels[0] if labels else "":
            score += 10

        import re
        digits = sum(1 for c in domain_lower if c.isdigit())
        if digits > 0:
            score += min(digits * 5, 20)

        return min(score, 100)

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if not isinstance(value, str) or not value.strip():
            return FailResult(
                error_message=f"Value is not a valid URL: {value}.",
            )

        try:
            result = urlparse(value)
            domain = result.netloc.split(":")[0]
            if not domain:
                return FailResult(
                    error_message=f"URL {value} does not contain a domain.",
                )
        except Exception:
            return FailResult(
                error_message=f"URL {value} is not valid.",
            )

        score = self._score_url(domain, value)
        if score >= self._threshold:
            if score >= 100:
                category = "malicious"
            elif score >= 80:
                category = "phishing"
            else:
                category = "suspicious"
            return FailResult(
                error_message=f"URL {value} is categorized as '{category}' "
                f"(risk score: {score}).",
            )

        return PassResult()
