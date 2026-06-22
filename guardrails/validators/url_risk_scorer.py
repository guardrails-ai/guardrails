import re
from difflib import SequenceMatcher
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

SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "buzz", "club",
    "work", "click", "link", "surf", "rest", "fit",
})

DEFAULT_KNOWN_DOMAINS = [
    "google.com", "facebook.com", "amazon.com", "apple.com",
    "microsoft.com", "github.com", "twitter.com", "linkedin.com",
    "youtube.com", "netflix.com", "instagram.com", "reddit.com",
    "wikipedia.org", "stackoverflow.com", "openai.com",
    "paypal.com", "chase.com", "bankofamerica.com",
]


def _extract_domain(value: str) -> Optional[str]:
    if "://" not in value:
        value = "https://" + value
    try:
        return urlparse(value).hostname
    except ValueError:
        return None


def _score_suspicious_tld(domain: str) -> float:
    tld = domain.rsplit(".", 1)[-1].lower()
    return 1.0 if tld in SUSPICIOUS_TLDS else 0.0


def _score_subdomain_depth(domain: str) -> float:
    parts = domain.split(".")
    depth = len(parts) - 2
    if depth <= 1:
        return 0.0
    if depth <= 3:
        return 0.5
    return 1.0


def _score_digit_mix(domain: str) -> float:
    name = domain.rsplit(".", 1)[0]
    if not name:
        return 0.0
    has_alpha = any(c.isalpha() for c in name)
    has_digit = any(c.isdigit() for c in name)
    if has_alpha and has_digit:
        digit_ratio = sum(c.isdigit() for c in name) / len(name)
        if 0.1 < digit_ratio < 0.6:
            return 1.0
    return 0.0


def _score_long_domain(domain: str) -> float:
    if len(domain) > 50:
        return 1.0
    if len(domain) > 30:
        return 0.5
    return 0.0


def _score_typosquatting(domain: str, known_domains: List[str]) -> float:
    base = domain.split(".")
    if len(base) >= 2:
        candidate = ".".join(base[-2:]).lower()
    else:
        candidate = domain.lower()

    best_ratio = 0.0
    for known in known_domains:
        if candidate == known:
            return 0.0
        ratio = SequenceMatcher(None, candidate, known).ratio()
        if ratio > best_ratio:
            best_ratio = ratio

    if best_ratio > 0.85:
        return 1.0
    if best_ratio > 0.75:
        return 0.5
    return 0.0


@register_validator(name="url-risk-scorer", data_type=["string"])
class URLRiskScorer(Validator):
    """Pattern-based URL risk scoring.

    Calculates a weighted risk score from multiple signals: suspicious TLDs,
    excessive subdomains, digit-letter mixing, long domains, and
    typosquatting similarity to well-known domains.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `url-risk-scorer`                 |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        threshold: Risk score above which the URL fails validation (0.0-1.0).
        known_domains: Domains used for typosquatting detection.
    """

    WEIGHTS = {
        "suspicious_tld": 0.20,
        "subdomain_depth": 0.15,
        "digit_mix": 0.20,
        "long_domain": 0.10,
        "typosquatting": 0.35,
    }

    def __init__(
        self,
        threshold: float = 0.7,
        known_domains: Optional[List[str]] = None,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(
            on_fail=on_fail,
            threshold=threshold,
            known_domains=known_domains,
        )
        self._threshold = threshold
        self._known_domains = known_domains or DEFAULT_KNOWN_DOMAINS

    def _score(self, domain: str) -> tuple:
        signals = {
            "suspicious_tld": _score_suspicious_tld(domain),
            "subdomain_depth": _score_subdomain_depth(domain),
            "digit_mix": _score_digit_mix(domain),
            "long_domain": _score_long_domain(domain),
            "typosquatting": _score_typosquatting(domain, self._known_domains),
        }

        total = sum(
            signals[k] * self.WEIGHTS[k] for k in signals
        )

        reasons = [k for k, v in signals.items() if v > 0]
        return total, reasons

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Scoring URL risk: {value}")

        domain = _extract_domain(value)
        if domain is None:
            return FailResult(
                error_message=f"Cannot extract domain from: {value}"
            )

        score, reasons = self._score(domain)

        if score > self._threshold:
            return FailResult(
                error_message=(
                    f"High risk URL (score={score:.2f}, "
                    f"threshold={self._threshold}): {', '.join(reasons)}"
                )
            )

        return PassResult()
