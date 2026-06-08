import socket
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


class _TTLCache:
    """Simple LRU cache with per-entry TTL."""

    def __init__(self, maxsize: int = 256, ttl: float = 3600.0):
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple] = OrderedDict()

    def get(self, key: str):
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.monotonic() - ts > self._ttl:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return value

    def put(self, key: str, value: bool):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.monotonic())
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


@register_validator(name="url-dns-resolver", data_type=["string"])
class URLDNSResolver(Validator):
    """Validates that the domain in a URL resolves via DNS.

    Uses socket.getaddrinfo to check whether the hostname actually exists.
    Results are cached with a configurable TTL to avoid repeated lookups.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `url-dns-resolver`                |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        timeout: DNS lookup timeout in seconds.
        cache_size: Maximum number of cached DNS results.
        cache_ttl: Cache entry lifetime in seconds.
    """

    def __init__(
        self,
        timeout: float = 5.0,
        cache_size: int = 256,
        cache_ttl: float = 3600.0,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(
            on_fail=on_fail,
            timeout=timeout,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
        )
        self._timeout = timeout
        self._cache = _TTLCache(maxsize=cache_size, ttl=cache_ttl)

    def _resolve(self, hostname: str) -> bool:
        cached = self._cache.get(hostname)
        if cached is not None:
            return cached

        old_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(self._timeout)
            socket.getaddrinfo(hostname, None)
            result = True
        except (socket.gaierror, socket.timeout, OSError):
            result = False
        finally:
            socket.setdefaulttimeout(old_timeout)

        self._cache.put(hostname, result)
        return result

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Resolving DNS for: {value}")

        url = value if "://" in value else "https://" + value
        try:
            hostname = urlparse(url).hostname
        except ValueError:
            hostname = None

        if not hostname:
            return FailResult(
                error_message=f"Cannot extract hostname from: {value}"
            )

        if self._resolve(hostname):
            return PassResult()

        return FailResult(
            error_message=f"Domain does not resolve: {hostname}"
        )
