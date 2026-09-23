"""Shared helpers for client-side Hub token handling."""

import base64
import json
import time
from typing import Any, Dict


class TokenExpiredError(Exception):
    """Raised when a client-side Hub JWT carries an ``exp`` claim in the past."""

    pass


class TokenInvalidError(Exception):
    """Raised when a client-side Hub JWT is malformed."""

    pass


def client_check_token_expiry(token: str) -> None:
    """Client-side check that a Hub JWT is not expired.

    This function does NOT validate the token signature. Signature
    validation is performed by the Guardrails Hub server on every request;
    this check exists only to fail fast on locally-known expired tokens
    before a request goes out.

    It intentionally avoids ``jwt.decode(..., verify_signature=False)``:
    only the ``exp`` claim is read, via a plain base64 decode of the
    payload segment, so the unverified-signature call shape does not appear
    in the codebase.

    Args:
        token: The JWT to check.

    Raises:
        TokenExpiredError: If the token carries an ``exp`` claim that is
            in the past.
        TokenInvalidError: If the token is not a well-formed JWT, its
            payload is not a JSON object, or its ``exp`` claim is present
            but not numeric.
    """
    try:
        _header_b64, payload_b64, _signature_b64 = token.split(".")
        # JWTs use base64url encoding without padding.
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload: Dict[str, Any] = json.loads(base64.urlsafe_b64decode(payload_b64))
    except ValueError as exc:
        raise TokenInvalidError("Token is not a well-formed JWT.") from exc

    if not isinstance(payload, dict):
        raise TokenInvalidError("Token payload is not a JSON object.")

    exp = payload.get("exp")
    if exp is None:
        # No expiry claim: nothing to check client-side.
        return
    # bool is a subclass of int; exclude it explicitly.
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        raise TokenInvalidError("Token 'exp' claim must be numeric.")
    if exp <= time.time():
        raise TokenExpiredError("Token has expired.")
