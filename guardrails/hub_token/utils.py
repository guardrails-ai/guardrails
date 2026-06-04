import base64
import json
import time

from guardrails.hub_token.token import (
    TOKEN_EXPIRED_MESSAGE,
    TOKEN_INVALID_MESSAGE,
    ExpiredTokenError,
    InvalidTokenError,
)


def client_check_token_expiry(token: str) -> None:
    """Client-side check that token is not expired.

    Does NOT validate the signature — the Guardrails Hub server validates
    the signature on every request. This function exists only to fail
    fast on locally-known expired tokens.
    """
    try:
        _header, payload_b64, _signature = token.split(".")
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64 + "===")
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidTokenError(TOKEN_INVALID_MESSAGE) from exc

    exp = payload.get("exp")
    if exp is not None and exp < time.time():
        raise ExpiredTokenError(TOKEN_EXPIRED_MESSAGE)