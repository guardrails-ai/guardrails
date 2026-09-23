"""Regression tests for the shared Hub token expiry check.

Covers guardrails.hub_token.utils.client_check_token_expiry and the two
get_jwt_token wrappers that consume it
(guardrails.hub_token.token and guardrails.cli.server.hub_client).

Trust model under test: the client-side check fails fast on expired or
malformed tokens only. It deliberately does NOT verify signatures --
signature validation is performed by the Guardrails Hub server on every
request.
"""

import base64
import datetime
import json
from datetime import timezone

import jwt
import pytest

from guardrails.classes.rc import RC
from guardrails.cli.server import hub_client
from guardrails.hub_token import token as hub_token_module
from guardrails.hub_token.utils import (
    TokenExpiredError,
    TokenInvalidError,
    client_check_token_expiry,
)

SECRET = "test-secret"


def _encode(payload: dict) -> str:
    return jwt.encode(payload, SECRET, algorithm="HS256")


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _craft_token(payload_segment: str) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    return f"{header}.{payload_segment}.fakesignature"


def _future_exp() -> datetime.datetime:
    return datetime.datetime.now(tz=timezone.utc) + datetime.timedelta(seconds=1000)


def _past_exp() -> datetime.datetime:
    return datetime.datetime.now(tz=timezone.utc) - datetime.timedelta(seconds=1000)


# ---------------------------------------------------------------------------
# client_check_token_expiry
# ---------------------------------------------------------------------------


def test_valid_token_passes():
    client_check_token_expiry(_encode({"exp": _future_exp()}))


def test_expired_token_raises():
    with pytest.raises(TokenExpiredError):
        client_check_token_expiry(_encode({"exp": _past_exp()}))


def test_token_without_exp_claim_passes():
    # Matches the previous jwt.decode(verify_exp=True) behaviour: no exp
    # claim means nothing to check client-side.
    client_check_token_expiry(_encode({"sub": "user-123"}))


@pytest.mark.parametrize(
    "bad_token",
    [
        "invalid",  # no segments at all
        "a.b",  # too few segments
        "a.b.c.d",  # too many segments
        _craft_token("!!!not-base64!!!"),  # undecodable payload segment
        _craft_token(_b64url(b"not json")),  # valid base64, invalid JSON
        _craft_token(_b64url(b"[1, 2, 3]")),  # valid JSON, not an object
        _craft_token(_b64url(b'"just a string"')),  # valid JSON, not an object
    ],
)
def test_malformed_tokens_raise_invalid(bad_token):
    with pytest.raises(TokenInvalidError):
        client_check_token_expiry(bad_token)


@pytest.mark.parametrize("bad_exp", ["1893456000", True, [1893456000], {"v": 1}])
def test_non_numeric_exp_raises_invalid(bad_exp):
    payload = _b64url(json.dumps({"exp": bad_exp}).encode())
    with pytest.raises(TokenInvalidError):
        client_check_token_expiry(_craft_token(payload))


def test_expired_token_with_tampered_signature_still_rejected():
    # Fail-fast on expiry must not depend on signature validity: the client
    # never verifies signatures, so a forged-but-expired token is still
    # rejected before any request goes out.
    token = _encode({"exp": _past_exp()})
    header, payload, _sig = token.split(".")
    with pytest.raises(TokenExpiredError):
        client_check_token_expiry(f"{header}.{payload}.tampered")


def test_valid_token_with_tampered_signature_passes_client_check():
    # Documents the trust model: an unexpired token with an invalid
    # signature passes the *client-side* check. The Hub server validates
    # the signature on every request and rejects forgeries there.
    token = _encode({"exp": _future_exp()})
    header, payload, _sig = token.split(".")
    client_check_token_expiry(f"{header}.{payload}.tampered")


# ---------------------------------------------------------------------------
# get_jwt_token wrappers -- behaviour must be identical to before the fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [hub_client, hub_token_module],
    ids=["cli.server.hub_client", "hub_token.token"],
)
def test_get_jwt_token_valid(module):
    valid = _encode({"exp": _future_exp()})
    assert module.get_jwt_token(RC.from_dict({"token": valid})) == valid


@pytest.mark.parametrize(
    "module",
    [hub_client, hub_token_module],
    ids=["cli.server.hub_client", "hub_token.token"],
)
def test_get_jwt_token_expired(module):
    expired = _encode({"exp": _past_exp()})
    with pytest.raises(module.ExpiredTokenError) as exc_info:
        module.get_jwt_token(RC.from_dict({"token": expired}))
    assert str(exc_info.value) == module.TOKEN_EXPIRED_MESSAGE


@pytest.mark.parametrize(
    "module",
    [hub_client, hub_token_module],
    ids=["cli.server.hub_client", "hub_token.token"],
)
def test_get_jwt_token_malformed(module):
    with pytest.raises(module.InvalidTokenError) as exc_info:
        module.get_jwt_token(RC.from_dict({"token": "invalid"}))
    assert str(exc_info.value) == module.TOKEN_INVALID_MESSAGE


@pytest.mark.parametrize(
    "module",
    [hub_client, hub_token_module],
    ids=["cli.server.hub_client", "hub_token.token"],
)
def test_get_jwt_token_none(module):
    assert module.get_jwt_token(RC.from_dict({"token": None})) is None


def test_no_unverified_jwt_decode_remains():
    """The verify_signature=False call shape must be gone from both modules."""
    import inspect

    for module in (hub_client, hub_token_module):
        source = inspect.getsource(module)
        assert "verify_signature" not in source
        assert "jwt.decode" not in source
