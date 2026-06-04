import time
import base64
import json
import pytest

from guardrails.hub_token.utils import client_check_token_expiry
from guardrails.hub_token.token import ExpiredTokenError, InvalidTokenError


def _make_token(exp_offset: int) -> str:
    """Helper to create a fake JWT-shaped token."""
    payload = {"exp": int(time.time()) + exp_offset}
    payload_b64 = (
        base64.urlsafe_b64encode(json.dumps(payload).encode())
        .rstrip(b"=")
        .decode()
    )
    return f"header.{payload_b64}.signature"


def test_valid_token_passes():
    """A token expiring in the future should pass without raising."""
    token = _make_token(exp_offset=3600)  # 1 hour from now
    client_check_token_expiry(token)  # should not raise


def test_expired_token_raises():
    """A token expired in the past should raise ExpiredTokenError."""
    token = _make_token(exp_offset=-3600)  # 1 hour ago
    with pytest.raises(ExpiredTokenError):
        client_check_token_expiry(token)


def test_malformed_token_raises():
    """A token that is not valid JWT shape should raise InvalidTokenError."""
    with pytest.raises(InvalidTokenError):
        client_check_token_expiry("not.a.valid.jwt.token.at.all")


def test_token_without_exp_passes():
    """A token with no exp claim should pass (no expiry enforced)."""
    payload = {"sub": "user123"}  # no exp field
    payload_b64 = (
        base64.urlsafe_b64encode(json.dumps(payload).encode())
        .rstrip(b"=")
        .decode()
    )
    token = f"header.{payload_b64}.signature"
    client_check_token_expiry(token)  # should not raise