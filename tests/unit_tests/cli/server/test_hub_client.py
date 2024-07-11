import datetime

import pytest
import jwt
from datetime import timezone


from guardrails.classes.credentials import Credentials
from guardrails.cli.server.hub_client import (
    TOKEN_EXPIRED_MESSAGE,
    TOKEN_INVALID_MESSAGE,
    InvalidTokenError,
    ExpiredTokenError,
    get_jwt_token,
)


# TODO
@pytest.mark.skip()
def test_fetch():
    assert 1 == 1


# TODO
@pytest.mark.skip()
def test_fetch_module_manifest():
    assert 1 == 1


# TODO
@pytest.mark.skip()
def test_fetch_module():
    assert 1 == 1


# TODO
@pytest.mark.skip()
def test_get_validator_manifest():
    assert 1 == 1


# TODO
@pytest.mark.skip()
def test_get_auth():
    assert 1 == 1


def test_get_jwt_token():
    # Create a JWT that expires in the future
    secret_key = "secret"
    timedelta = datetime.timedelta(seconds=1000)
    expiration = datetime.datetime.now(tz=timezone.utc) + timedelta
    valid_jwt = jwt.encode({"exp": expiration}, secret_key, algorithm="HS256")
    creds = Credentials.from_dict({"token": valid_jwt})

    # Test valid token
    assert get_jwt_token(creds) == valid_jwt

    # Test with an expired JWT
    with pytest.raises(ExpiredTokenError) as e:
        expired = datetime.datetime.now(tz=timezone.utc) - timedelta
        expired_jwt = jwt.encode({"exp": expired}, secret_key, algorithm="HS256")
        get_jwt_token(Credentials.from_dict({"token": expired_jwt}))

    assert str(e.value) == TOKEN_EXPIRED_MESSAGE

    # Test with an invalid token format
    with pytest.raises(InvalidTokenError) as e:
        invalid_jwt = "invalid"
        get_jwt_token(Credentials.from_dict({"token": invalid_jwt}))

    assert str(e.value) == TOKEN_INVALID_MESSAGE
