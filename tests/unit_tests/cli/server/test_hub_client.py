import datetime
import math

import pytest
from jwt import JWT, jwk_from_dict

from guardrails.classes.credentials import Credentials
from guardrails.cli.server.hub_client import (
    TOKEN_EXPIRED_MESSAGE,
    TOKEN_INVALID_MESSAGE,
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
    expiration = math.floor(datetime.datetime.now().timestamp() + 1000)

    jwk = jwk_from_dict(
        {
            "alg": "HS256",
            "kty": "oct",
            "kid": "050bf691-4348-4891-940f-99af8354e82b",
            "k": "eCE35cBrbRsO1GhrbxLXnGrVATgUFZDrPyyuOar4crw",
        }
    )

    valid_jwt = JWT().encode(
        {
            "exp": expiration,
        },
        jwk,
        "HS256",
    )
    creds = {"token": valid_jwt}
    assert get_jwt_token(Credentials.from_dict(creds)) == valid_jwt

    with pytest.raises(Exception) as e:
        expiration = math.floor(datetime.datetime.now().timestamp() - 1000)
        expired_jwt = JWT().encode(
            {
                "exp": expiration,
            },
            jwk,
            "HS256",
        )
        get_jwt_token(Credentials.from_dict({"token": expired_jwt}))
    assert str(e.value) == TOKEN_EXPIRED_MESSAGE

    with pytest.raises(Exception) as e:
        get_jwt_token(Credentials.from_dict({"token": "invalid_token"}))
    assert str(e.value) == TOKEN_INVALID_MESSAGE
