import os
from guardrails.classes.credentials import Credentials
import jwt
from jwt import ExpiredSignatureError, DecodeError
from typing import Optional

FIND_NEW_TOKEN = "You can find a new token at https://hub.guardrailsai.com/keys"

TOKEN_EXPIRED_MESSAGE = f"""Your token has expired. Please run `guardrails configure`\
to update your token.
{FIND_NEW_TOKEN}"""
TOKEN_INVALID_MESSAGE = f"""Your token is invalid. Please run `guardrails configure`\
to update your token.
{FIND_NEW_TOKEN}"""


class AuthenticationError(Exception):
    pass


class ExpiredTokenError(Exception):
    pass


class InvalidTokenError(Exception):
    pass


class HttpError(Exception):
    status: int
    message: str


VALIDATOR_HUB_SERVICE = os.getenv(
    "GR_VALIDATOR_HUB_SERVICE", "https://hub.api.guardrailsai.com"
)


def get_jwt_token(creds: Credentials) -> Optional[str]:
    token = creds.token

    # check for jwt expiration
    if token:
        try:
            jwt.decode(token, options={"verify_signature": False, "verify_exp": True})
        except ExpiredSignatureError:
            raise ExpiredTokenError(TOKEN_EXPIRED_MESSAGE)
        except DecodeError:
            raise InvalidTokenError(TOKEN_INVALID_MESSAGE)
    return token
