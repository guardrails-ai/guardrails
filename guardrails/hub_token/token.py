import os
from typing import Optional

from guardrails.classes.rc import RC
from guardrails.hub_token.utils import (
    TokenExpiredError as HubTokenExpiredError,
    TokenInvalidError as HubTokenInvalidError,
    client_check_token_expiry,
)

FIND_NEW_TOKEN = "You can find a new token at https://guardrailsai.com/hub/keys"

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


def get_jwt_token(rc: RC) -> Optional[str]:
    token = rc.token

    # check for jwt expiration
    if token:
        try:
            client_check_token_expiry(token)
        except HubTokenExpiredError:
            raise ExpiredTokenError(TOKEN_EXPIRED_MESSAGE)
        except HubTokenInvalidError:
            raise InvalidTokenError(TOKEN_INVALID_MESSAGE)
    return token
