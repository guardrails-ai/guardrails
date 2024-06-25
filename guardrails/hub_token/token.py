from guardrails.classes.credentials import Credentials
from jwt import JWT
from jwt.exceptions import JWTDecodeError
from typing import Optional

FIND_NEW_TOKEN = "You can find a new token at https://hub.guardrailsai.com/tokens"

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


VALIDATOR_HUB_SERVICE = "https://so4sg4q4pb.execute-api.us-east-1.amazonaws.com"


def get_jwt_token(creds: Credentials) -> Optional[str]:
    token = creds.token

    # check for jwt expiration
    if token:
        try:
            JWT().decode(token, do_verify=False)
        except JWTDecodeError as e:
            # if the error message includes "Expired", then the token is expired
            if "Expired" in str(e):
                raise ExpiredTokenError(TOKEN_EXPIRED_MESSAGE)
            else:
                raise InvalidTokenError(TOKEN_INVALID_MESSAGE)
    return token
