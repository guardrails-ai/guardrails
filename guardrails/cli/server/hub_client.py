import sys
import os
from string import Template
from typing import Any, Dict, Optional

import requests
from guardrails_hub_types import Manifest
import jwt
from jwt import ExpiredSignatureError, DecodeError


from guardrails.classes.credentials import Credentials
from guardrails.cli.logger import logger
from guardrails.version import GUARDRAILS_VERSION

FIND_NEW_TOKEN = "You can find a new token at https://hub.guardrailsai.com/keys"

TOKEN_EXPIRED_MESSAGE = f"""Your token has expired. Please run `guardrails configure`\
to update your token.
{FIND_NEW_TOKEN}"""
TOKEN_INVALID_MESSAGE = f"""Your token is invalid. Please run `guardrails configure`\
to update your token.
{FIND_NEW_TOKEN}"""

VALIDATOR_HUB_SERVICE = os.getenv(
    "GR_VALIDATOR_HUB_SERVICE", "https://hub.api.guardrailsai.com"
)
validator_manifest_endpoint = Template(
    "validator/${namespace}/${validator_name}/manifest"
)


class AuthenticationError(Exception):
    pass


class ExpiredTokenError(Exception):
    pass


class InvalidTokenError(Exception):
    pass


class HttpError(Exception):
    status: int
    message: str


def fetch(url: str, token: Optional[str], anonymousUserId: Optional[str]):
    try:
        # For Debugging
        # headers = { "Authorization": f"Bearer {token}", "x-anonymous-user-id": anonymousUserId, "Cache-Control": "no-cache" }  # noqa
        headers = {
            "Authorization": f"Bearer {token}",
            "x-anonymous-user-id": anonymousUserId,
            "x-guardrails-version": GUARDRAILS_VERSION,
        }
        req = requests.get(url, headers=headers)
        body = req.json()

        if not req.ok:
            logger.error(req.status_code)
            logger.error(body.get("message"))
            http_error = HttpError()
            http_error.status = req.status_code
            http_error.message = body.get("message")
            raise http_error

        return body
    except HttpError as http_e:
        raise http_e
    except Exception as e:
        logger.error("An unexpected error occurred!", e)
        sys.exit(1)


def fetch_module_manifest(
    module_name: str, token: Optional[str], anonymousUserId: Optional[str] = None
) -> Dict[str, Any]:
    namespace, validator_name = module_name.split("/", 1)
    manifest_path = validator_manifest_endpoint.safe_substitute(
        namespace=namespace, validator_name=validator_name
    )
    manifest_url = f"{VALIDATOR_HUB_SERVICE}/{manifest_path}"
    return fetch(manifest_url, token, anonymousUserId)


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


def fetch_module(module_name: str) -> Optional[Manifest]:
    creds = Credentials.from_rc_file(logger)
    token = get_jwt_token(creds)

    module_manifest_json = fetch_module_manifest(module_name, token, creds.id)
    return Manifest.from_dict(module_manifest_json)


def fetch_template(template_address: str) -> Dict[str, Any]:
    creds = Credentials.from_rc_file(logger)
    token = get_jwt_token(creds)

    namespace, template_name = template_address.replace("hub:template://", "").split(
        "/", 1
    )
    template_path = f"guard-templates/{namespace}/{template_name}"
    template_url = f"{VALIDATOR_HUB_SERVICE}/{template_path}"
    return fetch(template_url, token, creds.id)


# GET /guard-templates/{namespace}/{guardTemplateName}
def get_guard_template(template_address: str):
    try:
        template = fetch_template(template_address)
        if not template:
            logger.error(f"Failed to install template {template_address}")
            sys.exit(1)
        return template
    except HttpError:
        logger.error(f"Failed to install template {template_address}")
        sys.exit(1)
    except (ExpiredTokenError, InvalidTokenError) as e:
        logger.error(AuthenticationError(e))
        sys.exit(1)
    except Exception as e:
        logger.error("An unexpected error occurred!", e)
        sys.exit(1)


# GET /validator/{namespace}/{validatorName}/manifest
def get_validator_manifest(module_name: str):
    try:
        module_manifest = fetch_module(module_name)
        if not module_manifest:
            logger.error(f"Failed to install hub://{module_name}")
            sys.exit(1)
        return module_manifest
    except HttpError as e:
        if e.message == "Unauthorized":
            logger.error(TOKEN_INVALID_MESSAGE)
            raise
        logger.error(f"Failed to install hub://{module_name}")
        sys.exit(1)
    except (ExpiredTokenError, InvalidTokenError) as e:
        logger.error(AuthenticationError(e))
        sys.exit(1)
    except Exception as e:
        logger.error("An unexpected error occurred!", e)
        sys.exit(1)


# GET /auth
def get_auth():
    try:
        creds = Credentials.from_rc_file(logger)
        token = get_jwt_token(creds)
        auth_url = f"{VALIDATOR_HUB_SERVICE}/auth"
        response = fetch(auth_url, token, creds.id)
        if not response:
            raise AuthenticationError("Failed to authenticate!")
    except HttpError as http_error:
        logger.error(http_error)
        raise AuthenticationError("Failed to authenticate!")
    except (ExpiredTokenError, InvalidTokenError) as e:
        raise AuthenticationError(e)
    except Exception as e:
        logger.error("An unexpected error occurred!", e)
        raise AuthenticationError("Failed to authenticate!")


def post_validator_submit(package_name: str, content: str):
    try:
        creds = Credentials.from_rc_file(logger)
        token = get_jwt_token(creds)
        submission_url = f"{VALIDATOR_HUB_SERVICE}/validator/submit"

        headers = {
            "Authorization": f"Bearer {token}",
        }
        request_body = {"packageName": package_name, "content": content}
        req = requests.post(submission_url, data=request_body, headers=headers)

        body = req.json()
        if not req.ok:
            logger.error(req.status_code)
            logger.error(body.get("message"))
            http_error = HttpError()
            http_error.status = req.status_code
            http_error.message = body.get("message")
            raise http_error

        return body
    except HttpError as http_e:
        raise http_e
    except Exception as e:
        logger.error("An unexpected error occurred!", e)
        sys.exit(1)
