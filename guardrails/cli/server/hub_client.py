import sys
from string import Template
from typing import Any, Dict, Optional

import requests

from guardrails.cli.logger import logger
from guardrails.cli.server.auth import authenticate
from guardrails.cli.server.credentials import Credentials
from guardrails.cli.server.module_manifest import ModuleManifest

validator_hub_service = "https://so4sg4q4pb.execute-api.us-east-1.amazonaws.com"
validator_manifest_endpoint = Template(
    "validator-manifests/${namespace}/${validator_name}"
)


class AuthenticationError(Exception):
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
    module_name: str, token: str, anonymousUserId: Optional[str] = None
) -> Dict[str, Any]:
    namespace, validator_name = module_name.split("/", 1)
    manifest_path = validator_manifest_endpoint.safe_substitute(
        namespace=namespace, validator_name=validator_name
    )
    manifest_url = f"{validator_hub_service}/{manifest_path}"
    return fetch(manifest_url, token, anonymousUserId)


def fetch_module(module_name: str) -> ModuleManifest:
    creds = Credentials.from_rc_file()
    token = authenticate(creds)

    try:
        module_manifest_json = fetch_module_manifest(module_name, token, creds.id)
        module_manifest = ModuleManifest.from_dict(module_manifest_json)
        if not module_manifest:
            logger.error(f"Failed to install hub://{module_name}")
            sys.exit(1)
        return module_manifest
    except HttpError:
        logger.error(f"Failed to install hub://{module_name}")
        sys.exit(1)
    except Exception as e:
        logger.error("An unexpected error occurred!", e)
        sys.exit(1)


def check_auth():
    try:
        creds = Credentials.from_rc_file()
        token = authenticate(creds)
        auth_url = f"{validator_hub_service}/auth"
        response = fetch(auth_url, token, creds.id)
        print("check_auth fetch response: ", response)
        if not response:
            raise AuthenticationError("Failed to authenticate!")
    except HttpError as http_error:
        logger.error(http_error)
        raise AuthenticationError("Failed to authenticate!")
    except Exception as e:
        logger.error("An unexpected error occurred!", e)
        raise AuthenticationError("Failed to authenticate!")
