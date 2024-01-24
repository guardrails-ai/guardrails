import sys
from string import Template
from typing import Any, Dict, Optional

import requests

from guardrails.cli.hub.credentials import Credentials
from guardrails.cli.logger import logger
from guardrails.cli.server.auth import authenticate
from guardrails.cli.server.module_manifest import ModuleManifest

validator_hub_service = "https://so4sg4q4pb.execute-api.us-east-1.amazonaws.com"
validator_manifest_endpoint = Template(
    "validator-manifests/{namespace}/{validator_name}"
)


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
            sys.exit(1)

        return body
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

    module_manifest_json = fetch_module_manifest(module_name, token, creds.id)
    module_manifest = ModuleManifest.from_dict(module_manifest_json)
    return module_manifest
