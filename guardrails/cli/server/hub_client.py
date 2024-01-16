import json
import sys
from typing import Any, Dict, List
from urllib.request import Request, urlopen

from guardrails.cli.hub.credentials import Credentials
from guardrails.cli.logger import logger
from guardrails.cli.server.auth import authenticate
from guardrails.cli.server.module_manifest import ModuleManifest

hub_url = "https://raw.githubusercontent.com/guardrails-ai/guardrails-hub"
branch = "install-script"


# TODO: Error handling
def fetch_content(url: str):
    req = Request(url)
    # For Debugging
    # req.add_header("Cache-Control", "no-cache")
    conn = urlopen(req)
    contents = conn.read()
    conn.close()
    return json.loads(contents)


def fetch_module_manifest(module_summary: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = module_summary.get("manifest")
    manifest_url = f"{hub_url}/{branch}/{manifest_path}"
    return fetch_content(manifest_url)


def fetch_hub_index() -> List[Dict[str, Any]]:
    index_path = "index.json"
    index_url = f"{hub_url}/{branch}/{index_path}"
    return fetch_content(index_url)


def fetch_module(module_name: str) -> ModuleManifest:
    creds = Credentials.from_rc_file()
    authenticate(creds)

    # Discovery
    hub_index = fetch_hub_index()
    try:
        module_summary = next(x for x in hub_index if x.get("id") == module_name)
    except StopIteration:
        logger.error("Not Found!")
        logger.error(f"{module_name} does not exist in the hub!")
        sys.exit(1)

    module_manifest_json = fetch_module_manifest(module_summary)
    module_manifest = ModuleManifest.from_dict(module_manifest_json)
    return module_manifest
