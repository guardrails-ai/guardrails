import http
import json
import sys

from guardrails.cli.hub.credentials import Credentials
from guardrails.cli.logger import logger


def authenticate(creds: Credentials):
    audience = "https://api.validator-hub.guardrailsai.com"
    conn = http.client.HTTPSConnection("guardrailsai.us.auth0.com")  # type: ignore
    payload = json.dumps(
        {
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "audience": audience,
            "grant_type": "client_credentials",
        }
    )
    headers = {"content-type": "application/json"}
    conn.request("POST", "/oauth/token", payload, headers)

    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    if not data.get("access_token"):
        logger.error("Unauthorized!")
        sys.exit(1)


def authorize(validator_module_name: str, creds: Credentials):
    logger.error("Not yet supported!")
