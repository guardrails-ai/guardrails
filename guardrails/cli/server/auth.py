import http.client
import json

from guardrails.cli.hub.credentials import Credentials


def authenticate(creds: Credentials) -> str:
    audience = "https://validator-hub-service.guardrailsai.com"
    conn = http.client.HTTPSConnection("guardrailsai.us.auth0.com")
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
    return data.get("access_token", "")
