# import http.client
# import json

# from guardrails.classes.credentials import Credentials


# unused - for now
# def get_auth_token(creds: Credentials) -> str:
#     if creds.client_id and creds.client_secret:
#         audience = "https://validator-hub-service.guardrailsai.com"
#         conn = http.client.HTTPSConnection("guardrailsai.us.auth0.com")
#         payload = json.dumps(
#             {
#                 "client_id": creds.client_id,
#                 "client_secret": creds.client_secret,
#                 "audience": audience,
#                 "grant_type": "client_credentials",
#             }
#         )
#         headers = {"content-type": "application/json"}
#         conn.request("POST", "/oauth/token", payload, headers)

#         res = conn.getresponse()
#         data = json.loads(res.read().decode("utf-8"))
#         return data.get("access_token", "")
#     return ""
