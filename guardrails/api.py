import os
from guard_rails_api_client import AuthenticatedClient
from guard_rails_api_client.models import Guard
from guard_rails_api_client.api.guard import update_guard

class GuardrailsApiClient:
    _client: AuthenticatedClient = None
    base_url: str = None
    api_key: str = None

    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url if base_url is not None else os.environ.get("GUARDRAILS_BASE_URL", "http://localhost:8000")
        self.api_key = api_key if api_key is not None else os.environ.get("GUARDRAILS_API_KEY")
        self._client =  AuthenticatedClient(base_url=self.base_url, follow_redirects=True, token=self.api_key)

    def upsert_guard (self, guard: Guard):
        update_guard.sync(guard_name=guard.name, client=self._client, json_body=guard)

    
