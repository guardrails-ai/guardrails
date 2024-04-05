import os
from typing import Optional

from guardrails_api_client import AuthenticatedClient
from guardrails_api_client.api.guard import update_guard, validate
from guardrails_api_client.models import Guard, ValidatePayload
from guardrails_api_client.types import UNSET
from httpx import Timeout


class GuardrailsApiClient:
    _client: AuthenticatedClient
    base_url: str
    api_key: str

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (
            base_url
            if base_url is not None
            else os.environ.get("GUARDRAILS_BASE_URL", "http://localhost:8000")
        )
        self.api_key = (
            api_key if api_key is not None else os.environ.get("GUARDRAILS_API_KEY", "")
        )
        self._client = AuthenticatedClient(
            base_url=self.base_url,  # type: ignore
            follow_redirects=True,  # type: ignore
            token=self.api_key,
            timeout=Timeout(300),  # type: ignore
        )

    def upsert_guard(self, guard: Guard):
        update_guard.sync(guard_name=guard.name, client=self._client, body=guard)

    def validate(
        self,
        guard: Guard,
        payload: ValidatePayload,
        openai_api_key: Optional[str] = None,
    ):
        _openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.environ.get("OPENAI_API_KEY", UNSET)
        )
        return validate.sync(
            guard_name=guard.name,
            client=self._client,
            body=payload,
            x_openai_api_key=_openai_api_key,
        )
