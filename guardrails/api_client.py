import os
from typing import Optional

from guardrails_api_client.configuration import Configuration
from guardrails_api_client.api_client import ApiClient
from guardrails_api_client.api.guard_api import GuardApi
from guardrails_api_client.api.validate_api import ValidateApi
from guardrails_api_client.models import Guard, ValidatePayload


class GuardrailsApiClient:
    _api_client: ApiClient
    _guard_api: GuardApi
    _validate_api: ValidateApi
    timeout: float
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
        self.timeout = 300
        self._api_client = ApiClient(
            configuration=Configuration(api_key=self.api_key, host=self.base_url)
        )
        self._guard_api = GuardApi(self._api_client)
        self._validate_api = ValidateApi(self._api_client)

    def upsert_guard(self, guard: Guard):
        self._guard_api.update_guard(
            guard_name=guard.name, body=guard, _request_timeout=self.timeout
        )

    def validate(
        self,
        guard: Guard,
        payload: ValidatePayload,
        openai_api_key: Optional[str] = None,
    ):
        _openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.environ.get("OPENAI_API_KEY")
        )
        return self._validate_api.validate(
            guard_name=guard.name,
            validate_payload=payload,
            x_openai_api_key=_openai_api_key,
        )
