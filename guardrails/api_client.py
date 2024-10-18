import json
import os
from typing import Any, Iterator, Optional

import requests
from guardrails_api_client.configuration import Configuration
from guardrails_api_client.api_client import ApiClient
from guardrails_api_client.api.guard_api import GuardApi
from guardrails_api_client.api.validate_api import ValidateApi
from guardrails_api_client.models import (
    Guard,
    ValidatePayload,
    ValidationOutcome as IValidationOutcome,
)

from guardrails_api_client.exceptions import BadRequestException
from guardrails.errors import ValidationError

from guardrails.logger import logger


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

    def fetch_guard(self, guard_name: str) -> Optional[Guard]:
        try:
            return self._guard_api.get_guard(guard_name=guard_name)
        except Exception as e:
            logger.error(f"Error fetching guard {guard_name}: {e}")
            return None

    def validate(
        self,
        guard: Guard,
        payload: ValidatePayload,
        openai_api_key: Optional[str] = None,
    ):
        try:
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
        except BadRequestException as e:
            raise ValidationError(f"{e.body}")

    def stream_validate(
        self,
        guard: Guard,
        payload: ValidatePayload,
        openai_api_key: Optional[str] = None,
    ) -> Iterator[Any]:
        _openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.environ.get("OPENAI_API_KEY")
        )

        url = f"{self.base_url}/guards/{guard.name}/validate"
        headers = {
            "Content-Type": "application/json",
            "x-openai-api-key": _openai_api_key,
        }

        s = requests.Session()

        with s.post(url, json=payload.to_dict(), headers=headers, stream=True) as resp:
            for line in resp.iter_lines():
                if not resp.ok:
                    raise ValueError(
                        f"status_code: {resp.status_code}"
                        " reason: {resp.reason} text: {resp.text}"
                    )
                if line:
                    json_output = json.loads(line)
                    yield IValidationOutcome.from_dict(json_output)

    def get_history(self, guard_name: str, call_id: str):
        return self._guard_api.get_guard_history(guard_name, call_id)
