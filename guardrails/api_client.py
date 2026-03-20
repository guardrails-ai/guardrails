import json
import os
from typing import Any, AsyncIterator, Iterator, Optional

from httpx import AsyncClient, Client, HTTPStatusError
from guardrails_ai.types import (
    Guard,
    ValidationOutcome as IValidationOutcome,
)

from guardrails.errors import ValidationError

from guardrails.logger import logger


class GuardrailsApiClient:
    ahttp_client: AsyncClient
    http_client: Client
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
            api_key
            if api_key is not None
            else os.environ.get("GUARDRAILS_API_KEY", "x-guardrailsai-api-key")
        )
        self.timeout = 300

        headers = {"x-guardrailsai-api-key": self.api_key}

        self.ahttp_client = AsyncClient(
            base_url=self.base_url, headers=headers, timeout=self.timeout
        )
        self.http_client = Client(
            base_url=self.base_url, headers=headers, timeout=self.timeout
        )

    async def aupsert_guard(self, guard: Guard) -> Guard:
        existing_guard: Guard | None = None
        try:
            existing_guard = await self.afetch_guard(guard.name)
        except Exception:
            pass

        if existing_guard:
            response = await self.ahttp_client.put(
                f"/guards/{existing_guard.id}",
                json=guard.model_dump(),
            )
        else:
            response = await self.ahttp_client.post(
                "/guards",
                json=guard.model_dump(),
            )
        response.raise_for_status()
        res_body = response.json()
        return Guard.model_validate(res_body)

    def upsert_guard(self, guard: Guard) -> Guard:
        existing_guard: Guard | None = None
        try:
            existing_guard = self.fetch_guard(guard.name)
        except Exception:
            pass

        if existing_guard:
            response = self.http_client.put(
                f"/guards/{existing_guard.id}",
                json=guard.model_dump(),
            )
        else:
            response = self.http_client.post(
                "/guards",
                json=guard.model_dump(),
            )
        response.raise_for_status()
        res_body = response.json()
        return Guard.model_validate(res_body)

    async def afetch_guard(self, guard_name: str) -> Optional[Guard]:
        try:
            response = await self.ahttp_client.get(
                f"/guards?name={guard_name}",
            )
            response.raise_for_status()
            res_body = response.json()
            first = res_body[0] if res_body and len(res_body) > 0 else None
            if not first:
                raise ValueError(f"No guard found for name {guard_name}")
            return Guard.model_validate(first)
        except Exception as e:
            logger.error(f"Error fetching guard {guard_name}: {e}")
            return None

    def fetch_guard(self, guard_name: str) -> Optional[Guard]:
        try:
            response = self.http_client.get(
                f"/guards?name={guard_name}",
            )
            response.raise_for_status()
            res_body = response.json()
            first = res_body[0] if res_body and len(res_body) > 0 else None
            if not first:
                raise ValueError(f"No guard found for name {guard_name}")
            return Guard.model_validate(first)
        except Exception as e:
            logger.error(f"Error fetching guard {guard_name}: {e}")
            return None

    async def adelete_guard(self, guard_name: str) -> Optional[Guard]:
        guard = await self.afetch_guard(guard_name)
        if guard and guard.id:
            response = await self.ahttp_client.delete(f"/guards/{guard.id}")
            response.raise_for_status()
            res_body = response.json()
            return Guard.model_validate(res_body)

    def delete_guard(self, guard_name: str) -> Optional[Guard]:
        guard = self.fetch_guard(guard_name)
        if guard and guard.id:
            response = self.http_client.delete(f"/guards/{guard.id}")
            response.raise_for_status()
            res_body = response.json()
            return Guard.model_validate(res_body)

    async def avalidate(
        self,
        guard: Guard,
        openai_api_key: Optional[str] = None,
        *,
        llm_output: Optional[str] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        try:
            _openai_api_key = (
                openai_api_key
                if openai_api_key is not None
                else os.environ.get("OPENAI_API_KEY")
            )
            headers = {}
            if _openai_api_key:
                headers = {"x-openai-api-key": _openai_api_key}

            response = await self.ahttp_client.post(
                f"/guards/{guard.id}/validate",
                json={
                    "llm_output": llm_output,
                    "num_reasks": num_reasks,
                    "prompt_params": prompt_params,
                    **kwargs,
                },
                headers=headers,
            )
            response.raise_for_status()
            res_body = response.json()
            return IValidationOutcome.model_validate(res_body)
        except HTTPStatusError as e:
            if e.response.status_code == 400:
                raise ValidationError(str(e)) from e

    def validate(
        self,
        guard: Guard,
        openai_api_key: Optional[str] = None,
        *,
        llm_output: Optional[str] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        try:
            _openai_api_key = (
                openai_api_key
                if openai_api_key is not None
                else os.environ.get("OPENAI_API_KEY")
            )
            headers = {}
            if _openai_api_key:
                headers = {"x-openai-api-key": _openai_api_key}

            response = self.http_client.post(
                f"/guards/{guard.id}/validate",
                json={
                    "llm_output": llm_output,
                    "num_reasks": num_reasks,
                    "prompt_params": prompt_params,
                    **kwargs,
                },
                headers=headers,
            )
            response.raise_for_status()
            res_body = response.json()
            return IValidationOutcome.model_validate(res_body)
        except HTTPStatusError as e:
            if e.response.status_code == 400:
                raise ValidationError(str(e)) from e

    async def astream_validate(
        self,
        guard: Guard,
        openai_api_key: Optional[str] = None,
        *,
        llm_output: Optional[str] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[Any]:
        _openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.environ.get("OPENAI_API_KEY")
        )

        headers = {"Content-Type": "application/json"}
        if _openai_api_key:
            headers["x-openai-api-key"] = _openai_api_key

        async with self.ahttp_client.stream(
            "POST",
            f"/guards/{guard.id}/validate",
            json={
                "llm_output": llm_output,
                "num_reasks": num_reasks,
                "prompt_params": prompt_params,
                "stream": True,
                **kwargs,
            },
        ) as response:
            if not response.is_success:
                response.raise_for_status()
                return

            async for raw_chunk in response.aiter_text():
                str_chunk = raw_chunk.strip()
                if str_chunk:
                    str_chunk_data = "".join(str_chunk.split("\n")).split("data:")[1]
                    chunk = json.loads(str_chunk_data)
                    if chunk.get("error"):
                        raise Exception(chunk.get("error").get("message"))
                    yield IValidationOutcome.model_validate(chunk)

    def stream_validate(
        self,
        guard: Guard,
        openai_api_key: Optional[str] = None,
        *,
        llm_output: Optional[str] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Iterator[Any]:
        _openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.environ.get("OPENAI_API_KEY")
        )

        headers = {"Content-Type": "application/json"}
        if _openai_api_key:
            headers["x-openai-api-key"] = _openai_api_key

        with self.http_client.stream(
            "POST",
            f"/guards/{guard.id}/validate",
            json={
                "llm_output": llm_output,
                "num_reasks": num_reasks,
                "prompt_params": prompt_params,
                "stream": True,
                **kwargs,
            },
        ) as response:
            if not response.is_success:
                response.raise_for_status()
                return

            for raw_chunk in response.iter_text():
                str_chunk = raw_chunk.strip()
                if str_chunk:
                    str_chunk_data = "".join(str_chunk.split("\n")).split("data:")[1]
                    chunk = json.loads(str_chunk_data)
                    if chunk.get("error"):
                        raise Exception(chunk.get("error").get("message"))
                    yield IValidationOutcome.model_validate(chunk)

    def get_history(self, guard_id: str, call_id: str) -> Any:
        response = self.http_client.get(f"/guards/{guard_id}/history/{call_id}")
        response.raise_for_status()
        res_body = response.json()
        return res_body
