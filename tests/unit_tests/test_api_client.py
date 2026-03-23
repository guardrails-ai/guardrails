import asyncio
import json
import os
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from httpx import HTTPStatusError, Request, Response

from guardrails.api_client import GuardrailsApiClient
from guardrails.errors import ValidationError
from guardrails_ai.types import Guard, ValidationOutcome as IValidationOutcome


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_http_error(status_code: int) -> HTTPStatusError:
    request = Request("POST", "http://test.com")
    response = Response(status_code=status_code, request=request)
    return HTTPStatusError(
        message=f"HTTP {status_code}", request=request, response=response
    )


def mock_sync_response(json_data=None, raise_error=None):
    r = Mock()
    r.json.return_value = json_data if json_data is not None else {}
    r.status_code = 200
    r.is_success = True
    if raise_error:
        r.raise_for_status.side_effect = raise_error
    return r


def mock_async_response(json_data=None, raise_error=None):
    # Use Mock (not AsyncMock) — httpx response methods are synchronous
    r = Mock()
    r.json.return_value = json_data if json_data is not None else {}
    r.status_code = 200
    r.is_success = True
    if raise_error:
        r.raise_for_status.side_effect = raise_error
    return r


@dataclass
class MockedClient:
    client: GuardrailsApiClient
    http: MagicMock
    ahttp: AsyncMock


def make_client(**kwargs) -> MockedClient:
    """Create a GuardrailsApiClient with mocked HTTP clients."""
    with (
        patch("guardrails.api_client.Client"),
        patch("guardrails.api_client.AsyncClient"),
    ):
        client = GuardrailsApiClient(**kwargs)
    http = MagicMock()
    ahttp = AsyncMock()
    client.http_client = http  # type: ignore[assignment]
    client.ahttp_client = ahttp  # type: ignore[assignment]
    return MockedClient(client=client, http=http, ahttp=ahttp)


def make_guard(guard_id="guard-id-123", name="test-guard") -> Mock:
    guard = Mock(spec=Guard)
    guard.id = guard_id
    guard.name = name
    guard.model_dump.return_value = {"id": guard_id, "name": name}
    return guard


def make_sse_chunk(data: dict) -> str:
    return f"{json.dumps(data)}\n"


def make_stream_ctx(chunks, is_success=True, raise_on_fail=None):
    """Sync context manager mock for http_client.stream()."""
    mock_resp = MagicMock()
    mock_resp.is_success = is_success
    mock_resp.iter_text.return_value = iter(chunks)
    if raise_on_fail:
        mock_resp.raise_for_status.side_effect = raise_on_fail
    ctx = MagicMock()
    ctx.__enter__ = Mock(return_value=mock_resp)
    ctx.__exit__ = Mock(return_value=False)
    return ctx


def make_async_stream_ctx(chunks, is_success=True, raise_on_fail=None):
    """Async context manager mock for ahttp_client.stream()."""

    async def aiter():
        for chunk in chunks:
            yield chunk

    mock_resp = MagicMock()
    mock_resp.is_success = is_success
    mock_resp.aiter_text.return_value = aiter()
    if raise_on_fail:
        mock_resp.raise_for_status.side_effect = raise_on_fail

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ─── Init ─────────────────────────────────────────────────────────────────────


class TestGuardrailsApiClientInit:
    def test_init_with_env_vars(self):
        with (
            patch("guardrails.api_client.Client"),
            patch("guardrails.api_client.AsyncClient"),
        ):
            with patch.dict(
                os.environ,
                {
                    "GUARDRAILS_BASE_URL": "http://env.com",
                    "GUARDRAILS_API_KEY": "env-key",
                },
            ):
                client = GuardrailsApiClient()

        assert client.base_url == "http://env.com"
        assert client.api_key == "env-key"
        assert client.timeout == 300

    def test_init_with_explicit_params(self):
        with (
            patch("guardrails.api_client.Client"),
            patch("guardrails.api_client.AsyncClient"),
        ):
            client = GuardrailsApiClient(
                base_url="http://custom.com", api_key="custom-key"
            )

        assert client.base_url == "http://custom.com"
        assert client.api_key == "custom-key"
        assert client.timeout == 300

    def test_init_default_values(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("GUARDRAILS_BASE_URL", "GUARDRAILS_API_KEY")
        }
        with (
            patch("guardrails.api_client.Client"),
            patch("guardrails.api_client.AsyncClient"),
        ):
            with patch.dict(os.environ, env, clear=True):
                client = GuardrailsApiClient()

        assert client.base_url == "http://localhost:8000"
        assert client.api_key == "x-guardrailsai-api-key"

    def test_init_explicit_params_override_env(self):
        with (
            patch("guardrails.api_client.Client"),
            patch("guardrails.api_client.AsyncClient"),
        ):
            with patch.dict(
                os.environ,
                {
                    "GUARDRAILS_BASE_URL": "http://env.com",
                    "GUARDRAILS_API_KEY": "env-key",
                },
            ):
                client = GuardrailsApiClient(
                    base_url="http://custom.com", api_key="custom-key"
                )

        assert client.base_url == "http://custom.com"
        assert client.api_key == "custom-key"

    def test_init_partial_params_uses_env_key(self):
        env = {k: v for k, v in os.environ.items() if k != "GUARDRAILS_API_KEY"}
        with (
            patch("guardrails.api_client.Client"),
            patch("guardrails.api_client.AsyncClient"),
        ):
            with patch.dict(
                os.environ,
                {**env, "GUARDRAILS_API_KEY": "env-key"},
                clear=True,
            ):
                client = GuardrailsApiClient(base_url="http://custom.com")

        assert client.base_url == "http://custom.com"
        assert client.api_key == "env-key"

    def test_init_creates_http_clients(self):
        client = GuardrailsApiClient()
        assert client.http_client is not None
        assert client.ahttp_client is not None


# ─── upsert_guard ─────────────────────────────────────────────────────────────


class TestUpsertGuard:
    def test_upsert_guard_updates_existing_guard(self):
        mc = make_client()
        guard = make_guard()
        guard_data = guard.model_dump.return_value
        existing_guard = make_guard(guard_id="existing-id-456")
        returned_guard = make_guard()

        mc.client.fetch_guard = Mock(return_value=existing_guard)
        mc.http.put.return_value = mock_sync_response(guard_data)

        with patch.object(Guard, "model_validate", return_value=returned_guard):
            result = mc.client.upsert_guard(guard)

        mc.client.fetch_guard.assert_called_once_with(guard.name)
        mc.http.put.assert_called_once_with("/guards/existing-id-456", json=guard_data)
        mc.http.put.return_value.raise_for_status.assert_called_once()
        mc.http.post.assert_not_called()
        assert result == returned_guard

    def test_upsert_guard_creates_new_guard_when_not_found(self):
        mc = make_client()
        guard = make_guard()
        guard_data = guard.model_dump.return_value
        returned_guard = make_guard()

        mc.client.fetch_guard = Mock(return_value=None)
        mc.http.post.return_value = mock_sync_response(guard_data)

        with patch.object(Guard, "model_validate", return_value=returned_guard):
            result = mc.client.upsert_guard(guard)

        mc.client.fetch_guard.assert_called_once_with(guard.name)
        mc.http.post.assert_called_once_with("/guards", json=guard_data)
        mc.http.put.assert_not_called()
        assert result == returned_guard

    def test_upsert_guard_raises_on_http_error(self):
        mc = make_client()
        guard = make_guard()
        guard.model_dump.return_value = {}
        existing_guard = make_guard(guard_id="existing-id-456")

        mc.client.fetch_guard = Mock(return_value=existing_guard)
        mc.http.put.return_value = mock_sync_response(raise_error=make_http_error(500))

        with pytest.raises(HTTPStatusError):
            mc.client.upsert_guard(guard)

    def test_aupsert_guard_updates_existing_guard(self):
        mc = make_client()
        guard = make_guard()
        guard_data = guard.model_dump.return_value
        existing_guard = make_guard(guard_id="existing-id-456")
        returned_guard = make_guard()

        mc.client.afetch_guard = AsyncMock(return_value=existing_guard)
        mc.ahttp.put = AsyncMock(return_value=mock_async_response(guard_data))

        with patch.object(Guard, "model_validate", return_value=returned_guard):
            result = asyncio.run(mc.client.aupsert_guard(guard))

        mc.client.afetch_guard.assert_called_once_with(guard.name)
        mc.ahttp.put.assert_called_once_with("/guards/existing-id-456", json=guard_data)
        mc.ahttp.post.assert_not_called()
        assert result == returned_guard

    def test_aupsert_guard_creates_new_guard_when_not_found(self):
        mc = make_client()
        guard = make_guard()
        guard_data = guard.model_dump.return_value
        returned_guard = make_guard()

        mc.client.afetch_guard = AsyncMock(return_value=None)
        mc.ahttp.post = AsyncMock(return_value=mock_async_response(guard_data))

        with patch.object(Guard, "model_validate", return_value=returned_guard):
            result = asyncio.run(mc.client.aupsert_guard(guard))

        mc.client.afetch_guard.assert_called_once_with(guard.name)
        mc.ahttp.post.assert_called_once_with("/guards", json=guard_data)
        mc.ahttp.put.assert_not_called()
        assert result == returned_guard

    def test_aupsert_guard_raises_on_http_error(self):
        mc = make_client()
        guard = make_guard()
        guard.model_dump.return_value = {}
        existing_guard = make_guard(guard_id="existing-id-456")

        mc.client.afetch_guard = AsyncMock(return_value=existing_guard)
        mc.ahttp.put = AsyncMock(
            return_value=mock_async_response(raise_error=make_http_error(500))
        )

        with pytest.raises(HTTPStatusError):
            asyncio.run(mc.client.aupsert_guard(guard))


# ─── fetch_guard ──────────────────────────────────────────────────────────────


class TestFetchGuard:
    def test_fetch_guard_success(self):
        mc = make_client()
        guard_data = {"id": "g1", "name": "my-guard"}
        expected = make_guard()

        mc.http.get.return_value = mock_sync_response([guard_data])

        with patch.object(Guard, "model_validate", return_value=expected):
            result = mc.client.fetch_guard("my-guard")

        mc.http.get.assert_called_once_with("/guards?name=my-guard")
        assert result == expected

    def test_fetch_guard_returns_none_on_empty_list(self):
        mc = make_client()
        mc.http.get.return_value = mock_sync_response([])

        result = mc.client.fetch_guard("missing-guard")

        assert result is None

    def test_fetch_guard_returns_none_on_exception(self):
        mc = make_client()
        mc.http.get.side_effect = Exception("connection error")

        result = mc.client.fetch_guard("my-guard")

        assert result is None

    def test_fetch_guard_logs_error_on_exception(self):
        mc = make_client()
        mc.http.get.side_effect = Exception("API Error")

        with patch("guardrails.api_client.logger") as mock_logger:
            result = mc.client.fetch_guard("my-guard")

        assert result is None
        mock_logger.error.assert_called_once()
        assert "Error fetching guard my-guard" in mock_logger.error.call_args[0][0]
        assert "API Error" in mock_logger.error.call_args[0][0]

    def test_afetch_guard_success(self):
        mc = make_client()
        guard_data = {"id": "g1", "name": "my-guard"}
        expected = make_guard()

        mc.ahttp.get = AsyncMock(return_value=mock_async_response([guard_data]))

        with patch.object(Guard, "model_validate", return_value=expected):
            result = asyncio.run(mc.client.afetch_guard("my-guard"))

        mc.ahttp.get.assert_called_once_with("/guards?name=my-guard")
        assert result == expected

    def test_afetch_guard_returns_none_on_empty_list(self):
        mc = make_client()
        mc.ahttp.get = AsyncMock(return_value=mock_async_response([]))

        result = asyncio.run(mc.client.afetch_guard("missing-guard"))

        assert result is None

    def test_afetch_guard_returns_none_on_exception(self):
        mc = make_client()
        mc.ahttp.get = AsyncMock(side_effect=Exception("connection error"))

        result = asyncio.run(mc.client.afetch_guard("my-guard"))

        assert result is None

    def test_afetch_guard_logs_error_on_exception(self):
        mc = make_client()
        mc.ahttp.get = AsyncMock(side_effect=Exception("API Error"))

        with patch("guardrails.api_client.logger") as mock_logger:
            result = asyncio.run(mc.client.afetch_guard("my-guard"))

        assert result is None
        mock_logger.error.assert_called_once()
        assert "Error fetching guard my-guard" in mock_logger.error.call_args[0][0]


# ─── delete_guard ─────────────────────────────────────────────────────────────


class TestDeleteGuard:
    def test_delete_guard_success(self):
        mc = make_client()
        guard_data = {"id": "g1", "name": "my-guard"}
        fetched_guard = make_guard(guard_id="g1")
        deleted_guard = make_guard(guard_id="g1")

        mc.http.get.return_value = mock_sync_response([guard_data])
        mc.http.delete.return_value = mock_sync_response(guard_data)

        with patch.object(
            Guard, "model_validate", side_effect=[fetched_guard, deleted_guard]
        ):
            result = mc.client.delete_guard("my-guard")

        mc.http.delete.assert_called_once_with("/guards/g1")
        assert result == deleted_guard

    def test_delete_guard_does_nothing_when_guard_not_found(self):
        mc = make_client()
        mc.http.get.return_value = mock_sync_response([])

        result = mc.client.delete_guard("missing-guard")

        mc.http.delete.assert_not_called()
        assert result is None

    def test_delete_guard_does_nothing_when_guard_has_no_id(self):
        mc = make_client()
        guard_data = {"name": "my-guard"}
        guard_no_id = make_guard()
        guard_no_id.id = None

        mc.http.get.return_value = mock_sync_response([guard_data])

        with patch.object(Guard, "model_validate", return_value=guard_no_id):
            result = mc.client.delete_guard("my-guard")

        mc.http.delete.assert_not_called()
        assert result is None

    def test_adelete_guard_success(self):
        mc = make_client()
        guard_data = {"id": "g1", "name": "my-guard"}
        fetched_guard = make_guard(guard_id="g1")
        deleted_guard = make_guard(guard_id="g1")

        mc.ahttp.get = AsyncMock(return_value=mock_async_response([guard_data]))
        mc.ahttp.delete = AsyncMock(return_value=mock_async_response(guard_data))

        with patch.object(
            Guard, "model_validate", side_effect=[fetched_guard, deleted_guard]
        ):
            result = asyncio.run(mc.client.adelete_guard("my-guard"))

        mc.ahttp.delete.assert_called_once_with("/guards/g1")
        assert result == deleted_guard

    def test_adelete_guard_does_nothing_when_guard_not_found(self):
        mc = make_client()
        mc.ahttp.get = AsyncMock(return_value=mock_async_response([]))

        result = asyncio.run(mc.client.adelete_guard("missing-guard"))

        mc.ahttp.delete.assert_not_called()
        assert result is None


# ─── validate ─────────────────────────────────────────────────────────────────


class TestValidate:
    def test_validate_success(self):
        mc = make_client()
        guard = make_guard()
        outcome_data = {"callId": "c1"}
        outcome = Mock(spec=IValidationOutcome)

        mc.http.post.return_value = mock_sync_response(outcome_data)

        with patch.object(IValidationOutcome, "model_validate", return_value=outcome):
            result = mc.client.validate(
                guard,
                openai_api_key="oai-key",
                llm_output="hello",
                num_reasks=1,
                prompt_params={"key": "val"},
            )

        assert result == outcome
        mc.http.post.assert_called_once_with(
            "/guards/guard-id-123/validate",
            json={
                "llm_output": "hello",
                "num_reasks": 1,
                "prompt_params": {"key": "val"},
            },
            headers={"x-openai-api-key": "oai-key"},
        )

    def test_validate_uses_env_openai_key(self):
        mc = make_client()
        guard = make_guard()
        mc.http.post.return_value = mock_sync_response({})

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-oai-key"}):
            with patch.object(
                IValidationOutcome, "model_validate", return_value=Mock()
            ):
                mc.client.validate(guard)

        call_kwargs = mc.http.post.call_args[1]
        assert call_kwargs["headers"] == {"x-openai-api-key": "env-oai-key"}

    def test_validate_omits_openai_header_when_no_key(self):
        mc = make_client()
        guard = make_guard()
        mc.http.post.return_value = mock_sync_response({})

        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(
                IValidationOutcome, "model_validate", return_value=Mock()
            ):
                mc.client.validate(guard)

        call_kwargs = mc.http.post.call_args[1]
        assert call_kwargs["headers"] == {}

    def test_validate_raises_validation_error_on_400(self):
        mc = make_client()
        guard = make_guard()
        mc.http.post.return_value = mock_sync_response(raise_error=make_http_error(400))

        with pytest.raises(ValidationError):
            mc.client.validate(guard)

    def test_validate_returns_none_on_non_400_http_error(self):
        # Non-400 HTTPStatusErrors are caught but not re-raised per current impl
        mc = make_client()
        guard = make_guard()
        mc.http.post.return_value = mock_sync_response(raise_error=make_http_error(500))

        result = mc.client.validate(guard)

        assert result is None

    def test_validate_passes_kwargs_in_body(self):
        mc = make_client()
        guard = make_guard()
        mc.http.post.return_value = mock_sync_response({})

        with patch.object(IValidationOutcome, "model_validate", return_value=Mock()):
            mc.client.validate(guard, extra_field="extra_value")

        call_json = mc.http.post.call_args[1]["json"]
        assert call_json["extra_field"] == "extra_value"

    def test_avalidate_success(self):
        mc = make_client()
        guard = make_guard()
        outcome_data = {"callId": "c1"}
        outcome = Mock(spec=IValidationOutcome)

        mc.ahttp.post = AsyncMock(return_value=mock_async_response(outcome_data))

        with patch.object(IValidationOutcome, "model_validate", return_value=outcome):
            result = asyncio.run(
                mc.client.avalidate(
                    guard,
                    openai_api_key="oai-key",
                    llm_output="hello",
                    num_reasks=2,
                )
            )

        assert result == outcome
        mc.ahttp.post.assert_called_once_with(
            "/guards/guard-id-123/validate",
            json={
                "llm_output": "hello",
                "num_reasks": 2,
                "prompt_params": None,
            },
            headers={"x-openai-api-key": "oai-key"},
        )

    def test_avalidate_raises_validation_error_on_400(self):
        mc = make_client()
        guard = make_guard()
        mc.ahttp.post = AsyncMock(
            return_value=mock_async_response(raise_error=make_http_error(400))
        )

        with pytest.raises(ValidationError):
            asyncio.run(mc.client.avalidate(guard))

    def test_avalidate_returns_none_on_non_400_http_error(self):
        mc = make_client()
        guard = make_guard()
        mc.ahttp.post = AsyncMock(
            return_value=mock_async_response(raise_error=make_http_error(500))
        )

        result = asyncio.run(mc.client.avalidate(guard))

        assert result is None


# ─── stream_validate ──────────────────────────────────────────────────────────


class TestStreamValidate:
    def test_stream_validate_yields_outcomes(self):
        mc = make_client()
        guard = make_guard()
        outcome = Mock(spec=IValidationOutcome)

        chunks = [
            make_sse_chunk({"callId": "c1", "validationPassed": True}),
            make_sse_chunk({"callId": "c2", "validationPassed": False}),
        ]
        mc.http.stream.return_value = make_stream_ctx(chunks)

        with patch.object(IValidationOutcome, "model_validate", return_value=outcome):
            results = list(mc.client.stream_validate(guard, openai_api_key="oai-key"))

        assert len(results) == 2
        assert all(r is outcome for r in results)
        call_args = mc.http.stream.call_args
        assert call_args[0] == ("POST", "/guards/guard-id-123/validate")
        assert call_args[1]["json"]["stream"] is True

    def test_stream_validate_skips_empty_chunks(self):
        mc = make_client()
        guard = make_guard()
        outcome = Mock(spec=IValidationOutcome)

        chunks = ["", "   ", make_sse_chunk({"callId": "c1"}), ""]
        mc.http.stream.return_value = make_stream_ctx(chunks)

        with patch.object(IValidationOutcome, "model_validate", return_value=outcome):
            results = list(mc.client.stream_validate(guard))

        assert len(results) == 1

    def test_stream_validate_raises_on_error_in_chunk(self):
        mc = make_client()
        guard = make_guard()

        chunks = [make_sse_chunk({"error": {"message": "Validation error occurred"}})]
        mc.http.stream.return_value = make_stream_ctx(chunks)

        with pytest.raises(Exception, match="Validation error occurred"):
            list(mc.client.stream_validate(guard))

    def test_stream_validate_raises_on_non_success_response(self):
        mc = make_client()
        guard = make_guard()

        ctx = make_stream_ctx([], is_success=False, raise_on_fail=make_http_error(500))
        mc.http.stream.return_value = ctx

        with pytest.raises(HTTPStatusError):
            list(mc.client.stream_validate(guard))

    def test_stream_validate_passes_llm_output_and_params(self):
        mc = make_client()
        guard = make_guard()
        mc.http.stream.return_value = make_stream_ctx([])

        list(
            mc.client.stream_validate(
                guard,
                llm_output="test output",
                num_reasks=3,
                prompt_params={"k": "v"},
            )
        )

        call_json = mc.http.stream.call_args[1]["json"]
        assert call_json["llm_output"] == "test output"
        assert call_json["num_reasks"] == 3
        assert call_json["prompt_params"] == {"k": "v"}
        assert call_json["stream"] is True

    def test_stream_validate_uses_env_openai_key(self):
        mc = make_client()
        guard = make_guard()
        mc.http.stream.return_value = make_stream_ctx([])

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-oai-key"}):
            list(mc.client.stream_validate(guard))

        mc.http.stream.assert_called_once()

    def test_astream_validate_yields_outcomes(self):
        mc = make_client()
        guard = make_guard()
        outcome = Mock(spec=IValidationOutcome)

        chunks = [
            make_sse_chunk({"callId": "c1"}),
            make_sse_chunk({"callId": "c2"}),
        ]
        # ahttp is AsyncMock, so .stream() returns a coroutine by default.
        # Override with a regular Mock so it returns the async context manager.
        mock_stream = Mock(return_value=make_async_stream_ctx(chunks))
        mc.ahttp.stream = mock_stream

        async def run():
            return [
                r
                async for r in mc.client.astream_validate(
                    guard, openai_api_key="oai-key"
                )
            ]

        with patch.object(IValidationOutcome, "model_validate", return_value=outcome):
            results = asyncio.run(run())

        assert len(results) == 2
        assert all(r is outcome for r in results)
        call_args = mock_stream.call_args
        assert call_args[0] == ("POST", "/guards/guard-id-123/validate")
        assert call_args[1]["json"]["stream"] is True

    def test_astream_validate_skips_empty_chunks(self):
        mc = make_client()
        guard = make_guard()
        outcome = Mock(spec=IValidationOutcome)

        chunks = ["", make_sse_chunk({"callId": "c1"}), "   "]
        mc.ahttp.stream = Mock(return_value=make_async_stream_ctx(chunks))

        async def run():
            return [r async for r in mc.client.astream_validate(guard)]

        with patch.object(IValidationOutcome, "model_validate", return_value=outcome):
            results = asyncio.run(run())

        assert len(results) == 1

    def test_astream_validate_raises_on_error_in_chunk(self):
        mc = make_client()
        guard = make_guard()

        chunks = [make_sse_chunk({"error": {"message": "Stream error"}})]
        mc.ahttp.stream = Mock(return_value=make_async_stream_ctx(chunks))

        async def run():
            return [r async for r in mc.client.astream_validate(guard)]

        with pytest.raises(Exception, match="Stream error"):
            asyncio.run(run())

    def test_astream_validate_raises_on_non_success_response(self):
        mc = make_client()
        guard = make_guard()

        ctx = make_async_stream_ctx(
            [], is_success=False, raise_on_fail=make_http_error(500)
        )
        mc.ahttp.stream = Mock(return_value=ctx)

        async def run():
            return [r async for r in mc.client.astream_validate(guard)]

        with pytest.raises(HTTPStatusError):
            asyncio.run(run())


# ─── get_history ──────────────────────────────────────────────────────────────


class TestGetHistory:
    def test_get_history_success(self):
        mc = make_client()
        history_data = {"calls": [{"callId": "c1"}]}
        mc.http.get.return_value = mock_sync_response(history_data)

        result = mc.client.get_history("g1", "c1")

        mc.http.get.assert_called_once_with("/guards/g1/history/c1")
        assert result == history_data

    def test_get_history_uses_guard_id_not_name(self):
        mc = make_client()
        mc.http.get.return_value = mock_sync_response({})

        mc.client.get_history("guard-uuid-123", "call-uuid-456")

        mc.http.get.assert_called_once_with(
            "/guards/guard-uuid-123/history/call-uuid-456"
        )

    def test_get_history_raises_on_http_error(self):
        mc = make_client()
        mc.http.get.return_value = mock_sync_response(raise_error=make_http_error(404))

        with pytest.raises(HTTPStatusError):
            mc.client.get_history("g1", "c1")

    def test_get_history_returns_raw_body(self):
        mc = make_client()
        raw_body = [{"id": "1"}, {"id": "2"}]
        mc.http.get.return_value = mock_sync_response(raw_body)

        result = mc.client.get_history("g1", "c1")

        assert result == raw_body
