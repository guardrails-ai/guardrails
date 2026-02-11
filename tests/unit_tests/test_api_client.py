import os
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from guardrails.api_client import GuardrailsApiClient
from guardrails.errors import ValidationError
from guardrails_api_client.exceptions import BadRequestException
from guardrails_api_client.models import (
    Guard,
    ValidatePayload,
    ValidationOutcome as IValidationOutcome,
)


class TestGuardrailsApiClientInit:
    """Test initialization of GuardrailsApiClient"""

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_init_with_no_params(
        self, _mock_config, _mock_api_client, _mock_guard_api, _mock_validate_api
    ):
        """Test initialization with no parameters uses environment variables"""
        with patch.dict(
            os.environ,
            {
                "GUARDRAILS_BASE_URL": "http://test-env.com",
                "GUARDRAILS_API_KEY": "test-env-key",
            },
        ):
            client = GuardrailsApiClient()

            assert client.base_url == "http://test-env.com"
            assert client.api_key == "test-env-key"
            assert client.timeout == 300

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_init_with_explicit_params(
        self, _mock_config, _mock_api_client, _mock_guard_api, _mock_validate_api
    ):
        """Test initialization with explicit parameters overrides environment"""
        with patch.dict(
            os.environ,
            {
                "GUARDRAILS_BASE_URL": "http://test-env.com",
                "GUARDRAILS_API_KEY": "test-env-key",
            },
        ):
            client = GuardrailsApiClient(
                base_url="http://custom.com", api_key="custom-key"
            )

            assert client.base_url == "http://custom.com"
            assert client.api_key == "custom-key"
            assert client.timeout == 300

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_init_default_values(
        self, _mock_config, _mock_api_client, _mock_guard_api, _mock_validate_api
    ):
        """Test initialization uses defaults when environment variables are missing"""
        with patch.dict(os.environ, {}, clear=True):
            client = GuardrailsApiClient()

            assert client.base_url == "http://localhost:8000"
            assert client.api_key == ""
            assert client.timeout == 300

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_init_partial_params(
        self, _mock_config, _mock_api_client, _mock_guard_api, _mock_validate_api
    ):
        """Test initialization with only base_url provided"""
        with patch.dict(os.environ, {"GUARDRAILS_API_KEY": "env-key"}, clear=True):
            client = GuardrailsApiClient(base_url="http://custom.com")

            assert client.base_url == "http://custom.com"
            assert client.api_key == "env-key"

    @patch("guardrails.api_client.sys.version_info")
    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_init_api_key_format_python_310_plus(
        self,
        mock_config,
        mock_api_client,
        mock_guard_api,
        mock_validate_api,
        mock_version,
    ):
        """Test API key format for Python 3.10+"""
        mock_version.minor = 10
        _ = GuardrailsApiClient(base_url="http://test.com", api_key="test-key")

        mock_config.assert_called_once()
        call_kwargs = mock_config.call_args[1]
        assert call_kwargs["api_key"] == {"ApiKeyAuth": "test-key"}
        assert call_kwargs["host"] == "http://test.com"

    @patch("guardrails.api_client.sys.version_info")
    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_init_api_key_format_python_39(
        self,
        mock_config,
        mock_api_client,
        mock_guard_api,
        mock_validate_api,
        mock_version,
    ):
        """Test API key format for Python 3.9"""
        mock_version.minor = 9
        _ = GuardrailsApiClient(base_url="http://test.com", api_key="test-key")

        mock_config.assert_called_once()
        call_kwargs = mock_config.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["host"] == "http://test.com"

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_init_creates_api_instances(
        self, _mock_config, _mock_api_client, mock_guard_api, mock_validate_api
    ):
        """Test that initialization creates GuardApi and ValidateApi instances"""
        client = GuardrailsApiClient()

        mock_guard_api.assert_called_once()
        mock_validate_api.assert_called_once()
        assert client._guard_api is not None
        assert client._validate_api is not None


class TestGuardrailsApiClientUpsertGuard:
    """Test upsert_guard method"""

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_upsert_guard_success(
        self, _mock_config, _mock_api_client, mock_guard_api_class, _mock_validate_api
    ):
        """Test upsert_guard calls update_guard with correct parameters"""
        mock_guard_api = Mock()
        mock_guard_api_class.return_value = mock_guard_api

        client = GuardrailsApiClient()
        guard = Mock(spec=Guard)
        guard.name = "test-guard"

        client.upsert_guard(guard)

        mock_guard_api.update_guard.assert_called_once_with(
            guard_name="test-guard", body=guard, _request_timeout=300
        )

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_upsert_guard_uses_timeout(
        self, _mock_config, _mock_api_client, mock_guard_api_class, _mock_validate_api
    ):
        """Test upsert_guard uses the client's timeout value"""
        mock_guard_api = Mock()
        mock_guard_api_class.return_value = mock_guard_api

        client = GuardrailsApiClient()
        client.timeout = 500  # Custom timeout
        guard = Mock(spec=Guard)
        guard.name = "test-guard"

        client.upsert_guard(guard)

        mock_guard_api.update_guard.assert_called_once()
        call_kwargs = mock_guard_api.update_guard.call_args[1]
        assert call_kwargs["_request_timeout"] == 500


class TestGuardrailsApiClientFetchGuard:
    """Test fetch_guard method"""

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_fetch_guard_success(
        self, _mock_config, _mock_api_client, mock_guard_api_class, _mock_validate_api
    ):
        """Test fetch_guard returns guard when successful"""
        mock_guard_api = Mock()
        mock_guard = Mock(spec=Guard)
        mock_guard.name = "test-guard"
        mock_guard_api.get_guard.return_value = mock_guard
        mock_guard_api_class.return_value = mock_guard_api

        client = GuardrailsApiClient()
        result = client.fetch_guard("test-guard")

        assert result == mock_guard
        mock_guard_api.get_guard.assert_called_once_with(guard_name="test-guard")

    @patch("guardrails.api_client.logger")
    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_fetch_guard_handles_exception(
        self,
        _mock_config,
        _mock_api_client,
        mock_guard_api_class,
        _mock_validate_api,
        mock_logger,
    ):
        """Test fetch_guard returns None and logs error on exception"""
        mock_guard_api = Mock()
        mock_guard_api.get_guard.side_effect = Exception("API Error")
        mock_guard_api_class.return_value = mock_guard_api

        client = GuardrailsApiClient()
        result = client.fetch_guard("test-guard")

        assert result is None
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "Error fetching guard test-guard" in error_msg
        assert "API Error" in error_msg

    @patch("guardrails.api_client.logger")
    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_fetch_guard_handles_not_found(
        self,
        _mock_config,
        _mock_api_client,
        mock_guard_api_class,
        _mock_validate_api,
        mock_logger,
    ):
        """Test fetch_guard handles 404 errors gracefully"""
        mock_guard_api = Mock()
        mock_guard_api.get_guard.side_effect = Exception("404 Not Found")
        mock_guard_api_class.return_value = mock_guard_api

        client = GuardrailsApiClient()
        result = client.fetch_guard("nonexistent-guard")

        assert result is None
        mock_logger.error.assert_called_once()


class TestGuardrailsApiClientDeleteGuard:
    """Test delete_guard method"""

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_delete_guard_success(
        self, _mock_config, _mock_api_client, mock_guard_api_class, _mock_validate_api
    ):
        """Test delete_guard calls delete_guard with correct parameters"""
        mock_guard_api = Mock()
        mock_guard_api_class.return_value = mock_guard_api

        client = GuardrailsApiClient()
        client.delete_guard("test-guard")

        mock_guard_api.delete_guard.assert_called_once_with(
            guard_name="test-guard", _request_timeout=300
        )

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_delete_guard_uses_timeout(
        self, _mock_config, _mock_api_client, mock_guard_api_class, _mock_validate_api
    ):
        """Test delete_guard uses the client's timeout value"""
        mock_guard_api = Mock()
        mock_guard_api_class.return_value = mock_guard_api

        client = GuardrailsApiClient()
        client.timeout = 600
        client.delete_guard("test-guard")

        call_kwargs = mock_guard_api.delete_guard.call_args[1]
        assert call_kwargs["_request_timeout"] == 600


class TestGuardrailsApiClientValidate:
    """Test validate method"""

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_validate_success(
        self, _mock_config, _mock_api_client, _mock_guard_api, mock_validate_api_class
    ):
        """Test validate calls validate API with correct parameters"""
        mock_validate_api = Mock()
        mock_outcome = Mock(spec=IValidationOutcome)
        mock_validate_api.validate.return_value = mock_outcome
        mock_validate_api_class.return_value = mock_validate_api

        guard = Mock(spec=Guard)
        guard.name = "test-guard"
        payload = Mock(spec=ValidatePayload)

        client = GuardrailsApiClient()
        result = client.validate(guard, payload, openai_api_key="test-key")

        assert result == mock_outcome
        mock_validate_api.validate.assert_called_once_with(
            guard_name="test-guard",
            validate_payload=payload,
            x_openai_api_key="test-key",
        )

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_validate_uses_env_openai_key(
        self, _mock_config, _mock_api_client, _mock_guard_api, mock_validate_api_class
    ):
        """Test validate uses environment OPENAI_API_KEY when not provided"""
        mock_validate_api = Mock()
        mock_validate_api_class.return_value = mock_validate_api

        guard = Mock(spec=Guard)
        guard.name = "test-guard"
        payload = Mock(spec=ValidatePayload)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}):
            client = GuardrailsApiClient()
            client.validate(guard, payload)

            mock_validate_api.validate.assert_called_once()
            call_kwargs = mock_validate_api.validate.call_args[1]
            assert call_kwargs["x_openai_api_key"] == "env-openai-key"

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_validate_no_openai_key(
        self, _mock_config, _mock_api_client, _mock_guard_api, mock_validate_api_class
    ):
        """Test validate passes None when no OpenAI key is available"""
        mock_validate_api = Mock()
        mock_validate_api_class.return_value = mock_validate_api

        guard = Mock(spec=Guard)
        guard.name = "test-guard"
        payload = Mock(spec=ValidatePayload)

        with patch.dict(os.environ, {}, clear=True):
            client = GuardrailsApiClient()
            client.validate(guard, payload)

            mock_validate_api.validate.assert_called_once()
            call_kwargs = mock_validate_api.validate.call_args[1]
            assert call_kwargs["x_openai_api_key"] is None

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_validate_raises_validation_error_on_bad_request(
        self, _mock_config, _mock_api_client, _mock_guard_api, mock_validate_api_class
    ):
        """Test validate raises ValidationError when BadRequestException occurs"""
        mock_validate_api = Mock()
        bad_request = BadRequestException(status=400, reason="Bad Request")
        bad_request.body = "Validation failed: invalid input"
        mock_validate_api.validate.side_effect = bad_request
        mock_validate_api_class.return_value = mock_validate_api

        guard = Mock(spec=Guard)
        guard.name = "test-guard"
        payload = Mock(spec=ValidatePayload)

        client = GuardrailsApiClient()

        with pytest.raises(ValidationError) as exc_info:
            client.validate(guard, payload)

        assert "Validation failed: invalid input" in str(exc_info.value)


class TestGuardrailsApiClientStreamValidate:
    """Test stream_validate method"""

    @patch("guardrails.api_client.requests.Session")
    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_stream_validate_success(
        self,
        _mock_config,
        _mock_api_client,
        _mock_guard_api,
        _mock_validate_api,
        mock_session,
    ):
        """Test stream_validate yields validation outcomes"""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_lines.return_value = [
            b'{"callId": "call1", "validationPassed": true, '
            b'"validatedOutput": "test1", "validationSummaries": []}',
            b'{"callId": "call2", "validationPassed": false, '
            b'"validatedOutput": "test2", "validationSummaries": []}',
        ]

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__enter__.return_value = mock_response
        mock_session.return_value = mock_session_instance

        guard = Mock(spec=Guard)
        guard.name = "test-guard"
        payload = Mock(spec=ValidatePayload)
        payload.to_dict.return_value = {"llm_output": "test"}

        client = GuardrailsApiClient(base_url="http://test.com")
        results = list(
            client.stream_validate(guard, payload, openai_api_key="test-key")
        )

        assert len(results) == 2
        assert all(isinstance(r, IValidationOutcome) for r in results)

        mock_session_instance.post.assert_called_once_with(
            "http://test.com/guards/test-guard/validate",
            json={"llm_output": "test"},
            headers={
                "Content-Type": "application/json",
                "x-openai-api-key": "test-key",
            },
            stream=True,
        )

    @patch("guardrails.api_client.requests.Session")
    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_stream_validate_uses_env_openai_key(
        self,
        _mock_config,
        _mock_api_client,
        _mock_guard_api,
        _mock_validate_api,
        mock_session,
    ):
        """Test stream_validate uses environment OPENAI_API_KEY when not provided"""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_lines.return_value = []

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__enter__.return_value = mock_response
        mock_session.return_value = mock_session_instance

        guard = Mock(spec=Guard)
        guard.name = "test-guard"
        payload = Mock(spec=ValidatePayload)
        payload.to_dict.return_value = {}

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            client = GuardrailsApiClient()
            list(client.stream_validate(guard, payload))

            call_args = mock_session_instance.post.call_args
            headers = call_args[1]["headers"]
            assert headers["x-openai-api-key"] == "env-key"

    @patch("guardrails.api_client.requests.Session")
    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_stream_validate_raises_on_error_response(
        self,
        _mock_config,
        _mock_api_client,
        _mock_guard_api,
        _mock_validate_api,
        mock_session,
    ):
        """Test stream_validate raises ValueError on non-OK response"""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 400
        mock_response.reason = "Bad Request"
        mock_response.text = "Invalid payload"
        mock_response.iter_lines.return_value = [b"error line"]

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__enter__.return_value = mock_response
        mock_session.return_value = mock_session_instance

        guard = Mock(spec=Guard)
        guard.name = "test-guard"
        payload = Mock(spec=ValidatePayload)
        payload.to_dict.return_value = {}

        client = GuardrailsApiClient()

        with pytest.raises(ValueError) as exc_info:
            list(client.stream_validate(guard, payload))

        assert "status_code: 400" in str(exc_info.value)

    @patch("guardrails.api_client.requests.Session")
    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_stream_validate_raises_on_error_in_json(
        self,
        _mock_config,
        _mock_api_client,
        _mock_guard_api,
        _mock_validate_api,
        mock_session,
    ):
        """Test stream_validate raises Exception when error in JSON response"""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_lines.return_value = [
            b'{"error": {"message": "Validation error occurred"}}'
        ]

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__enter__.return_value = mock_response
        mock_session.return_value = mock_session_instance

        guard = Mock(spec=Guard)
        guard.name = "test-guard"
        payload = Mock(spec=ValidatePayload)
        payload.to_dict.return_value = {}

        client = GuardrailsApiClient()

        with pytest.raises(Exception) as exc_info:
            list(client.stream_validate(guard, payload))

        assert "Validation error occurred" in str(exc_info.value)

    @patch("guardrails.api_client.requests.Session")
    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_stream_validate_skips_empty_lines(
        self,
        _mock_config,
        _mock_api_client,
        _mock_guard_api,
        _mock_validate_api,
        mock_session,
    ):
        """Test stream_validate skips empty lines in response"""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_lines.return_value = [
            b"",
            b'{"callId": "call1", "validationPassed": true, "validationSummaries": []}',
            b"",
            b'{"callId": "call2", "validationPassed": false, '
            b'"validationSummaries": []}',
            b"",
        ]

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__enter__.return_value = mock_response
        mock_session.return_value = mock_session_instance

        guard = Mock(spec=Guard)
        guard.name = "test-guard"
        payload = Mock(spec=ValidatePayload)
        payload.to_dict.return_value = {}

        client = GuardrailsApiClient()
        results = list(client.stream_validate(guard, payload))

        # Should only get 2 results, skipping empty lines
        assert len(results) == 2


class TestGuardrailsApiClientGetHistory:
    """Test get_history method"""

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_get_history_success(
        self, _mock_config, _mock_api_client, mock_guard_api_class, _mock_validate_api
    ):
        """Test get_history calls get_guard_history with correct parameters"""
        mock_guard_api = Mock()
        mock_history = {"call_id": "123", "data": "test"}
        mock_guard_api.get_guard_history.return_value = mock_history
        mock_guard_api_class.return_value = mock_guard_api

        client = GuardrailsApiClient()
        result = client.get_history("test-guard", "call-123")

        assert result == mock_history
        mock_guard_api.get_guard_history.assert_called_once_with(
            "test-guard", "call-123"
        )

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_get_history_with_different_ids(
        self, _mock_config, _mock_api_client, mock_guard_api_class, _mock_validate_api
    ):
        """Test get_history with various guard names and call IDs"""
        mock_guard_api = Mock()
        mock_guard_api_class.return_value = mock_guard_api

        client = GuardrailsApiClient()

        # Test various combinations
        client.get_history("guard-1", "call-1")
        client.get_history("guard-2", "call-2")
        client.get_history("my-special-guard", "abc-123-def")

        assert mock_guard_api.get_guard_history.call_count == 3
        calls = mock_guard_api.get_guard_history.call_args_list
        assert calls[0] == call("guard-1", "call-1")
        assert calls[1] == call("guard-2", "call-2")
        assert calls[2] == call("my-special-guard", "abc-123-def")


class TestGuardrailsApiClientIntegration:
    """Integration-style tests for GuardrailsApiClient"""

    @patch("guardrails.api_client.requests.Session")
    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_complete_workflow(
        self,
        _mock_config,
        _mock_api_client,
        mock_guard_api_class,
        mock_validate_api_class,
        _mock_session,
    ):
        """Test a complete workflow: create, validate, fetch history, delete"""
        mock_guard_api = Mock()
        mock_validate_api = Mock()
        mock_guard_api_class.return_value = mock_guard_api
        mock_validate_api_class.return_value = mock_validate_api

        # Setup mocks
        guard = Mock(spec=Guard)
        guard.name = "workflow-guard"
        payload = Mock(spec=ValidatePayload)
        mock_outcome = Mock(spec=IValidationOutcome)
        mock_validate_api.validate.return_value = mock_outcome
        mock_guard_api.get_guard.return_value = guard
        mock_guard_api.get_guard_history.return_value = {"calls": []}

        client = GuardrailsApiClient(base_url="http://test.com", api_key="test-key")

        # Upsert guard
        client.upsert_guard(guard)
        assert mock_guard_api.update_guard.called

        # Validate
        result = client.validate(guard, payload, openai_api_key="openai-key")
        assert result == mock_outcome

        # Fetch guard
        fetched = client.fetch_guard("workflow-guard")
        assert fetched == guard

        # Get history
        history = client.get_history("workflow-guard", "call-1")
        assert history == {"calls": []}

        # Delete guard
        client.delete_guard("workflow-guard")
        assert mock_guard_api.delete_guard.called

    @patch("guardrails.api_client.ValidateApi")
    @patch("guardrails.api_client.GuardApi")
    @patch("guardrails.api_client.ApiClient")
    @patch("guardrails.api_client.Configuration")
    def test_client_maintains_state(
        self, _mock_config, _mock_api_client, _mock_guard_api, _mock_validate_api
    ):
        """Test that client maintains its state across method calls"""
        client = GuardrailsApiClient(base_url="http://custom.com", api_key="custom-key")

        assert client.base_url == "http://custom.com"
        assert client.api_key == "custom-key"

        # Modify timeout
        client.timeout = 1000

        # Ensure state is maintained
        assert client.base_url == "http://custom.com"
        assert client.api_key == "custom-key"
        assert client.timeout == 1000
