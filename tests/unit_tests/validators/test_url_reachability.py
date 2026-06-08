from unittest.mock import patch, MagicMock

import pytest
from guardrails.validators.url_reachability import URLReachability
from guardrails.validator_base import PassResult, FailResult


class TestURLReachability:
    def setup_method(self):
        self.validator = URLReachability(timeout=5.0)

    @patch("guardrails.validators.url_reachability.httpx.head")
    def test_reachable_url_passes(self, mock_head):
        mock_head.return_value = MagicMock(status_code=200)
        result = self.validator.validate("https://example.com", {})
        assert isinstance(result, PassResult)

    @patch("guardrails.validators.url_reachability.httpx.head")
    def test_redirect_passes(self, mock_head):
        mock_head.return_value = MagicMock(status_code=301)
        result = self.validator.validate("https://example.com", {})
        assert isinstance(result, PassResult)

    @patch("guardrails.validators.url_reachability.httpx.head")
    def test_404_fails(self, mock_head):
        mock_head.return_value = MagicMock(status_code=404)
        result = self.validator.validate("https://example.com/missing", {})
        assert isinstance(result, FailResult)
        assert "404" in result.error_message

    @patch("guardrails.validators.url_reachability.httpx.head")
    def test_500_fails(self, mock_head):
        mock_head.return_value = MagicMock(status_code=500)
        result = self.validator.validate("https://example.com/error", {})
        assert isinstance(result, FailResult)

    @patch("guardrails.validators.url_reachability.httpx.head")
    def test_connection_error(self, mock_head):
        import httpx
        mock_head.side_effect = httpx.ConnectError("Connection refused")
        result = self.validator.validate("https://nonexistent.example", {})
        assert isinstance(result, FailResult)
        assert "connection failed" in result.error_message

    @patch("guardrails.validators.url_reachability.httpx.head")
    def test_timeout_error(self, mock_head):
        import httpx
        mock_head.side_effect = httpx.TimeoutException("Timed out")
        result = self.validator.validate("https://slow.example.com", {})
        assert isinstance(result, FailResult)
        assert "timeout" in result.error_message

    @patch("guardrails.validators.url_reachability.httpx.head")
    def test_tls_error(self, mock_head):
        import httpx
        mock_head.side_effect = httpx.HTTPError("SSL certificate verify failed")
        result = self.validator.validate("https://expired-cert.com", {})
        assert isinstance(result, FailResult)
        assert "TLS" in result.error_message or "SSL" in result.error_message

    @patch("guardrails.validators.url_reachability.httpx.head")
    def test_custom_accept_status(self, mock_head):
        mock_head.return_value = MagicMock(status_code=404)
        v = URLReachability(accept_status=[200, 404])
        result = v.validate("https://example.com", {})
        assert isinstance(result, PassResult)

    def test_invalid_url_fails(self):
        result = self.validator.validate("", {})
        assert isinstance(result, FailResult)

    @patch("guardrails.validators.url_reachability.httpx.head")
    def test_check_tls_disabled(self, mock_head):
        mock_head.return_value = MagicMock(status_code=200)
        v = URLReachability(check_tls=False)
        v.validate("https://self-signed.example.com", {})
        mock_head.assert_called_once()
        _, kwargs = mock_head.call_args
        assert kwargs["verify"] is False

    @patch("guardrails.validators.url_reachability.httpx.head")
    def test_bare_domain_gets_scheme(self, mock_head):
        mock_head.return_value = MagicMock(status_code=200)
        result = self.validator.validate("example.com", {})
        assert isinstance(result, PassResult)
        call_url = mock_head.call_args[0][0]
        assert call_url.startswith("https://")
