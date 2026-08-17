import socket
from unittest.mock import patch

import pytest
from guardrails.validators.url_dns_resolver import URLDNSResolver
from guardrails.validator_base import PassResult, FailResult


class TestURLDNSResolver:
    def setup_method(self):
        self.validator = URLDNSResolver(timeout=2.0, cache_size=16)

    @patch("guardrails.validators.url_dns_resolver.socket.getaddrinfo")
    def test_resolvable_domain_passes(self, mock_dns):
        mock_dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ]
        result = self.validator.validate("https://example.com", {})
        assert isinstance(result, PassResult)
        mock_dns.assert_called_once_with("example.com", None)

    @patch("guardrails.validators.url_dns_resolver.socket.getaddrinfo")
    def test_unresolvable_domain_fails(self, mock_dns):
        mock_dns.side_effect = socket.gaierror("Name resolution failed")
        result = self.validator.validate("https://fake-domain-xyz-12345.com", {})
        assert isinstance(result, FailResult)
        assert "does not resolve" in result.error_message

    @patch("guardrails.validators.url_dns_resolver.socket.getaddrinfo")
    def test_timeout_fails(self, mock_dns):
        mock_dns.side_effect = socket.timeout("Timed out")
        result = self.validator.validate("https://slow-domain.com", {})
        assert isinstance(result, FailResult)

    @patch("guardrails.validators.url_dns_resolver.socket.getaddrinfo")
    def test_cache_hit_avoids_second_lookup(self, mock_dns):
        mock_dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 0))
        ]
        self.validator.validate("https://cached.com", {})
        self.validator.validate("https://cached.com/other-path", {})
        assert mock_dns.call_count == 1

    @patch("guardrails.validators.url_dns_resolver.socket.getaddrinfo")
    def test_failed_result_cached(self, mock_dns):
        mock_dns.side_effect = socket.gaierror("fail")
        self.validator.validate("https://bad.com", {})
        self.validator.validate("https://bad.com", {})
        assert mock_dns.call_count == 1

    def test_bare_domain_input(self):
        with patch("guardrails.validators.url_dns_resolver.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 0))]
            result = self.validator.validate("example.com", {})
            assert isinstance(result, PassResult)

    def test_empty_input_fails(self):
        result = self.validator.validate("", {})
        assert isinstance(result, FailResult)

    @patch("guardrails.validators.url_dns_resolver.socket.getaddrinfo")
    def test_url_with_path_extracts_host(self, mock_dns):
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 0))]
        result = self.validator.validate("https://example.com/path/to/page?q=1", {})
        assert isinstance(result, PassResult)
        mock_dns.assert_called_once_with("example.com", None)
