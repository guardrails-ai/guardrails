"""Integration tests: composing multiple URL validators together."""

import socket
from unittest.mock import patch, MagicMock

import pytest
from guardrails.validators.url_format import URLFormatValidator
from guardrails.validators.url_dns_resolver import URLDNSResolver
from guardrails.validators.domain_blocklist import DomainBlocklist
from guardrails.validators.url_risk_scorer import URLRiskScorer
from guardrails.validator_base import PassResult, FailResult


class TestValidatorComposition:
    """Test validators used in sequence, simulating Guard().use_many()."""

    def _run_chain(self, validators, value):
        for v in validators:
            result = v.validate(value, {})
            if isinstance(result, FailResult):
                return result
        return PassResult()

    def test_legitimate_url_passes_all(self):
        validators = [
            URLFormatValidator(),
            DomainBlocklist(blocked=["evil.com"]),
            URLRiskScorer(threshold=0.7),
        ]
        result = self._run_chain(validators, "https://github.com/user/repo")
        assert isinstance(result, PassResult)

    def test_blocked_domain_caught_early(self):
        validators = [
            URLFormatValidator(),
            DomainBlocklist(blocked=["malware.com"]),
            URLRiskScorer(threshold=0.7),
        ]
        result = self._run_chain(validators, "https://malware.com/payload")
        assert isinstance(result, FailResult)
        assert "blocked" in result.error_message

    def test_bad_format_caught_first(self):
        validators = [
            URLFormatValidator(),
            DomainBlocklist(blocked=["evil.com"]),
        ]
        result = self._run_chain(validators, "not-a-url")
        assert isinstance(result, FailResult)
        assert "scheme" in result.error_message.lower() or "missing" in result.error_message.lower()

    @patch("guardrails.validators.url_dns_resolver.socket.getaddrinfo")
    def test_hallucinated_url_caught_by_dns(self, mock_dns):
        mock_dns.side_effect = socket.gaierror("NXDOMAIN")
        validators = [
            URLFormatValidator(),
            URLDNSResolver(timeout=2.0),
        ]
        result = self._run_chain(validators, "https://fake-company-xyz-12345.com")
        assert isinstance(result, FailResult)
        assert "does not resolve" in result.error_message

    def test_typosquatting_caught_by_risk_scorer(self):
        validators = [
            URLFormatValidator(),
            URLRiskScorer(threshold=0.3),
        ]
        result = self._run_chain(validators, "https://gooogle.com")
        assert isinstance(result, FailResult)
        assert "typosquatting" in result.error_message

    def test_strict_allowlist_mode(self):
        validators = [
            URLFormatValidator(require_https=True),
            DomainBlocklist(
                allowed=["api.internal.com", "docs.internal.com"],
                mode="allowlist",
            ),
        ]
        result = self._run_chain(validators, "https://api.internal.com/v1")
        assert isinstance(result, PassResult)

        result = self._run_chain(validators, "https://external.com")
        assert isinstance(result, FailResult)
