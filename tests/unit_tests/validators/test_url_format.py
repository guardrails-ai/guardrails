import pytest
from guardrails.validators.url_format import URLFormatValidator
from guardrails.validator_base import PassResult, FailResult


class TestURLFormatValidator:
    def setup_method(self):
        self.validator = URLFormatValidator()

    def test_valid_http_url(self):
        result = self.validator.validate("http://example.com", {})
        assert isinstance(result, PassResult)

    def test_valid_https_url(self):
        result = self.validator.validate("https://example.com/path?q=1", {})
        assert isinstance(result, PassResult)

    def test_missing_scheme(self):
        result = self.validator.validate("example.com", {})
        assert isinstance(result, FailResult)

    def test_missing_host(self):
        result = self.validator.validate("https://", {})
        assert isinstance(result, FailResult)

    def test_empty_string(self):
        result = self.validator.validate("", {})
        assert isinstance(result, FailResult)

    def test_require_https_rejects_http(self):
        v = URLFormatValidator(require_https=True)
        result = v.validate("http://example.com", {})
        assert isinstance(result, FailResult)
        assert "HTTPS required" in result.error_message

    def test_require_https_accepts_https(self):
        v = URLFormatValidator(require_https=True)
        result = v.validate("https://example.com", {})
        assert isinstance(result, PassResult)

    def test_disallowed_scheme(self):
        v = URLFormatValidator(allowed_schemes=["https"])
        result = v.validate("ftp://files.example.com", {})
        assert isinstance(result, FailResult)
        assert "not allowed" in result.error_message

    def test_ip_rejected_by_default(self):
        result = self.validator.validate("http://192.168.1.1/path", {})
        assert isinstance(result, FailResult)
        assert "IP addresses" in result.error_message

    def test_ip_allowed_when_configured(self):
        v = URLFormatValidator(allow_ip=True)
        result = v.validate("http://192.168.1.1/path", {})
        assert isinstance(result, PassResult)

    def test_consecutive_dots_rejected(self):
        result = self.validator.validate("https://example..com", {})
        assert isinstance(result, FailResult)
        assert "consecutive dots" in result.error_message

    def test_single_part_domain_rejected(self):
        result = self.validator.validate("https://localhost", {})
        assert isinstance(result, FailResult)

    def test_valid_subdomain(self):
        result = self.validator.validate("https://sub.example.com", {})
        assert isinstance(result, PassResult)

    def test_long_domain_rejected(self):
        long_host = "a" * 250 + ".com"
        result = self.validator.validate(f"https://{long_host}", {})
        assert isinstance(result, FailResult)
        assert "too long" in result.error_message
