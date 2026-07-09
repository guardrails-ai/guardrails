import pytest

from guardrails.validators.url_validators import (
    IsValidURL,
    IsValidEmail,
    IsValidDomain,
    IsValidIPAddress,
    URLCategorization,
)
from guardrails.validator_base import FailResult, PassResult


class TestIsValidURL:
    def test_valid_http_url(self):
        v = IsValidURL()
        result = v.validate("http://example.com", {})
        assert isinstance(result, PassResult)

    def test_valid_https_url(self):
        v = IsValidURL()
        result = v.validate("https://example.com/path?q=1", {})
        assert isinstance(result, PassResult)

    def test_valid_url_with_port(self):
        v = IsValidURL()
        result = v.validate("https://example.com:8080/path", {})
        assert isinstance(result, PassResult)

    def test_invalid_url_no_scheme(self):
        v = IsValidURL()
        result = v.validate("example.com", {})
        assert isinstance(result, FailResult)

    def test_invalid_url_garbage(self):
        v = IsValidURL()
        result = v.validate("not a url at all !!!", {})
        assert isinstance(result, FailResult)

    def test_empty_string(self):
        v = IsValidURL()
        result = v.validate("", {})
        assert isinstance(result, FailResult)

    def test_require_https_rejects_http(self):
        v = IsValidURL(require_https=True)
        result = v.validate("http://example.com", {})
        assert isinstance(result, FailResult)
        assert "https" in result.error_message

    def test_require_https_accepts_https(self):
        v = IsValidURL(require_https=True)
        result = v.validate("https://example.com", {})
        assert isinstance(result, PassResult)

    def test_allowed_schemes(self):
        v = IsValidURL(allowed_schemes=["https"])
        result = v.validate("ftp://example.com", {})
        assert isinstance(result, FailResult)

    def test_none_value(self):
        v = IsValidURL()
        result = v.validate(None, {})
        assert isinstance(result, FailResult)


class TestIsValidEmail:
    def test_valid_email(self):
        v = IsValidEmail()
        result = v.validate("user@example.com", {})
        assert isinstance(result, PassResult)

    def test_valid_email_with_plus(self):
        v = IsValidEmail()
        result = v.validate("user+tag@example.co.uk", {})
        assert isinstance(result, PassResult)

    def test_valid_email_with_dots(self):
        v = IsValidEmail()
        result = v.validate("first.last@example.com", {})
        assert isinstance(result, PassResult)

    def test_valid_email_with_numbers(self):
        v = IsValidEmail()
        result = v.validate("user123@example.com", {})
        assert isinstance(result, PassResult)

    def test_invalid_email_no_at(self):
        v = IsValidEmail()
        result = v.validate("userexample.com", {})
        assert isinstance(result, FailResult)

    def test_invalid_email_no_domain(self):
        v = IsValidEmail()
        result = v.validate("user@", {})
        assert isinstance(result, FailResult)

    def test_invalid_email_no_tld(self):
        v = IsValidEmail()
        result = v.validate("user@example", {})
        assert isinstance(result, FailResult)

    def test_invalid_email_spaces(self):
        v = IsValidEmail()
        result = v.validate("user @example.com", {})
        assert isinstance(result, FailResult)

    def test_empty_string(self):
        v = IsValidEmail()
        result = v.validate("", {})
        assert isinstance(result, FailResult)

    def test_none_value(self):
        v = IsValidEmail()
        result = v.validate(None, {})
        assert isinstance(result, FailResult)

    def test_valid_email_subdomain(self):
        v = IsValidEmail()
        result = v.validate("user@sub.example.com", {})
        assert isinstance(result, PassResult)


class TestIsValidDomain:
    def test_valid_domain(self):
        v = IsValidDomain()
        result = v.validate("example.com", {})
        assert isinstance(result, PassResult)

    def test_valid_domain_subdomain(self):
        v = IsValidDomain()
        result = v.validate("sub.example.com", {})
        assert isinstance(result, PassResult)

    def test_valid_domain_multi_level(self):
        v = IsValidDomain()
        result = v.validate("a.b.c.example.com", {})
        assert isinstance(result, PassResult)

    def test_invalid_domain_no_tld(self):
        v = IsValidDomain()
        result = v.validate("example", {})
        assert isinstance(result, FailResult)

    def test_invalid_domain_with_spaces(self):
        v = IsValidDomain()
        result = v.validate("exa mple.com", {})
        assert isinstance(result, FailResult)

    def test_invalid_domain_starting_with_hyphen(self):
        v = IsValidDomain()
        result = v.validate("-example.com", {})
        assert isinstance(result, FailResult)

    def test_invalid_domain_ending_with_hyphen(self):
        v = IsValidDomain()
        result = v.validate("example-.com", {})
        assert isinstance(result, FailResult)

    def test_empty_string(self):
        v = IsValidDomain()
        result = v.validate("", {})
        assert isinstance(result, FailResult)

    def test_none_value(self):
        v = IsValidDomain()
        result = v.validate(None, {})
        assert isinstance(result, FailResult)

    def test_label_too_long(self):
        label = "a" * 64
        v = IsValidDomain()
        result = v.validate(f"{label}.com", {})
        assert isinstance(result, FailResult)

    def test_total_length_too_long(self):
        label = "a" * 63
        domain = ".".join([label] * 5) + ".com"
        v = IsValidDomain()
        result = v.validate(domain, {})
        assert isinstance(result, FailResult)

    def test_require_tld_false(self):
        v = IsValidDomain(require_tld=False)
        result = v.validate("example", {})
        assert isinstance(result, PassResult)

    def test_domain_with_numbers(self):
        v = IsValidDomain()
        result = v.validate("example123.com", {})
        assert isinstance(result, PassResult)


class TestIsValidIPAddress:
    def test_valid_ipv4(self):
        v = IsValidIPAddress()
        result = v.validate("192.168.1.1", {})
        assert isinstance(result, PassResult)

    def test_valid_ipv6(self):
        v = IsValidIPAddress()
        result = v.validate("::1", {})
        assert isinstance(result, PassResult)

    def test_valid_ipv6_full(self):
        v = IsValidIPAddress()
        result = v.validate("2001:0db8:85a3:0000:0000:8a2e:0370:7334", {})
        assert isinstance(result, PassResult)

    def test_invalid_ip_address(self):
        v = IsValidIPAddress()
        result = v.validate("999.999.999.999", {})
        assert isinstance(result, FailResult)

    def test_invalid_string(self):
        v = IsValidIPAddress()
        result = v.validate("not-an-ip", {})
        assert isinstance(result, FailResult)

    def test_ipv4_only_accepts_ipv4(self):
        v = IsValidIPAddress(ip_version=4)
        result = v.validate("10.0.0.1", {})
        assert isinstance(result, PassResult)

    def test_ipv4_only_rejects_ipv6(self):
        v = IsValidIPAddress(ip_version=4)
        result = v.validate("::1", {})
        assert isinstance(result, FailResult)

    def test_ipv6_only_accepts_ipv6(self):
        v = IsValidIPAddress(ip_version=6)
        result = v.validate("::1", {})
        assert isinstance(result, PassResult)

    def test_ipv6_only_rejects_ipv4(self):
        v = IsValidIPAddress(ip_version=6)
        result = v.validate("192.168.1.1", {})
        assert isinstance(result, FailResult)

    def test_empty_string(self):
        v = IsValidIPAddress()
        result = v.validate("", {})
        assert isinstance(result, FailResult)

    def test_none_value(self):
        v = IsValidIPAddress()
        result = v.validate(None, {})
        assert isinstance(result, FailResult)

    def test_loopback(self):
        v = IsValidIPAddress()
        result = v.validate("127.0.0.1", {})
        assert isinstance(result, PassResult)


class TestURLCategorization:
    def test_safe_url_default(self):
        v = URLCategorization()
        result = v.validate("https://example.com", {})
        assert isinstance(result, PassResult)

    def test_malicious_domain(self):
        v = URLCategorization(
            malicious_domains=["evil.com"],
        )
        result = v.validate("https://evil.com", {})
        assert isinstance(result, FailResult)
        assert "malicious" in result.error_message

    def test_phishing_domain(self):
        v = URLCategorization(
            phishing_domains=["phishy.com"],
        )
        result = v.validate("https://phishy.com", {})
        assert isinstance(result, FailResult)
        assert "phishing" in result.error_message

    def test_safe_domain_override(self):
        v = URLCategorization(
            safe_domains=["example.com"],
            malicious_domains=["example.com"],
        )
        result = v.validate("https://example.com", {})
        assert isinstance(result, PassResult)

    def test_suspicious_tld(self):
        v = URLCategorization(threshold=20)
        result = v.validate("https://example.tk", {})
        assert isinstance(result, FailResult)

    def test_suspicious_keywords(self):
        v = URLCategorization(threshold=10)
        result = v.validate("https://example.com/login", {})
        assert isinstance(result, FailResult)

    def test_high_threshold_passes(self):
        v = URLCategorization(threshold=100)
        result = v.validate("https://example.tk/login/secure/verify", {})
        assert isinstance(result, PassResult)

    def test_invalid_url(self):
        v = URLCategorization()
        result = v.validate("not a url", {})
        assert isinstance(result, FailResult)

    def test_empty_string(self):
        v = URLCategorization()
        result = v.validate("", {})
        assert isinstance(result, FailResult)

    def test_none_value(self):
        v = URLCategorization()
        result = v.validate(None, {})
        assert isinstance(result, FailResult)
