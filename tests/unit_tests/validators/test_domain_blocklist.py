import pytest
from guardrails.validators.domain_blocklist import DomainBlocklist
from guardrails.validator_base import PassResult, FailResult


class TestDomainBlocklist:
    def test_blocked_domain_rejected(self):
        v = DomainBlocklist(blocked=["evil.com"])
        result = v.validate("https://evil.com/malware", {})
        assert isinstance(result, FailResult)
        assert "blocked" in result.error_message

    def test_clean_domain_passes_blocklist(self):
        v = DomainBlocklist(blocked=["evil.com"])
        result = v.validate("https://google.com", {})
        assert isinstance(result, PassResult)

    def test_subdomain_matched_by_default(self):
        v = DomainBlocklist(blocked=["evil.com"])
        result = v.validate("https://sub.evil.com/page", {})
        assert isinstance(result, FailResult)

    def test_subdomain_not_matched_when_disabled(self):
        v = DomainBlocklist(blocked=["evil.com"], match_subdomains=False)
        result = v.validate("https://sub.evil.com/page", {})
        assert isinstance(result, PassResult)

    def test_allowlist_mode_passes_allowed(self):
        v = DomainBlocklist(allowed=["trusted.com"], mode="allowlist")
        result = v.validate("https://trusted.com/api", {})
        assert isinstance(result, PassResult)

    def test_allowlist_mode_rejects_unknown(self):
        v = DomainBlocklist(allowed=["trusted.com"], mode="allowlist")
        result = v.validate("https://unknown.com", {})
        assert isinstance(result, FailResult)
        assert "not in the allowlist" in result.error_message

    def test_allowlist_with_subdomain(self):
        v = DomainBlocklist(allowed=["trusted.com"], mode="allowlist")
        result = v.validate("https://api.trusted.com", {})
        assert isinstance(result, PassResult)

    def test_case_insensitive(self):
        v = DomainBlocklist(blocked=["Evil.COM"])
        result = v.validate("https://evil.com", {})
        assert isinstance(result, FailResult)

    def test_bare_domain_input(self):
        v = DomainBlocklist(blocked=["evil.com"])
        result = v.validate("evil.com", {})
        assert isinstance(result, FailResult)

    def test_empty_blocklist_passes_all(self):
        v = DomainBlocklist(blocked=[])
        result = v.validate("https://anything.com", {})
        assert isinstance(result, PassResult)

    def test_invalid_mode_raises(self):
        with pytest.raises(AssertionError):
            DomainBlocklist(mode="invalid")

    def test_unparseable_input(self):
        v = DomainBlocklist(blocked=["evil.com"])
        result = v.validate("", {})
        assert isinstance(result, FailResult)
