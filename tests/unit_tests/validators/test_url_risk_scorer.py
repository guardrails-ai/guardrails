import pytest
from guardrails.validators.url_risk_scorer import URLRiskScorer
from guardrails.validator_base import PassResult, FailResult


class TestURLRiskScorer:
    def setup_method(self):
        self.validator = URLRiskScorer(threshold=0.7)

    def test_legitimate_domain_passes(self):
        result = self.validator.validate("https://google.com", {})
        assert isinstance(result, PassResult)

    def test_legitimate_domain_with_path_passes(self):
        result = self.validator.validate("https://github.com/user/repo", {})
        assert isinstance(result, PassResult)

    def test_suspicious_tld(self):
        v = URLRiskScorer(threshold=0.15)
        result = v.validate("https://something.tk", {})
        assert isinstance(result, FailResult)
        assert "suspicious_tld" in result.error_message

    def test_typosquatting_detected(self):
        v = URLRiskScorer(threshold=0.3)
        result = v.validate("https://gooogle.com", {})
        assert isinstance(result, FailResult)
        assert "typosquatting" in result.error_message

    def test_digit_mix_detected(self):
        v = URLRiskScorer(threshold=0.15)
        result = v.validate("https://g00gle.com", {})
        assert isinstance(result, FailResult)

    def test_deep_subdomains(self):
        v = URLRiskScorer(threshold=0.1)
        result = v.validate("https://a.b.c.d.e.example.com", {})
        assert isinstance(result, FailResult)
        assert "subdomain_depth" in result.error_message

    def test_long_domain(self):
        v = URLRiskScorer(threshold=0.05)
        long_name = "a" * 55 + ".com"
        result = v.validate(f"https://{long_name}", {})
        assert isinstance(result, FailResult)
        assert "long_domain" in result.error_message

    def test_combined_risk_signals(self):
        v = URLRiskScorer(threshold=0.3)
        result = v.validate("https://g00gle.tk", {})
        assert isinstance(result, FailResult)

    def test_custom_known_domains(self):
        v = URLRiskScorer(
            threshold=0.3,
            known_domains=["mycompany.com"]
        )
        result = v.validate("https://myconpany.com", {})
        assert isinstance(result, FailResult)

    def test_exact_match_known_domain_passes(self):
        result = self.validator.validate("https://google.com", {})
        assert isinstance(result, PassResult)

    def test_empty_input_fails(self):
        result = self.validator.validate("", {})
        assert isinstance(result, FailResult)

    def test_high_threshold_passes_more(self):
        v = URLRiskScorer(threshold=0.99)
        result = v.validate("https://something-weird.tk", {})
        assert isinstance(result, PassResult)
