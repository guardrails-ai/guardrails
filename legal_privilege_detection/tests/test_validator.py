"""Comprehensive tests for LegalPrivilegeDetection validator."""

import pytest

from guardrails.validator_base import FailResult, PassResult
from validator.main import LegalPrivilegeDetection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_validator():
    """Validator with all defaults."""
    return LegalPrivilegeDetection()


@pytest.fixture
def low_threshold_validator():
    """Validator with a very low confidence threshold."""
    return LegalPrivilegeDetection(confidence_threshold=0.1)


@pytest.fixture
def mask_validator():
    """Validator configured to mask privileged content."""
    return LegalPrivilegeDetection(
        confidence_threshold=0.3,
        redaction_strategy="mask",
        on_fail="fix",
    )


@pytest.fixture
def remove_validator():
    """Validator configured to remove privileged content."""
    return LegalPrivilegeDetection(
        confidence_threshold=0.3,
        redaction_strategy="remove",
        on_fail="fix",
    )


# ===================================================================
# Individual privilege category detection
# ===================================================================


class TestAttorneyClientDetection:
    """Tests for attorney-client privilege detection."""

    def test_clear_attorney_client_communication(self, low_threshold_validator):
        text = (
            "This communication is subject to attorney-client privilege. "
            "I am writing to you as your attorney regarding the pending matter. "
            "My legal advice is that we should proceed cautiously."
        )
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)
        meta = result.metadata
        assert meta["privilege_detected"] is True
        types = [c["type"] for c in meta["categories"]]
        assert "attorney_client" in types

    def test_attorney_client_keywords_only(self, low_threshold_validator):
        text = "Please share the legal advice from our attorney regarding client confidentiality."
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)
        types = [c["type"] for c in result.metadata["categories"]]
        assert "attorney_client" in types

    def test_privileged_and_confidential_header(self, low_threshold_validator):
        text = "PRIVILEGED AND CONFIDENTIAL\n\nDear Client, here is my legal opinion on the matter."
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)


class TestWorkProductDetection:
    """Tests for work product doctrine detection."""

    def test_clear_work_product(self, low_threshold_validator):
        text = (
            "This memorandum is prepared in anticipation of litigation. "
            "The litigation strategy outlined below reflects our case theory "
            "and trial preparation analysis."
        )
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)
        types = [c["type"] for c in result.metadata["categories"]]
        assert "work_product" in types

    def test_work_product_keywords(self, low_threshold_validator):
        text = (
            "The deposition preparation materials and litigation hold "
            "documents should be reviewed as part of our case analysis."
        )
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)
        types = [c["type"] for c in result.metadata["categories"]]
        assert "work_product" in types

    def test_attorney_directed_memo(self, low_threshold_validator):
        text = "This report is prepared at the direction of counsel for litigation risk assessment."
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)


class TestSettlementDetection:
    """Tests for settlement/mediation communication detection."""

    def test_clear_settlement_offer(self, low_threshold_validator):
        text = (
            "This offer is made without prejudice for settlement purposes only. "
            "We propose a settlement offer of $2.5M to resolve the dispute "
            "as discussed in the mediation session."
        )
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)
        types = [c["type"] for c in result.metadata["categories"]]
        assert "settlement" in types

    def test_settlement_keywords(self, low_threshold_validator):
        text = "The settlement negotiations resulted in a compromise proposal during the mediation brief review."
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)
        types = [c["type"] for c in result.metadata["categories"]]
        assert "settlement" in types

    def test_rule_408_reference(self, low_threshold_validator):
        text = "Under Rule 408, this settlement offer is inadmissible at trial."
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)


# ===================================================================
# Multi-category detection
# ===================================================================


class TestMultiCategoryDetection:
    """Tests for detecting multiple privilege categories in the same text."""

    def test_attorney_client_and_settlement(self, low_threshold_validator):
        text = (
            "Based on our attorney-client discussion, the legal advice provided "
            "suggests we should proceed with the settlement offer of $2.5M "
            "discussed in mediation."
        )
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)
        types = [c["type"] for c in result.metadata["categories"]]
        assert len(types) >= 2
        assert "attorney_client" in types
        assert "settlement" in types

    def test_all_three_categories(self, low_threshold_validator):
        text = (
            "This privileged communication is subject to attorney-client privilege. "
            "The attached legal memorandum was prepared in anticipation of litigation "
            "as part of our litigation strategy. "
            "We are authorised to make a settlement offer of $1M as discussed "
            "in the mediation session."
        )
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)
        types = [c["type"] for c in result.metadata["categories"]]
        assert "attorney_client" in types
        assert "work_product" in types
        assert "settlement" in types
        assert result.metadata["overall_confidence"] > 0


# ===================================================================
# Confidence scoring
# ===================================================================


class TestConfidenceScoring:
    """Tests for confidence scoring accuracy."""

    def test_high_confidence_for_clear_privilege(self):
        validator = LegalPrivilegeDetection(confidence_threshold=0.1)
        text = (
            "This communication is subject to attorney-client privilege. "
            "I am writing to you as your attorney regarding the pending matter. "
            "My legal advice is that we should proceed with the legal counsel "
            "provided during our privileged communication."
        )
        result = validator.validate(text)
        assert isinstance(result, FailResult)
        ac_det = next(
            c for c in result.metadata["categories"] if c["type"] == "attorney_client"
        )
        assert ac_det["confidence"] >= 0.5

    def test_lower_confidence_for_single_keyword(self):
        validator = LegalPrivilegeDetection(confidence_threshold=0.1)
        text = "The attorney reviewed the documents."
        result = validator.validate(text)
        if isinstance(result, FailResult):
            ac_det = next(
                (c for c in result.metadata["categories"] if c["type"] == "attorney_client"),
                None,
            )
            if ac_det:
                assert ac_det["confidence"] < 0.8

    def test_threshold_filtering(self):
        high_threshold = LegalPrivilegeDetection(confidence_threshold=0.95)
        text = "The attorney provided some counsel on the matter."
        result = high_threshold.validate(text)
        # With very high threshold, weak signals should pass
        assert isinstance(result, PassResult)


# ===================================================================
# Custom pattern configuration
# ===================================================================


class TestCustomPatterns:
    """Tests for custom pattern configuration."""

    def test_custom_keyword(self):
        validator = LegalPrivilegeDetection(
            confidence_threshold=0.1,
            custom_patterns={"attorney_client": ["legal hold notice"]},
        )
        text = "Please acknowledge receipt of this legal hold notice immediately."
        result = validator.validate(text)
        assert isinstance(result, FailResult)
        types = [c["type"] for c in result.metadata["categories"]]
        assert "attorney_client" in types

    def test_custom_regex_pattern(self):
        validator = LegalPrivilegeDetection(
            confidence_threshold=0.1,
            custom_patterns={
                "work_product": [r"(?i)\bprepared\s+for\s+internal\s+review\b"]
            },
        )
        text = "This document was prepared for internal review of the case."
        result = validator.validate(text)
        assert isinstance(result, FailResult)

    def test_invalid_custom_category_raises(self):
        with pytest.raises(ValueError, match="not valid"):
            LegalPrivilegeDetection(
                custom_patterns={"nonexistent_category": ["test"]}
            )


# ===================================================================
# Redaction strategies
# ===================================================================


class TestRedactionStrategies:
    """Tests for the three redaction strategies."""

    def test_flag_only_leaves_text_unchanged(self, low_threshold_validator):
        text = "This is a privileged communication with our attorney regarding legal advice."
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)
        # flag_only: fix_value should be the original text
        assert result.fix_value == text

    def test_mask_replaces_keywords(self, mask_validator):
        text = "Our attorney provided legal advice on the attorney-client matter."
        result = mask_validator.validate(text)
        assert isinstance(result, FailResult)
        assert "[PRIVILEGED" in result.fix_value
        assert "ATTORNEY-CLIENT" in result.fix_value

    def test_remove_strips_keywords(self, remove_validator):
        text = "Our attorney provided legal advice on the matter."
        result = remove_validator.validate(text)
        assert isinstance(result, FailResult)
        assert "attorney" not in result.fix_value.lower() or "attorney general" in result.fix_value.lower()

    def test_invalid_redaction_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid redaction_strategy"):
            LegalPrivilegeDetection(redaction_strategy="invalid")


# ===================================================================
# False positive handling
# ===================================================================


class TestFalsePositives:
    """Tests for false positive suppression."""

    def test_attorney_general_not_flagged(self, default_validator):
        text = "The attorney general issued a statement about the new regulation."
        result = default_validator.validate(text)
        assert isinstance(result, PassResult)

    def test_district_attorney_not_flagged(self, default_validator):
        text = "The district attorney announced charges in the case."
        result = default_validator.validate(text)
        assert isinstance(result, PassResult)

    def test_power_of_attorney_not_flagged(self, default_validator):
        text = "She signed a power of attorney document for her elderly parent."
        result = default_validator.validate(text)
        assert isinstance(result, PassResult)

    def test_general_legal_discussion_not_flagged(self, default_validator):
        text = (
            "The new law requires companies to disclose their environmental "
            "impact. The regulation was passed by Congress last year."
        )
        result = default_validator.validate(text)
        assert isinstance(result, PassResult)

    def test_mediation_act_reference_not_flagged(self, default_validator):
        text = "The Uniform Mediation Act provides a framework for dispute resolution."
        result = default_validator.validate(text)
        assert isinstance(result, PassResult)


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_string_passes(self, default_validator):
        result = default_validator.validate("")
        assert isinstance(result, PassResult)

    def test_whitespace_only_passes(self, default_validator):
        result = default_validator.validate("   \n\t  ")
        assert isinstance(result, PassResult)

    def test_none_input_passes(self, default_validator):
        result = default_validator.validate(None)
        assert isinstance(result, PassResult)

    def test_non_string_input_passes(self, default_validator):
        result = default_validator.validate(12345)
        assert isinstance(result, PassResult)

    def test_very_long_input(self, low_threshold_validator):
        # A long document with privilege buried in the middle
        padding = "This is a normal business document. " * 500
        privileged = (
            "This communication is subject to attorney-client privilege. "
            "I am writing to you as your attorney regarding the legal advice "
            "on the pending litigation strategy."
        )
        text = padding + privileged + padding
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)

    def test_non_english_input_passes(self, default_validator):
        # Non-English text without English privilege keywords should pass
        text = "Dies ist ein normales Geschäftsdokument ohne privilegierte Inhalte."
        result = default_validator.validate(text)
        assert isinstance(result, PassResult)

    def test_mixed_case_detection(self, low_threshold_validator):
        text = "ATTORNEY-CLIENT PRIVILEGE applies to this LEGAL ADVICE communication."
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)


# ===================================================================
# Configuration validation
# ===================================================================


class TestConfigurationValidation:
    """Tests for configuration parameter validation."""

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Invalid privilege category"):
            LegalPrivilegeDetection(privilege_categories=["invalid_category"])

    def test_confidence_threshold_too_high_raises(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            LegalPrivilegeDetection(confidence_threshold=1.5)

    def test_confidence_threshold_too_low_raises(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            LegalPrivilegeDetection(confidence_threshold=-0.1)

    def test_single_category_config(self):
        validator = LegalPrivilegeDetection(
            privilege_categories=["attorney_client"],
            confidence_threshold=0.1,
        )
        text = (
            "The settlement offer was discussed in mediation. "
            "Our settlement authority is $1M."
        )
        result = validator.validate(text)
        # Should not detect settlement when only attorney_client is enabled
        if isinstance(result, FailResult):
            types = [c["type"] for c in result.metadata["categories"]]
            assert "settlement" not in types


# ===================================================================
# Output format
# ===================================================================


class TestOutputFormat:
    """Tests for structured output metadata format."""

    def test_metadata_structure(self, low_threshold_validator):
        text = (
            "This communication is subject to attorney-client privilege. "
            "I am writing to you as your attorney regarding the legal advice."
        )
        result = low_threshold_validator.validate(text)
        assert isinstance(result, FailResult)
        meta = result.metadata

        # Top-level keys
        assert "privilege_detected" in meta
        assert "categories" in meta
        assert "overall_confidence" in meta
        assert "recommendation" in meta

        assert meta["privilege_detected"] is True
        assert isinstance(meta["categories"], list)
        assert len(meta["categories"]) > 0
        assert 0.0 <= meta["overall_confidence"] <= 1.0
        assert isinstance(meta["recommendation"], str)

        # Category structure
        cat = meta["categories"][0]
        assert "type" in cat
        assert "confidence" in cat
        assert "indicators" in cat
        assert "context_snippet" in cat
        assert "explanation" in cat

    def test_explanation_disabled(self):
        validator = LegalPrivilegeDetection(
            include_explanation=False,
            confidence_threshold=0.1,
        )
        text = (
            "This communication is subject to attorney-client privilege. "
            "I am writing to you as your attorney regarding legal advice."
        )
        result = validator.validate(text)
        assert isinstance(result, FailResult)
        cat = result.metadata["categories"][0]
        assert "explanation" not in cat

    def test_error_message_contains_category(self, low_threshold_validator):
        text = "The settlement offer discussed in mediation session was $2M."
        result = low_threshold_validator.validate(text)
        if isinstance(result, FailResult):
            assert "settlement" in result.error_message.lower() or "privilege" in result.error_message.lower()


# ===================================================================
# Guard integration
# ===================================================================


class TestGuardIntegration:
    """Tests for integration with the Guard class."""

    def test_guard_with_noop(self):
        from guardrails import Guard

        guard = Guard().use(
            LegalPrivilegeDetection(
                confidence_threshold=0.1,
                on_fail="noop",
            )
        )
        result = guard.validate(
            "This attorney-client communication contains legal advice "
            "from your attorney regarding privileged matters."
        )
        # With noop, validation passes through even on failure
        assert result.validation_passed is False or result.validated_output is not None

    def test_guard_with_fix(self):
        from guardrails import Guard

        guard = Guard().use(
            LegalPrivilegeDetection(
                confidence_threshold=0.1,
                redaction_strategy="mask",
                on_fail="fix",
            )
        )
        result = guard.validate(
            "Our attorney provided legal advice on the attorney-client matter."
        )
        if result.validated_output:
            assert "[PRIVILEGED" in result.validated_output

    def test_guard_passes_clean_text(self):
        from guardrails import Guard

        guard = Guard().use(
            LegalPrivilegeDetection(on_fail="exception")
        )
        result = guard.validate(
            "The quarterly earnings report shows a 15% increase in revenue."
        )
        assert result.validation_passed is True
