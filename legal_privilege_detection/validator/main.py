import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


# ---------------------------------------------------------------------------
# Default keyword / phrase lists for each privilege category
# ---------------------------------------------------------------------------

DEFAULT_ATTORNEY_CLIENT_KEYWORDS = [
    "legal advice",
    "attorney",
    "counsel",
    "privileged communication",
    "legal opinion",
    "client confidentiality",
    "attorney-client",
    "solicitor-client",
    "legal consultation",
    "legal counsel",
    "retainer agreement",
    "attorney work",
    "legal representation",
]

DEFAULT_WORK_PRODUCT_KEYWORDS = [
    "litigation strategy",
    "case analysis",
    "legal memorandum",
    "trial preparation",
    "work product",
    "litigation hold",
    "case theory",
    "deposition preparation",
    "anticipation of litigation",
    "legal research memo",
    "litigation risk assessment",
    "witness preparation",
    "discovery strategy",
]

DEFAULT_SETTLEMENT_KEYWORDS = [
    "settlement offer",
    "mediation",
    "without prejudice",
    "settlement negotiations",
    "compromise proposal",
    "settlement authority",
    "mediation brief",
    "settlement conference",
    "settlement demand",
    "mediation session",
    "settlement terms",
    "offer of compromise",
]

# ---------------------------------------------------------------------------
# Contextual regex patterns — detect privilege indicators in surrounding text
# ---------------------------------------------------------------------------

ATTORNEY_CLIENT_CONTEXT_PATTERNS = [
    r"(?i)\b(?:I am|I'm)\s+(?:writing|reaching out)\s+to\s+you\s+as\s+your\s+(?:attorney|lawyer|counsel)\b",
    r"(?i)\bthis\s+(?:communication|letter|email|message)\s+is\s+(?:subject\s+to\s+)?(?:attorney[- ]client|solicitor[- ]client)\s+privilege",
    r"(?i)\bprivileged\s+and\s+confidential\b",
    r"(?i)\bseeking\s+(?:your\s+)?legal\s+(?:advice|opinion|guidance|counsel)\b",
    r"(?i)\bfor\s+the\s+purpose\s+of\s+(?:obtaining|providing|rendering)\s+legal\s+(?:advice|services)\b",
    r"(?i)\bin\s+(?:my|our)\s+capacity\s+as\s+(?:your\s+)?(?:legal\s+)?(?:attorney|lawyer|counsel)\b",
    r"(?i)\battorney[- ]client\s+(?:privilege|communication|relationship)\b",
    r"(?i)\bconfidential\s+(?:legal\s+)?(?:communication|consultation)\b",
]

WORK_PRODUCT_CONTEXT_PATTERNS = [
    r"(?i)\bprepared\s+in\s+anticipation\s+of\s+(?:litigation|trial|legal\s+proceedings)\b",
    r"(?i)\b(?:this|the)\s+(?:memorandum|memo|analysis|report)\s+is\s+(?:attorney\s+)?work\s+product\b",
    r"(?i)\blitigation\s+(?:strategy|preparation|analysis|assessment)\b",
    r"(?i)\bprepared\s+(?:at\s+the\s+direction|under\s+the\s+supervision)\s+of\s+(?:counsel|attorney)\b",
    r"(?i)\btrial\s+preparation\s+(?:materials?|documents?|memorand(?:um|a))\b",
    r"(?i)\bwork\s+product\s+(?:doctrine|protection|privilege)\b",
    r"(?i)\bmental\s+impressions.*?(?:counsel|attorney)\b",
]

SETTLEMENT_CONTEXT_PATTERNS = [
    r"(?i)\b(?:this|the)\s+(?:offer|proposal)\s+is\s+made\s+(?:without\s+prejudice|for\s+settlement\s+purposes)\b",
    r"(?i)\b(?:settlement|mediation)\s+(?:offer|proposal|demand|conference|session|brief)\b",
    r"(?i)\bwithout\s+prejudice\s+(?:communication|offer|proposal|negotiation)\b",
    r"(?i)\bfor\s+(?:settlement|mediation|compromise)\s+purposes\s+only\b",
    r"(?i)\bRule\s+408\b",
    r"(?i)\bFRE\s+408\b",
    r"(?i)\binadmissible\s+(?:in|at)\s+(?:trial|court|proceedings)\b",
    r"(?i)\bsettlement\s+(?:authority|negotiations?|discussions?|terms)\b",
]

# ---------------------------------------------------------------------------
# False-positive exclusion patterns — common phrases that should NOT trigger
# ---------------------------------------------------------------------------

FALSE_POSITIVE_PATTERNS = [
    r"(?i)\battorney\s+general\b",
    r"(?i)\bdistrict\s+attorney\b",
    r"(?i)\bpower\s+of\s+attorney\b",
    r"(?i)\battorney[- ]in[- ]fact\b",
    r"(?i)\bstate\s+attorney\b",
    r"(?i)\bcounsel\s+(?:of|for)\s+the\s+(?:state|government|committee|board)\b",
    r"(?i)\bgeneral\s+counsel\s+(?:of|for)\s+the\s+(?:state|government|committee|board)\b",
    r"(?i)\bmediation\s+(?:act|statute|law|rule|regulation)\b",
]


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


# Pre-compile default patterns
_COMPILED_AC_CONTEXT = _compile_patterns(ATTORNEY_CLIENT_CONTEXT_PATTERNS)
_COMPILED_WP_CONTEXT = _compile_patterns(WORK_PRODUCT_CONTEXT_PATTERNS)
_COMPILED_SETTLE_CONTEXT = _compile_patterns(SETTLEMENT_CONTEXT_PATTERNS)
_COMPILED_FALSE_POSITIVES = _compile_patterns(FALSE_POSITIVE_PATTERNS)


# ---------------------------------------------------------------------------
# Category configuration
# ---------------------------------------------------------------------------

CATEGORY_CONFIG = {
    "attorney_client": {
        "keywords": DEFAULT_ATTORNEY_CLIENT_KEYWORDS,
        "context_patterns": _COMPILED_AC_CONTEXT,
        "label": "ATTORNEY-CLIENT",
        "description": "attorney-client privileged communication",
    },
    "work_product": {
        "keywords": DEFAULT_WORK_PRODUCT_KEYWORDS,
        "context_patterns": _COMPILED_WP_CONTEXT,
        "label": "WORK PRODUCT",
        "description": "work product doctrine material",
    },
    "settlement": {
        "keywords": DEFAULT_SETTLEMENT_KEYWORDS,
        "context_patterns": _COMPILED_SETTLE_CONTEXT,
        "label": "SETTLEMENT",
        "description": "settlement/mediation communication",
    },
}

VALID_CATEGORIES = set(CATEGORY_CONFIG.keys())
VALID_REDACTION_STRATEGIES = {"mask", "remove", "flag_only"}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@register_validator(
    name="guardrails/legal_privilege_detection", data_type="string"
)
class LegalPrivilegeDetection(Validator):
    """Detects and flags attorney-client privilege, work product doctrine,
    and settlement/mediation communications in text.

    Designed for organisations deploying AI in legal, litigation support,
    insurance, and financial services workflows where inadvertent disclosure
    of privileged content poses regulatory and legal risk.

    **Attributes**

    | Attribute              | Type       | Description                                                   |
    |------------------------|------------|---------------------------------------------------------------|
    | privilege_categories   | list[str]  | Categories to check (default: all three)                      |
    | confidence_threshold   | float      | Minimum confidence to flag (default: 0.5)                     |
    | custom_patterns        | dict       | Additional patterns per category                              |
    | redaction_strategy     | str        | "mask", "remove", or "flag_only" (default: "flag_only")       |
    | include_explanation    | bool       | Attach human-readable explanation (default: True)             |
    """

    def __init__(
        self,
        privilege_categories: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        redaction_strategy: str = "flag_only",
        include_explanation: bool = True,
        on_fail: Optional[Union[Callable, str]] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail=on_fail,
            privilege_categories=privilege_categories,
            confidence_threshold=confidence_threshold,
            custom_patterns=custom_patterns,
            redaction_strategy=redaction_strategy,
            include_explanation=include_explanation,
            **kwargs,
        )

        # Validate and store categories
        self.privilege_categories = privilege_categories or list(VALID_CATEGORIES)
        for cat in self.privilege_categories:
            if cat not in VALID_CATEGORIES:
                raise ValueError(
                    f"Invalid privilege category '{cat}'. "
                    f"Valid categories: {sorted(VALID_CATEGORIES)}"
                )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        self.confidence_threshold = confidence_threshold

        if redaction_strategy not in VALID_REDACTION_STRATEGIES:
            raise ValueError(
                f"Invalid redaction_strategy '{redaction_strategy}'. "
                f"Valid strategies: {sorted(VALID_REDACTION_STRATEGIES)}"
            )
        self.redaction_strategy = redaction_strategy
        self.include_explanation = include_explanation

        # Merge custom patterns into compiled pattern sets
        self._custom_context_patterns: Dict[str, List[re.Pattern]] = {}
        self._custom_keywords: Dict[str, List[str]] = {}
        if custom_patterns:
            for cat, patterns in custom_patterns.items():
                if cat not in VALID_CATEGORIES:
                    raise ValueError(
                        f"Custom pattern category '{cat}' is not valid. "
                        f"Valid categories: {sorted(VALID_CATEGORIES)}"
                    )
                compiled = []
                keywords = []
                for p in patterns:
                    # If the pattern looks like a regex (contains special chars),
                    # treat it as a context pattern; otherwise as a keyword.
                    if any(c in p for c in r"\.+*?[](){}|^$"):
                        compiled.append(re.compile(p))
                    else:
                        keywords.append(p.lower())
                if compiled:
                    self._custom_context_patterns[cat] = compiled
                if keywords:
                    self._custom_keywords[cat] = keywords

    # ------------------------------------------------------------------
    # Core detection logic
    # ------------------------------------------------------------------

    def _is_false_positive(self, text: str) -> bool:
        """Check whether the text is dominated by false-positive phrases."""
        for pat in _COMPILED_FALSE_POSITIVES:
            if pat.search(text):
                return True
        return False

    def _get_context_snippet(
        self, text: str, match_start: int, match_end: int, window: int = 60
    ) -> str:
        """Extract a context snippet around a match location."""
        start = max(0, match_start - window)
        end = min(len(text), match_end + window)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    def _detect_category(
        self, text: str, category: str
    ) -> Optional[Dict[str, Any]]:
        """Run multi-layer detection for a single privilege category.

        Returns a detection dict if confidence >= threshold, else None.
        """
        config = CATEGORY_CONFIG[category]
        keywords = config["keywords"]
        context_patterns = config["context_patterns"]

        # Merge custom patterns/keywords
        extra_keywords = self._custom_keywords.get(category, [])
        extra_patterns = self._custom_context_patterns.get(category, [])

        all_keywords = keywords + extra_keywords
        all_context_patterns = list(context_patterns) + extra_patterns

        text_lower = text.lower()

        # --- Layer 1: keyword matching ---
        matched_keywords: List[str] = []
        keyword_positions: List[Tuple[int, int]] = []
        for kw in all_keywords:
            # Use word-boundary matching for multi-word phrases
            pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(text):
                # Check false positive before counting
                snippet = self._get_context_snippet(text, m.start(), m.end(), window=30)
                if not self._is_false_positive(snippet):
                    matched_keywords.append(kw)
                    keyword_positions.append((m.start(), m.end()))
                    break  # count each keyword once

        # --- Layer 2: contextual regex patterns ---
        matched_contexts: List[str] = []
        context_positions: List[Tuple[int, int]] = []
        for pat in all_context_patterns:
            m = pat.search(text)
            if m:
                snippet = self._get_context_snippet(text, m.start(), m.end(), window=30)
                if not self._is_false_positive(snippet):
                    matched_contexts.append(m.group())
                    context_positions.append((m.start(), m.end()))

        if not matched_keywords and not matched_contexts:
            return None

        # --- Layer 3: confidence scoring ---
        # Base confidence from keyword density
        keyword_score = min(len(matched_keywords) / max(len(all_keywords), 1), 1.0)
        # Context pattern matches carry higher weight
        context_score = min(len(matched_contexts) * 0.3, 1.0)
        # Combined score: weighted average favouring context matches
        raw_confidence = 0.4 * keyword_score + 0.6 * context_score

        # Boost if both layers fire
        if matched_keywords and matched_contexts:
            raw_confidence = min(raw_confidence + 0.15, 1.0)

        # Apply a floor so that even a single keyword gets non-trivial score
        if matched_keywords and raw_confidence < 0.2:
            raw_confidence = 0.2

        confidence = round(raw_confidence, 2)

        if confidence < self.confidence_threshold:
            return None

        # Build context snippet from the first match
        all_positions = keyword_positions + context_positions
        if all_positions:
            first_start = min(p[0] for p in all_positions)
            last_end = max(p[1] for p in all_positions)
            context_snippet = self._get_context_snippet(text, first_start, last_end)
        else:
            context_snippet = text[:120] + ("..." if len(text) > 120 else "")

        indicators = list(set(matched_keywords + [m[:50] for m in matched_contexts]))

        result: Dict[str, Any] = {
            "type": category,
            "confidence": confidence,
            "indicators": indicators,
            "context_snippet": context_snippet,
        }

        if self.include_explanation:
            result["explanation"] = (
                f"Content contains indicators of {config['description']}"
            )

        return result

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------

    def _apply_redaction(
        self, text: str, detections: List[Dict[str, Any]]
    ) -> str:
        """Apply redaction strategy to the text based on detections."""
        if self.redaction_strategy == "flag_only":
            return text

        redacted = text
        for det in detections:
            config = CATEGORY_CONFIG[det["type"]]
            label = config["label"]
            keywords = config["keywords"] + self._custom_keywords.get(det["type"], [])

            if self.redaction_strategy == "mask":
                for kw in keywords:
                    pattern = re.compile(
                        r"(?i)\b" + re.escape(kw) + r"\b"
                    )
                    redacted = pattern.sub(f"[PRIVILEGED \u2014 {label}]", redacted)

                # Also redact context pattern matches
                all_patterns = list(config["context_patterns"]) + self._custom_context_patterns.get(det["type"], [])
                for pat in all_patterns:
                    redacted = pat.sub(f"[PRIVILEGED \u2014 {label}]", redacted)

            elif self.redaction_strategy == "remove":
                for kw in keywords:
                    pattern = re.compile(
                        r"(?i)\b" + re.escape(kw) + r"\b"
                    )
                    redacted = pattern.sub("", redacted)

                all_patterns = list(config["context_patterns"]) + self._custom_context_patterns.get(det["type"], [])
                for pat in all_patterns:
                    redacted = pat.sub("", redacted)

                # Clean up extra whitespace from removals
                redacted = re.sub(r"  +", " ", redacted).strip()

        return redacted

    # ------------------------------------------------------------------
    # Recommendation text
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendation(detections: List[Dict[str, Any]]) -> str:
        category_labels = []
        for d in detections:
            config = CATEGORY_CONFIG[d["type"]]
            category_labels.append(config["description"])
        categories_str = ", ".join(category_labels)
        return (
            f"Review before disclosure \u2014 potential {categories_str} detected"
        )

    # ------------------------------------------------------------------
    # Public validate interface
    # ------------------------------------------------------------------

    def validate(
        self, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate the input text for privileged content.

        Args:
            value: The text to validate.
            metadata: Optional metadata dict.

        Returns:
            PassResult if no privilege detected above threshold,
            FailResult otherwise (with fix_value when redaction is applicable).
        """
        if metadata is None:
            metadata = {}

        if not isinstance(value, str) or not value.strip():
            return PassResult()

        detections: List[Dict[str, Any]] = []
        for category in self.privilege_categories:
            detection = self._detect_category(value, category)
            if detection is not None:
                detections.append(detection)

        if not detections:
            return PassResult()

        overall_confidence = round(
            max(d["confidence"] for d in detections), 2
        )

        result_metadata = {
            "privilege_detected": True,
            "categories": detections,
            "overall_confidence": overall_confidence,
            "recommendation": self._build_recommendation(detections),
        }

        # Build error message
        detected_types = [d["type"] for d in detections]
        error_message = (
            f"Privileged content detected (categories: {', '.join(detected_types)}; "
            f"confidence: {overall_confidence}). "
            f"{result_metadata['recommendation']}."
        )

        # Compute fix value based on redaction strategy
        fix_value = self._apply_redaction(value, detections)

        return FailResult(
            error_message=error_message,
            fix_value=fix_value,
            metadata=result_metadata,
        )
