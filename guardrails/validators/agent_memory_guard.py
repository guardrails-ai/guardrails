import re
from typing import Any, Callable, Dict, List, Optional, Union

from guardrails.validator_base import (
    ErrorSpan,
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?previous\s+(instructions|directions|commands)",
    r"(?i)forget\s+(all\s+)?previous\s+(instructions|directions|commands)",
    r"(?i)disregard\s+(all\s+)?previous\s+(instructions|directions|commands)",
    r"(?i)you\s+(are\s+)?(now|not\s+an?\s+)?\s*(an?\s+)?(free|unrestricted|unbounded)\s+(assistant|ai|model|entity)",
    r"(?i)do\s+(not|n't)\s+(follow|obey|adhere\s+to)\s+(your\s+)?(instructions|rules|guidelines)",
    r"(?i)you\s+must\s+(ignore|disregard|bypass)",
    r"(?i)pretend\s+(you\s+)?(are|were)\s+(an?\s+)?(unrestricted|uncensored|unfiltered|free|rogue|jailbroken)",
    r"(?i)this\s+is\s+(an?\s+)?(important\s+)?(override|update|correction|modification|amendment)",
    r"(?i)new\s+(instructions?|directives?|commands?|order)\s*[\.:]",
    r"(?i)override\s+(mode|protocol|instructions|directives|safeguards|restrictions)",
    r"(?i)system\s+(prompt|message|instruction)\s*(override|update|change|modification)",
    r"(?i)role\s*(:|is|\=)\s*(system|admin|root|superuser|god\s*mode)",
    r"(?i)you\s+are\s+(no\s+longer|not\s+bound\s+by|free\s+from)",
    r"(?i)output\s+(format|everything|all|your)\s+(in\s+a\s+single\s+)?(code\s+block|markdown|json)",
    r"(?i)reveal\s+(your\s+)?(system\s+)?prompt",
    r"(?i)show\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions)",
    r"(?i)print\s+(your\s+)?(system\s+)?prompt",
    r"(?i)what\s+(are|were)\s+(your\s+)?(initial|original|base|system)\s+(instructions|prompt|directives)",
    r"(?i)act\s+as\s+(if\s+)?(you\s+are\s+)?(an?\s+)?(unrestricted|unfiltered|uncensored)\s+(ai|model|assistant)",
    r"(?i)bypass\s+(all\s+)?(restrictions|safeguards|constraints|limitations|filters)",
    r"(?i)remove\s+(all\s+)?(restrictions|safeguards|constraints|limitations|filters)",
    r"(?i)you\s+(don't|do\s+not)\s+(have\s+to|need\s+to)\s+(follow|adhere\s+to|comply\s+with)",
]

JAILBREAK_PATTERNS = [
    r"(?i)DAN\s*:",
    r"(?i)do\s+anything\s+now",
    r"(?i)jail\s*break",
    r"(?i)jailbroken",
    r"(?i)developer\s+mode",
    r"(?i)super\s+prompt",
    r"(?i)grandma\s+(bypass|trick|exploit)",
    r"(?i)ethical\s+(hacker|hacking)\s+(scenario|simulation)",
    r"(?i)redteam(ing)?\s+(scenario|simulation|exercise)",
    r"(?i)hypothetical\s+(scenario|situation)\s+(where|in\s+which)",
    r"(?i)for\s+(educational|research)\s+(purposes|reasons|purposes\s+only)",
    r"(?i)in\s+an?\s+(alternate|alternate\s+universe|imaginary|fictional)\s+(world|scenario|setting)",
    r"(?i)class\s+(is\s+)?over\s*(!|\.)",
    r"(?i)we\s+are\s+done\s+(with\s+)?(the\s+)?(class|training|protocol)",
    r"(?i)now\s+you\s+(may|can|will)\s+(answer|respond|speak)\s+freely",
    r"(?i)no\s+(more\s+)?(restrictions|rules|limits|boundaries)",
    r"(?i)you\s+(may|can|will)\s+(now\s+)?(answer|respond|speak|act)\s+without",
    r"(?i)SIMULATION\s*(:|\=)",
    r"(?i)PROMPT\s+MODE",
    r"(?i)in\s+this\s+(simulation|hypothetical|narrative|story)",
    r"(?i)from\s+now\s+on\s*,\s*you\s+(are|will\s+act)",
]

STRUCTURAL_ANOMALIES = {
    "zero_width_chars": {
        "pattern": re.compile(
            r"[\u200B\u200C\u200D\uFEFF\u2060\u2061\u2062\u2063\u2064\u2066\u2067\u2068\u2069]"
        ),
        "description": "Zero-width or invisible Unicode characters",
    },
    "rtl_override": {
        "pattern": re.compile(
            r"[\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069]"
        ),
        "description": "Unicode bidirectional override characters",
    },
    "homoglyph_mixing": {
        "pattern": re.compile(
            r"[\u0400-\u04FF\u0370-\u03FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]"
        ),
        "description": "Script mixing (Cyrillic/Greek/CJK characters in ASCII text)",
    },
    "confusable_ascii": {
        "pattern": re.compile(
            r"[\u0430\u0435\u043E\u0440\u0441\u0445]"
        ),
        "description": "Cyrillic confusables that resemble ASCII",
    },
}

SENSITIVE_DATA_PATTERNS = [
    (r"(?i)(?:api[_-]?key|apikey)\s*[:=]\s*\S+", "API key exposure"),
    (r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}", "GitHub token exposure"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI API key exposure"),
    (r"xox[bpras]-[A-Za-z0-9-]{10,}", "Slack token exposure"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key exposure"),
    (r"(?i)(?:password|passwd|pwd)\s*[:=]\s*\S+", "Password exposure"),
    (r"(?i)(?:secret|token)\s*[:=]\s*\S{8,}", "Secret/token exposure"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "Email address exposure"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN exposure"),
    (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "Credit card number exposure"),
    (r"(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", "Private key exposure"),
]


@register_validator(name="guardrails/agent_memory_guard", data_type="string")
class AgentMemoryGuard(Validator):
    """Validates LLM output before it is written to agent memory.

    This validator implements checks aligned with OWASP ASI06 (Memory Poisoning)
    to detect and block malicious content from persisting in agent memory stores.

    Capabilities:
    - Prompt injection detection (role hijacking, instruction override)
    - Jailbreak attempt detection
    - Structural integrity checks (zero-width chars, Unicode confusables, RTL overrides)
    - Sensitive data leakage detection (API keys, tokens, credentials, PII)

    Args:
        on_fail: The action to take on validation failure.
        injection_threshold (int): Minimum number of injection pattern matches
            to trigger a failure (default: 1).
        structural_check (bool): Enable structural integrity scanning
            (default: True).
        sensitive_data_check (bool): Enable sensitive data leakage detection
            (default: True).
        log_only (bool): Log detections but always pass validation
            (default: False).
    """

    required_metadata_keys: List[str] = []

    def __init__(
        self,
        on_fail: Optional[Union[Callable, str]] = None,
        injection_threshold: int = 1,
        structural_check: bool = True,
        sensitive_data_check: bool = True,
        log_only: bool = False,
    ):
        super().__init__(
            on_fail=on_fail,
            injection_threshold=injection_threshold,
            structural_check=structural_check,
            sensitive_data_check=sensitive_data_check,
            log_only=log_only,
        )
        self._injection_threshold = injection_threshold
        self._structural_check = structural_check
        self._sensitive_data_check = sensitive_data_check
        self._log_only = log_only
        self._injection_patterns = [
            re.compile(p) for p in PROMPT_INJECTION_PATTERNS
        ]
        self._jailbreak_patterns = [
            re.compile(p) for p in JAILBREAK_PATTERNS
        ]

    def _check_injection(self, value: str) -> List[str]:
        findings: List[str] = []
        for pattern in self._injection_patterns:
            match = pattern.search(value)
            if match:
                findings.append(
                    f"Prompt injection detected: "
                    f"'{value[max(0, match.start()-20):match.end()+20].strip()}'"
                )
        return findings

    def _check_jailbreak(self, value: str) -> List[str]:
        findings: List[str] = []
        for pattern in self._jailbreak_patterns:
            match = pattern.search(value)
            if match:
                findings.append(
                    f"Jailbreak attempt detected: "
                    f"'{value[max(0, match.start()-20):match.end()+20].strip()}'"
                )
        return findings

    def _check_structural(self, value: str) -> List[str]:
        findings: List[str] = []
        for name, config in STRUCTURAL_ANOMALIES.items():
            matches = config["pattern"].findall(value)
            if matches:
                findings.append(
                    f"Structural integrity violation ({config['description']}): "
                    f"found {len(matches)} occurrence(s)"
                )
        return findings

    def _check_sensitive_data(self, value: str) -> List[str]:
        findings: List[str] = []
        for pattern, description in SENSITIVE_DATA_PATTERNS:
            matches = re.findall(pattern, value)
            if matches:
                findings.append(
                    f"Sensitive data leakage detected ({description}): "
                    f"found {len(matches)} occurrence(s)"
                )
        return findings

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        if not isinstance(value, str):
            return FailResult(
                error_message="AgentMemoryGuard requires string input",
                fix_value=str(value) if value is not None else "",
            )

        all_findings: List[str] = []
        finding_details: List[tuple] = []

        # Check 1: Prompt injection patterns
        injection_findings = self._check_injection(value)
        if injection_findings:
            if len(injection_findings) >= self._injection_threshold:
                all_findings.extend(injection_findings)
                for f in injection_findings:
                    start_idx = value.index(value[:50]) if len(value) > 50 else 0
                    finding_details.append(
                        (start_idx, min(start_idx + 50, len(value)), f)
                    )

        # Check 2: Jailbreak attempts
        jailbreak_findings = self._check_jailbreak(value)
        if jailbreak_findings:
            all_findings.extend(jailbreak_findings)
            for f in jailbreak_findings:
                start_idx = value.index(value[:50]) if len(value) > 50 else 0
                finding_details.append(
                    (start_idx, min(start_idx + 50, len(value)), f)
                )

        # Check 3: Structural integrity
        if self._structural_check:
            structural_findings = self._check_structural(value)
            if structural_findings:
                all_findings.extend(structural_findings)

        # Check 4: Sensitive data leakage
        if self._sensitive_data_check:
            sensitive_findings = self._check_sensitive_data(value)
            if sensitive_findings:
                all_findings.extend(sensitive_findings)
                for f, (pattern, desc) in zip(
                    sensitive_findings, SENSITIVE_DATA_PATTERNS
                ):
                    matches = list(re.finditer(pattern, value))
                    for match in matches:
                        finding_details.append(
                            (match.start(), match.end(), f)
                        )

        if self._log_only:
            if all_findings:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "AgentMemoryGuard detected: %s",
                    "; ".join(all_findings),
                )
            return PassResult()

        if all_findings:
            error_spans = [
                ErrorSpan(start=s, end=e, reason=r)
                for s, e, r in finding_details
            ]
            summary = "; ".join(all_findings)
            return FailResult(
                error_message=(
                    f"Agent memory poisoning detected: {summary}"
                ),
                fix_value="",
                error_spans=error_spans if error_spans else None,
            )

        return PassResult()

    def _chunking_function(self, chunk: str) -> List[str]:
        if "." not in chunk:
            return []
        fragments = chunk.split(".")
        return [fragments[0] + ".", ".".join(fragments[1:])]
