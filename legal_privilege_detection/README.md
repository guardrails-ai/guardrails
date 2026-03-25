# LegalPrivilegeDetection

A [Guardrails AI](https://github.com/guardrails-ai/guardrails) validator that detects attorney-client privilege, work product doctrine, and settlement/mediation communications in LLM outputs.

Designed for organisations deploying AI in legal, litigation support, insurance, and financial services workflows where inadvertent disclosure of privileged content poses regulatory and legal risk.

## Why This Matters

AI systems are increasingly used in regulated industries for document review, claims processing, and legal research. Recent case law (*United States v. Heppner*, S.D.N.Y. 2026) has established that AI-generated content may not be protected by attorney-client privilege when created using platforms that permit third-party data disclosure. This creates urgent demand for runtime guardrails that prevent AI systems from inadvertently disclosing privileged information.

This validator aligns with the **NIST AI Risk Management Framework (AI RMF)** MANAGE function — specifically MG-2.6 (mechanisms for tracking and responding to known and emergent AI risks) and MG-3.2 (pre-deployment and ongoing testing for risks) — by providing configurable runtime risk detection and mitigation.

## Installation

```bash
pip install legal-privilege-detection
```

Or install from source:

```bash
git clone https://github.com/guardrails-ai/legal-privilege-detection.git
cd legal-privilege-detection
pip install -e ".[dev]"
```

## Quick Start

```python
from guardrails import Guard
from validator import LegalPrivilegeDetection

guard = Guard().use(
    LegalPrivilegeDetection(
        privilege_categories=["attorney_client", "work_product", "settlement"],
        confidence_threshold=0.5,
        redaction_strategy="flag_only",
        on_fail="exception",
    )
)

# Clean text passes through
result = guard.validate("The quarterly revenue report shows a 15% increase.")
print(result.validation_passed)  # True

# Privileged content is flagged
try:
    result = guard.validate(
        "This communication is subject to attorney-client privilege. "
        "I am writing to you as your attorney regarding the pending matter."
    )
except Exception as e:
    print(e)  # Raises validation error with privilege details
```

## Configuration Options

| Parameter              | Type         | Default          | Description                                                        |
|------------------------|--------------|------------------|--------------------------------------------------------------------|
| `privilege_categories` | `list[str]`  | All three        | Categories to check: `attorney_client`, `work_product`, `settlement` |
| `confidence_threshold` | `float`      | `0.5`            | Minimum confidence score (0.0–1.0) to flag content                 |
| `custom_patterns`      | `dict`       | `None`           | Additional keywords/regex patterns per category                    |
| `redaction_strategy`   | `str`        | `"flag_only"`    | How to handle detected content: `mask`, `remove`, or `flag_only`   |
| `include_explanation`  | `bool`       | `True`           | Attach human-readable explanation to each detection                |
| `on_fail`              | `str`        | `"exception"`    | Guardrails on-fail action: `exception`, `fix`, `reask`, `noop`, `refrain` |

## Detection Approach

The validator uses a multi-layer detection system:

1. **Pattern matching** — Configurable keyword and phrase lists for each privilege category
2. **Contextual analysis** — Regex patterns that detect privilege indicators in surrounding context (e.g., "I am writing to you as your attorney regarding...")
3. **Confidence scoring** — Each detection returns a confidence score (0.0–1.0) based on the number and strength of matched indicators
4. **False positive suppression** — Common non-privileged legal terms (e.g., "attorney general", "power of attorney") are excluded

### Privilege Categories

**Attorney-Client Privilege** — Communications between lawyer and client for the purpose of obtaining or providing legal advice.

**Work Product Doctrine** — Materials prepared in anticipation of litigation, including case analysis, legal memoranda, and trial preparation documents.

**Settlement/Mediation Communications** — Protected settlement discussions, offers of compromise, and mediation communications (cf. FRE 408).

## Example Output

When privilege is detected, the validator returns structured metadata:

```json
{
  "privilege_detected": true,
  "categories": [
    {
      "type": "attorney_client",
      "confidence": 0.85,
      "indicators": ["attorney-client", "legal advice"],
      "context_snippet": "...the attorney-client communication regarding...",
      "explanation": "Content contains indicators of attorney-client privileged communication"
    }
  ],
  "overall_confidence": 0.85,
  "recommendation": "Review before disclosure — potential attorney-client privileged communication detected"
}
```

## Redaction Strategies

When used with `on_fail="fix"`, the validator modifies detected content based on the chosen strategy:

- **`flag_only`** (default) — Content is unchanged; metadata is attached for downstream review
- **`mask`** — Privileged segments are replaced with labels like `[PRIVILEGED — ATTORNEY-CLIENT]`
- **`remove`** — Privileged segments are stripped from the output

```python
guard = Guard().use(
    LegalPrivilegeDetection(
        redaction_strategy="mask",
        on_fail="fix",
    )
)

result = guard.validate(
    "Our attorney provided legal advice on the attorney-client matter."
)
print(result.validated_output)
# "Our [PRIVILEGED — ATTORNEY-CLIENT] provided [PRIVILEGED — ATTORNEY-CLIENT] on the [PRIVILEGED — ATTORNEY-CLIENT] matter."
```

## Custom Patterns

Add domain-specific keywords or regex patterns:

```python
guard = Guard().use(
    LegalPrivilegeDetection(
        custom_patterns={
            "attorney_client": [
                "legal hold notice",
                r"(?i)\bprivileged\s+and\s+exempt\b",
            ],
            "work_product": [
                "internal investigation memo",
            ],
        },
        confidence_threshold=0.3,
        on_fail="exception",
    )
)
```

Plain strings are treated as keywords; strings containing regex special characters are compiled as regex patterns.

## Use Cases

- **Litigation support AI** — Prevent AI-assisted document review tools from surfacing privileged communications in production outputs
- **Legal document processing** — Flag privileged content in automated document summarisation and extraction pipelines
- **Insurance claims processing** — Detect work product and settlement communications in claims AI workflows
- **Financial compliance** — Guard against inadvertent privilege waiver in AI-generated reports and communications
- **Corporate AI governance** — Runtime guardrail for any enterprise AI system that processes legal or regulatory content

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
