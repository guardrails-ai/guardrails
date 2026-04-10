# Semantic Validation with semantix-ai

Guardrails validates **structure and rules** — required fields, type constraints, regex patterns, and custom validators. But it doesn't validate **meaning**: did the model actually answer the question asked? Does the output semantically match the intent?

[semantix-ai](https://github.com/labrat-akhona/semantix-ai) fills that gap. It validates LLM outputs against natural language intents using local NLI (Natural Language Inference) models — no API calls, no external services, ~15ms per check.

Together, Guardrails + semantix-ai give you full-stack output validation: structural correctness *and* semantic correctness.

---

## Installation

```bash
pip install guardrails-ai semantix-ai
```

semantix-ai downloads a small quantized NLI model on first use (cached locally). No API keys required.

---

## What is semantix-ai?

semantix-ai is a semantic type system for AI outputs. Instead of checking *format*, it checks *meaning* — using a locally-run NLI model to score whether an LLM output entails a given intent.

Key properties:
- **Local inference** — runs on CPU, ~15ms per validation
- **Zero API cost** — no calls to OpenAI, Anthropic, or any external service
- **NLI-based** — not LLM-based; uses a quantized entailment model (e.g. `cross-encoder/nli-MiniLM2-L6-H768`)
- **Self-healing** — built-in retry logic regenerates outputs that fail semantic validation
- **Training signal** — captures correction pairs for downstream fine-tuning

---

## Basic Usage

### Defining an Intent

```python
from semantix import Intent

# Define what the output should semantically mean
summarize_intent = Intent(
    "The response is a concise summary of the provided text, "
    "covering the main points without unnecessary detail."
)
```

### Validating an Output

```python
from semantix import SemanticValidator

validator = SemanticValidator()

output = "The article discusses climate change impacts on coastal regions."
result = validator.validate(output, summarize_intent)

print(result.passed)   # True / False
print(result.score)    # 0.0 – 1.0 entailment confidence
```

### Using the `@validate_intent` Decorator

```python
from semantix import validate_intent, Intent

intent = Intent("The answer explains the concept clearly and accurately.")

@validate_intent(intent, threshold=0.75)
def explain_concept(topic: str) -> str:
    # Your LLM call here
    return call_llm(f"Explain {topic} in simple terms.")

result = explain_concept("gradient descent")
# Raises SemanticValidationError if output doesn't meet the intent
```

---

## Combining with Guardrails

Use Guardrails for structural validation and semantix for semantic validation in the same pipeline:

```python
import guardrails as gd
from semantix import Intent, SemanticValidator

# Guardrails: enforce structure
guard = gd.Guard.from_pydantic(output_class=SummaryOutput)

# semantix: enforce meaning
intent = Intent(
    "The summary accurately reflects the source document's main argument "
    "and does not introduce information not present in the original."
)
semantic_validator = SemanticValidator(threshold=0.78)

def validated_summarize(document: str) -> SummaryOutput:
    # Step 1: Guardrails structural validation
    raw_output, validated, *rest = guard(
        llm_api=openai.chat.completions.create,
        prompt=f"Summarize the following:\n\n{document}",
        model="gpt-4o",
    )

    # Step 2: semantix semantic validation
    result = semantic_validator.validate(validated.summary, intent)
    if not result.passed:
        raise ValueError(
            f"Output failed semantic validation (score={result.score:.2f}): "
            f"{validated.summary}"
        )

    return validated
```

---

## Self-Healing Retries

semantix includes built-in retry logic. If the output doesn't meet the intent, it re-prompts with corrective context:

```python
from semantix import Intent, SelfHealingValidator

intent = Intent(
    "The response provides a specific, actionable recommendation — "
    "not a generic or hedged answer."
)

healer = SelfHealingValidator(intent=intent, max_retries=3, threshold=0.80)

output = healer.validate_with_retry(
    generate_fn=lambda: call_llm("What should I do first when debugging a memory leak?"),
)

print(output.final_text)
print(f"Attempts: {output.attempts}")
```

---

## Training Collector

Every failed validation is a learning signal. semantix captures the original output, the intent that failed, and the corrected output — structured correction pairs ready for fine-tuning:

```python
from semantix import Intent, SemanticValidator, TrainingCollector

collector = TrainingCollector(output_path="./corrections.jsonl")

intent = Intent("The response answers the user's question directly and completely.")
validator = SemanticValidator(threshold=0.75, collector=collector)

# During normal use — corrections are logged automatically
result = validator.validate(llm_output, intent, corrected=human_approved_output)
```

The `.jsonl` file contains structured entries:

```json
{
  "intent": "The response answers the user's question directly and completely.",
  "original": "There are many factors to consider when...",
  "corrected": "You should start by checking the error logs in /var/log/app.log.",
  "score_before": 0.41,
  "score_after": 0.91,
  "timestamp": "2025-10-14T09:22:11Z"
}
```

These pairs can be used directly with fine-tuning pipelines (e.g. OpenAI fine-tuning, Axolotl, Unsloth).

---

## Why Not Just Use Another LLM as a Judge?

LLM-based judges are expensive, slow, and introduce a second point of API dependency. semantix uses a local NLI model:

| | LLM-as-judge | semantix-ai |
|---|---|---|
| Latency | 500ms–3s | ~15ms |
| Cost | Per-token API cost | Free (local) |
| Privacy | Output sent to external API | Stays on your machine |
| Determinism | Low | High |

---

## Links

- **PyPI:** [pypi.org/project/semantix-ai](https://pypi.org/project/semantix-ai/)
- **Repository:** [github.com/labrat-akhona/semantix-ai](https://github.com/labrat-akhona/semantix-ai)
