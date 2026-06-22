# Semantic Intent Validation with semantix-ai

[SemanticIntent](https://pypi.org/project/semantix-ai/) is a Guardrails validator that checks whether LLM output matches a given semantic intent. It runs locally with no API calls required (~15ms per validation) and implements the standard Guardrails `Validator` interface.

## Installation

```bash
pip install 'semantix-ai[guardrails]'
```

## Basic Usage

```python
from guardrails import Guard
from semantix.integrations.guardrails import SemanticIntent

guard = Guard().use(SemanticIntent("must be polite and professional"))
result = guard.validate("Thank you for your patience.")
```

## With Threshold

You can set a confidence threshold and an `on_fail` action for automatic retry:

```python
guard = Guard().use(
    SemanticIntent("must be polite", threshold=0.7, on_fail="reask")
)
```

## With Custom Judge

Use an LLM-based judge for higher accuracy on complex intents:

```python
from semantix import LLMJudge

guard = Guard().use(
    SemanticIntent("must be polite", judge=LLMJudge(model="gpt-4o-mini"))
)
```

## Composing with Other Validators

SemanticIntent works alongside any existing Guardrails validator in a pipeline:

```python
from guardrails import Guard
from guardrails.hub import RegexMatch
from semantix.integrations.guardrails import SemanticIntent

guard = Guard().use_many(
    SemanticIntent("must be a professional customer service response"),
    RegexMatch(regex=r"^(?!.*\b(damn|hell)\b).*$"),  # no profanity
)
```

## Key Characteristics

- **Local inference**: ~15ms per validation, no external API calls by default
- **Standard interface**: Implements the Guardrails `Validator` class, registered as `"semantix/semantic_intent"`
- **Retry support**: Works with `on_fail="reask"` for automatic retry on validation failure
- **Composable**: Use with `Guard.use()` or `Guard.use_many()` alongside any other validator
