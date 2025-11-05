# Getting Started with the Hub

## 1. Install the Hub CLI

The Guardrails Hub CLI is a part of the Guardrails AI package. You can install it using pip:

```bash
pip install guardrails-ai
```

## 2. Configure the Hub CLI

To configure the Hub CLI, you need to set your API key. You can get your API key from the Guardrails Hub website.

```bash
guardrails configure
```

## 3. Install a Guardrail

You can download a guardrail from the Hub using the `install` command. For example, to download the `regex_match` guardrail, you can run:

```bash
guardrails hub install hub://guardrails/regex_match
```

## 4. Use the Guardrail

You can use the guardrail in your code by creating a `Guard` object and passing it to your LLM API calls. For example, to use the `regex_match` guardrail with OpenAI's GPT-3, you can run:

```python
# Import Guard and Validator
from guardrails.hub import RegexMatch
from guardrails import Guard

# Initialize the Guard with 
val = Guard().use(
    RegexMatch(regex="^[A-Z][a-z]*$")
)

guard.parse("Caesar")  # Guardrail Passes
guard.parse("Caesar is a great leader")  # Guardrail Fails
```

## 5. Run multiple Guardrails

You can run multiple guardrails in a single `Guard` object. For example, here's how to make sure any LLM generated text doesn't contain any toxic language or any mention of your competitors:

First, install the necessary guardrails:

```bash
guardrails hub install hub://guardrails/competitor_check
guardrails hub install hub://guardrails/toxic_language
```

```python
from guardrails import Guard
from guardrails.hub import CompetitorCheck, ToxicLanguage

competitors = ["Apple", "Samsung"]

guard = Guard().use(
    CompetitorCheck(
        competitors=[competitors]
).use(
    ToxicLanguage(
        validation_method=sentence,
        threshold=0.5
    )
)

guard.validate("My favorite phone is BlackBerry.")  # Guardrail Passes
```
