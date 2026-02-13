# Use multiple guardrails in a single `Guard` object

You can use multiple guardrails in a single `Guard` object.

## Install the necessary guardrails

First, install the necessary guardrails:

```bash
guardrails hub install hub://guardrails/competitor_check
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/regex_match
```

## The `use` methods

You can utilze the `use` methods to add one or more guardrails to a single `Guard` object. For example, here's how to make sure any LLM generated text doesn't contain any toxic language or any mention of your competitors:

```python
from guardrails import Guard
from guardrails.hub import CompetitorCheck, ToxicLanguage, RegexMatch

competitors = ["Apple", "Samsung"]

guard = Guard().use(
    CompetitorCheck(competitors=[competitors]),
    ToxicLanguage(
        validation_method=sentence,
        threshold=0.5
    ),
    RegexMatch(regex="^[A-Z].*")
)

guard.validate("My favorite phone is BlackBerry.")  # Guardrail Passes
```
