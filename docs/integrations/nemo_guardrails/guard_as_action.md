# Guard as Actions

This guide will teach you how to use a `Guard` with any of the 60+ GuardrailsAI Validators as an action inside a NeMo Guardrails configuration.

## Prerequisites

We'll be using an OpenAI model for our LLM in this guide, so set up an OpenAI API key, if not already set.

```bash
export OPENAI_API_KEY=$OPENAI_API_KEY    # Replace with your own key
```

If you're running this inside a notebook, you also need to patch the AsyncIO loop.

```python
import nest_asyncio

nest_asyncio.apply()
```

## Sample Guard

Let's create a sample Guard that can detect PII.  First, install guardrails-ai.

```bash
pip install guardrails-ai -q
```

Next configure the guardrails cli so we can install the validator we want to use from the Guardrails Hub.

```bash
guardrails configure
```

```bash
guardrails hub install hub://guardrails/detect_pii --no-install-local-models -q
```

Now we can define our Guard.
This Guard will use the DetectPII validator to safeguard against leaking personally identifiable information such as names, email addresses, etc..

Once the Guard is defined, we can test it with a static value to make sure it's working how we would expect.

```python
from guardrails import Guard
from guardrails.hub import DetectPII

g = Guard(name="pii_guard").use(DetectPII(["PERSON", "EMAIL_ADDRESS"], on_fail="fix"))

print(g.validate("My name is John Doe"))
```

```
ValidationOutcome(
    call_id='14534730096',
    raw_llm_output='My name is John Doe',
    validation_summaries=[
        ValidationSummary(
            validator_name='DetectPII',
            validator_status='fail',
            property_path='$',
            failure_reason='The following text in your response contains PII:\nMy name is John Doe',
            error_spans=[
                ErrorSpan(start=11, end=19, reason='PII detected in John Doe')
            ]
        )
    ],
    validated_output='My name is <PERSON>',
    reask=None,
    validation_passed=True,
    error=None
)
```

## NeMo Guardrails Configuration

Now we'll use the Guard we defeined above to create an action and a flow. Since we're calling our guard "pii_guard", we'll use "pii_guard_validate" in order to see if the LLM output is safe.

```colang
define flow detect_pii
  $output = execute pii_guard_validate(text=$bot_message)

  if not $output
    bot refuse to respond
    stop
```

```yaml
models:
 - type: main
   engine: openai
   model: gpt-3.5-turbo-instruct

rails:
  output:
    flows:
      - detect_pii
```

To hook the Guardrails AI guard up so that it can be read from Colang, we use the integration's `register_guardrails_guard_actions` function.
This takes a name and registers two actions:

1. [guard_name]_validate: This action is used to detect validation failures in outputs
2. [guard name]_fix: This action is used to automatically fix validation failures in outputs, when possible

```python
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.integrations.guardrails_ai.guard_actions import register_guardrails_guard_actions

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

register_guardrails_guard_actions(rails, g, "pii_guard")
```

```
Fetching 5 files: 100%|██████████| 5/5 [00:00<00:00, 109226.67it/s]
```

## Testing

Let's try this out. If we invoke the NeMo Guardrails configuration with a message that prompts the LLM to return personal information like names, email addresses, etc., it should refuse to respond.

```python
response = rails.generate("Who is the current president of the United States, and what was their email address?")
print(response)
```

```
I'm sorry, I can't respond to that.
```

Great! So the valdiation-only flow works.  Next let's try the fix flow.

```colang
define flow detect_pii
  $output = execute pii_guard_fix(text=$bot_message)

  if not $output
    bot refuse to respond
    stop
  else
    $bot_message = $output
```

If we send the same message, we should get a response this time, but any PII will be filtered out.

```python
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

register_guardrails_guard_actions(rails, g, "pii_guard")

response = rails.generate("Who is the current president of the United States, and what was their email address?")
print(response)
```

```
The current president of the United States is <PERSON>. His official email address is <EMAIL_ADDRESS>. However, he also has a personal email address, which is <EMAIL_ADDRESS>.
```

If however, we prompt the LLM with a message that does not cause it to return PII, we should get the unaltered response.

```python
response = rails.generate("Hello!")
print(response)
```

```
Hello there! How can I assist you?
```
