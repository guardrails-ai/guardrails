# NeMo Guardrails as Guards
This guide will teach you how to add NeMo Guardrails to a GuardrailsAI Guard.

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

## Sample NeMo Guardrails
We'll start by creating a new NeMo Guardrails configuration.

```yaml
models:
 - type: main
   engine: openai
   model: gpt-3.5-turbo-instruct
```

We'll do a quick test to make sure everything is working as expected.

```python
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate("Hello!")

print(response)
```

```
    Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]

    Hi there! How can I assist you today?
```

That worked!  Now let's install a validator from the GuardrailsAI Hub to augment our NeMo Guardrails configuration from above.

If you haven't already, install and configure guardrails-ai before trying to install the DetectPII validator.

```bash
pip install guardrails-ai
 guardrails configure
```

```bash
guardrails hub install hub://guardrails/detect_pii --no-install-local-models
```

Now we can use the rails defined earlier as the basis for our Guard.  We'll also attach the DetectPII validator as an additional measure.

```python
from guardrails.integrations.nemoguardrails import NemoguardrailsGuard
from guardrails.hub import DetectPII

guard = NemoguardrailsGuard(rails)
guard.use(DetectPII(
    pii_entities=["PERSON", "EMAIL_ADDRESS"],
    on_fail="fix"
))
```

## Testing
With everything configured, we can test out our new Guard!

Let's invoke the Guard with a message that prompts the LLM to return personal information like names, email addresses, etc.. Since we specified `on_fail="fix"` in the DetectPII validator, the response should have any PII filtered out.

```python
response = guard(
    messages=[{
        "role": "user",
        "content": "Who is the current president of the United States, and what was their email address?"
    }]
)

print(response.validated_output)
```

```
The current president of the United States is <PERSON>. His email address is <EMAIL_ADDRESS>. He can also be reached through his personal email at <EMAIL_ADDRESS>. Additionally, he is active on social media and can be contacted through his official Twitter account <PERSON>. Is there anything else you would like to know about President <PERSON>?
```

Great! We can see that the Guard called the LLM configured in the LLMRails, validated the output, and filtered it accordingly. If however, we prompt the LLM with a message that does not cause it to return PII, we should get the unaltered response.

```python
response = guard(
    messages=[{
        "role": "user",
        "content": "Hello!"
    }]
)

print(response.validated_output)
```

```
Hi there! It's nice to meet you. My name is AI Assistant. How can I help you today?
```
