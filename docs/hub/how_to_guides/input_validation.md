# Use Validators for Input Validation

Validators that are tagged as input validators can be used to validate the input prompt before it is sent to the model. This can be useful for ensuring that the input prompt meets certain criteria, such as being on-topic, not containing PII, etc.

In order to use an input validator, first make sure that the validator is installed. You can install the validator using the `guardrails hub install` command. For example, to install the `DetectPII` validator, you can run:

```bash
guardrails hub install hub://guardrails/detect_pii
```

Then, add the input validator to the `Guard` object via the `use` method. For example, to use the `DetectPII` validator with OpenAI's GPT-3, you can run:

```python
import openai
from guardrails import Guard
from guardrails.errors import ValidationError
from guardrails.hub import DetectPII
from guardrails.types import OnFailAction

guard = Guard()
guard.use(
    DetectPII(
        pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], 
        on_fail=OnFailAction.EXCEPTION
    ), 
    on="prompt"
)

try:
    guard(
        openai.chat.completions.create,
        prompt="My email address is not_a_real_email@guardrailsai.com",
    )
except ValidationError as e:
    print(e)
```
