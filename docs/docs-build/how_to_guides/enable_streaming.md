# Stream Validated Output

To stream validated output, you need only pass the `stream=True` flag through the `guard` function. This will return a generator that will yield `GuardResult` objects as they are processed. 

```python
from guardrails import Guard
import os

# Set your OpenAI API key
# os.environ['OPENAI_API_KEY'] = ""

guard = Guard()

stream_chunk_generator = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="gpt-3.5-turbo",
    stream=True
)

# Print the validated output as it is processed
for chunk in stream_chunk_generator:
    print(f"{chunk.validated_output}")

```

## Using Validators with Streaming

Using validators with streaming works the same way. Note that not all `on_fail` types are supported with streaming. See the full list [here](../concepts/error_remediation).

```bash
guardrails hub install hub://guardrails/profanity_free
```

```python
from guardrails import Guard
from guardrails.hub import ProfanityFree
import os

# Set your OpenAI API key
# os.environ['OPENAI_API_KEY'] = ""

guard = Guard().use(ProfanityFree())

stream_chunk_generator = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="gpt-3.5-turbo",
    stream=True
)

# Print the validated output as it is processed
for chunk in stream_chunk_generator:
    print(f"{chunk.validated_output}")
```

## Learn more
Read more about streaming in our concept docs:

- [Streaming](../concepts/streaming)
- [Async Stream-validate LLM responses](../concepts/async_streaming)
- [Streaming Structured Data](../concepts/streaming_structured_data)