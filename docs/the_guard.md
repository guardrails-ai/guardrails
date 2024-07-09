# What is a Guard?

The guard object is the main interface for GuardRails. It can be used without configuration for string-based LLM apps, and accepts a pydantic object for structured data usecases. The Guard is then used to run the GuardRails AI engine. It is the object that wraps LLM calls, orchestrates validation, and keeps track of call history.


## How it works

![Guardrails Logo](./img/guardrails_arch_white_vertical.png#only-light)
![Guardrails Logo](./img/guardrails_arch_dark_vertical.png#only-dark)


## Two main flows
### __call__
After initializing a guard, you can invoke it using a model name and messages for your prompt, similarly to how you would invoke a regular call to an LLM SDK. Guardrails will call the LLM and then validate the output against the guardrails you've configured. The output will be returned as a `GuardResponse` object, which contains the raw LLM output, the validated output, and whether or not validation was successful.

```py
from guardrails import Guard
import os

os.environ["OPENAI_API_KEY"] = [YOUR API KEY]

guard = Guard()

res = guard(
    model="gpt-3.5-turbo-instruct",
    messages=[{
        "role": "user",
        "content": "How do I make a cake?"
    }]
)

print(res.raw_llm_output)
print(res.validated_output)
print(res.validation_passed)
``` 

### parse
If you would rather call the LLM yourself, or at least make the first call yourself, you can use `Guard.parse` to apply your RAIL specification to the LLM output as a post process.  You can also allow Guardrails to make re-asks to the LLM by specifying the `num_reasks` argument, or keep it purely as a post-processor by setting it to zero.  `Guard.parse` returns the same fields as `__call__`.

Calling `Guard.parse` as a post-processor:
```py
import openai
from guardrails import Guard

guard = Guard()

output = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role": "user",
        "content": "How do I make a cake?"
    }]
).choices[0].message.content

res = guard.parse(
    llm_output=output
)

print(res.validated_output) # Validated output
```

Calling `Guard.parse` with reasks:
```py
import openai
from guardrails import Guard

guard = Guard()

output = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role": "user",
        "content": "How do I make a cake?"
    }]
).choices[0].message.content

res = guard.parse(
    llm_output=output,
    model="gpt-3.5-turbo-instruct",
    num_reasks=1
)

print(res.validated_output) # Validated output
print(guard.history.last.reasks) # A list of reasks
```

## Error Handling and Retries
Guardtails currently performs automatic retries with exponential backoff when any of the following errors occur when calling the LLM:

- openai.error.APIConnectionError
- openai.error.APIError
- openai.error.TryAgain
- openai.error.Timeout
- openai.error.RateLimitError
- openai.error.ServiceUnavailableError
- An incorrect structure was returned from the LLM

Note that this list is not exhaustive of the possible errors that could occur.  In the event that errors other than these arise during LLM calls, an exception will be raised.  The messaging of this exception is intended to help troubleshoot common problems, especially with custom LLM wrappers, as well as communicate the underlying error.  This type of exception would look like the following:
```log
The callable `fn` passed to `Guard(fn, ...)` failed with the following error: `{Root error message here!}`. Make sure that `fn` can be called as a function that takes in a single prompt string and returns a string.
```

In situations where the exception can be handled and retried, that is the exception is in the list above, the call to the LLM will be retried with exponential backoff until a max wait time between requests of sixty (60) seconds is reached.