# Guard


The guard object is the main interface for GuardRails. It is seeded with a RailSpec, and then used to run the GuardRails AI engine. It is the object that accepts changing prompts, wraps LLM prompts, and keeps track of call history.


## How it works

```mermaid
graph
    A[Create `RAIL` spec] --> B[Initialize `guard` from spec];
    C[Collect LLM callable information] --> D[Invoke `guard`];
    B --> D;
    E[Invoke guard with prompt and instructions] --> D;
    D --> F[Guard invokes LLM API];
    F --> G[LLM API returns];
    G --> H[LLM metadata is stored and logged];
    H --> I[LLM output is validated];
    I --> J[Valid];
    I --> K[Invalid];
    K --> L[Check the on-fail action set in failed validator];
    L --> M[default: no-op];
    M --> N[Return output];
    L --> O[reask: reask LLM];
    O --> F;
    L --> P[filter: filter out output];
    P --> N;
    L --> Q[fix: fix output based on validator fix function]
    Q --> N;
    J --> N;
```

## Two main flows
### __call__
After instantiating a Guard you can call it in order to wrap an LLM with Guardrails and validate the output of that LLM according to the RAIL specification you provided.  Calling the guard in this way returns a tuple with the raw output from the LLM first, and the validatd output second.
```py
import openai
from guardrails import Guard

guard = Guard.from_rail(...)

raw_output, validated_output = guard(
    openai.Completion.create,
    engine="text-davinci-003",
    max_tokens=1024,
    temperature=0.3
)

print(raw_output)
print(validated_output)
``` 

### parse
If you would rather call the LLM yourself, at at least make the first call yourself, you can use `Guard.parse` to apply your RAIL specification to the LLM output as a post process.  You can also allow Guardrails to make re-asks to the LLM by specifying the `num_reasks` argument, or keep it purely as a post-processor by setting it to zero.  Unlike `__call__`, `Guard.parse` only returns the validated output.

Calling `Guard.parse` with reasks:
```py
import openai
from guardrails import Guard

guard = Guard.from_rail(...)

output = call_my_llm()

validated_output = guard.parse(
    llm_output=output,
    llm_api=openai.Completion.create,
    engine="text-davinci-003",
    max_tokens=1024,
    temperature=0.3,
    num_reasks=2
)

print(validated_output)
```

Calling `Guard.parse` as a post-processor:
```py
import openai
from guardrails import Guard

guard = Guard.from_rail(...)

output = call_my_llm()

validated_output = guard.parse(
    llm_output=output,
    num_reasks=0
)

print(validated_output)
```

## Error Handling and Retries
GuardRails currently performs automatic retries with exponential backoff when any of the following errors occur when calling the LLM:
    - openai.error.APIConnectionError
    - openai.error.APIError
    - openai.error.TryAgain
    - openai.error.Timeout
    - openai.error.RateLimitError
    - openai.error.ServiceUnavailableError

Note that this list is not exhaustive of the possible errors that could occur.  In the event that errors other than these arise during LLM calls, an exception will be raised.  The messaging of this exception is intended to help troubleshoot common problems, especially with custom LLM wrappers, as well as communicate the underlying error.  This type of exception would look like the following:
```log
Received an exception when calling with_handling:  The callable `fn` passed to `Guard(fn, ...)` failed with the following error: `{Root error message here!}`. Make sure that `fn` can be called as a function that takes in a single prompt string and returns a string.
```

In situations where the exception can be handled and retried, that is the exception is in the list above, the call to the LLM will be retried with exponential backoff until a max wait time between requests of sixty (60) seconds is reached.