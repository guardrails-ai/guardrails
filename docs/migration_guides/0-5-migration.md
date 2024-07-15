# Migrating to 0.5.0


## New Features

### Run Guardrails as a local server

Guardrails 0.5.0 introduces the `start` command to the guardrails cli.  This allows you to run the Guardrails validation engine as a local python server.  

Benefits of using Guardrails this way include:

- Less strain on your main process/thread
- The Guardrails server utilizes Gunicorn to take advantage of multiple threads
    - Supported on Linux and MacOS by default, supported on Windows when using WSL
- Declare your Guards in a separate config and reference them by name in your app to keep your code slim and clean

Example:

1. Install
```sh
pip install "guardarils-ai>=0.5.0"
guardrails hub install hub://guardrails/regex_match
```

2. Create a `config.py`
```py
from guardrails import Guard
from guardrails.hub import RegexMatch


Guard(
    name='name-case',
    description='Checks that a string is in Name Case format.'
).use(
    RegexMatch(regex="^[A-Z][a-z\\s]*$")
)
```

3. Start the Guardrails server
```sh
guardrails start --config=config.py
```

4. Use the Guard in your application
```py
from rich import print
from guardrails import Guard

name_case = Guard(name='name-case')

result = name_case.validate("Zayd")

print(result)
```


### Generate structured data with smaller models:

As part of Guardrails 0.5.0, we're launching constrained decoding support for HuggingFace models.  This allow you to generate structured data that matches your schema with confidence.

Example:
```py
from guardrails import Guard
from pydantic import BaseModel

class Dog(BaseModel):
    name: str
    color: str
    weight_kg: float

class NewFriends(BaseModel):
    dogs: list[Dog]

guard = Guard.from_pydantic(NewFriends, output_formatter="jsonformer")

# JSONFormer is only compatible with HF Pipelines and HF Models:
from transformers import pipeline
tiny_llama_pipeline = pipeline("text-generation", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Inference is straightforward:
response = guard(tiny_llama_pipeline, prompt="Please enjoy this list of good dogs:")

# `out` is a dict.  Format it as JSON for readability:
import json
print(json.dumps(response.validated_output, indent=2))
```


## Improvements

### LiteLLM is now easier to use within Guardrails

When calling models through LiteLLM, specifying the `llm_api` argument is now optional. Instead, just pass the model name.

Example:

```py
from rich import print
from guardrails import Guard
from guardrails.hub import RegexMatch

guard = Guard().use(RegexMatch("95", match_type="search"))

response = guard(
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    prompt="How many moons does jupiter have?",
)

print(response)
```

### New public interface for generating JSON schema-based function calling tools

Guardrails has supported function calling for OpenAI Chat models for a while and previously would auto-insert a function to specify the schema when a Guard was created via a Pydantic model.

In Guardrails 0.5.0, you can use this same pattern regardless of how the Guard was initialized.  We also made the process more transparent by allowing you to generate the tool first and decide when to pass it as a keyword argument.  For models that support openai tool/function calling (`gpt-4o`, `gpt-4-turbo`, or `gpt-3.5-turbo`), you can extend your existing `tools` with `Guard.json_function_calling_tool()`

Example:
```py
from guardrails import Guard
from guardrails.hub import RegexMatch
from pydantic import BaseModel, Field
from typing import List

NAME_REGEX = "^[A-Z][a-z]+\s[A-Z][a-z]+$"

class Delivery(BaseModel):
    custome_name: str=Field(validators=[RegexMatch(regex=NAME_REGEX)], description="customer name")
    pickup_time: str=Field(description="date and time of pickup")
    pickup_location: str=Field(description="address of pickup")
    dropoff_time: str=Field(description="date and time of dropoff")
    dropoff_location: str=Field(description="address of dropoff")
    price: str = Field(description="price of delivery with currency symbol included")

class Schedule(BaseModel):
    deliveries: List[Delivery]

guard = Guard.from_pydantic(Schedule)
chat_history="""
nelson and murdock: i need a pickup 797 9th Avenue, manila envelope, June 3 10:00am with dropoff 10:30am Courthouse, 61 Center Street C/O frank james
operator: quote - $23.00
neslon and murdock: perfect, we accept the quote
operator: 797 9th ave, 10:00am pickup comfirmed
abc flowers: i need a pickup of a flowers from abc flowers at 21 3rd street at 11:00am on june 2 with a dropoff at 75th Ave at 5:30pm same day
operator: 21 3rd street flowers quote - $14.50
abc flowers: accepted
polk and wardell: i need a pickup of a bagels from Bakers Co at 331 5th street at 11:00am on june 3 with a dropoff at 75th Ave at 5:30pm same day
operator: 331 5th street bagels quote - $34.50
polk and wardell: accepted
"""

prompt = """
From the chat exchanges below extract a schedule of deliveries.
Chats:
${chat_history}
"""

tools = [] # an open ai compatible list of tools

response = guard(
    openai.chat.completions.create,
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    prompt=prompt,
    prompt_params={"chat_history": chat_history},
    tools=guard.json_function_calling_tool(tools),
    tool_choice="required",
)
```

### `Guard.use()` now works for all Guards

Previously, constructing a Guard via the `use` method was only supported for unstructured response schemas.  It now supports specifying validators for any Guard regardless of the initialization method (`Guard()`, `Guard.from_rail()`, `Guard.from_pydantic()`, etc.).  `Guard.use()` is also the new method of applying input validations to a Guard.

Example of applying input validation to the Prompt:
```py
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

To utilize `Guard.use()` on a Guard with structured output, you can specify a JSON Path to identify which property the Validator(s) should be assigned to.

Example:
```py
import json
from pydantic import BaseModel, Field
from guardrails import Guard, OnFailAction
from guardrails.errors import ValidationError
from guardrails.hub import RegexMatch, ValidRange

class Person(BaseModel):
    name: str
    # Existing way of assigning validators; still valid
    age: int = Field(validators=[ValidRange(0, 100, on_fail=OnFailAction.EXCEPTION)])
    is_employed: bool

guard = Guard.from_pydantic(Person)

# Use a regex to make sure the name is Title Case
guard.use(
    RegexMatch("^(?:[A-Z][^\\s]*\\s?)+$", on_fail=OnFailAction.EXCEPTION),
    on="$.name"
)

try:
    guard.validate(json.dumps({
        "name": "john doe",
        "age": 30,
        "is_employed": True
    }))
except ValidationError as e:
    print(e)
```

## Backwards-incompatible changes

### Args vs Kwargs
In previous versions, most of the Guardrails interfaces utilized positional arguments for most parameters. This could be tedious when specifying optional arguments.

In 0.5.0, for our public interfaces, only required arguments are positional; all optional arguments are keyword only.

If you previously called you Guard like this:
```py
guard(
    openai.chat.completions.create,,
    { "topic": "recursion" }, # prompt parameters
    2,  # number of reasks
    "Write a short statement about ${topic}", # prompt
)
```

You will now call it like this:
```py
guard(
    openai.chat.completions.create,
    prompt_params={ "topic": "recursion" },
    num_reasks=2,
    prompt="Write a short statement about ${topic}",
)
```

### Validators have moved
We've moved validators to the [Guardrails Hub](https://hub.guardrailsai.com), reducing the core package size for faster installations and smoother workflows.

Targeted validation: Install only the validators you need, streamlining your project and dependencies.

New naming: Some validator names changed for clarity and consistency. Head to the hub to find the updated names and grab your validators!

### AsyncGuard's for Async LLMs
In v0.4.4, we introduced a new `AsyncGuard` for use with asynchronous LLM's.  As of 0.5.0, support for async LLM's was removed from the `Guard` class and is now only supported in the `AsyncGuard` class.  This should provide better type hinting while developing as well as make the interface simpler and easier to use.

### Prompt Primitives have moved
In v0.4.5, we introduced `xml` prompt primitives to replace the previous `json` constants.  In 0.5.0, the `json` prompt primitives have a different meaning and will likely continue to evolve.  If you wish to keep the same constructed prompts as before, you must utilize the new `xml` prompt primitives.

### Removal of support for older dependency versions
As of 0.5.0, we no longer directly support to following versions of dependencies:

- Python 3.8
- Pydantic 1.x
- OpenAI 0.x