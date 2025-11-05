# Use Supported LLMs

Guardrails has support for 100+ LLMs through its integration with LiteLLM. This integration is really useful because it allows the Guardrails call API to use the same clean interface that LiteLLM and OpenAI use. This means that you can use  similar code to make LLM requests with Guardrails as you would with OpenAI.

To interact with a model, set the desired LLM API KEY such as the OPENAI_API_KEY and the desired model with the model property. Examples are below for some common ones.

## OpenAI

### Basic Usage

```python
from guardrails import Guard

os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_API_KEY"

guard = Guard()

result = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="gpt-4o",
)

print(f"{result.validated_output}")
```

### Streaming

```python
from guardrails import Guard

os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_API_KEY"

guard = Guard()

stream_chunk_generator = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="gpt-4o",
    stream=True,
)

for chunk in stream_chunk_generator
    print(f"{chunk.validated_output}")
```

### Tools/Function Calling

```python
from pydantic import BaseModel, Field
from typing import List
from guardrails import Guard

os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_API_KEY"

class Fruit(BaseModel):
    name: str
    color: str

class Basket(BaseModel):
    fruits: List[Fruit]
    
guard = Guard.for_pydantic(Basket)

result = guard(
    messages=[{"role":"user", "content":"Generate a basket of 5 fruits"}],
    model="gpt-4o",
    tools=guard.json_function_calling_tool([]),
    tool_choice="required",
)

print(f"{result.validated_output}")
```

## Anthropic

### Basic Usage

```python
from guardrails import Guard
import os

guard = Guard()

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

result = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="claude-3-opus-20240229"
)

print(f"{result.validated_output}")
```

### Streaming

```python
from guardrails import Guard
import os

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

guard = Guard()

stream_chunk_generator = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="claude-3-opus-20240229",
    stream=True,
)

for chunk in stream_chunk_generator
    print(f"{chunk.validated_output}")
```

## Azure OpenAI

### Basic Usage

```python
from guardrails import Guard
import os
os.environ["AZURE_API_KEY"] = "" # "my-azure-api-key"
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2023-05-15"

guard = Guard()

result = guard(
    model="azure/<your_deployment_name>",
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
)

print(f"{result.validated_output}")
```

### Streaming

```python
from guardrails import Guard

os.environ["AZURE_API_KEY"] = "" # "my-azure-api-key"
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2023-05-15"

guard = Guard()

stream_chunk_generator = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="azure/<your_deployment_name>", 
    stream=True
)

for chunk in stream_chunk_generator
    print(f"{chunk.validated_output}")
```

### Tools/Function Calling

```python
from pydantic import BaseModel, Field
from typing import List
from guardrails import Guard

os.environ["AZURE_API_KEY"] = "" # "my-azure-api-key"
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2023-05-15"

class Fruit(BaseModel):
    name: str
    color: str

class Basket(BaseModel):
    fruits: List[Fruit]
    
guard = Guard.for_pydantic(Basket)

result = guard(
    messages=[{"role":"user", "content":"Generate a basket of 5 fruits"}],
    model="azure/<your_deployment_name>", 
    tools=guard.add_json_function_calling_tool([]),
    tool_choice="required",
)

print(f"{result.validated_output}")
```

## Gemini

### Basic Usage

```python
from guardrails import Guard
import os

os.environ['GEMINI_API_KEY'] = ""
guard = Guard()

result = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="gemini/gemini-pro"
)

print(f"{result.validated_output}")
```

### Streaming

```python
from guardrails import Guard
import os

os.environ['GEMINI_API_KEY'] = ""
guard = Guard()
stream_chunk_generator = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="gemini/gemini-pro",
    stream=True
)

for chunk in stream_chunk_generator
    print(f"{chunk.validated_output}")
```

### COMING SOON - Tools/Function calling

```python
from pydantic import BaseModel, Field
from typing import List
from guardrails import Guard

os.environ['GEMINI_API_KEY'] = ""

class Fruit(BaseModel):
    name: str
    color: str

class Basket(BaseModel):
    fruits: List[Fruit]
    
guard = Guard.for_pydantic(Basket)

result = guard(
    messages=[{"role":"user", "content":"Generate a basket of 5 fruits"}],
    model="gemini/gemini-pro"
    tools=guard.add_json_function_calling_tool([])
)

print(f"{result.validated_output}")
```

## Databricks

### Basic Usage

```python
from guardrails import Guard

os.environ["DATABRICKS_API_KEY"] = "" # your databricks key
os.environ["DATABRICKS_API_BASE"] = "" # e.g.: https://abc-123ab12a-1234.cloud.databricks.com

guard = Guard()

result = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="databricks/databricks-dbrx-instruct",
)

print(f"{result.validated_output}")
```

### Streaming

```python
from guardrails import Guard

os.environ["DATABRICKS_API_KEY"] = "" # your databricks key
os.environ["DATABRICKS_API_BASE"] = "" # e.g.: https://abc-123ab12a-1234.cloud.databricks.com

guard = Guard()

stream_chunk_generator = guard(
    messages=[{"role":"user", "content":"How many moons does Jupiter have?"}],
    model="databricks/databricks-dbrx-instruct",
    stream=True,
)

for chunk in stream_chunk_generator
    print(f"{chunk.validated_output}")
```

## Other LLMs
As mentioned at the top of this page, over 100 LLMs are supported through our litellm integration, including (but not limited to)

- Anthropic
- AWS Bedrock
- Anyscale
- Huggingface
- Mistral
- Predibase
- Fireworks


Find your LLM in LiteLLM’s documentation [here](https://docs.litellm.ai/docs/providers). Then, follow those same steps and set the same environment variables they guide you to use, but invoke a `Guard` object instead of the litellm object.

Guardrails will wire through the arguments to litellm, run the Guarding process, and return a validated outcome.

## Custom LLM Wrappers
In case you're using an LLM that isn't natively supported by Guardrails and you don't want to use LiteLLM, you can build a custom LLM API wrapper. In order to use a custom LLM, create a function that accepts a positional argument for the prompt as a string and any other arguments that you want to pass to the LLM API as keyword args. The function should return the output of the LLM API as a string.
Install ProfanityFree from hub:
```
guardrails hub install hub://guardrails/profanity_free
```
```python
from guardrails import Guard
from guardrails.hub import ProfanityFree

# Create a Guard class
guard = Guard().use(ProfanityFree())

# Function that takes the prompt as a string and returns the LLM output as string
def my_llm_api(
    *,
    **kwargs
) -> str:
    """Custom LLM API wrapper.

    At least one of messages should be provided.

    Args:
        **kwargs: Any additional arguments to be passed to the LLM API

    Returns:
        str: The output of the LLM API
    """
    messages = kwargs.pop("messages", [])
    updated_messages = some_message_processing(messages)
    # Call your LLM API here
    # What you pass to the llm will depend on what arguments it accepts.
    llm_output = some_llm(updated_messages, **kwargs)

    return llm_output

# Wrap your LLM API call
validated_response = guard(
    my_llm_api,
    messages=[{"role":"user","content":"Can you generate a list of 10 things that are not food?"}],
    **kwargs,
)
```
