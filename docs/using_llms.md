## Interacting with different LLMs

Guardrails has support for many many LLMs through its integration with LiteLLM. To interact with a model set the desired LLM API KEY such as the OPENAI_API_KEY and the desired model with the model property. Examples are below for some common ones.

## OpenAI

### Basic Usage

```python
from guardrails import Guard

os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_API_KEY"

guard = Guard()

validated_output, *rest = guard(
	prompt="How many moons does Jupiter have?",
    model="gpt-4o"
)

print(f"{validated_output}")
```

### Streaming

```python
from guardrails import Guard

os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_API_KEY"

guard = Guard()

stream_chunk_generator = guard(
	prompt="How many moons does Jupiter have?",
    model="gpt-4o",
    stream=True,
)

for chunk in stream_chunk_generator
    validated_chunk_output, *rest = chunk
    print(f"{validated_output}")
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
    
guard = Guard.from_pydantic(Basket)

validated_output, *rest = guard(
	prompt="Generate a basket of 5 fruits",
    model="gpt-4o",
    tools=guard.add_json_function_calling_tool([]),
    tool_choice="required",
)

print(f"{validated_output}")
```

## Anthropic

### Basic Usage

```python
from guardrails import Guard
import os

guard = Guard()

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

validated_output, *rest = guard(
	prompt="How many moons does Jupiter have?",
    model="claude-3-opus-20240229"
)

print(f"{validated_output}")
```

### Streaming

```python
from guardrails import Guard
import os

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

guard = Guard()

stream_chunk_generator = guard(
	prompt="How many moons does Jupiter have?",
    model="claude-3-opus-20240229",
    stream=True,
)

for chunk in stream_chunk_generator
    validated_output, *rest = chunk
    print(f"{validated_output}")
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

validated_output, *rest = guard(
	model="azure/<your_deployment_name>", 
    prompt="How many moons does Jupiter have?",
)

print(f"{validated_output}")
```

### Streaming

```python
from guardrails import Guard

os.environ["AZURE_API_KEY"] = "" # "my-azure-api-key"
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2023-05-15"

guard = Guard()

stream_chunk_generator = guard(
	prompt="How many moons does Jupiter have?",
    model="azure/<your_deployment_name>", 
    stream=True
)

for chunk in stream_chunk_generator
    validated_output, *rest = chunk
    print(f"{validated_output}")
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
    
guard = Guard.from_pydantic(Basket)

validated_output, *rest = guard(
	prompt="Generate a basket of 5 fruits",
    model="azure/<your_deployment_name>", 
    tools=guard.add_json_function_calling_tool([]),
    tool_choice="required",
)

print(f"{validated_output}")
```

## Gemini

### Basic Usage

```python
from guardrails import Guard
import os

os.environ['GEMINI_API_KEY'] = ""
guard = Guard()

validated_output, *rest = guard(
	prompt="How many moons does Jupiter have?",
    model="gemini/gemini-pro"
)

print(f"{validated_output}")
```

### Streaming

```python
from guardrails import Guard
import os

os.environ['GEMINI_API_KEY'] = ""
guard = Guard()
stream_chunk_generator = guard(
	prompt="How many moons does Jupiter have?",
    model="gemini/gemini-pro",
    stream=True
)

for chunk in stream_chunk_generator
    validated_output, *rest = chunk
    print(f"{validated_output}")
```

### Tools/Function calling

NOTE guardrails will not support until upgrade to 1.40.16 see https://github.com/BerriAI/litellm/issues/3086 need https://github.com/guardrails-ai/guardrails/pull/872

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
    
guard = Guard.from_pydantic(Basket)

validated_output, *rest = guard(
	prompt="Generate a basket of 5 fruits",
    model="gemini/gemini-pro"
    tools=guard.add_json_function_calling_tool([])
)

print(f"{validated_output}")
```

## Other LLMs

See LiteLLM’s documentation [here](https://docs.litellm.ai/docs/providers) for details on many other llms.