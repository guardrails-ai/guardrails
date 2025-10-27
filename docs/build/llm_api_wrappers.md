# Use Guardrails with LLM APIs

Guardrails' `Guard` wrappers provide a simple way to add Guardrails to your LLM API calls. The wrappers are designed to be used with any LLM API.


Here are some examples of how to use the wrappers with different LLM providers and models:

## OpenAI

### Completion Models (e.g. GPT-3)

```python
import openai
import guardrails as gd


# Create a Guard class
guard = gd.Guard.for_rail(...)

# Wrap openai API call
raw_llm_output, guardrail_output, *rest = guard(
    openai.completions.create,
    prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
    model="gpt-3.5-turbo-instruct",
    max_tokens=100,
    temperature=0.0,
)
```

### ChatCompletion Models (e.g. ChatGPT)

```python
import openai
import guardrails as gd

# Create a Guard class
guard = gd.Guard.for_rail(...)

# Wrap openai API call
raw_llm_output, guardrail_output, *rest = guard(
    openai.chat.completions.create,
    prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
    system_prompt="You are a helpful assistant...",
    model="gpt-3.5-turbo",
    max_tokens=100,
    temperature=0.0,
)
```

## Cohere

### Generate (e.g. command)

```python
import cohere
import guardrails as gd

# Create a Guard class
guard = gd.Guard.for_rail(...)

# Create a Cohere client
cohere_client = cohere.Client(api_key="my_api_key")

# Wrap cohere API call
raw_llm_output, guardrail_output, *rest = guard(
    cohere_client.generate,
    prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
    model="command-nightly",
    max_tokens=100,
    ...
)
```

## Anthropic

### Completion

```python
from anthropic import Anthropic
import guardrails as gd

# Create a Guard class
guard = gd.Guard.for_rail(...)

# Create an Anthropic client
anthropic_client = Anthropic(api_key="my_api_key")

# Wrap Anthropic API call
raw_llm_output, guardrail_output, *rest = guard(
    anthropic_client.completions.create,
    prompt_params={
        "prompt_param_1": "value_1", 
        "prompt_param_2": "value_2",
        ...
    },
    model="claude-2",
    max_tokens_to_sample=100,
    ...
)
```


## Hugging Face

### Text Generation Models
```py
from guardrails import Guard
from guardrails.validators import ValidLength, ToxicLanguage
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Create your prompt or starting text
prompt = "Hello, I'm a language model,"

# Setup torch
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate your tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Instantiate your model
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)

# Customize your model inputs if desired.
# If you don't pass and inputs (`input_ids`, `input_values`, `input_features`, or `pixel_values`)
# We'll try to do something similar to below using the tokenizer and the prompt.
# We strongly suggest passing in your own inputs.
model_inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)


# Create the Guard
guard = Guard.for_string(
    validators=[
        ValidLength(
            min=48,
            on_fail=OnFailAction.FIX
        ),
        ToxicLanguage(
            on_fail=OnFailAction.FIX
        )
    ],
    prompt=prompt
)

# Run the Guard
response = guard(
    llm_api=model.generate,
    max_new_tokens=40,
    tokenizer=tokenizer,
    **model_inputs,
)

# Check the output
if response.validation_passed:
    print("validated_output: ", response.validated_output)
else:
    print("error: ", response.error)

```

### Pipelines
```py
from guardrails import Guard
from guardrails.validators import ValidLength, ToxicLanguage
import torch
from transformers import pipeline


# Create your prompt or starting text
prompt = "What are we having for dinner?"

# Setup pipeline
generator = pipeline("text-generation", model="facebook/opt-350m")


# Create the Guard
guard = Guard.for_string(
    validators=[
        ValidLength(
            min=48,
            on_fail=OnFailAction.FIX
        ),
        ToxicLanguage(
            on_fail=OnFailAction.FIX
        )
    ],
    prompt=prompt
)

# Run the Guard
response = guard(
    llm_api=generator,
    max_new_tokens=40
)

if response.validation_passed:
    print("validated_output: ", response.validated_output)
else:
    print("error: ", response.error)

```


## Using Manifest
[Manifest](https://github.com/HazyResearch/manifest) is a wrapper around most model APIs and supports hosting local models. It can be used as a LLM API.

```python
import guardrails as gd
import manifest

# Create a Guard class
guard = gd.Guard.for_rail(...)

# Create a Manifest client - this one points to GPT-4
# and caches responses in SQLLite
manifest = manifest.Manifest(
    client_name="openai",
    engine="gpt-4",
    cache_name="sqlite",
    cache_connection="my_manifest_cache.db"
)

# Wrap openai API call
raw_llm_output, guardrail_output, *rest = guard(
    manifest,
    prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
    max_tokens=100,
    temperature=0.0,
)
```


## Using a custom LLM API

```python
import guardrails as gd

# Create a Guard class
guard = gd.Guard.for_rail(...)

# Function that takes the prompt as a string and returns the LLM output as string
def my_llm_api(prompt: str, **kwargs) -> str:
    """Custom LLM API wrapper.

    Args:
        prompt (str): The prompt to be passed to the LLM API
        **kwargs: Any additional arguments to be passed to the LLM API

    Returns:
        str: The output of the LLM API
    """

    # Call your LLM API here
    return ...


# Wrap your LLM API call
raw_llm_output, guardrail_output, *rest = guard(
    my_llm_api,
    prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
    **kwargs,
)
```
