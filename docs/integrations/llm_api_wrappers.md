# Use Guardrails with LLM APIs

Guardrails' `Guard` wrappers provide a simple way to add Guardrails to your LLM API calls. The wrappers are designed to be used with any LLM API.


Here are some examples of how to use the wrappers with different LLM providers and models:

## OpenAI

### Completion Models (e.g. GPT-3)

```python
import openai
import guardrails as gd


# Create a Guard class
guard = gd.Guard.from_rail(...)

# Wrap openai API call
raw_llm_output, guardrail_output = guard(
    openai.Completion.create,
    prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
    engine="text-davinci-003",
    max_tokens=100,
    temperature=0.0,
)
```

### ChatCompletion Models (e.g. ChatGPT)

```python
import openai
import guardrails as gd

# Create a Guard class
guard = gd.Guard.from_rail(...)

# Wrap openai API call
raw_llm_output, guardrail_output = guard(
    openai.ChatCompletion.create,
    prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
    system_prompt="You are a helpful assistant...",
    model="gpt-3.5-turbo",
    max_tokens=100,
    temperature=0.0,
)
```

## Cohere

### Generate (e.g. GPT-3)

```python
import cohere
import guardrails as gd

# Create a Guard class
guard = gd.Guard.from_rail(...)

# Create a Cohere client
cohere_client = cohere.Client(api_key="my_api_key")

# Wrap cohere API call
raw_llm_output, guardrail_output = guard(
    cohere_client.generate,
    prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
    model="command-nightly",
    max_tokens=100,
    ...
)
```

## Using Manifest
[Manifest](https://github.com/HazyResearch/manifest) is a wrapper around most model APIs and supports hosting local models. It can be used as a LLM API.

```python
import guardrails as gd
import manifest

# Create a Guard class
guard = gd.Guard.from_rail(...)

# Create a Manifest client - this one points to GPT-4
# and caches responses in SQLLite
manifest = manifest.Manifest(
    client_name="openai",
    engine="gpt-4",
    cache_name="sqlite",
    cache_connection="my_manifest_cache.db"
)

# Wrap openai API call
raw_llm_output, guardrail_output = guard(
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
guard = gd.Guard.from_rail(...)

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
raw_llm_output, guardrail_output = guard(
    my_llm_api,
    prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
    **kwargs,
)
```
