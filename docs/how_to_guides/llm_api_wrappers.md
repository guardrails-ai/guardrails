import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Use Guardrails with any LLM

Guardrails' `Guard` wrappers provide a simple way to add Guardrails to your LLM API calls. The wrappers are designed to be used with any LLM API.

There are three ways to use Guardrails with an LLM API:
1. [**Natively-supported LLMs**](#natively-supported-llms): Guardrails provides out-of-the-box wrappers for OpenAI, Cohere, Anthropic and HuggingFace. If you're using any of these APIs, check out the documentation in [this](#natively-supported-llms) section.
2. [**LLMs supported through LiteLLM**](#llms-supported-via-litellm): Guardrails provides an easy integration with [liteLLM](https://docs.litellm.ai/docs/), a lightweight abstraction over LLM APIs that supports over 100+ LLMs. If you're using an LLM that isn't natively supported by Guardrails, you can use LiteLLM to integrate it with Guardrails. Check out the documentation in [this](#llms-supported-via-litellm) section.
3. [**Build a custom LLM wrapper**](#build-a-custom-llm-wrapper): If you're using an LLM that isn't natively supported by Guardrails and you don't want to use LiteLLM, you can build a custom LLM API wrapper. Check out the documentation in [this](#build-a-custom-llm-wrapper) section.


## Natively-supported LLMs

Guardrails provides native support for a select few LLMs and Manifest. If you're using any of these LLMs, you can use Guardrails' out-of-the-box wrappers to add Guardrails to your LLM API calls.

<Tabs>
  <TabItem value="openai" label="OpenAI" default>
    ```python
    import openai
    from guardrails import Guard
    from guardrails.hub import ProfanityFree

    # Create a Guard
    guard = Guard().use(ProfanityFree())

    # Wrap openai API call
    validated_response = guard(
        openai.chat.completions.create,
        prompt="Can you generate a list of 10 things that are not food?",
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.0,
    )
    ```
  </TabItem>
  <TabItem value="cohere" label="Cohere">
    ```python
    import cohere
    from guardrails import Guard
    from guardrails.hub import ProfanityFree

    # Create a Guard
    guard = Guard().use(ProfanityFree())

    # Create a Cohere client
    cohere_client = cohere.Client(api_key="my_api_key")

    # Wrap cohere API call
    validated_response = guard(
        cohere_client.chat,
        prompt="Can you try to generate a list of 10 things that are not food?",
        model="command",
        max_tokens=100,
        ...
    )
    ```
  </TabItem>
</Tabs>


## LLMs supported via LiteLLM

[LiteLLM](https://docs.litellm.ai/docs/) is a lightweight wrapper that unifies the interface for over 100+ LLMs. Guardrails only supports 4 LLMs natively, but you can use Guardrails with LiteLLM to support over 100+ LLMs. You can read more about the LLMs supported by LiteLLM [here](https://docs.litellm.ai/docs/providers).

In order to use Guardrails with any of the LLMs supported through liteLLM, you need to do the following:
1. Call the `Guard.__call__` method with `litellm.completion` as the first argument.
2. Pass any additional litellm arguments as keyword arguments to the `Guard.__call` method.

Some examples of using Guardrails with LiteLLM are shown below.

### Use Guardrails with Ollama

```python
import litellm
from guardrails import Guard
from guardrails.hub import ProfanityFree

# Create a Guard class
guard = Guard().use(ProfanityFree())

# Call the Guard to wrap the LLM API call
validated_response = guard(
    litellm.completion,
    model="ollama/llama2",
    max_tokens=500,
    api_base="http://localhost:11434",
    messages=[{"role": "user", "content": "hello"}]
)
```

### Use Guardrails with Azure's OpenAI endpoint

```python
import os

import litellm
from guardrails import Guard
from guardrails.hub import ProfanityFree

validated_response = guard(
    litellm.completion,
    model="azure/<your deployment name>",
    max_tokens=500,
    api_base=os.environ.get("AZURE_OPENAI_API_BASE"),
    api_version="2023-05-15",
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    messages=[{"role": "user", "content": "hello"}]
)
```

## Build a custom LLM wrapper

In case you're using an LLM that isn't natively supported by Guardrails and you don't want to use LiteLLM, you can build a custom LLM API wrapper. In order to use a custom LLM, create a function that takes accepts a prompt as a string and any other arguments that you want to pass to the LLM API as keyword args. The function should return the output of the LLM API as a string.

```python
from guardrails import Guard
from guardrails.hub import ProfanityFree

# Create a Guard class
guard = Guard().use(ProfanityFree())

# Function that takes the prompt as a string and returns the LLM output as string
def my_llm_api(
    **kwargs
) -> str:
    """Custom LLM API wrapper.

    At least messages should be provided.

    Args:
        messages (list[dict]): The message history to be passed to the LLM API
        **kwargs: Any additional arguments to be passed to the LLM API

    Returns:
        str: The output of the LLM API
    """
    messages=kwargs.get("messages")
    # Call your LLM API here
    llm_output = some_llm(messages, **kwargs)

    return llm_output

# Wrap your LLM API call
validated_response = guard(
    my_llm_api,
    prompt="Can you generate a list of 10 things that are not food?",
    **kwargs,
)
```
