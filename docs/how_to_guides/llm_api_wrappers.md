import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Use Guardrails with any LLMs

Guardrails' `Guard` wrappers provide a simple way to add Guardrails to your LLM API calls. The wrappers are designed to be used with any LLM API.

There are three ways to use Guardrails with an LLM API:
1. [**Natively-supported LLMs**](#natively-supported-llms): Guardrails provides out-of-the-box wrappers for OpenAI, Cohere, Anthropic and HuggingFace. If you're using any of these APIs, check out the documentation in the [Natively-supported LLMs](#natively-supported-llms) section.
2. [**LLMs supported through LiteLLM**](#litellm): Guardrails provides an easy integration with LiteLLM, a lightweight abstraction over LLM APIs that supports over 100+ LLMs. If you're using an LLM that isn't natively supported by Guardrails, you can use LiteLLM to integrate it with Guardrails. Check out the documentation in the [LiteLLM](#litellm) section.
3. [**Custom LLM**](#custom-llm): If you're using an LLM that isn't natively supported by Guardrails and you don't want to use LiteLLM, you can build a custom LLM API wrapper. Check out the documentation in this [Custom LLM](#custom-llm) section.


## Natively-supported LLMs

<Tabs>
  <TabItem value="openai" label="OpenAI" default>
    ```python
    import openai
    import guardrails as gd

    # Create a Guard class
    guard = gd.Guard.from_rail(...)

    # Wrap openai API call
    raw_llm_output, guardrail_output, *rest = guard(
        openai.ChatCompletion.create,
        prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
        system_prompt="You are a helpful assistant...",
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.0,
    )
    ```
  </TabItem>
  <TabItem value="cohere" label="Cohere">
    ```python
    import cohere
    import guardrails as gd

    # Create a Guard class
    guard = gd.Guard.from_rail(...)

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
  </TabItem>
  <TabItem value="anthropic" label="Anthropic">
    ```python
    from anthropic import Anthropic
    import guardrails as gd

    # Create a Guard class
    guard = gd.Guard.from_rail(...)

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
  </TabItem>
  <TabItem value="huggingface" label="🤗 HuggingFace">
    ```python
    import torch
    from guardrails import Guard
    from guardrails.validators import ValidLength, ToxicLanguage
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)

    # Customize your model inputs if desired.
    # If you don't pass and inputs (`input_ids`, `input_values`, `input_features`, or `pixel_values`)
    # We'll try to do something similar to below using the tokenizer and the prompt.
    # We strongly suggest passing in your own inputs.
    model_inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)

    # Create your prompt or starting text
    prompt = "Hello, I'm a language model,"

    # Create the Guard
    guard = Guard.from_string(
        validators=[
            ValidLength(min=48, on_fail="fix"),
            ToxicLanguage(on_fail="fix")
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
  </TabItem>
  <TabItem value="huggingface_pipelines" label="🤗 Pipelines">
    ```python
    from guardrails import Guard
    from guardrails.validators import ValidLength, ToxicLanguage
    import torch
    from transformers import pipeline


    # Create your prompt or starting text
    prompt = "What are we having for dinner?"

    # Setup pipeline
    generator = pipeline("text-generation", model="facebook/opt-350m")


    # Create the Guard
    guard = Guard.from_string(
        validators=[
            ValidLength(
                min=48,
                on_fail="fix"
            ),
            ToxicLanguage(
                on_fail="fix"
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
  </TabItem>
  <TabItem value="manifest" label="Manifest">
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
    raw_llm_output, guardrail_output, *rest = guard(
        manifest,
        prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
        max_tokens=100,
        temperature=0.0,
    )
    ```
  </TabItem>
</Tabs>


## LLMs supported via LiteLLM

LiteLLM is a lightweight wrapper that unifies the interface for over 100+ LLMs. Guardrails only supports 4 LLMs natively, but you can use Guardrails with LiteLLM to support over 100+ LLMs.

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
response = guard(
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

response = guard(
    litellm.completion,
    model="azure/<your deployment name>",
    prompt="Please help me write a poem about Kubernetes in the style of Frost.",
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
    llm_output = some_llm(prompt, **kwargs)

    return llm_output


# Wrap your LLM API call
raw_llm_output, guardrail_output, *rest = guard(
    my_llm_api,
    prompt_params={"prompt_param_1": "value_1", "prompt_param_2": "value_2", ..},
    **kwargs,
)
```
