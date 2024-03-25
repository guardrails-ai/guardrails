<div align="center">

<img src="https://raw.githubusercontent.com/guardrails-ai/guardrails/main/docs/img/Guardrails-ai-logo-for-dark-bg.svg#gh-dark-mode-only" alt="Guardrails AI Logo" width="600px">
<img src="https://raw.githubusercontent.com/guardrails-ai/guardrails/main/docs/img/Guardrails-ai-logo-for-white-bg.svg#gh-light-mode-only" alt="Guardrails AI Logo" width="600px">

<hr>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/guardrails-ai)
[![CI](https://github.com/guardrails-ai/guardrails/actions/workflows/ci.yml/badge.svg)](https://github.com/guardrails-ai/guardrails/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/guardrails-ai/guardrails/graph/badge.svg?token=CPkjw91Ngo)](https://codecov.io/gh/guardrails-ai/guardrails)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/guardrails_ai)](https://x.com/guardrails_ai)
[![Discord](https://img.shields.io/discord/1085077079697150023?logo=discord&label=support&link=https%3A%2F%2Fdiscord.gg%2Fgw4cR9QvYE)](https://discord.gg/U9RKkZSBgx)
[![Static Badge](https://img.shields.io/badge/Docs-blue?link=https%3A%2F%2Fwww.guardrailsai.com%2Fdocs)](https://www.guardrailsai.com/docs)
[![Static Badge](https://img.shields.io/badge/Blog-blue?link=https%3A%2F%2Fwww.guardrailsai.com%2Fblog)](https://www.guardrailsai.com/blog)

</div>

## What is Guardrails?

Guardrails is a Python framework that helps build reliable AI applications by performing two key functions:
1. Guardrails runs Input/Output Guards in your application that detect, quantify and mitigate the presence of specific types of risks. To look at the full suite of risks, check out [Guardrails Hub](https://hub.guardrailsai.com/).
2. Guardrails help you generate structured data from LLMs.


<div align="center">
<img src="https://raw.githubusercontent.com/guardrails-ai/guardrails/main/docs/img/with_and_without_guardrails.svg" alt="Guardrails in your application" width="1500px">
</div>


### Guardrails Hub

Guardrails Hub is a collection of pre-built measures of specific types of risks (called 'validators'). Multiple validators can be combined together into Input and Output Guards that intercept the inputs and outputs of LLMs. Visit [Guardrails Hub](https://hub.guardrailsai.com/) to see the full list of validators and their documentation.

<div align="center">
<img src="https://raw.githubusercontent.com/guardrails-ai/guardrails/main/docs/img/guardrails_hub.gif" alt="Guardrails Hub gif" width="600px">
</div>


## Installation

```python
pip install guardrails-ai
```


## Getting Started


### Create Input and Output Guards for LLM Validation

1. Download and configure the Guardrails Hub CLI.
    
    ```bash
    pip install guardrails-ai
    guardrails configure
    ```
2. Install a guardrail from Guardrails Hub.

    ```bash
    guardrails hub install hub://guardrails/regex_match
    ```
3. Create a Guard from the installed guardrail.

    ```python
    # Import Guard and Validator
    from guardrails.hub import RegexMatch
    from guardrails import Guard

    # Initialize the Guard with 
    val = Guard().use(
        RegexMatch(regex="^[A-Z][a-z]*$")
    )

    guard.parse("Caesar")  # Guardrail Passes
    guard.parse("Caesar is a great leader")  # Guardrail Fails
    ```
4. Run multiple guardrails within a Guard.
    First, install the necessary guardrails from Guardrails Hub.

    ```bash
    guardrails hub install hub://guardrails/competitor_check
    guardrails hub install hub://guardrails/toxic_language
    ```

    Then, create a Guard from the installed guardrails.
    
    ```python
    from guardrails.hub import RegexMatch, ValidLength
    from guardrails import Guard

    guard = Guard().use(
        RegexMatch(regex="^[A-Z][a-z]*$"),
        ValidLength(min=1, max=32)
    )

    guard.parse("Caesar")  # Guardrail Passes
    guard.parse("Caesar is a great leader")  # Guardrail Fails
    ```


### Use Guardrails to generate structured data from LLMs


Let's go through an example where we ask an LLM to generate fake pet names. To do this, we'll create a Pydantic [BaseModel](https://docs.pydantic.dev/latest/api/base_model/) that represents the structure of the output we want.

```py
from pydantic import BaseModel, Field

class Pet(BaseModel):
    pet_type: str = Field(description="Species of pet")
    name: str = Field(description="a unique pet name")
```

Now, create a Guard from the `Pet` class. The Guard can be used to call the LLM in a manner so that the output is formatted to the `Pet` class. Under the hood, this is done by either of two methods:
1. Function calling: For LLMs that support function calling, we generate structured data using the function call syntax.
2. Prompt optimization: For LLMs that don't support function calling, we add the schema of the expected output to the prompt so that the LLM can generate structured data.

```py
from guardrails import Guard
import openai

prompt = """
    What kind of pet should I get and what should I name it?

    ${gr.complete_json_suffix_v2}
"""
guard = Guard.from_pydantic(output_class=Pet, prompt=prompt)

validated_output, *rest = guard(
    llm_api=openai.completions.create,
    engine="gpt-3.5-turbo-instruct"
)

print(f"{validated_output}")
```

This prints: 
```
{
    "pet_type": "dog",
    "name": "Buddy
}
```

## FAQ

#### I'm running into issues with Guardrails. Where can I get help?

You can reach out to us on [Discord](https://discord.gg/gw4cR9QvYE) or [Twitter](https://twitter.com/guardrails_ai).

#### Can I use Guardrails with any LLM?

Yes, Guardrails can be used with proprietary and open-source LLMs. Check out this guide on [how to use Guardrails with any LLM](https://www.guardrailsai.com/docs/how_to_guides/llm_api_wrappers).

#### Can I create my own validators?

Yes, you can create your own validators and contribute them to Guardrails Hub. Check out this guide on [how to create your own validators](https://www.guardrailsai.com/docs/hub/how_to_guides/custom_validator).

#### Does Guardrails support other languages?

Guardrails can be used with Python and JavaScript. Check out the docs on how to use Guardrails from JavaScript. We are working on adding support for other languages. If you would like to contribute to Guardrails, please reach out to us on [Discord](https://discord.gg/gw4cR9QvYE) or [Twitter](https://twitter.com/guardrails_ai).


## Contributing

We welcome contributions to Guardrails!

Get started by checking out Github issues and check out the [Contributing Guide](CONTRIBUTING.md). Feel free to open an issue, or reach out if you would like to add to the project!
