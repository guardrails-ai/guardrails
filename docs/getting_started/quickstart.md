import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quickstart

## Introduction

Guardrails is a framework that validates and structures data from language models. These validations range simple checks like regex matching to more complex checks like competitor analysis. Guardrails can be used with any language model.

## Installation

### Download Guardrails (required)

Guardrails consists of two parts:

* The core Guardrails package (Python or Javascript)
* Guardrails Hub, which enables downloading from our library of publicly available validators. 

**Note**: The Javascript library works via an I/O bridge to run the underlying Python library. You must have Python 3.16 or greater installed on your system to use the Javascript library. 

First, install Guardrails for your desired language: 

<Tabs>

<TabItem value="py" label="Python">

```bash
pip install guardrails-ai
```

</TabItem>
<TabItem value="js" label="JavaScript">

```bash
npm i @guardrails-ai/core
```

</TabItem>

</Tabs>

### Configure the Guardrails CLI (required)
    
```bash
guardrails configure
```

## Usage

1. Install a guardrail from [Guardrails Hub](https://hub.guardrailsai.com).

    ```bash
    guardrails hub install hub://guardrails/regex_match
    ```
2. Create a Guard from the installed guardrail.

    ```python
    # Import Guard and Validator
    from guardrails.hub import RegexMatch
    from guardrails import Guard

    # Initialize the Guard with 
    guard = Guard().use(
        RegexMatch(regex="^[A-Z][a-z]*$")
    )

    guard.parse("Caesar")  # Guardrail Passes
    guard.parse("Caesar is a great leader")  # Guardrail Fails
    ```
3. Run multiple guardrails within a Guard.
    First, install the necessary guardrails from Guardrails Hub.

    ```bash
    guardrails hub install hub://guardrails/competitor_check
    guardrails hub install hub://guardrails/toxic_language
    ```

    Then, create a Guard from the installed guardrails.
    
    ```python
    from guardrails.hub import RegexMatch, ValidLength
    from guardrails import Guard

    guard = Guard().use_many(
        RegexMatch(regex="^[A-Z][a-z]*$"),
        ValidLength(min=1, max=32)
    )

    guard.parse("Caesar")  # Guardrail Passes
    guard.parse("Caesar is a great leader")  # Guardrail Fails
    ```

## Structured data example

Now, let's go through an example where we ask an LLM to generate fake pet names. 

1. Create a Pydantic [BaseModel](https://docs.pydantic.dev/latest/api/base_model/) that represents the structure of the output we want.

```py
from pydantic import BaseModel, Field

class Pet(BaseModel):
    pet_type: str = Field(description="Species of pet")
    name: str = Field(description="a unique pet name")
```

2. create a Guard from the `Pet` class. The Guard can be used to call the LLM in a manner so that the output is formatted to the `Pet` class. Under the hood, this is done by either of two methods:

(1) Function calling: For LLMs that support function calling, we generate structured data using the function call syntax.

(2) Prompt optimization: For LLMs that don't support function calling, we add the schema of the expected output to the prompt so that the LLM can generate structured data.

```py
from guardrails import Guard
import openai

prompt = """
    What kind of pet should I get and what should I name it?

    ${gr.complete_json_suffix_v2}
"""
guard = Guard.from_pydantic(output_class=Pet)

res = guard(
    engine="gpt-3.5-turbo",
    messages=[{
        "role": "user",
        "content": prompt
    }]
)

print(f"{res.validated_output}")
```

This prints: 
```
{
    "pet_type": "dog",
    "name": "Buddy
}
```

## Advanced installation instructions

### Install specific version

<Tabs>

<TabItem value="py" label="Python">

To install a specific version in Python, run:

```bash
# pip install guardrails-ai==[version-number]

# Example:
pip install guardrails-ai==0.2.0a6
```

</TabItem>
<TabItem value="js" label="JavaScript">

To install a pre-release version with Javascript, install it with the intended semantic version. 

</TabItem>

</Tabs>

### Install from GitHub

Installing directly from GitHub is useful when a release has not yet been cut with the changes pushed to a branch that you need. Non-released versions may include breaking changes, and may not yet have full test coverage. We recommend using a released version whenever possible.

<Tabs>

<TabItem value="py" label="Python">

```bash
# pip install git+https://github.com/guardrails-ai/guardrails.git@[branch/commit/tag]
# Examples:
pip install git+https://github.com/guardrails-ai/guardrails.git@main
pip install git+https://github.com/guardrails-ai/guardrails.git@dev
```

</TabItem>
<TabItem value="js" label="JavaScript">

```bash
npm i git+https://github.com/guardrails-ai/guardrails-js.git
```

</TabItem>

</Tabs>