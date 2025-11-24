import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quickstart: In-Application

## Introduction

Guardrails is a framework that validates and structures data from language models. These validations range simple checks like regex matching to more complex checks like competitor analysis. Guardrails can be used with any language model.

## Installation

### Download Guardrails (required)
First, install Guardrails for your desired language: 

```bash
pip install guardrails-ai
```

### Generate an API Key (required)
Next, get a free API key from [Guardrails Hub](https://hub.guardrailsai.com/keys).  Copy and keep track of this API key because you'll use it in the next step to configure the CLI.

### Configure the Guardrails CLI (required)

Configure the Guardrails CLI with the command:
    
```bash
guardrails configure
```

The configuration process will ask you three questions:

1. Whether you want to enable metrics reporting.
2. Whether you want to use our hosted remote inference endpoints for validators that utilize large ML models.
3. To enter the API key you generated in the previous step.  This is required in order to install validators in the next step.


### Install a Validator from the Guardrails Hub
In order to perform any validation on LLM output with Guardrails, you will need to install an appropriate validator for your use case from the [Guardrails Hub](https://hub.guardrailsai.com).  Each validator has a unique and identifying Hub URI that can be used in the Guardrails CLI to install it into your current environment.  You can find the exact install command for a validator on it's details page in the [Guardrails Hub](https://hub.guardrailsai.com).  For example, the [Detect PII](https://hub.guardrailsai.com/validator/guardrails/detect_pii) validator can be installed via:

    ```bash
    guardrails hub install hub://guardrails/detect_pii
    ```

## Usage
1. Create a Guard with an installed validator.
    First, install the validator you want to use from the Guardrails Hub:
    
    ```bash
    guardrails hub install hub://guardrails/regex_match --quiet
    ```

    Next, you can import this validator from the `guardrails.hub` module and use it to construct a Guard.
    ```python
    # Import Guard and Validator
    from guardrails.hub import RegexMatch
    from guardrails import Guard

    # Initialize the Guard with 
    guard = Guard().use(
        RegexMatch(regex="^[A-Z][a-z]*$")
    )

    print(guard.parse("Caesar").validation_passed)  # Guardrail Passes
    print(
        guard.parse("Caesar Salad")
        .validation_passed
    )  # Guardrail Fails
    ```
2. Run multiple validators within a Guard.
    First, install the necessary validators from Guardrails Hub.

    ```bash
    guardrails hub install hub://guardrails/regex_match hub://guardrails/valid_length --quiet
    ```

    Then, create a Guard from the installed validators.
    
    ```python
    from guardrails.hub import RegexMatch, ValidLength
    from guardrails import Guard

    guard = Guard().use_many(
        RegexMatch(regex="^[A-Z][a-z]*$"),
        ValidLength(min=1, max=12)
    )

    print(guard.parse("Caesar").validation_passed)  # Guardrail Passes
    print(
        guard.parse("Caesar Salad")
        .validation_passed
    )  # Guardrail Fails due to regex match
    print(
        guard.parse("Caesarisagreatleader")
        .validation_passed
    )  # Guardrail Fails due to length
    ```

## Structured Data Generation and Validation

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

prompt = """
    What kind of pet should I get and what should I name it?

    ${gr.complete_json_suffix_v2}
"""
guard = Guard.for_pydantic(output_class=Pet)

res = guard(
    model="gpt-3.5-turbo",
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

### Install the Javascript library

**Note**: The Javascript library works via an I/O bridge to run the underlying Python library. You must have Python 3.10 or greater installed on your system to use the Javascript library. 


```bash
npm i @guardrails-ai/core
```

### Install specific version

<Tabs>

<TabItem value="py" label="Python">

To install a specific version in Python, run:

```bash
# pip install guardrails-ai==[version-number]

# Example:
pip install guardrails-ai==0.5.0a13
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
# Example:
pip install git+https://github.com/guardrails-ai/guardrails.git@main
```

</TabItem>
<TabItem value="js" label="JavaScript">

```bash
npm i git+https://github.com/guardrails-ai/guardrails-js.git
```

</TabItem>

</Tabs>