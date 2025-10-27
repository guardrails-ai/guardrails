# Getting Started

## Installation

Install the Guardrails core package and CLI using pip.

```python
pip install guardrails-ai
```

## Create Input and Output Guards for LLM Validation

1. Configure the Guardrails Hub CLI.
    
    ```bash
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
    guard = Guard().use(
        RegexMatch(regex="^[A-Z][a-z]*$")
    )

    guard.parse("Caesar")  # Guardrail Passes
    guard.parse("Caesar is a great leader")  # Guardrail Fails
    ```
4. Run multiple guardrails within a Guard.
    First, install the necessary guardrails from Guardrails Hub.

    ```bash
    guardrails hub install hub://guardrails/regex_match
    guardrails hub install hub://guardrails/valid_length
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


## Use Guardrails to generate structured data from LLMs


Let's go through an example where we ask an LLM to generate fake pet names. To do this, we'll create a Pydantic [BaseModel](https://docs.pydantic.dev/latest/api/base_model/) that represents the structure of the output we want.

```py
from pydantic import BaseModel, Field

class Pet(BaseModel):
    pet_type: str = Field(description="Species of pet")
    name: str = Field(description="a unique pet name")
```

Now, create a Guard from the `Pet` class. The Guard can be used to call the LLM in a manner so that the output is formatted to match the `Pet` class. Under the hood, this is done by either of two methods:
1. Function calling: For LLMs that support function calling, we generate structured data using the function call syntax.
2. Prompt optimization: For LLMs that don't support function calling, we add the schema of the expected output to the prompt so that the LLM can generate structured data.

```py
from guardrails import Guard

prompt = """
    What kind of pet should I get and what should I name it?

    ${gr.complete_xml_suffix_v2}
"""
guard = Guard.for_pydantic(output_class=Pet)

res = guard(
    messages=[{"role": "user", "content": prompt}],
    model="gpt-3.5-turbo"
)

print(res.validated_output)
```

This prints:
```
{'pet_type': 'dog', 'name': 'Buddy'}
```

This output is a dict that matches the structure of the `Pet` class.