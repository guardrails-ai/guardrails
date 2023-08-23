# Pydantic, XML, and Strings

Guardrails is currently only available in python, but we do want to expand to other languages as demand becomes apparent. 

As you've seen in other concepts pages, xml and pydantic are used to model RailSpecs. We recommend using pydantic instead of xml. The modeling capabilities are easier to use and more powerful, and it integrates more seamlessly with a python codebase. All examples in these docs have representation for both xml and pydantic.

Guardrails additionally supports single-field, string outputs. This is useful for running validations on simple, single-value outputs.

### Simple Example

In this simple example, we initialize a guard that validates a string output and ensures that it is within 10-20 characters.

=== "XML"

```py
from guardrails import Guard
railspec = """
<rail version="0.1">
    <output 
        type="string"
        format="length: 10 20"
        description="Puppy name"
    />
</rail>
"""

guard = Guard.from_rail_string(railspec)
```

=== "Pydantic"

N/A - Pydantic models can only be used to reprensent structured output (i.e. JSON).

=== "Single-Field String"

```py
from guardrails import Guard
from guardrails.validators import ValidLength

validators = [ValidLength(10, 20)]

guard = Guard.from_string(
    validators=validators
    description="Puppy name"
)
```
