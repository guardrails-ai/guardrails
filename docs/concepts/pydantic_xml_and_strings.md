# Pydantic, XML, and Strings

Guardrails is currently only available in python, but we do want to expand to other languages as demand becomes apparent. 

As you've seen in other concepts pages, xml and pydantic are used to model RailSpecs. We recommend using pydantic2 instead of xml. The modeling capabilities are easier to use and more powerful, and it integrates more seamlessly with a python codebase. All examples in these docs have representation for both xml and pydantic.

Guardrails additionally supports single-field, string outputs. This is useful for running validations on simple, single-value outputs.

### Simple Example

In this simple example, we initialize a guard that validates a string output and ensures that it is within 10-20 characters.

=== "XML"

```xml
from guardrails import Guard
railspec = """
<rail version="0.1">
    <output>
        <string name="puppy_name" format="length: 10 20"/>
    </output>
</rail>
"""

guard = Guard.from_railspec(railspec)
```

=== "Pydantic"

```python
from pydantic import Field, 
```

=== "Single-Field String"

hi
