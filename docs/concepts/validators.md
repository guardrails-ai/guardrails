# Validators

Validators are how we apply quality controls to the schemas specified in our `RAIL` specs.  They specify the criteria to measure whether an output is valid, as well as what actions to take when an output does not meet those criteria.

## How do Validators work?
When a validator is applied to a property on a schema, and output is provided for that schema, either by wrapping the LLM call or passing in the LLM output, the validators are executed against the values for the properties they were applied to.  If the value for the property passes the criteria defined, a `PassResult` is returned from the validator.  This `PassResult` tells Guardrails to treat the value as if it is valid.  In most cases this means returning that value for that property at the end; other advanced cases, like using a value override, will be covered in other sections.  If, however, the value for the property does not pass the criteria, a `FailResult` is returned.  This in turn tells Guardrails to take any corrective actions defined for the property and validation.  Corrective actions are defined by the `on-fail-...` attributes in a `RAIL` spec.  You can read more about what corrective actions are available [here](/concepts/output/#specifying-corrective-actions).

## Validator Structure
### Arguments
Now we know that a validator is some method that checks the value of a given schema property.  By this definition we can assume that the validator uses that schema property's value at runtime.  In some cases the validator may require other arguments to successfully execute.  These arguments are defined in the validator's method signature and assigned values in the `RAIL` spec.  For example, the `ValidLength` (aka `length`) validator needs to know the bounds of the sized object (string, list, dict, etc.).  The bounds are specified by a `min` and `max` argument.

This means that the `ValidLength`'s constructor accepts these arguments so they can be used during validation later:
```python
@register_validator(name="length", data_type=["string", "list"])
class ValidLength(Validator):
    def __init__(
        self, min: int = None, max: int = None, on_fail: Optional[Callable] = None
    ):
    # ...
```

The values for these arguments are provided when assigning this validator within a `RAIL` spec as positional arguments.

For example:
```xml
<output
    type="string"
    format="length: 1 120"
/>
```
will apply the `ValidLength` formater to the string output requiring a minimum character count of 1 and a max of 120.

### Metadata
Sometimes validators need addtional parameters that are only availble during runtime.  This is where metadata comes in.  Metadata could be data generated during the execution of a validtor (*important if you're writing your own validators*), or could just be a container for runtime arguments.  For example, the `ExtractedSummarySentencesMatch` validator accepts a `filepaths` property in the metadata dictionary to specify what source files to compare the summary against to ensure similarity.  Unlike arguments which are specified in the `RAIL` spec, metadata is specified when calling the guard:
```python
guard = Guard.from_rail("my_railspec.rail")

raw_output, guarded_output = guard(
    llm_api=openai.ChatCompletion.create,
    model="gpt-3.5-turbo",
    num_reasks=3,
    metadata={
        "filepaths": [
            "./my_data/article1.txt",
            "./my_data/article2.txt",
        ]
    }
)
```

#### How do I know what metadata is required?
First step is to check the docs.  Each validator has an API reference that documents both its initialization arguments and any required metadata that must be supplied at runtime.  Continuing with the example used above, `ExtractedSummarySentencesMatch` accepts an optional threshold argument which defaults to `0.7`; it also requires an entry in the metadata called `filepaths` which is an array of strings specifying which documents to use for the similarity comparison.  You can see an example of a Validator's metadata documentation [here](../api_reference/validators.md/#guardrails.validators.ExtractedSummarySentencesMatch).

Secondly, if a piece of metadata is required and not present, a `RuntimeError` will be raised.  For example, if the metadata requirements are not met for the above validator, an `RuntimeError` will be raised with the following message:

> extracted-sentences-summary-match validator expects `filepaths` key in metadata

## Custom Validators
If you need to perform a validation that is not currently supported by the [validators](../api_reference/validators.md) included in guardrails, you can create your own custom validators to be used in your local python environment.

A custom validator can be as simple as a single function if you do not require addtional arguments:
```py
from typing import Dict
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult
)

@register_validator(name="starts-with-a", data_type="string")
def starts_with_a(value: str, metadata: Dict) -> ValidationResult:
    if value.startswith("a"):
        return PassResult(metadata)
    
    return FailResult(
        error_message=f"Value {value} does not start with a."
    )

```

If you need to perform more complex operations or require addtional arguments to perform the validation, then the validator can be specified as a class that inherits from our base Validator class:
```py
from typing import Callable, Dict, Optional
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
    Validator
)

@register_validator(name="starts-with", data_type="string")
class StartsWith(Validator):
    def __init__(self, prefix: str, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, prefix=prefix)
        self.prefix = prefix

    def validate(value: str, metadata: Dict) -> ValidationResult:
        if value.startswith(self.prefix):
            return PassResult(metadata)
        
        return FailResult(
            error_message=f"Value {value} does not start with {self.prefix}.",
            fix_value: f"{self.prefix}{value}" # To enable the "fix" option for on-fail
        )
```

Custom validators must be defined before creating a `Guard` or `RAIL` spec in the code, but otherwise can be used like built in validators:
```py
import openai
from guardrails import Guard
from .my_custom_validators import starts_with_a, StartsWith

rail_str = """
<rail version="0.1">

<output>
    <string name="a-string" type="string" format="starts-with-a">
    <string name="custom-string" type="string" format="starts-with: my-prefix">
</output>


<prompt>
Generate a dataset of fake word pairs within a JSON object.
The "a-string" property should start with the letter "a",
and the "custom-string" property should start with "my-prefix".

${guardrails.complete_json_suffix}
</prompt>

</rail>
"""

guard = Guard.from_rail_string(rail_string=rail_str)

raw_output, guarded_output = guard(
    llm_api=openai.ChatCompletion.create,
    model="gpt-3.5-turbo"
)

print("validated output: ", guarded_output)
``` 