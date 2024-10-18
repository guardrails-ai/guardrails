# Validators

Validators are how we apply quality controls to the outputs of LLMs.  They specify the criteria to measure whether an output is valid, as well as what actions to take when an output does not meet those criteria.

## How do Validators work?
Each validator is a method that encodes some criteria, and checks if a given value meets that criteria.

- If the value passes the criteria defined, the validator returns `PassResult`. In most cases this means returning that value unchanged. In very few advanced cases, there may be a a value override (the specific validator will document this).
- If the value does not pass the criteria, a `FailResult` is returned.  In this case, the validator applies the user-configured `on_fail` policies (see [On-Fail Policies](/docs/hub/concepts/on_fail_policies.md) for more details).

## Runtime Metadata

Occasionally, validators need addtional metadata that is only available during runtime. Metadata could be data generated during the execution of a validtor (*important if you're writing your own validators*), or could just be a container for runtime arguments.

As an example, the `ExtractedSummarySentencesMatch` validator accepts a `filepaths` property in the metadata dictionary to specify what source files to compare the summary against to ensure similarity.  Unlike arguments which are specified at validator initialization, metadata is specified when calling `guard.validate`. For more information on how to use metadata, see [How to use Metadata](/docs/hub/how_to_guides/metadata.md).

```python
guard = Guard.for_rail("my_railspec.rail")

outcome = guard(
    llm_api=openai.chat.completions.create,
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

## Custom Validators

If you need to perform a validation that is not currently supported by the hub, you can create your own custom validators.

A custom validator can be as simple as a single function if you do not require addtional arguments:

```py
from typing import Dict
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
)

@register_validator(name="starts-with-a", data_type="string")
def starts_with_a(value: str, metadata: Dict) -> ValidationResult:
    if value.startswith("a"):
        return PassResult()

    return FailResult(
        error_message=f"Value {value} does not start with a.",
        fix_value="a" + value,
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
    Validator,
)

@register_validator(name="starts-with", data_type="string")
class StartsWith(Validator):
    def __init__(self, prefix: str, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, prefix=prefix)
        self.prefix = prefix

    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        if value.startswith(self.prefix):
            return PassResult()

        return FailResult(
            error_message=f"Value {value} does not start with {self.prefix}.",
            fix_value=self.prefix + value,  # To enable the "fix" option for on-fail
        )
```

Custom validators must be defined before creating a `Guard` or `RAIL` spec in the code, 
but otherwise can be used like built in validators. It can be used in a `RAIL` spec OR
a `Pydantic` model like so:
