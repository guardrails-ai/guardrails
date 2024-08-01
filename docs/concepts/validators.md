# Validators

Validators are how we apply quality controls to the outputs of LLMs.  They specify the criteria to measure whether an output is valid, as well as what actions to take when an output does not meet those criteria.

## How do Validators work?
Each validator is a method that encodes some criteria, and checks if a given value meets that criteria.

- If the value passes the criteria defined, the validator returns `PassResult`. In most cases this means returning that value unchanged. In very few advanced cases, there may be a a value override (the specific validator will document this).
- If the value does not pass the criteria, a `FailResult` is returned.  In this case, the validator applies the user-configured `on_fail` policies (see [On-Fail Policies](/docs/hub/concepts/on_fail_policies.md) for more details).

## Runtime Metadata

Occasionally, validators need additional metadata that is only available during runtime. Metadata could be data generated during the execution of a validator (*important if you're writing your own validators*), or could just be a container for runtime arguments.

As an example, the `ExtractedSummarySentencesMatch` validator accepts a `filepaths` property in the metadata dictionary to specify what source files to compare the summary against to ensure similarity.  Unlike arguments which are specified at validator initialization, metadata is specified when calling `guard.validate`. For more information on how to use metadata, see [How to use Metadata](/docs/hub/how_to_guides/metadata.md).

```python
guard = Guard.from_rail("my_railspec.rail")

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

## Submitting a Custom Validator to the Hub

There are two ways to create a new validator and submit it to the Hub.

1. For lightweight validators, use the `hub` CLI to create a new validator and submit it to the Hub.
2. For more complex validators, clone the Validator-Template repository and register the validator via the Guardrails Hub website.

### Creating a new validator using the `hub` CLI

The `hub` CLI provides a simple way to create a new validator and submit it to the Hub. The `hub` CLI will create a new validator in the current directory and submit it to the Hub.

To create a new validator using the `hub` CLI, run the following command:

```bash
guardrails hub create-validator my_validator
```

This will create a new file called `my_validator.py` in the current directory. The file will contain a template and instructions for creating a new validator.

```bash
guardrails hub submit my_validator
```

### Creating a new validator using the Validator-Template repository

For more complex validators, you can clone the [Validator-Template repository](https://github.com/guardrails-ai/validator-template) and register the validator via a Google Form on the Guardrails Hub website.

```bash
git clone git@github.com:guardrails-ai/validator-template.git
```

Once the repository is cloned and the validator is created, you can register the validator via this [Google Form](https://forms.gle/N6UaE9611niuMxZj7).



