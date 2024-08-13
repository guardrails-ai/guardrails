# Installing Validators

## Guardrails Hub

Validators can be combined together into Input and Output Guards that intercept the inputs and outputs of LLMs. There are a large collection of Validators which can be found at the [Guardrails Hub](https://hub.guardrailsai.com/).

<div align="center">
<img src="https://raw.githubusercontent.com/guardrails-ai/guardrails/main/docs/img/guardrails_hub.gif" alt="Guardrails Hub gif" width="600px" />
</div>


## Installing

Once you have found a Validator on the hub, you can click on the Validator REAMDE to find the install link.

### Using CLI

You can install a validator using the Guardrails CLI. For example the [Toxic Language](https://hub.guardrailsai.com/validator/guardrails/toxic_language) validator can be installed with:

```bash
guardrails hub install hub://guardrails/toxic_language
```

At which point you can start to use the Validator:

```python
from guardrails.hub import ToxicLanguage
from guardrails import Guard

guard = Guard().use(
    ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
)

guard.validate("My landlord is an asshole!") 
```

### In Code Installs - Pattern A

You can also install validators using the Guardrails SDK which simplifies development particularly when using Jupyter Notebooks:

```python
from guardrails import install

install("hub://guardrails/toxic_language")

from guardrails.hub import ToxicLanguage

guard = Guard().use(
    ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
)

guard.validate("My landlord is an asshole!") 
```
> Note: Invoking `install` always installs the validator module so it's recommended for the install to be in a seperate code block when using Notebooks.

### In Code Installs - Pattern B

You can also extract the validator directly from the installed module as follows:

```python
from guardrails import install

ToxicLanguage = install("hub://guardrails/toxic_language").ToxicLanguage

guard = Guard().use(
    ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
)

guard.validate("My landlord is an asshole!") 
```

> Note: Invoking `install` always installs the validator module so it's recommended for the install to be in a seperate code block when using Notebooks.