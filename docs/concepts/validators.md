# Validators

Validators are how we apply quality controls to the outputs of LLMs.  They specify the criteria to measure whether an output is valid, as well as what actions to take when an output does not meet those criteria.

## How do Validators work?
Each validator is a method that encodes some criteria, and checks if a given value meets that criteria.

- If the value passes the criteria defined, the validator returns `PassResult`. In most cases this means returning that value unchanged. In very few advanced cases, there may be a a value override (the specific validator will document this).
- If the value does not pass the criteria, a `FailResult` is returned.  In this case, the validator applies the user-configured `on_fail` policies (see [On-Fail Policies](/concepts/validator_on_fail_actions)).

## Runtime Metadata

Occasionally, validators need additional metadata that is only available during runtime. Metadata could be data generated during the execution of a validator (*important if you're writing your own validators*), or could just be a container for runtime arguments.

As an example, the `ExtractedSummarySentencesMatch` validator accepts a `filepaths` property in the metadata dictionary to specify what source files to compare the summary against to ensure similarity.  Unlike arguments which are specified at validator initialization, metadata is specified when calling `guard.validate` or `guard.__call__` (this is the `guard()` function).

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

If multiple validators require metadata, create a single metadata dictionary that contains the metadata keys for each validator. In the example below, both the `Provenance_LLM` and `DetectPII` validators require metadata.

```python
from guardrails import Guard
from guardrails.hub import DetectPII, ProvenanceLLM

from sentence_transformers import SentenceTransformer


# Setup Guard with multiple validators
guard = Guard().use_many(
    ProvenanceLLM(validation_method="sentence"),
    DetectPII()
)

# Setup metadata for provenance validator
sources = [
    "The sun is a star.",
    "The sun rises in the east and sets in the west."
]
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def embed_function(sources: list[str]) -> np.array:
    return model.encode(sources)

# Setup metadata for PII validator
pii_entities = ["EMAIL_ADDRESS", "PHONE_NUMBER"]

# Create a single metadata dictionary containing metadata keys for each validator
metadata = {
    'pii_entities': pii_entities,
    'sources': sources,
    'embed_function': embed_function
}

# Pass the metadata to the guard.validate method
guard.validate("some text", metadata=metadata)
```

## Custom Validators

Custom validators can extend the ability of Guardrails beyond the hub. Documentation for them can be found [here](/docs/how_to_guides/custom_validators).

## Submitting a Validator to the Hub

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


## Installing Validators

### Guardrails Hub

Validators can be combined together into Input and Output Guards that intercept the inputs and outputs of LLMs. There are a large collection of Validators which can be found at the [Guardrails Hub](https://hub.guardrailsai.com/).

<div align="center">
<img src="https://raw.githubusercontent.com/guardrails-ai/guardrails/main/docs/img/guardrails_hub.gif" alt="Guardrails Hub gif" width="600px" />
</div>

Once you have found a Validator on the hub, you can click on the Validator `README` to find the install link.

### Using CLI

You can install a validator using the Guardrails CLI. For example the [Toxic Language](https://hub.guardrailsai.com/validator/guardrails/toxic_language) validator can be installed with:

```bash
guardrails hub install hub://guardrails/toxic_language
```

> This will not download local models if you opted into remote inferencing during `guardrails configure`

> If you want to control if associated models are downloaded or not you can use the `--install-local-models` or `--no-install-local-models` flags respectively during `guardrails hub install`

After installing the validator with the CLI you can start to use the validator in your guards:

```python
from guardrails.hub import ToxicLanguage
from guardrails import Guard

guard = Guard().use(
    ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
)

guard.validate("My landlord is an asshole!") 
```

### In Code Installs

You can also install validators using the Guardrails SDK which simplifies development particularly when using Jupyter Notebooks.

```python
from guardrails import install

install(
    "hub://guardrails/toxic_language",
    install_local_models=True, # defaults to `None` - which will not download local models if you opted into remote inferencing.
    quiet=False # defaults to `True`
)
```

### In Code Installs - Pattern A

After an `install` invocation you can import a validator as you typically would: 

```python
from guardrails import install

install("hub://guardrails/toxic_language")

from guardrails.hub import ToxicLanguage

guard = Guard().use(
    ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
)

guard.validate("My landlord is an asshole!") 
```

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


> Note: Invoking the `install` SDK  always installs the validator module so it's recommended for the install to be in a separate code block when using Notebooks.
