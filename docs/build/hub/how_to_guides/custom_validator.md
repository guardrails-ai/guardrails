# Create a new Validator and submit to the Hub

There are two ways to create a new validator and submit it to the Hub.

1. For lightweight validators, use the `hub` CLI to create a new validator and submit it to the Hub.
2. For more complex validators, clone the Validator-Template repository and register the validator via the Guardrails Hub website.

## Creating a new validator using the `hub` CLI

The `hub` CLI provides a simple way to create a new validator and submit it to the Hub. The `hub` CLI will create a new validator in the current directory and submit it to the Hub.

To create a new validator using the `hub` CLI, run the following command:

```bash
guardrails hub create-validator my_validator
```

This will create a new file called `my_validator.py` in the current directory. The file will contain a template and instructions for creating a new validator.

```bash
guardrails hub submit my_validator
```

## Creating a new validator using the Validator-Template repository

For more complex validators, you can clone the [Validator-Template repository](https://github.com/guardrails-ai/validator-template) and register the validator via a Google Form on the Guardrails Hub website.

```bash
git clone git@github.com:guardrails-ai/validator-template.git
```

Once the repository is cloned and the validator is created, you can register the validator via this [Google Form](https://forms.gle/N6UaE9611niuMxZj7).



