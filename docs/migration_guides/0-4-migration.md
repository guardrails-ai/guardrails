# Migrating to 0.4.0

Get ready for a major upgrade with Guardrails 0.4.0! Introducing the Guardrails Hub, a free marketplace bursting with powerful validator tools. This innovative addition allows you to access specialized functionality without bloating the core Guardrails package, keeping it lean and efficient.

To pave the way for the Hub, some backwards compatibility adjustments have been made, primarily affecting validator usage.

## New Features

### New `validate` function on Guard

Guard 0.4.0 introduces validate, replacing parse for string validation. It's clearer and less likely to be misinterpreted. parse is deprecated, so switch to validate now for future compatibility.

Example:

```python
guard.parse("some_string")  # Old
guard.validate("some_string")  # New (preferred)
```

### Unlock new tools with the Guardrails Hub CLI:

Interact directly with the [Guardrails Hub](https://hub.guardrailsai.com) for expanded functionality.

Install: `pip install guardrails-ai`

Get started: `guardrails --help`


### New Approach to Guard Construction

We've introduced a novel way to define and combine guards, simplifying the overall process. The previous method required constructing guards from specific validation types (`from_pydantic`, `from_string`), but now you can leverage a validator-first approach with assumed string validation.

Single Validator Usage:

```python
from guardrails.hub import ValidatorOfChoice

Guard().use(ValidatorOfChoice(args))(
    llm_api=...,
    model=...,
    prompt=...,
)
```

In this example:

- ValidatorOfChoice: Replace with the actual validator you want to use.
- args: Pass any necessary arguments to the validator constructor.
- llm_api, model, prompt: Provide values for these parameters as usual.

#### Multiple Validator Usage:

Multiple validators can be combined in two ways:

<b>1. linking `use`:</b>

```python
from guardrails.hub import ValidatorA, ValidatorB
Guard() \
    .use(ValidatorA()) \
    .use(ValidatorB()) \
    .validate("Some text")
```

<b>2. `use_many` for Concise Composition:</b>

```python
from guardrails.hub import ValidatorA, ValidatorB

Guard().use_many(ValidatorA(), ValidatorB())(
    llm_api=...,
    model=...,
    prompt=...,
)
```


## Backwards-incompatible changes

We've moved validators to the [Guardrails Hub](https://hub.guardrailsai.com), reducing the core package size for faster installations and smoother workflows.

Targeted validation: Install only the validators you need, streamlining your project and dependencies.

New naming: Some validator names changed for clarity and consistency. Head to the hub to find the updated names and grab your validators!

