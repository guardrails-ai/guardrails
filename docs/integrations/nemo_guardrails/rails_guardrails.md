:::
note: This will exist in the NeMo Guardrails docs
:::


# Introduction

Integrating Guardrails AI with NeMo Guardrails combines the strengths of both frameworks:

Guardrails AI's extensive hub of validators can enhance NeMo Guardrails' input and output checking capabilities. 
NeMo Guardrails' flexible configuration system can provide a powerful context for applying Guardrails AI validators. 
Users of both frameworks can benefit from a seamless integration, reducing development time and improving overall safety measures.
This integration allows developers to leverage the best features of both frameworks, creating more robust and secure LLM applications.

# Overview
This document provides a guide to using a Guardrails AI Guard as an action within a NeMo Guardrails Rails application. This can be done either by defining an entire Guard and registering it, or by registering a validator directly.

## Registering a Guard as an action

First, we install our validators and define our Guard

```python
from guardrails import Guard, install
install("hub://guardrails/toxic_language")
from guardrails.hub import ToxicLanguage

guard = Guard().use(
  ToxicLanguage()
)
```

Next, we register our `guard` using the nemoguardrails registration API

```python
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config)

rails.register_action(guard, "custom_guard_action")
```

Now, the `custom_guard_action` can be used as an action within the Rails specification. This action can be used on input or output, and may be used in any number of flows.

```yaml
define flow
  ...
  $result = execute custom_guard_action
  ...
```


