# Guardrails AI & NeMo Guardrails

Integrating Guardrails AI with NeMo Guardrails combines the strengths of both frameworks:

Guardrails AI's extensive hub of validators can enhance NeMo Guardrails' input and output checking capabilities. 
NeMo Guardrails' flexible configuration system can provide a powerful context for applying Guardrails AI validators. 
Users of both frameworks can benefit from a seamless integration, reducing development time and improving overall safety measures.
This integration allows developers to leverage the best features of both frameworks, creating more robust and secure LLM applications.

## Registering a Guard as an Action

```bash
guardrails hub install hub://guardrails/toxic_language
```

```python
from guardrails import Guard
from guardrails.hub import ToxicLanguage
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.integrations.guardrails_ai.guard_actions import register_guardrails_guard_actions

guard = Guard().use(
  ToxicLanguage()
)

config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config)

register_guardrails_guard_actions(rails, guard, "custom_guard_action")
```

Now, the `custom_guard_action` can be used as an action within the Rails specification. This action can be used on input or output, and may be used in any number of flows.

```yaml
define flow
  ...
  $result = execute custom_guard_action
  ...
```

## Using LLMRails in a Guard

```bash
guardrails hub install hub://guardrails/toxic_language
```

```yaml
# config.yml
models:
 - type: main
   engine: openai
   model: gpt-3.5-turbo-instruct
```

```python 
from guardrails import Guard
from guardrails.hub import ToxicLanguage
from nemoguardrails import RailsConfig, LLMRails
from guardrails.integrations.nemoguardrails import NemoguardrailsGuard

config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config)

guard = NemoguardrailsGuard(rails)
guard.use(
  ToxicLanguage()
)
```
