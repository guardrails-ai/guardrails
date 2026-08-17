# EPI Recorder Integration for Guardrails AI

Produces tamper-evident, cryptographically signed **.epi** artifacts from every Guardrails validation execution.

## Quick Start

```bash
pip install epi-recorder
```

```python
from guardrails.integrations.epi_recorder import EPIInstrumentor

instrumentor = EPIInstrumentor()
instrumentor.instrument()

# Every Guard execution now produces a signed .epi artifact
guard = Guard.from_rail("my.rail")
result = guard(llm_api, prompt)
# -> guardrails_run.epi written

instrumentor.uninstrument()
```

## Verification

```bash
pip install epi-recorder
epi verify guardrails_run.epi --aiuc1
```

## Options

```python
EPIInstrumentor(
    output_path="my_run.epi",
    auto_sign=True,
    goal="Validate LLM output",
    tags=["guardrails", "prod"],
)
```
