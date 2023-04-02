# Using Guardrails from a CLI

Guardrails can be used from the command line to validate the output of an LLM. Currently, the guardrails CLI doesn't support reasking.


## Usage

```bash
guardrails validate <path to rail spec> <llm output as string> --out <output path for validated JSON>
```