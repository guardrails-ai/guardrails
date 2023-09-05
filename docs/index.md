# Guardrails.ai

_Note: Guardrails is an alpha release, so expect sharp edges and bugs._

## 🛤️ What is Guardrails?

Guardrails is a Python package that lets a user add more reliability to their LLM applications. Guardrails:

- is a framework to perform efficient validation on the inputs and outputs of LLMs. This includes semantic validation (e.g. checking for bias in generated text, checking for bugs in generated code, etc.) and syntactic validation (e.g. checking for valid JSON, valid Python code, etc.),
- takes corrective actions (e.g. reasking LLM, filtering incorrect outputs, etc.) when validation fails,
- contains a library of validators and corrective actions for common use cases,
- provides a framework to easily create custom validators and orchestrate them via Guardrails.

## 🚒 Under the hood

Guardrails provides a format (`.rail`) for enforcing a specification on an LLM output, and a lightweight wrapper around LLM API calls to implement this spec.

1. `rail` (**R**eliable **AI** markup **L**anguage) files for specifying structure and type information, validators and corrective actions over LLM outputs.
2. `gd.Guard` wraps around LLM API calls to structure, validate and correct the outputs.

``` mermaid
graph LR
    A[Create `RAIL` spec] --> B["Initialize `guard` from spec"];
    B --> C["Wrap LLM API call with `guard`"];
```

Check out the [Getting Started](getting_started.ipynb) guide to learn how to use Guardrails.

### 📜 `RAIL` spec

At the heart of Guardrails is the `rail` spec. `rail` is intended to be a language-agnostic, human-readable format for specifying structure and type information, validators and corrective actions over LLM outputs.

`rail` is a flavor of XML that lets users specify:

1. The expected structure and types of the LLM output (e.g. JSON),
2. The quality criteria for the output to be considered valid (e.g. generated text should be bias-free, generated code should be bug-free),
3. Corrective actions to be taken if the output is invalid (e.g. reask the LLM, filter out the invalid output, etc.)

To learn more about the `rail` spec and the design decisions behind it, check out the [Rail Specification](rail/index.md). To learn how to write your own `rail` spec, check out [specifying `output` elements in RAIL](rail/output.md).

## 📍 Roadmap

- [ ] Adding more examples, new use cases and domains
- [x] Adding integrations with langchain, gpt-index, minichain, manifest
- [ ] Expanding validators offering
- [ ] More compilers from `.rail` -> LLM prompt (e.g. `.rail` -> TypeScript)
- [ ] Informative logging
- [x] Improving reasking logic
- [ ] A guardrails.js implementation
- [ ] VSCode extension for `.rail` files
- [ ] Next version of `.rail` format
- [x] Add more LLM providers
