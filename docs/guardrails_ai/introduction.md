# Guardrails.ai

_Note: Guardrails is an alpha release, so expect sharp edges and bugs._

## 🛤️ What is Guardrails?

Guardrails AI is a fully open source library that assures interactions with Large Language Models (LLMs). It offers

✅ Framework for creating custom validators

✅ Orchestration of prompting → verification → re-prompting

✅ Library of commonly used validators for multiple use cases

✅ Specification language for communicating requirements to LLM

## 🚒 Under the hood

Guardrails provides an object definition called a `Rail` for enforcing a specification on an LLM output, and a lightweight wrapper called a `Guard` around LLM API calls to implement this spec.

1. `rail` (**R**eliable **AI** markup **L**anguage) files for specifying structure and type information, validators and corrective actions over LLM outputs. The concept of a Rail has evolved from markup - Rails can be defined in either <a href='/defining_guards/pydantic'>Pydantic</a> or <a href='/defining_guards/rail'>rail</a> for structured outputs, or directly in <a href='/defining_guards/strings'>Python</a> for string outputs.
2. `Guard` wraps around LLM API calls to structure, validate and correct the outputs.

``` mermaid
graph LR
    A[Create `RAIL` spec] --> B["Initialize `guard` from spec"];
    B --> C["Wrap LLM API call with `guard`"];
```

Check out the [Getting Started](getting_started.ipynb) guide to learn how to use Guardrails.

## 📍 Roadmap

- [ ] Adding more examples, new use cases and domains
- [x] Adding integrations with langchain, gpt-index, minichain, manifest
- [~] Expanding validators offering
- [ ] Wider variety of language support (TypeScript, Go, etc)
- [ ] Informative logging
- [x] Improving reasking logic
- [ ] VSCode extension for `.rail` files
- [ ] Next version of `.rail` format
- [x] Add more LLM providers
