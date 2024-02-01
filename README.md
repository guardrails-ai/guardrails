# 🛤️ Guardrails AI

<div align="center">

[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/Jsey3mX98B) [![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/guardrails_ai)

Guardrails is an open-source Python package for specifying structure and type, validating and correcting the outputs of large language models (LLMs).

[**Docs**](https://docs.guardrailsai.com)
</div>

## 🧩 What is Guardrails?

Guardrails is a Python package that lets a user add structure, type and quality guarantees to the outputs of large language models (LLMs). Guardrails:

- does pydantic-style validation of LLM outputs (including semantic validation such as checking for bias in generated text, checking for bugs in generated code, etc.)
- takes corrective actions (e.g. reasking LLM) when validation fails,
- enforces structure and type guarantees (e.g. JSON).


## 🚒 Under the hood

Guardrails provides a file format (`.rail`) for enforcing a specification on an LLM output, and a lightweight wrapper around LLM API calls to implement this spec.

1. `rail` (**R**eliable **AI** markup **L**anguage) files for specifying structure and type information, validators and corrective actions over LLM outputs.
2. `gd.Guard` wraps around LLM API calls to structure, validate and correct the outputs.

``` mermaid
graph LR
    A[Create `RAIL` spec] --> B["Initialize `guard` from spec"];
    B --> C["Wrap LLM API call with `guard`"];
```

Check out the [Getting Started](https://docs.guardrailsai.com/guardrails_ai/getting_started/) guide to learn how to use Guardrails.

### 📜 `RAIL` spec

At the heart of Guardrails is the `rail` spec. `rail` is intended to be a language-agnostic, human-readable format for specifying structure and type information, validators and corrective actions over LLM outputs.

`rail` is a flavor of XML that lets users specify:

1. the expected structure and types of the LLM output (e.g. JSON)
2. the quality criteria for the output to be considered valid (e.g. generated text should be bias-free, generated code should be bug-free)
3. and corrective actions to be taken if the output is invalid (e.g. reask the LLM, filter out the invalid output, etc.)


To learn more about the `RAIL` spec and the design decisions behind it, check out the [docs](https://docs.guardrailsai.com/defining_guards/rail/). To learn how to write your own `RAIL` spec, check out [this link](https://docs.guardrailsai.com/api_reference/rail/).



## 📦 Installation

```python
pip install guardrails-ai
```

## 📍 Roadmap
- [ ] Javascript SDK
- [ ] Wider variety of language support (TypeScript, Go, etc)
- [ ] Informative logging
- [ ] VSCode extension for `.rail` files
- [ ] Next version of `.rail` format
- [ ] Validator playground
- [x] Input Validation
- [x] Pydantic 2.0
- [x] Improving reasking logic
- [x] Integration with LangChain
- [x] Add more LLM providers

## 🚀 Getting Started
Let's go through an example where we ask an LLM to generate fake pet names. To do this, we'll use Pydantic, a popular data validation library for Python.  

### 📝 Creating Structured Outputs

In order to create a LLM that generates fake pet names, we can create a class `Pet` that inherits from the Pydantic class [Link BaseModel](https://docs.pydantic.dev/latest/api/base_model/): 

```py
from pydantic import BaseModel, Field

class Pet(BaseModel):
    pet_type: str = Field(description="Species of pet")
    name: str = Field(description="a unique pet name")
```

We can now pass in this new `Pet` class as the `output_class` parameter in our Guard. When we run the code, the LLM's output is formatted to the pydnatic structure. We also add `${gr.complete_json_suffix_v2}` to the prompt which tells our LLM to only respond with JSON: 

```py
from guardrails import Guard
import openai

prompt = """
    What kind of pet should I get and what should I name it?

    ${gr.complete_json_suffix_v2}
"""
guard = Guard.from_pydantic(output_class=Pet, prompt=prompt)

validated_output, *rest = guard(
    llm_api=openai.completions.create,
    engine="gpt-3.5-turbo-instruct"
)

print(f"{validated_output}")
```

This prints: 
```
{
    "pet_type": "dog",
    "name": "Buddy
}
```

## Structured Outputs with Validation 
We can add validation to our Guard instead of just structuring the ouput in a specific format. In the below code, we add a Validator that checks if the pet name generated is of valid length. If it does not pass the validation, the reask is triggered and the query is reasked to the LLM. Check out the [Link Validators API Spec](https://www.guardrailsai.com/docs/api_reference_markdown/validators/) for a list of supported validators.

```py
from guardrails.validators import ValidLength, TwoWords
from rich import print

class Pet(BaseModel):
    pet_type: str = Field(description="Species of pet")
    name: str = Field(description="a unique pet name", validators=[ValidLength(min=1, max=32, on_fail='reask')])

guard = Guard.from_pydantic(output_class=Pet, prompt=prompt)

raw_llm_output, validated_output, *rest = guard(
    llm_api=openai.chat.completions.create,
    model="gpt-3.5-turbo",
    max_tokens=1024,
    temperature=0.5
)

print(guard.history.last.tree)
```

## 🛠️ Contributing

Get started by checking out Github issues and of course using Guardrails to familiarize yourself with the project. Guardrails is still actively under development and any support is gladly welcomed. Feel free to open an issue, or reach out if you would like to add to the project!
