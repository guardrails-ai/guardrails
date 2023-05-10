# 🛤️ Guardrails

<div align="center">

[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/Jsey3mX98B) [![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/guardrails_ai)

Guardrails is an open-source Python package for specifying structure and type, validating and correcting the outputs of large language models (LLMs).

[**Docs**](http://shreyar.github.io/guardrails/)
</div>

_Note: Guardrails is an alpha release, so expect sharp edges and bugs._

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

Check out the [Getting Started](https://shreyar.github.io/guardrails/getting_started) guide to learn how to use Guardrails.

### 📜 `RAIL` spec

At the heart of Guardrails is the `rail` spec. `rail` is intended to be a language-agnostic, human-readable format for specifying structure and type information, validators and corrective actions over LLM outputs.

`rail` is a flavor of XML that lets users specify:

1. the expected structure and types of the LLM output (e.g. JSON)
2. the quality criteria for the output to be considered valid (e.g. generated text should be bias-free, generated code should be bug-free)
3. and corrective actions to be taken if the output is invalid (e.g. reask the LLM, filter out the invalid output, etc.)


To learn more about the `RAIL` spec and the design decisions behind it, check out the [docs](https://shreyar.github.io/guardrails/rail). To learn how to write your own `RAIL` spec, check out [this link](https://shreyar.github.io/guardrails/rail/output).



## 📦 Installation

```python
pip install guardrails-ai
```

## 📍 Roadmap
- [ ] Adding more examples, new use cases and domains
- [ ] Adding integrations with langchain, gpt-index, minichain, manifest
- [ ] Expanding validators offering
- [ ] More compilers from `.rail` -> LLM prompt (e.g. `.rail` -> TypeScript)
- [ ] Informative logging
- [ ] Improving reasking logic
- [ ] A guardrails.js implementation
- [ ] VSCode extension for `.rail` files
- [ ] Next version of `.rail` format
- [ ] Add more LLM providers

## 🚀 Getting Started
Let's go through an example where we ask an LLM to explain what a "bank run" is in a tweet, and generate URLs to relevant news articles. We'll generate a `.rail` spec for this and then use Guardrails to enforce it. You can see more examples in the docs.

### 📝 Creating a `RAIL` spec

We create a `RAIL` spec to describe the expected structure and types of the LLM output, the quality criteria for the output to be considered valid, and corrective actions to be taken if the output is invalid.

Using `RAIL`, we:
- Request the LLM to generate an object with two fields: `explanation` and `follow_up_url`.
- For the `explanation` field, ensure the max length of the generated string should be between 200 and 280 characters.
  - If the explanation is not of valid length, `reask` the LLM.
- For the `follow_up_url` field, the URL should be reachable.
  - If the URL is not reachable, we will `filter` it out of the response.


```xml
<rail version="0.1">
<output>
    <object name="bank_run" format="length: 2">
        <string
            name="explanation"
            description="A paragraph about what a bank run is."
            format="length: 200 280"
            on-fail-length="reask"
        />
        <url
            name="follow_up_url"
            description="A web URL where I can read more about bank runs."
            format="valid-url"
            on-fail-valid-url="filter"
        />
    </object>
</output>

<prompt>
Explain what a bank run is in a tweet.

@xml_prefix_prompt

{output_schema}

@json_suffix_prompt_v2_wo_none
</prompt>
</rail>
```

We specify our quality criteria (generated length, URL reachability) in the `format` fields of the `RAIL` spec below. We `reask` if `explanation` is not valid, and filter the `follow_up_url` if it is not valid.

### 🛠️ Using Guardrails to enforce the `RAIL` spec

Next, we'll use the `RAIL` spec to create a `Guard` object. The `Guard` object will wrap the LLM API call and enforce the `RAIL` spec on its output.

```python
import guardrails as gd

guard = gd.Guard.from_rail(f.name)
```

The `Guard` object compiles the `RAIL` specification and adds it to the prompt. (Right now this is a passthrough operation, more compilers are planned to find the best way to express the spec in a prompt.)

Here's what the prompt looks like after the `RAIL` spec is compiled and added to it.

```xml
Explain what a bank run is in a tweet.

Given below is XML that describes the information to extract from this document and the tags to extract it into.

<output>
    <object name="bank_run" format="length: 2">
        <string name="explanation" description="A paragraph about what a bank run is." format="length: 200 280" on-fail-length="reask" />
        <url name="follow_up_url" description="A web URL where I can read more about bank runs." required="true" format="valid-url" on-fail-valid-url="filter" />
    </object>
</output>

ONLY return a valid JSON object (no other text is necessary). The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise.

JSON Output:
```

Call the `Guard` object with the LLM API call as the first argument and add any additional arguments to the LLM API call as the remaining arguments.


```python
import openai

# Wrap the OpenAI API call with the `guard` object
raw_llm_output, validated_output = guard(
    openai.Completion.create,
    engine="text-davinci-003",
    max_tokens=1024,
    temperature=0.3
)

print(validated_output)
```
```python
{
    'bank_run': {
        'explanation': 'A bank run is when a large number of people withdraw their deposits from a bank due to concerns about its solvency. This can cause a financial crisis if the bank is unable to meet the demand for withdrawals.',
        'follow_up_url': 'https://www.investopedia.com/terms/b/bankrun.asp'
    }
}

```
## 🚴‍ Guardrails Activity Report

To help the Guardrails community stay informed about the project's progress, [Blueprint AI](https://blueprint.ai) has developed a Github activity summarizer for Guardrails. This concise report displays a summary of all contributions to the Guardrails repository over the past 7 days (continuously updated), making it easy for you to keep track of the latest developments.

To view the Guardrails 7-day activity report, go here: [https://app.blueprint.ai/github/ShreyaR/guardrails](https://app.blueprint.ai/github/ShreyaR/guardrails)

## 🛠️ Contributing

Get started by checking out Github issues and of course using Guardrails to familiarize yourself with the project. Guardrails is still actively under development and any support is gladly welcomed. Feel free to open an issue, or reach out if you would like to add to the project!
