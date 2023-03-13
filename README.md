# 🛤️ Guardrails 

<div align="center">
Guardrails is an open-source Python package for specifying structure and type, validating and correcting the outputs of large language models (LLMs).

[**Docs**](http://shreyar.github.io/guardrails/)
</div>

_Note: Guardrails is an alpha release, so expect sharp edges and bugs._

Guardrails provides a format (`.rail`) for enforcing a specification on an LLM output, and a lightweight wrapper around LLM API calls to implement this spec.

1. `rail` (`reliable AI markup language`) files for specifying structure and type information, validators and corrective actions over LLM outputs.
2. `gd.Guard` wraps around LLM API calls to structure, validate and correct the outputs.

## 📦 Installation

```python
pip install guardrails-ai
```

## 🛣️ Roadmap
- adding more examples, new use cases and domains
- integrations with langchain, gpt-index, minichain, manifest
- expanding offered validators
- more transpilers from `.rail` -> LLM prompt (e.g. `.rail` -> TypeScript)
- informative logging
- improving reasking logic
- a guardrails.js implementation
- VSCode extension for `.rail` files
- next version of `.rail` format

## 🚀 Getting Started
Let's go through an example where we ask an LLM to explain what a "bank run" is, and generate URL links to relevant news articles. We'll generate a `.rail` spec for this and then use Guardrails to enforce it. You can see more examples in the docs.

### 📝 Creating a `RAIL` spec

`RAIL` (with extension `.rail`) is a flavor of XML (stands for `Reliable AI Markup Language`) that describes the expected structure and types of the LLM output, the quality criteria for the output to be considered valid, and corrective actions to be taken if the output is invalid.

- Create a `RAIL` spec that requests the LLM to generate an object with two fields: `explanation` and `follow_up_url`.
- For the `explanation` field, the max length of the generated string should be 280 characters. If the explanation is not of valid length, reask the LLM.
- For the `follow_up_url` field, the URL should be reachable. If the URL is not reachable, we will filter it out of the response.


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
            required="true"
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
"""
```
We specify our quality criteria (generated length, URL reachability) in the `format` fields of the `RAIL` spec below. We `reask` if `explanation` is not valid, and filter the `follow_up_url` if it is not valid.

### 🛠️ Using Guardrails to enforce the `RAIL` spec

Next, we'll use the `RAIL` spec to create a `Guard` object. The `Guard` object will wrap the LLM API call and enforce the `RAIL` spec on its output.

```python
import guardrails as gd

guard = gd.Guard.from_rail(f.name)
```

The `Guard` object compiles the `RAIL` specification and adds it to the prompt. (Right now this is a passthrough operation, more compilers are planned to find the best way to express the spec in a prompt.)


```python
print(guard.base_prompt)
```

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
raw_llm_output, validated_output = guard(openai.Completion.create, engine="text-davinci-003", max_tokens=1024, temperature=0.3)

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
