# Overview

## 🤖 What is `RAIL`?

`.RAIL` is a dialect of XML. It stands for "**R**eliable **AI** markup **L**anguage", and it can be used to define:

1. The structure of the expected outcome of the LLM. (E.g. JSON)
2. The type of each field in the expected outcome. (E.g. string, integer, list, object)
3. The quality criteria for the expected outcome to be considered valid. (E.g. generated text should be bias-free, generated code should be bug-free)
4. The corrective action to take in case the quality criteria is not met. (E.g. reask the question, filter the LLM, progrmatically fix, etc.)

<details>

<summary>Expand to see an example of a RAIL specification.</summary>

```xml
<rail version="0.1">

<output>
    <list name="fees" description="What fees and charges are associated with my account?">
        <object>
            <integer name="index" format="1-indexed" />
            <string name="name" format="lower-case; two-words" on-fail-lower-case="noop" on-fail-two-words="reask"/>
            <string name="explanation" format="one-line" on-fail-one-line="noop" />
            <float name="value" format="percentage"/>
        </object>
    </list>
    <string name='interest_rates' description='What are the interest rates offered by the bank on savings and checking accounts, loans, and credit products?' format="one-line" on-fail-one-line="noop"/>
</output>


<prompt>

Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 'None'.

${document}

${gr.xml_prefix_prompt}

${output_schema}

${gr.json_suffix_prompt}</prompt>

</rail>
```

</details>

## 🤔 Why `RAIL`?

1. **Language agnostic:** `RAIL` Specifications can be enforced in any language.
2. **Simple and familiar:** `RAIL` should be familiar to anyone familiar with HTML, and should be easy to learn.
3. **Validation and correction**: `RAIL` can be used to define quality criteria for the expected output, and corrective actions to take in case the quality criteria is not met.
4. **Can define complex structures:** `RAIL` can be used to define arbitrarily complex structures, such as nested lists, nested objects, etc.
5. **Code assistance**: In the future, we plan to support code completion and IntelliSense for `RAIL` specifications, which will make it very easy to write `RAIL` specifications.

**Design inspiration**

- HTML, CSS and Javascript: `RAIL` spec is a dialect of XML, and so is similar to HTML. Specifying quality criteria is done via the `format` attribute, which is similar to CSS `style` tags. Corrective actions are specified via the `on-fail-*` attributes, which is similar to Javascript event handlers.
- OpenAPI as an open standard for creating machine-readable RESTful APIs.

## 📚 Components of an `RAIL` Specification

The `RAIL` specification contains 2 main components:

1. `Output`: Contains information about the expected output of the LLM. It contains the spec for the overall structure of the LLM output, type info for each field, and the quality criteria for each field and the corrective action to be taken in case quality criteria is not met.
   This is the main component of the `RAIL` specification, which enforces the guarantees that the LLM should provide.
   Check out the [RAIL Output](../concepts/output.md) page for more details, including the full specifcation of how to create complex output schemas.
2. `Prompt`: Prompt template, and contains the high level instructions that are sent to the LLM. Check out the [RAIL Prompt](../concepts/prompt.md) page for more details.

Let's see an example of an `RAIL` specification in action:

```xml
<rail version="0.1">

<output> <!-- (1)! -->
...
</output>


<prompt> <!-- (2)! -->
...
</prompt>

</rail>
```

1. The `output` element contains the structure of the expected output of the LLM. It contains the spec for the overall structure of the LLM output, type info for each field, and the quality criteria for each field and the corrective action to be taken in case quality criteria is not met.
2. The `prompt` element contains the high level instructions that are sent to the LLM. Check out the [RAIL Prompt](../concepts/prompt.md) page for more details.

## 📖 How to use `RAIL` in Guardrails?

After creating a `RAIL` specification, you can use it to get corrected output from LLMs by wrapping your LLM API call with a `Guard` module. Here's an example of doing that:

```python

import guardrails as gd

# Create a Guard object
guard = gd.Guard.from_rail('path/to/rail/spec.xml')  # (1)!
validated_output = guard(
    openai.Completion.create,  # (2)!
    **prompt_args,
    *args,
    **kwargs
)

```

1. A `Guard` object is created from a `RAIL` specification. This object manages the validation and correction of the output of the LLM, as well as the prompt that is sent to the LLM.
2. Wrap the LLM API call (`openai.Completion.create`) with the `Guard` object, and add any additional arguments that you want to pass to the LLM API call. Instead of returning the raw text object, the `Guard` object will return a JSON object that is validated and corrected according to the `RAIL` specification.
