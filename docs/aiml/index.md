# Overview

## 🤖 What is `AIML`?

`.AIML` is a dialect of XML. It stands for `AI Markup Language`, and it can be used to define:

1. The structure of the expected outcome of the LLM. (E.g. JSON)
2. The type of each field in the expected outcome. (E.g. string, integer, list, object)
3. The quality criteria for the expected outcome to be considered valid. (E.g. generated text should be bias-free, generated code should be bug-free)
4. The corrective action to take in case the quality criteria is not met. (E.g. reask the question, filter the LLM, progrmatically fix, etc.)


<details>

<summary>Expand to see an example of an AIML specification.</summary>

```xml
<aiml>

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

{document}

@xml_prefix_prompt

{{output_schema}}

@json_suffix_prompt</prompt>


<script language='python'>
    from guardrails.validators import Validator, EventDetail, register_validator
    import random


    @register_validator(name="custom", data_type="any")
    class CustomValidator(Validator):
        def __init__(self, *args, **kwargs):
            super(CustomValidator, self).__init__(*args, **kwargs)

        def validate(self, key: str, value: Any, schema: Union[Dict, List]):
            """Validate that a value is within a range."""

            logger.debug(f"Validating {value} is in choices {self._choices}...")

            if random.random() > 0.5:
                raise EventDetail(
                    key,
                    value,
                    schema,
                    f"Value {value} is not in choices {self._choices}.",
                    None,
                )

            return schema
</script>

</aiml>
```


</details>


## 🤔 Why `AIML`?

1. **Language agnostic:** `AIML` Specifications can be enforced in any language.
2. **Simple and familiar:** `AIML` should be familiar to anyone familiar with HTML, and should be easy to learn.
3. **Validation and correction**: `AIML` can be used to define quality criteria for the expected output, and corrective actions to take in case the quality criteria is not met.
4. **Can define complex structures:** `AIML` can be used to define arbitrarily complex structures, such as nested lists, nested objects, etc.
5. **Supports writing custom code:** If needed, `AIML` supports writing code for using validators, custom corrective actions, etc.


**Design inspiration**

- HTML, CSS and Javascript: `AIML` spec is a dialect of XML, and so is similar to HTML. Specifying quality criteria is done via the `format` attribute, which is similar to CSS `style` tags. Corrective actions are specified via the `on-fail-*` attributes, which is similar to Javascript event handlers.
- OpenAPI as an open standard for creating machine-readable RESTful APIs.


## 📚 Components of an `AIML` Specification

The `AIML` specification contains 3 main components:

1. `Output`: Contains information about the expected output of the LLM. It contains the spec for the overall structure of the LLM output, type info for each field, and the quality criteria for each field and the corrective action to be taken in case quality criteria is not met.
   This is the main component of the `AIML` specification, which enforces the guarantees that the LLM should provide.
   Check out the [AIML Output](output.md) page for more details, including the full specifcation of how to create complex output schemas.
2. `Prompt`: Prompt template, and contains the high level instructions that are sent to the LLM. Check out the [AIML Prompt](prompt.md) page for more details.
3. (Experimental) (Optional) `Script`: Contains any custom code for implementing the schema. This is useful for implementing custom validators, custom corrective actions, etc. Check out the [AIML Script](script.md) page for more details.

Let's see an example of an `AIML` specification in action:


```xml
<aiml>

<output> <!-- (1)! -->
...
</output>


<prompt> <!-- (2)! -->
...
</prompt>


<script language=python> <!-- (3)! -->
...
</script>

</aiml>
```

1. The `output` element contains the structure of the expected output of the LLM. It contains the spec for the overall structure of the LLM output, type info for each field, and the quality criteria for each field and the corrective action to be taken in case quality criteria is not met.
2. The `prompt` element contains the high level instructions that are sent to the LLM. Check out the [AIML Prompt](prompt.md) page for more details.
3. The `script` element is optional, and contains any custom code for implementing the schema.


## 📖 How to use `AIML` in Guardrails?

After creating an `AIML` specification, you can use to get corrected output from LLMs by wrapping your LLM API call with a `Guard` module. Here's an example of doing that:

```python

import guardrails as gd

# Create a Guard object
guard = gd.Guard.from_aiml('path/to/aiml/spec.xml')  # (1)!
validated_output = guard(
    openai.Completion.create,  # (2)!
    **prompt_args,
    *args,
    **kwargs
)

```

1. A `Guard` object is created from an `AIML` specification. This object manages the validation and correction of the output of the LLM, as well as the prompt that is sent to the LLM.
2. Wrap the LLM API call (`openai.Completion.create`) with the `Guard` object, and add any additional arguments that you want to pass to the LLM API call. Instead of returning the raw text object, the `Guard` object will return a JSON object that is validated and corrected according to the `AIML` specification.