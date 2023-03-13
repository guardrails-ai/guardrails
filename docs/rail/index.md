# Overview

## 🤖 What is `RAIL`?

`.RAIL` is a dialect of XML. It stands for `**R**eliable **AI** markup **L**anguage`, and it can be used to define:

1. The structure of the expected outcome of the LLM. (E.g. JSON)
2. The type of each field in the expected outcome. (E.g. string, integer, list, object)
3. The quality criteria for the expected outcome to be considered valid. (E.g. generated text should be bias-free, generated code should be bug-free)
4. The corrective action to take in case the quality criteria is not met. (E.g. reask the question, filter the LLM, progrmatically fix, etc.)


<details>

<summary>Expand to see an example of an RAIL specification.</summary>

```xml
<rail>

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

</rail>
```


</details>


## 🤔 Why `RAIL`?

1. **Language agnostic:** `RAIL` Specifications can be enforced in any language.
2. **Simple and familiar:** `RAIL` should be familiar to anyone familiar with HTML, and should be easy to learn.
3. **Validation and correction**: `RAIL` can be used to define quality criteria for the expected output, and corrective actions to take in case the quality criteria is not met.
4. **Can define complex structures:** `RAIL` can be used to define arbitrarily complex structures, such as nested lists, nested objects, etc.
5. **Supports writing custom code:** If needed, `RAIL` supports writing code for using validators, custom corrective actions, etc.


**Design inspiration**

- HTML, CSS and Javascript: `RAIL` spec is a dialect of XML, and so is similar to HTML. Specifying quality criteria is done via the `format` attribute, which is similar to CSS `style` tags. Corrective actions are specified via the `on-fail-*` attributes, which is similar to Javascript event handlers.
- OpenAPI as an open standard for creating machine-readable RESTful APIs.


## 📚 Components of an `RAIL` Specification

The `RAIL` specification contains 3 main components:

1. `Output`: Contains information about the expected output of the LLM. It contains the spec for the overall structure of the LLM output, type info for each field, and the quality criteria for each field and the corrective action to be taken in case quality criteria is not met.
   This is the main component of the `RAIL` specification, which enforces the guarantees that the LLM should provide.
   Check out the [RAIL Output](output.md) page for more details, including the full specifcation of how to create complex output schemas.
2. `Prompt`: Prompt template, and contains the high level instructions that are sent to the LLM. Check out the [RAIL Prompt](prompt.md) page for more details.
3. (Experimental) (Optional) `Script`: Contains any custom code for implementing the schema. This is useful for implementing custom validators, custom corrective actions, etc. Check out the [RAIL Script](script.md) page for more details.

Let's see an example of an `RAIL` specification in action:


```xml
<rail>

<output> <!-- (1)! -->
...
</output>


<prompt> <!-- (2)! -->
...
</prompt>


<script language=python> <!-- (3)! -->
...
</script>

</rail>
```

1. The `output` element contains the structure of the expected output of the LLM. It contains the spec for the overall structure of the LLM output, type info for each field, and the quality criteria for each field and the corrective action to be taken in case quality criteria is not met.
2. The `prompt` element contains the high level instructions that are sent to the LLM. Check out the [RAIL Prompt](prompt.md) page for more details.
3. The `script` element is optional, and contains any custom code for implementing the schema.


## 📖 How to use `RAIL` in Guardrails?

After creating an `RAIL` specification, you can use to get corrected output from LLMs by wrapping your LLM API call with a `Guard` module. Here's an example of doing that:

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

1. A `Guard` object is created from an `RAIL` specification. This object manages the validation and correction of the output of the LLM, as well as the prompt that is sent to the LLM.
2. Wrap the LLM API call (`openai.Completion.create`) with the `Guard` object, and add any additional arguments that you want to pass to the LLM API call. Instead of returning the raw text object, the `Guard` object will return a JSON object that is validated and corrected according to the `RAIL` specification.