# Use Guardrails via RAIL

## What is `RAIL`?

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

## Why `RAIL`?

1. **Language agnostic:** `RAIL` Specifications can be enforced in any language.
2. **Simple and familiar:** `RAIL` should be familiar to anyone familiar with HTML, and should be easy to learn.
3. **Validation and correction**: `RAIL` can be used to define quality criteria for the expected output, and corrective actions to take in case the quality criteria is not met.
4. **Can define complex structures:** `RAIL` can be used to define arbitrarily complex structures, such as nested lists, nested objects, etc.
5. **Code assistance**: In the future, we plan to support code completion and IntelliSense for `RAIL` specifications, which will make it very easy to write `RAIL` specifications.

**Design inspiration**

- HTML, CSS and Javascript: `RAIL` spec is a dialect of XML, and so is similar to HTML. Specifying quality criteria is done via the `format` attribute, which is similar to CSS `style` tags. Corrective actions are specified via the `on-fail-*` attributes, which is similar to Javascript event handlers.
- OpenAPI as an open standard for creating machine-readable RESTful APIs.

## Components of an `RAIL` Specification

The `RAIL` specification contains 2 main components:

1. `Output`: Contains information about the expected output of the LLM. It contains the spec for the overall structure of the LLM output, type info for each field, and the quality criteria for each field and the corrective action to be taken in case quality criteria is not met.
   This is the main component of the `RAIL` specification, which enforces the guarantees that the LLM should provide.
   Check out the [RAIL Output](#-specifying-output-structure) page for more details, including the full specifcation of how to create complex output schemas.
2. `Prompt`: Prompt template, and contains the high level instructions that are sent to the LLM. Check out the [RAIL Prompt](#components-of-a-prompt-element) page for more details.

Let's see an example of an `RAIL` specification in action:

```xml
<rail version="0.1">

<output> <!-- (1)! -->
...
</output>


<messages> <!-- (2)! -->
<message role="user">
...
</message>
</message>

</rail>
```

1. The `output` element contains the structure of the expected output of the LLM. It contains the spec for the overall structure of the LLM output, type info for each field, and the quality criteria for each field and the corrective action to be taken in case quality criteria is not met.
2. The `messages` element contains the high level instructions that are sent to the LLM. Check out the [RAIL Prompt](#components-of-a-message-element) page for more details.

## 📖 How to use `RAIL` in Guardrails?

After creating a `RAIL` specification, you can use it to get corrected output from LLMs by wrapping your LLM API call with a `Guard` module. Here's an example of doing that:

```python

import guardrails as gd

# Create a Guard object
guard = gd.Guard.for_rail('path/to/rail/spec.xml')  # (1)!
_, validated_output, *rest = guard(
    openai.Completion.create,  # (2)!
    **prompt_args,
    *args,
    **kwargs
)

```

1. A `Guard` object is created from a `RAIL` specification. This object manages the validation and correction of the output of the LLM, as well as the prompt that is sent to the LLM.
2. Wrap the LLM API call (`openai.Completion.create`) with the `Guard` object, and add any additional arguments that you want to pass to the LLM API call. Instead of returning the raw text object, the `Guard` object will return a JSON object that is validated and corrected according to the `RAIL` specification.

# `Messages` Element

The `<messages></messages>` element contains instructions and the query that describes the high level task.

## 📚 Components of a Prompt Element

In addition to the high level task description, messages also contains the following:

| Component         | Syntax                   | Description                                                                                                                                                                                                                                                                                                                             |
|-------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Variables         | `${variable_name}`        | These are provided by the user at runtime, and substituted in the prompt.                                                                                                                                                                                                                                                               |
| Output Schema     | `${output_schema}`      | This is the schema of the expected output, and is compiled based on the  `output` element.  For more information on how the output schema is compiled for the prompt, check out [`output` element compilation](/docs/concepts/output/#adding-compiled-output-element-to-prompt).                                                                    |
| Prompt Primitives | `${gr.prompt_primitive_name}` | These are pre-constructed prompts that are useful for common tasks. E.g., some primitives may contain information that helps the LLM understand the output schema better.  To see the full list of prompt primitives, check out [`guardrails/constants.xml`](https://github.com/guardrails-ai/guardrails/blob/main/guardrails/constants.xml). |

```xml
<rail version="0.1">
<messages>
<message role="system">
<!-- (1)! -->
You are a helpful assistant only capable of communicating with valid JSON, and no other text.
</message>
<message role="user">
<!-- (2)! -->
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 'None'.

${document} <!-- (3)! -->


${gr.xml_prefix_prompt}  <!-- (4)! -->


${output_schema}  <!-- (5)! -->


${gr.json_suffix_prompt}  <!-- (6)! -->
</message>
</message>
</rail>
```

1. The instructions element contains high level background information for the LLM containing textual context and constraints.
2. The prompt contains high level task information.
3. The variable `${document}` is provided by the user at runtime.
4. `${gr.xml_prefix_prompt}` is a prompt primitive provided by guardrails. It is equivalent to typing the following lines in the prompt: `Given below is XML that describes the information to extract from this document and the tags to extract it into.`
5. `${output_schema}` is the output schema and contains information about , which is compiled based on the `output` element.
6. `${gr.json_suffix_prompt}` is a prompt primitive provided by guardrails. It is equivalent to typing the following lines in the prompt:
```
ONLY return a valid JSON object (no other text is necessary). The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`.
```

The messages element is made up of message elements with role attributes. Messages with the role system are intended to be system level prompt. Messages with the role assistant are intended to be messages from the llm to be repassed to itself as additional context and history. Messages with role user are input from the user and also convey history of the conversation.

# `Output` Element

The `<output>...</output>` element of a `RAIL` spec is used to give precise specification of the expected output of the LLM. It specifies

1. the structure of the expected output (e.g. JSON),
2. the type of each field,
3. the quality criteria for each field to be considered valid (e.g. generated text should be bias-free, generated code should be bug-free), and
4. the corrective action to take in case the quality criteria is not met (e.g. reask the question to the LLM, filter offending values, progrmatically fix, etc.)

Example:
<!-- TODO add formatting so that there's nesting between the spec and outputs for each output type -->

=== "JSON RAIL Spec"

    ```xml
    <rail version="0.1">
        <output>
            <string name="text" description="The generated text" format="two-words" on-fail-two-words="reask"/>
            <float name="score" description="The score of the generated text" format="min-val: 0" on-fail-min-val="fix"/>
            <object name="metadata" description="The metadata associated with the generated text">
                <string name="key_1" description="description of key_1" />
                ...
            </object>
        </output>
    </rail>
    ```

=== "Output JSON"
    
    ```json
    {
        "text": "string output",
        "score": 0.0,
        "metadata": {
            "key_1": "string",
            ...
        }
    }
    ```

=== "String RAIL Spec"


    ```xml
    <rail version="0.1">
        <output
            type="string"
            description="The generated text"
            format="two-words"
            on-fail-two-words="reask"
        />
    </rail>
    ```

=== "Output String"
    
    ```
    string output
    ```

## ⚡ Specifying output structure

You can combine `RAIL` elements to create an arbitrarily complex output structure.

### Flat JSON output

=== "RAIL Spec"

    ```xml
    <rail version="0.1">
        <output>
            <string name="some_key" ..../>
            <integer name="some_other_key" ..../>
        </output>
    </rail>
    ```

=== "Output JSON"
    ```json
    {
        "some_key": "string",
        "some_other_key": 0
    }
    ```


### JSON output with objects

`object` elements can be used to specify a JSON object, which is a collection of key-value pairs.

- A child of an `object` element represents a key in the JSON object. The child element can be any RAIL element, including another `list` or `object` elements. The value of the key is generated by the LLM based on the info provided by the child element.
- An object element can have multiple children, each of which can be any RAIL element, including another `list` or `object` elements.
- Formatters can be applied to the child elements of an object element. For example, if the child element is a `string` element, the `format` attribute can be used to specify the quality criteria for the strings in the list.


=== "RAIL Spec"

    ```xml
    <rail version="0.1">
        <output>
            <object name="some_object">
                <string name="some_str_key" description="What should the value for this key represent?" validators="guardrails/uppercase; guardrails/two_words" />
                <integer name="some_other_key" description="What should this integer represent?" format="min-val: 0"/>
            </object>
        </output>
    </rail>
    ```

=== "Output JSON"
    ```json
    {
        "some_object": {
            "some_str_key": "SOME STRING",
            "some_other_key": 0
        }
    }
    ```

In the above example, `"SOME STRING"` is the value for the `some_str_key` key, and is generated based on the name, description and quality criteria provided by the `<string name="some_str_key" ... />` element.


!!! note
    The `object` element doesn't *need* to have children. If child elements are not provided, the LLM will automatically generate keys and values for the object based on the `name`, `description` and `format` attributes of the `object` element.

    Providing child elements is useful when you want to specify the keys and values that the LLM should generate.

### JSON output with lists

`list` elements can be used to specify a list of values.

- Currently, a list element can only contain a single child element. This means that a list can only contain a single type of data. For example, a list can only contain strings, or a list can only contain integers, but a list cannot contain both strings and integers.
- This child element can be any RAIL element, including another `list` or `object` elements.
- The child of a list element doesn't need to have a `name` attribute, since items in a list don't have names.
- Formatters can be applied to the child element of a list element. For example, if the child element is a `string` element, the `validators` attribute can be used to specify the quality criteria for the strings in the list.

=== "RAIL Spec"

    ```xml
    <rail version="0.1">
        <output>
            <list name="some_list" format="min-len: 2">
                <string validators="guardrails/uppercase; guardrails/two_words" />
            </list>
        </output>
    </rail>
    ```

=== "Output JSON"
    ```json
    {
        "some_list": [
            "STRING 1", "STRING 2"
        ]
    }
    ```


!!! note
    The `list` element doesn't *need* to have a child element. If a child element is not provided, the LLM will automatically generate values for the list based on the `name`, `description` and `format` attributes of the `list` element.

    Providing a child element is useful when you want to have more control over the values that the LLM should generate.

### String output

Generate simple strings by specifying `type="string"` in the `<output ... />` element.
All the formatters supported by the `string` element can be used to specify the quality criteria for the generated string.

=== "RAIL Spec"

    ```xml
    <rail version="0.1">
        <output
            type="string" 
            format="two-words" 
            on-fail-two-words="reask"
        />
    </rail>
    ```

=== "Output"
    ```
    string output
    ```

## `RAIL` Elements

At the heart of the `RAIL` specification is the use of elements. Each element's tag represents a type of data. For example, in the element `<string ... />`, the tag represents a string, the `<integer ... />` elements represents an integer, the `<object ...></object>` element represents an object, etc.


!!! note
    The tag of RAIL element is the same as the "type" of the data it represents.

    E.g. `<string .../>` element will generate a string, `<integer .../>` element will generate an integer, etc.

### Supported types

Guardrails supports all [JSON](https://www.w3schools.com/js/js_json_datatypes.asp) and pydantic compatible datatypes. 


#### Scalar vs Non-scalar types

Guardrails supports two types of data types: scalar and non-scalar.


| Scalar                                                                  | Non Scalar                                                                           |
|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Scalar types are void elements, and can't have any child elements.      | Non-scalar types can be non-void, and can have closing tags and child elements.                       |
| Syntax:   ``` <string ... /> ```                                        | Syntax: ```<list ...>     <string /> </list>```|
| Examples: `string`, `integer`, `float`, `bool`, `url`, `email`, etc. | Examples: `list` and `object` are the only non-scalar types supported by Guardrails. |


### Supported attributes

Each element can have attributes that specify additional information about the data, such as:

1. `name` attribute that specifies the name of the field. This will be the key in the output JSON. E.g.

=== "RAIL Spec"

    ```xml
    <rail version="0.1">
        <output>
            <string name="some_key" />
        </output>
    </rail>
    ```

=== "Output JSON"

    ```json
    {
        "some_key": "..."
    }
    ```

2. `description` attribute that specifies the description of the field. This is similar to a prompt that will be provided to the LLM. It can contain more context to help the LLM generate the correct output.
3. (Coming soon!) `required` attribute that specifies whether the field is required or not. If the field is required, the LLM will be asked to generate the field until it is generated correctly. If the field is not required, the LLM will not be asked to generate the field if it is not generated correctly.
4. `validators` attribute that specifies the quality criteria that the field should respect. The `validators` attribute can contain multiple quality criteria separated by a colon (`;`). For example, `guardrails/uppercase; guardrails/two_words`.
5. `on-fail-{quality-criteria}` attribute that specifies the corrective action to take in case the quality criteria is not met. For example, `on-fail-two-words="reask"` specifies that if the field does not have two words, the LLM should be asked to re-generate the field.


E.g.,

=== "RAIL Spec"
    ```xml
    <rail version="0.1">
        <output>
            <string
                name="some_key"
                description="Detailed description of what the value of the key should be"
                required="true"
                validators="guardrails/uppercase; guardrails/two_words"
                on-fail-guardrails_two_words="reask"
                on-fail-guardrails_uppercase="noop"
            />
        </output>
    </rail>
    ```

=== "Output JSON"

    ```json
    {
        "some_key": "SOME STRING"
    }
    ```

## Specifying quality criteria

The `format` attribute allows specifying the quality criteria for each field in the expected output. The `format` attribute can contain multiple quality criteria separated by a colon (`;`). For example,

```xml
<rail version="0.1">
    <output>
        <string
            name="text"
            description="The generated text"
            validators="guardrails/uppercase; guardrails/two_words"
            on-fail-guardrails_two_words="reask"
            on-fail-guardrails_uppercase="noop"
        />
    </output>
</rail>

```

The above example specifies that the `text` field should be a string with two words and the text should be returned in upper case.


### Quality criteria under the hood

Under the hood, the `format` attribute is parsed into a list of quality criteria.

Each quality criteria is backed by a `Validator` class that checks if the generated output meets the quality criteria. For example, the `two-words` quality criteria is backed by the `TwoWords` class, which checks if the generated output has two words.

Each quality criteria is then checked against the generated output. If the quality criteria is not met, the corrective action specified by the `on-fail-{quality-criteria}` attribute is taken.


### Supported criteria

- Each quality critera is relevant to a specific data type. For example, the `two-words` quality criteria is only relevant to strings, and the `positive` quality criteria is only relevant to integers and floats.
- To see the full list of supported quality criteria, check out the [Validation](/docs/hub/concepts/validators) page.


## 🛠️ Specifying corrective actions

The `on-fail-{quality-criteria}` attribute allows specifying the corrective action that should be taken if the quality criteria is not met. The corrective action can be one of the following:

| Action    | Behavior                                                                                                                                                                                               |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `reask`   | Reask the LLM to generate an output that meets the quality criteria.  The prompt used for reasking contains information about which quality criteria failed, which is auto-generated by the validator. |
| `fix`     | Programmatically fix the generated output to meet the quality criteria.  E.g. for the formatter `two-words`, the programatic `fix` simply takes the first 2 words of the generated string.             |
| `filter`  | Filter the incorrect value. This only filters the field that fails, and will return the rest of the generated output.                                                                                  |
| `refrain` | Refrain from returning an output. If a formatter has the corrective action refrain, then on failure there will be a `None` output returned instead of the JSON.                                        |
| `noop`    | Do nothing. The failure will still be recorded in the logs, but no corrective action will be taken.                                                                                                    |
| `exception`  | Raise an exception when validation fails.                                                                                                                                                      |
| `fix_reask` | First, fix the generated output deterministically, and then rerun validation with the deterministically fixed output. If validation fails, then perform reasking.             |


## Adding compiled `output` element to prompt

In order to generate the correct LLM output, the `output` schema needs to be compiled and added to the prompt. This is handled automatically by the `Guardrails` library.

The `output` element can be compiled into different formats to be used in the prompt. Currently, only a passthrough compilation into `XML` is supported, but in the future we will support additional compilation formats like `TypeScript`.


### Passthrough (`XML`) compilation

By default, the `output` element will be compiled into `XML` and added to the prompt. Compilation into `XML` involves removing any `on-fail-{quality-criteria}` attributes, and adding the `output` element to the prompt.

An example of the compiled `output` element:

=== "RAIL Spec"

    ```xml
    <rail version="0.1">
        <output>
            <string
                name="text"
                description="The generated text"
                validators="guardrails/uppercase; guardrails/two_words"
            />
        </output>
    </rail>
    ```

=== "Compiled XML added to prompt"

    ```xml
    <output>
        <string
            name="text"
            description="The generated text"
        />
    </output>
    ```


### `TypeScript` Compilation

Coming soon!



## Unsupported tags and attributes

- By default, Guardrails will not throw an error if you add an unsupported type, attribute or quality criteria. Instead, it will treat the unsupported type as a string, and will not perform any quality checks on the field. Often, LLMs will generate a string for an unsupported type, so this behavior is useful.
- Unsupported tags and attributes will still be included in the output schema definition that is appended to the prompt.
- This behavior can be changed by setting the `strict` attribute of the `<output>` element to `true`. If `strict` is set to `true`, Guardrails will throw an error if you add an unsupported type, attribute or quality criteria.

    ```xml
    <rail version="0.1">
        <output strict="true">
            <unsupported-type ... />
        </output>
    </rail>
    ```

    This will throw an error:

    ```bash
    ❌ Error: Unsupported type: unsupported-type
    ```


