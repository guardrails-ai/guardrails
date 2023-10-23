# `Instructions` Element

The `<instructions></instructions>` element contains the high level instructions sent to the LLM (e.g. the system message for chat models).

## 📚 Components of an Instructions Element

In addition to the high level task description, the instructions element also contains the following:

| Component         | Syntax                   | Description                                                                                                                                                                                                                                                                                                                             |
|-------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Variables         | `${variable_name}`        | These are provided by the user at runtime, and substituted in the prompt.                                                                                                                                                                                                                                                               |
| Output Schema     | `${output_schema}`      | This is the schema of the expected output, and is compiled based on the  `output` element.  For more information on how the output schema is compiled for the prompt, check out [`output` element compilation](../output/#adding-compiled-output-element-to-prompt).                                                                    |
| Prompt Primitives | `${gr.prompt_primitive_name}` | These are pre-constructed prompts that are useful for common tasks. E.g., some primitives may contain information that helps the LLM understand the output schema better.  To see the full list of prompt primitives, check out [`guardrails/constants.xml`](https://github.com/guardrails-ai/guardrails/blob/main/guardrails/constants.xml). |

```xml
<rail version="0.1">
<instructions>
<!-- (1)! -->
You are a helpful assistant only capable of communicating with valid JSON, and no other text.

${gr.json_suffix_prompt_examples}  <!-- (2)! -->
</instructions>
</rail>
```


1. The instructions element contains high level task information.
2. `${gr.json_suffix_prompt_examples}` is a prompt primitive provided by guardrails. It is equivalent to typing the following lines in the prompt: `Given below is XML that describes the information to extract from this document and the tags to extract it into.`

```
You are a helpful assistant only capable of communicating with valid JSON, and no other text.

ONLY return a valid JSON object (no other text is necessary).
The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types.
Be correct and concise. If you are unsure anywhere, enter `null`.

Here are examples of simple (XML, JSON) pairs that show the expected behavior:
- `<string name='foo' format='two-words lower-case' />` => `{'foo': 'example one'}`
- `<list name='bar'><string format='upper-case' /></list>` => `{"bar": ['STRING ONE', 'STRING TWO', etc.]}`
- `<object name='baz'><string name="foo" format="capitalize two-words" /><integer name="index" format="1-indexed" /></object>` => `{'baz': {'foo': 'Some String', 'index': 1}}`
```
