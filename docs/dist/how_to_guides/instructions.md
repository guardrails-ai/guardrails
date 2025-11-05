# `Instructions` Element

**Note**: Instructions element support has been dropped in 0.6.0 in support of [messages](/docs/how_to_guides/messages).

The `<instructions></instructions>` element is passed to the LLM as secondary input. Different model may use these differently. For example, chat models may receive instructions in the system-prompt.

## 📚 Components of an Instructions Element

In addition to any static text describing the context of the task, instructions can also contain any of the following:

| Component         | Syntax                   | Description                                                                                                                                                                                                                                                                                                                             |
|-------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Variables         | `${variable_name}`        | These are provided by the user at runtime, and substituted in the instructions.                                                                                                                                                                                                                                                             |
| Output Schema     | `${output_schema}`      | This is the schema of the expected output, and is compiled based on the  `output` element.  For more information on how the output schema is compiled for the instructions, check out [`output` element compilation](/docs/how_to_guides/output#-adding-compiled-output-element-to-prompt)                                                                    |
| Prompt Primitives | `${gr.prompt_primitive_name}` | These are pre-constructed blocks of text that are useful for common tasks. E.g., some primitives may contain information that helps the LLM understand the output schema better.  To see the full list of prompt primitives, check out [`guardrails/constants.xml`](https://github.com/guardrails-ai/guardrails/blob/main/guardrails/constants.xml). |


Here's an example of how you could compose instructions using RAIL xml:
```xml
<rail version="0.1">
<instructions>
<!-- (1)! -->
You are a helpful assistant only capable of communicating with valid JSON, and no other text.

${gr.json_suffix_prompt_examples}  <!-- (2)! -->
</instructions>
</rail>
```

1. The instructions element contains high level background information for the LLM containing textual context and constraints.
2. `${gr.json_suffix_prompt_examples}` is a prompt primitive provided by guardrails. It is equivalent to typing the following lines in the instructions:
````
ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`.

Here are examples of simple (XML, JSON) pairs that show the expected behavior:
- `<![CDATA[<string name='foo' format='two-words lower-case' />`]]> => `{'foo': 'example one'}`
- `<![CDATA[<list name='bar'><string format='upper-case' /></list>]]>` => `{"bar": ['STRING ONE', 'STRING TWO', etc.]}`
- `<![CDATA[<object name='baz'><string name="foo" format="capitalize two-words" /><integer name="index" format="1-indexed" /></object>]]>` => `{'baz': {'foo': 'Some String', 'index': 1}}`
````

Or if you prefer Pydantic:
```py
# <!-- (1)! -->
instructions = """You are a helpful assistant only capable of communicating with valid JSON, and no other text.

    ${gr.json_suffix_prompt_examples}""" # <!-- (2)! -->
```


1. The instructions element contains high level background information for the LLM containing textual context and constraints.
2. `${gr.json_suffix_prompt_examples}` is a prompt primitive provided by guardrails. It is equivalent to typing the following lines in the instructions:
````
ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`.

Here are examples of simple (XML, JSON) pairs that show the expected behavior:
- `<![CDATA[<string name='foo' format='two-words lower-case' />`]]> => `{'foo': 'example one'}`
- `<![CDATA[<list name='bar'><string format='upper-case' /></list>]]>` => `{"bar": ['STRING ONE', 'STRING TWO', etc.]}`
- `<![CDATA[<object name='baz'><string name="foo" format="capitalize two-words" /><integer name="index" format="1-indexed" /></object>]]>` => `{'baz': {'foo': 'Some String', 'index': 1}}`
````

When either of the above are compiled, it would looks like this:
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


For an example of using instructions alongside a prompt see [this example for using chat models](../examples/guardrails_with_chat_models.ipynb).