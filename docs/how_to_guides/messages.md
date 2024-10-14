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