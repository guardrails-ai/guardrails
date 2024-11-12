# `Prompt` Element


**Note**: Prompt element support has been dropped in 0.6.0 in support of [messages](./messages).

The `<prompt></prompt>` element contains the query that describes the high level task.

## 📚 Components of a Prompt Element

In addition to the high level task description, the prompt also contains the following:

| Component         | Syntax                   | Description                                                                                                                                                                                                                                                                                                                             |
|-------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Variables         | `${variable_name}`        | These are provided by the user at runtime, and substituted in the prompt.                                                                                                                                                                                                                                                               |
| Output Schema     | `${output_schema}`      | This is the schema of the expected output, and is compiled based on the  `output` element.  For more information on how the output schema is compiled for the prompt, check out [`output` element compilation](/docs/concepts/output/#adding-compiled-output-element-to-prompt).                                                                    |
| Prompt Primitives | `${gr.prompt_primitive_name}` | These are pre-constructed prompts that are useful for common tasks. E.g., some primitives may contain information that helps the LLM understand the output schema better.  To see the full list of prompt primitives, check out [`guardrails/constants.xml`](https://github.com/guardrails-ai/guardrails/blob/main/guardrails/constants.xml). |

```xml
<rail version="0.1">
<prompt>
<!-- (1)! -->
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 'None'.

${document} <!-- (2)! -->


${gr.xml_prefix_prompt}  <!-- (3)! -->


${output_schema}  <!-- (4)! -->


${gr.json_suffix_prompt}  <!-- (5)! -->

</prompt>
</rail>
```

1. The prompt contains high level task information.
2. The variable `${document}` is provided by the user at runtime.
3. `${gr.xml_prefix_prompt}` is a prompt primitive provided by guardrails. It is equivalent to typing the following lines in the prompt: `Given below is XML that describes the information to extract from this document and the tags to extract it into.`
4. `${output_schema}` is the output schema and contains information about , which is compiled based on the `output` element.
5. `${gr.json_suffix_prompt}` is a prompt primitive provided by guardrails. It is equivalent to typing the following lines in the prompt:
```
ONLY return a valid JSON object (no other text is necessary). The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`.
```
