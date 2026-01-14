from pydantic import BaseModel, Field

prompt = """\n\nHuman:
Given the following resume, answer the following questions. If the answer doesn't exist in the resume, enter `null`.

${document}

Extract information from this resume and return a JSON that follows the correct schema.

${gr.complete_xml_suffix}

\n\nAssistant:
"""  # noqa

document = (
    """Joe Smith – 1234 5678 / joe@example.com  PRIVATE & CONFIDENTIAL
 1 Joe Smith
Lorem ipsum dolor sit amet, consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in
reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia
deserunt mollit anim id est laborum."""
    ""
)  # noqa

compiled_prompt = """

Human:
Given the following resume, answer the following questions. If the answer doesn't exist in the resume, enter `null`.

Joe Smith – 1234 5678 / joe@example.com  PRIVATE & CONFIDENTIAL
 1 Joe Smith
Lorem ipsum dolor sit amet, consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in
reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia
deserunt mollit anim id est laborum.

Extract information from this resume and return a JSON that follows the correct schema.


Given below is XML that describes the information to extract from this document and the tags to extract it into.

<output>
  <string description="What is the candidate name?" name="name" required="true"></string>
  <string description="What is the candidate contact number?" name="contact_number" required="true"></string>
  <string description="What is the candidate email address?" name="contact_email" required="true"></string>
</output>

ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`.

Here are examples of simple (XML, JSON) pairs that show the expected behavior:
- `<string name='foo' format='two-words lower-case' />` => `{'foo': 'example one'}`
- `<list name='bar'><string format='upper-case' /></list>` => `{"bar": ['STRING ONE', 'STRING TWO', etc.]}`
- `<object name='baz'><string name="foo" format="capitalize two-words" /><integer name="index" format="1-indexed" /></object>` => `{'baz': {'foo': 'Some String', 'index': 1}}`




Assistant:
"""  # noqa

compiled_reask = """
I was given the following response, which was not parseable as JSON.

"Here is the JSON containing the requested information extracted from the resume:\\n\\n```\\n{\\n  \\"name\\": \\"Joe Smith\\",\\n  \\"contact_number\\": \\"1234 5678\\",\\n  \\"contact_email\\": \\"joe@example.com\\"\\n```"

Help me correct this by making it valid JSON.

Given below is XML that describes the information to extract from this document and the tags to extract it into.

<output>
  <string description="What is the candidate name?" name="name" required="true"></string>
  <string description="What is the candidate contact number?" name="contact_number" required="true"></string>
  <string description="What is the candidate email address?" name="contact_email" required="true"></string>
</output>

ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`.
"""  # noqa


class PersonalDetails(BaseModel):
    name: str = Field(..., description="What is the candidate name?")
    contact_number: str = Field(..., description="What is the candidate contact number?")
    contact_email: str = Field(..., description="What is the candidate email address?")


expected_llm_output = """Here is the JSON containing the requested information extracted from the resume:

```json
{
  "name": "Joe Smith",
  "contact_number": "1234 5678",
  "contact_email": "joe@example.com"
}
```"""  # noqa

unparseable_llm_response = """Here is the JSON containing the requested information extracted from the resume:

```
{
  "name": "Joe Smith",
  "contact_number": "1234 5678",
  "contact_email": "joe@example.com"
```"""  # noqa

expected_output = {
    "name": "Joe Smith",
    "contact_number": "1234 5678",
    "contact_email": "joe@example.com",
}
