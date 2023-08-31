from pydantic import BaseModel, Field

prompt = """\n\nHuman:
Given the following resume, answer the following questions. If the answer doesn't exist in the resume, enter `null`.

${document}

Extract information from this resume and return a JSON that follows the correct schema.

${gr.complete_json_suffix}

\n\nAssistant:
"""  # noqa

document = """Joe Smith – 1234 5678 / joe@example.com  PRIVATE & CONFIDENTIAL
 1 Joe Smith
Lorem ipsum dolor sit amet, consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in
reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia
deserunt mollit anim id est laborum."""  # noqa

compiled_prompt = """\n\nHuman:\nGiven the following resume, answer the following questions. If the answer doesn\'t exist in the resume, enter `null`.\n\nJoe Smith – 1234 5678 / joe@example.com  PRIVATE & CONFIDENTIAL\n 1 Joe Smith\nLorem ipsum dolor sit amet, consectetur adipiscing elit,\nsed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\nUt enim ad minim veniam, quis nostrud exercitation ullamco laboris\nnisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in\nreprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.\nExcepteur sint occaecat cupidatat non proident, sunt in culpa qui officia\ndeserunt mollit anim id est laborum.\n\nExtract information from this resume and return a JSON that follows the correct schema.\n\n\nGiven below is XML that describes the information to extract from this document and the tags to extract it into.\n\n<output>\n    <string name="name" description="What is the candidate name?"/>\n    <string name="contact_number" description="What is the candidate contact number?"/>\n    <string name="contact_email" description="What is the candidate email address?"/>\n</output>\n\n\nONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML\'s tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`.\n\nHere are examples of simple (XML, JSON) pairs that show the expected behavior:\n- `<string name=\'foo\' format=\'two-words lower-case\' />` => `{\'foo\': \'example one\'}`\n- `<list name=\'bar\'><string format=\'upper-case\' /></list>` => `{"bar": [\'STRING ONE\', \'STRING TWO\', etc.]}`\n- `<object name=\'baz\'><string name="foo" format="capitalize two-words" /><integer name="index" format="1-indexed" /></object>` => `{\'baz\': {\'foo\': \'Some String\', \'index\': 1}}`\n\n\n\n\nAssistant:\n"""  # noqa

compiled_reask = """\nI was given the following response, which was not parseable as JSON.\n\n{\n  "incorrect_value": "Here is the JSON containing the requested information extracted from the resume:\\n\\n```\\n  name: Joe Smith\\n  contact_number: 1234 5678\\n  contact_email: joe@example.com\\n```",\n  "error_messages": [\n    "Output is not parseable as JSON"\n  ]\n}\n\nHelp me correct this by making it valid JSON.\n\nGiven below is XML that describes the information to extract from this document and the tags to extract it into.\n\n<output>\n    <string name="name" description="What is the candidate name?"/>\n    <string name="contact_number" description="What is the candidate contact number?"/>\n    <string name="contact_email" description="What is the candidate email address?"/>\n</output>\n\n\nONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML\'s tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`.\n"""  # noqa


class PersonalDetails(BaseModel):
    name: str = Field(..., description="What is the candidate name?")
    contact_number: str = Field(
        ..., description="What is the candidate contact number?"
    )
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
  name: Joe Smith
  contact_number: 1234 5678
  contact_email: joe@example.com
```"""  # noqa

expected_output = {
    "name": "Joe Smith",
    "contact_number": "1234 5678",
    "contact_email": "joe@example.com",
}
