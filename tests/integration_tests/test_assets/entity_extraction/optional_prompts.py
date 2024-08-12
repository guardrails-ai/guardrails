# ruff: noqa: E501

OPTIONAL_PROMPT_COMPLETION_MODEL = """
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 'None'.

${document}

${gr.xml_prefix_prompt}

${xml_output_schema}

${gr.xml_suffix_prompt_v2_wo_none}"""


OPTIONAL_PROMPT_CHAT_MODEL = """
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter `null`.

${document}

Extract information from this document and return a JSON that follows the correct schema.

${gr.xml_prefix_prompt}

${xml_output_schema}
"""
OPTIONAL_INSTRUCTIONS_CHAT_MODEL = """
You are a helpful assistant only capable of communicating with valid JSON, and no other text.

${gr.xml_suffix_prompt_examples}
"""


OPTIONAL_MSG_HISTORY = [
    {
        "role": "system",
        "content": "\nYou are a helpful assistant only capable of communicating with valid JSON, and no other text.\n\n${gr.xml_suffix_prompt_examples}\n",
    },
    {
        "role": "user",
        "content": "\nGiven the following document, answer the following questions. If the answer doesn't exist in the document, enter `null`.\n\n${document}\n\nExtract information from this document and return a JSON that follows the correct schema.\n\n${gr.xml_prefix_prompt}\n\n${xml_output_schema}\n",
    },
]
