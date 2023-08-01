# flake8: noqa: E501

OPTIONAL_PROMPT_COMPLETION_MODEL = """
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 'None'.

{{document}}

@xml_prefix_prompt

{output_schema}

@json_suffix_prompt_v2_wo_none"""


OPTIONAL_PROMPT_CHAT_MODEL = """
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter `null`.

{{document}}

Extract information from this document and return a JSON that follows the correct schema.

@xml_prefix_prompt

{output_schema}
"""
OPTIONAL_INSTRUCTIONS_CHAT_MODEL = """
You are a helpful assistant only capable of communicating with valid JSON, and no other text.

@json_suffix_prompt_examples
"""
