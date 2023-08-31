from typing import Dict, List

from pydantic import BaseModel, Field

from guardrails.validators import LowerCase, OneLine, TwoWords


class FeeDetailsFilter(BaseModel):
    index: int = Field(validators=("1-indexed", "noop"))
    name: str = Field(
        validators=[LowerCase(on_fail="filter"), TwoWords(on_fail="filter")]
    )
    explanation: str = Field(validators=OneLine(on_fail="filter"))
    value: float = Field(validators=("percentage", "noop"))


class ContractDetailsFilter(BaseModel):
    fees: List[FeeDetailsFilter] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings "
        "and checking accounts, loans, and credit products?"
    )


class FeeDetailsFix(BaseModel):
    index: int = Field(validators=("1-indexed", "noop"))
    name: str = Field(validators=[LowerCase(on_fail="fix"), TwoWords(on_fail="fix")])
    explanation: str = Field(validators=OneLine(on_fail="fix"))
    value: float = Field(validators=("percentage", "noop"))


class ContractDetailsFix(BaseModel):
    fees: List[FeeDetailsFix] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings "
        "and checking accounts, loans, and credit products?"
    )


class FeeDetailsNoop(BaseModel):
    index: int = Field(validators=("1-indexed", "noop"))
    name: str = Field(validators=[LowerCase(on_fail="noop"), TwoWords(on_fail="noop")])
    explanation: str = Field(validators=OneLine(on_fail="noop"))
    value: float = Field(validators=("percentage", "noop"))


class ContractDetailsNoop(BaseModel):
    fees: List[FeeDetailsNoop] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings "
        "and checking accounts, loans, and credit products?"
    )


class FeeDetailsReask(BaseModel):
    index: int = Field(validators=("1-indexed", "noop"))
    name: str = Field(validators=[LowerCase(on_fail="noop"), TwoWords(on_fail="reask")])
    explanation: str = Field(validators=OneLine(on_fail="noop"))
    value: float = Field(validators=("percentage", "noop"))


class ContractDetailsReask(BaseModel):
    fees: List[FeeDetailsReask] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings "
        "and checking accounts, loans, and credit products?"
    )


class FeeDetailsRefrain(BaseModel):
    index: int = Field(validators=("1-indexed", "noop"))
    name: str = Field(
        validators=[LowerCase(on_fail="refrain"), TwoWords(on_fail="refrain")]
    )
    explanation: str = Field(validators=OneLine(on_fail="refrain"))
    value: float = Field(validators=("percentage", "noop"))


class ContractDetailsRefrain(BaseModel):
    fees: List[FeeDetailsRefrain] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings "
        "and checking accounts, loans, and credit products?"
    )


PROMPT = """
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 'None'.

${document}

${gr.xml_prefix_prompt}

${output_schema}

${gr.json_suffix_prompt_v2_wo_none}"""  # noqa: E501


INSTRUCTIONS_CHAT_MODEL = """
You are a helpful assistant only capable of communicating with valid JSON, and no other text.

${gr.json_suffix_prompt_examples}
"""  # noqa: E501


PROMPT_CHAT_MODEL = """
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter `null`.

${document}

Extract information from this document and return a JSON that follows the correct schema.

${gr.xml_prefix_prompt}

${output_schema}
"""  # noqa: E501
