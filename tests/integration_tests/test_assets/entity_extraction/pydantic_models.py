from typing import Dict, List

from pydantic import BaseModel, Field

from guardrails.validator_base import OnFailAction
from guardrails.validators import LowerCase, OneLine, TwoWords


class FeeDetailsFilter(BaseModel):
    index: int = Field(validators=("1-indexed", OnFailAction.NOOP))
    name: str = Field(
        validators=[
            LowerCase(on_fail=OnFailAction.FILTER),
            TwoWords(on_fail=OnFailAction.FILTER),
        ]
    )
    explanation: str = Field(validators=OneLine(on_fail=OnFailAction.FILTER))
    value: float = Field(validators=("percentage", OnFailAction.NOOP))


class ContractDetailsFilter(BaseModel):
    fees: List[FeeDetailsFilter] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings "
        "and checking accounts, loans, and credit products?"
    )


class FeeDetailsFix(BaseModel):
    index: int = Field(validators=("1-indexed", OnFailAction.NOOP))
    name: str = Field(
        validators=[
            LowerCase(on_fail=OnFailAction.FIX),
            TwoWords(on_fail=OnFailAction.FIX),
        ]
    )
    explanation: str = Field(validators=OneLine(on_fail=OnFailAction.FIX))
    value: float = Field(validators=("percentage", OnFailAction.NOOP))


class ContractDetailsFix(BaseModel):
    fees: List[FeeDetailsFix] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings "
        "and checking accounts, loans, and credit products?"
    )


class FeeDetailsNoop(BaseModel):
    index: int = Field(validators=("1-indexed", OnFailAction.NOOP))
    name: str = Field(
        validators=[
            LowerCase(on_fail=OnFailAction.NOOP),
            TwoWords(on_fail=OnFailAction.NOOP),
        ]
    )
    explanation: str = Field(validators=OneLine(on_fail=OnFailAction.NOOP))
    value: float = Field(validators=("percentage", OnFailAction.NOOP))


class ContractDetailsNoop(BaseModel):
    fees: List[FeeDetailsNoop] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings "
        "and checking accounts, loans, and credit products?"
    )


class FeeDetailsReask(BaseModel):
    index: int = Field(validators=("1-indexed", OnFailAction.NOOP))
    name: str = Field(
        validators=[
            LowerCase(on_fail=OnFailAction.NOOP),
            TwoWords(on_fail=OnFailAction.REASK),
        ]
    )
    explanation: str = Field(validators=OneLine(on_fail=OnFailAction.NOOP))
    value: float = Field(validators=("percentage", OnFailAction.NOOP))


class ContractDetailsReask(BaseModel):
    fees: List[FeeDetailsReask] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings "
        "and checking accounts, loans, and credit products?"
    )


class FeeDetailsRefrain(BaseModel):
    index: int = Field(validators=("1-indexed", OnFailAction.NOOP))
    name: str = Field(
        validators=[
            LowerCase(on_fail=OnFailAction.REFRAIN),
            TwoWords(on_fail=OnFailAction.REFRAIN),
        ]
    )
    explanation: str = Field(validators=OneLine(on_fail=OnFailAction.REFRAIN))
    value: float = Field(validators=("percentage", OnFailAction.NOOP))


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

${xml_output_schema}

${gr.xml_suffix_prompt_v2_wo_none}"""  # noqa: E501


INSTRUCTIONS_CHAT_MODEL = """
You are a helpful assistant only capable of communicating with valid JSON, and no other text.

${gr.xml_suffix_prompt_examples}
"""  # noqa: E501


PROMPT_CHAT_MODEL = """
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter `null`.

${document}

Extract information from this document and return a JSON that follows the correct schema.

${gr.xml_prefix_prompt}

${xml_output_schema}
"""  # noqa: E501
