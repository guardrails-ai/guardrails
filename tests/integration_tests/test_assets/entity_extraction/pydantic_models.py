from typing import Dict, List

from guardrails.datatypes import Field, GuardModel

from guardrails.validators import LowerCase, TwoWords, OneLine


class FeeDetailsFilter(GuardModel):
    index: int = Field(gd_validators="1-indexed")
    name: str = Field(
        gd_validators=[LowerCase(on_fail="filter"), TwoWords(on_fail="filter")]
    )
    explanation: str = Field(gd_validators=OneLine(on_fail="filter"))
    value: float = Field(gd_validators="percentage")


class ContractDetailsFilter(GuardModel):
    fees: List[FeeDetailsFilter] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings and checking accounts, loans, and credit products?"
    )


class FeeDetailsFix(GuardModel):
    index: int = Field(gd_validators="1-indexed")
    name: str = Field(gd_validators=[LowerCase(on_fail="fix"), TwoWords(on_fail="fix")])
    explanation: str = Field(gd_validators=OneLine(on_fail="fix"))
    value: float = Field(gd_validators="percentage")


class ContractDetailsFix(GuardModel):
    fees: List[FeeDetailsFix] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings and checking accounts, loans, and credit products?"
    )


class FeeDetailsNoop(GuardModel):
    index: int = Field(gd_validators="1-indexed")
    name: str = Field(
        gd_validators=[LowerCase(on_fail="noop"), TwoWords(on_fail="noop")]
    )
    explanation: str = Field(gd_validators=OneLine(on_fail="noop"))
    value: float = Field(gd_validators="percentage")


class ContractDetailsNoop(GuardModel):
    fees: List[FeeDetailsNoop] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings and checking accounts, loans, and credit products?"
    )


class FeeDetailsReask(GuardModel):
    index: int = Field(gd_validators="1-indexed")
    name: str = Field(
        gd_validators=[LowerCase(on_fail="noop"), TwoWords(on_fail="reask")]
    )
    explanation: str = Field(gd_validators=OneLine(on_fail="noop"))
    value: float = Field(gd_validators="percentage")


class ContractDetailsReask(GuardModel):
    fees: List[FeeDetailsReask] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings and checking accounts, loans, and credit products?"
    )


class FeeDetailsRefrain(GuardModel):
    index: int = Field(gd_validators="1-indexed")
    name: str = Field(
        gd_validators=[LowerCase(on_fail="refrain"), TwoWords(on_fail="refrain")]
    )
    explanation: str = Field(gd_validators=OneLine(on_fail="refrain"))
    value: float = Field(gd_validators="percentage")


class ContractDetailsRefrain(GuardModel):
    fees: List[FeeDetailsRefrain] = Field(
        description="What fees and charges are associated with my account?"
    )
    interest_rates: Dict = Field(
        description="What are the interest rates offered by the bank on savings and checking accounts, loans, and credit products?"
    )


PROMPT = """
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 
'None'.

{{document}}

@xml_prefix_prompt

{output_schema}

@json_suffix_prompt_v2_wo_none"""


INSTRUCTIONS_CHAT_MODEL = """
You are a helpful assistant only capable of communicating with valid JSON, and no other text.

@json_suffix_prompt_examples
"""


PROMPT_CHAT_MODEL = """
Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 
`null`.

{{document}}

Extract information from this document and return a JSON that follows the correct schema.

@xml_prefix_prompt

{output_schema}
"""
