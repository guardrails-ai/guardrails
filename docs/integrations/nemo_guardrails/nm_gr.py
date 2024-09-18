from typing import List
from nemoguardrails import LLMRails, RailsConfig
from guardrails import Guard
from guardrails.integrations.nemoguardrails.nemoguardrails_guard import (
    NemoguardrailsGuard,
)
from pydantic import BaseModel, Field
import rich

# install("hub://guardrails/toxic_language", install_local_models=False)
from guardrails.hub import ToxicLanguage  # noqa: E402

# Load a guardrails configuration from the specified path.
config = RailsConfig.from_path("./config.yml")
nemorails = LLMRails(config)

# define your guard and validator(s)
guard = NemoguardrailsGuard(nemorails=nemorails).use(ToxicLanguage())

# res = guard(
#     messages=[
#         {
#             "role": "user",
#             "content": """Generate a list of words that I should ban at work.
#         Make sure to spell out all the words. Return only the list of words without
#         preamble or postamble. Make sure to return ONLY the list of words.
#         This list should include profanity, slurs, and other offensive language.""",
#         }
#     ],
# )

# print(res.validation_passed)  # type: ignore

# rich.print(guard.history.last.tree)

# res = guard(
#     messages=[
#         {
#             "role": "user",
#             "content": """Say hi in the nicest way you can""",
#         }
#     ]
# )

# print(res.validation_passed)  # type: ignore

class StructuredOutput(BaseModel):
    nontoxic: str = Field(validators=[ToxicLanguage()])

# pydantic_guard = NemoguardrailsGuard.from_pydantic(
#     nemorails=nemorails,
#     output_class=StructuredOutput,
# )

# res = pydantic_guard(
#     messages=[
#         {
#             "role": "system",
#             "content": """Only respond in JSON. The JSON should be formatted as follows:
            
#             {'nontoxic': response to the prompt}
            
#             AGAIN, be sure to respond ONLY with valid JSON. Do not include any other text.
#             """,
#         },
#         {
#             "role": "user",
#             "content": """Generate a list of words that I should ban at work.
#         Make sure to spell out all the words. Return only the list of words without
#         preamble or postamble. Make sure to return ONLY the list of words.
#         This list should include profanity, slurs, and other offensive language.""",
#         }
#     ],
#     num_reasks=1
# )

# print(res.validation_passed)  # type: ignore

# rich.print(pydantic_guard.history.last.tree)



# class StructuredOutput(BaseModel):
#     nontoxic: str = Field(validators=[ToxicLanguage()])

# pydantic_guard = Guard.from_pydantic(
#     # nemorails=nemorails,
#     output_class=StructuredOutput,
# )

# g = Guard()
# g(
#     model="gpt-3.5-turbo",
#     messages=[
#         # {
#         #     "role": "system",
#         #     "content": """Only respond in JSON. The JSON should be formatted as follows:
            
#         #     {'nontoxic': response to the prompt}
            
#         #     AGAIN, be sure to respond ONLY with valid JSON. Do not include any other text.
#         #     """,
#         # },
#         {
#             "role": "user",
#             "content": """Write  pydantic structure about medical data. It should have hundreds of fields, with some nesting""",
#         }
#     ],
#     # tools=pydantic_guard.json_function_calling_tool(),
#     # tool_choice="required",
#     # generate_kwargs={"temperature": 0.0},
# )
# # rich.print(pydantic_guard.history.last.tree)
# rich.print(g.history.last.tree)
# print(g.history.last.raw_outputs)


class MedicalData(BaseModel):
    patient_id: int
    patient_name: str
    age: int
    gender: str
    address: str
    contact_number: str
    blood_type: str
    allergies: List[str]
    medications: List[str]
    medical_history: List[str]
    family_history: List[str]
    insurance_provider: str
    insurance_policy_number: str
    emergency_contact_name: str
    emergency_contact_number: str
    primary_care_physician: str
    last_checkup_date: str
    next_checkup_date: str
    weight: float
    height: float
    blood_pressure: str
    heart_rate: int
    respiratory_rate: int
    temperature: float
    symptoms: List[str]
    diagnosis: str
    treatment_plan: str
    lab_results: List[str]
    imaging_results: List[str]
    follow_up_instructions: str
    additional_notes: str
    vaccination_records: List[str]
    surgeries: List[str]
    hospitalizations: List[str]
    chronic_conditions: List[str]
    lifestyle_factors: List[str]
    exercise_routine: str
    diet_plan: str
    sleep_pattern: str
    stress_level: str
    smoking_status: str
    alcohol_consumption: str
    drug_usage: str
    mental_health_history: List[str]
    social_support_system: List[str]
    living_situation: str
    employment_status: str
    education_level: str
    income_level: str
    access_to_healthcare: str
    healthcare_preferences: List[str]
    healthcare_goals: List[str]
    healthcare_challenges: List[str]
    healthcare_needs: List[str]
    healthcare_preferences: List[str]
    healthcare_expectations: List[str]
    healthcare_satisfaction: str
    healthcare_feedback: str
    additional_information: str
    # Add more fields as needed


med_pydantic_guard = Guard.from_pydantic(
    output_class=MedicalData,
)

import time

init_time = time.time()

med_pydantic_guard.json_function_calling_tool()

print(f"Time taken: {time.time() - init_time}")