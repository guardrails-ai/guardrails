from pydantic import BaseModel
from typing import Any, Callable, Dict, Optional, Union
import openai
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
@register_validator(name="llm_critic", data_type="string")
class LLMCritic(Validator):
    def __init__(self,
                 input_doc: str,
                 rating_schema: BaseModel,
                 rating_prompt_tmplt: str,
                 rating_model: str = "gpt-4",
                 thresh: Union[int, dict] = 5,
                 on_fail: Optional[Callable] = None,
                 rating_guard_kwargs: dict = {}
                 ):
        super().__init__(input_doc=input_doc,
                         rating_schema=rating_schema,
                         rating_prompt_tmplt=rating_prompt_tmplt,
                         on_fail=on_fail,
                         rating_model=rating_model,
                         thresh=thresh,
                         rating_guard_kwargs=rating_guard_kwargs)
        self._input_doc = input_doc
        self._rating_schema = rating_schema
        #Extract rating criteria from the Pydantic object
        criteria = set([k for k, v in self._rating_schema.__fields__.items()])
        #If per-criteria threshold is used, ensure that all criteria are addressed.
        if isinstance(thresh, dict):
            assert set(thresh.keys()) == criteria, "Must include a threshold for each criteria"
        #Store the threshold per criteria
        self._thresh = thresh if isinstance(thresh, dict) else {c:thresh for c in criteria}
        self._rating_model = rating_model
        from langchain import PromptTemplate
        self._prompt_tmplt = PromptTemplate.from_template(rating_prompt_tmplt)
        self._is_chat_model = "gpt" in self._rating_model
        self._llm_api = openai.ChatCompletion.create if self._is_chat_model else openai.Completion.create
        from guardrails import Guard
        self._rating_guard = Guard.from_pydantic(output_class=self._rating_schema)
        self._rating_guard_kwargs = rating_guard_kwargs

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        """Validates that the value receives ratings of at least self._thresh from the LLM critic."""
        #Get a rating from the rating model
        rating_prompt = self._prompt_tmplt.format(input_doc=self._input_doc, summary=value)
        rating_prompt += "\n\n${gr.xml_prefix_prompt}\n${output_schema}\n${gr.complete_json_suffix_v2}"
        raw_llm_response, ratings = self._rating_guard(
            self._llm_api,
            prompt=rating_prompt,
            engine=self._rating_model,
            **self._rating_guard_kwargs
        )
        #On which criteria was the threshold not met?
        failed_criteria = set([])
        for criteria,score in ratings.items():
            if score < self._thresh[criteria]:
                failed_criteria.add(criteria)
        if len(failed_criteria) > 0:
            criteria2descrip = {k:v.field_info.description for k,v in self._rating_schema.__fields__.items()}
            fix_instructions = "Your last attempt failed according to the following criteria:\n"
            for criteria in failed_criteria:
                fix_instructions += (f"\n- It was not {criteria} enough. "
                                     f"You can make it more {criteria} by meeting the following goal: {criteria2descrip[criteria]}.")
            return FailResult(
                error_message=fix_instructions
            )
        return PassResult()