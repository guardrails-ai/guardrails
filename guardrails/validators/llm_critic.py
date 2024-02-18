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
                 rating_schema: BaseModel,
                 critic_prompt_tmplt: str,
                 critic_llm_api: Callable,
                 thresh: Union[int, dict] = 5,
                 on_fail: Optional[Callable] = None,
                 critic_guard_kwargs: dict = {}
                 ):
        super().__init__(
                         rating_schema=rating_schema,
                         critic_prompt_tmplt=critic_prompt_tmplt,
                         critic_llm_api=critic_llm_api,
                         on_fail=on_fail,
                         thresh=thresh,
                         critic_guard_kwargs=critic_guard_kwargs)
        self._rating_schema = rating_schema
        #Extract rating criteria from the Pydantic object
        criteria = set([k for k, v in self._rating_schema.__fields__.items()])
        #If per-criteria threshold is used, ensure that all criteria are addressed.
        if isinstance(thresh, dict):
            assert set(thresh.keys()) == criteria, "Must include a threshold for each criteria"
        #Store the threshold per criteria
        self._thresh = thresh if isinstance(thresh, dict) else {c:thresh for c in criteria}
        self._critic_llm_api = critic_llm_api
        from string import Formatter
        self._prompt_tmplt = critic_prompt_tmplt
        prompt_vars = [fn for _, fn, _, _ in Formatter().parse(self._prompt_tmplt) if fn is not None]
        assert len(prompt_vars) == 1, ("The prompt template for the critic must have exactly one "
                                                              "input variable, for the document the critic should rate.")
        self._prompt_var = prompt_vars[0]
        from guardrails import Guard
        self._rating_guard = Guard.from_pydantic(output_class=self._rating_schema)
        self._critic_guard_kwargs = critic_guard_kwargs

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        """Validates that the value receives ratings of at least self._thresh from the LLM critic."""
        #Get a rating from the rating model
        rating_prompt = self._prompt_tmplt.format(**{self._prompt_var : value})
        rating_prompt += "\n\n${gr.xml_prefix_prompt}\n${output_schema}\n${gr.complete_json_suffix_v2}"
        raw_llm_response, ratings = self._rating_guard(
            self._critic_llm_api,
            prompt=rating_prompt,
            #engine=self._rating_model,
            **self._critic_guard_kwargs
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