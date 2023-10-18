import importlib
from typing import Dict

import openai
import pytest
from integration_tests.mock_llm_outputs import MockOpenAICallable

from guardrails.guard import Guard
from guardrails.validators import FailResult

from .validators.test_parameters import (
    validator_test_pass_fail,
    validator_test_prompt,
    validator_test_python_str,
    validator_test_xml,
)


# TODO: Spread this object, so essentially each validator will be its own test case
@pytest.mark.parametrize("validator_test_data", [(validator_test_pass_fail)])
def test_validator_validate(validator_test_data: Dict[str, Dict[str, str]]):
    for validator_name in validator_test_data:
        print("testing validator: ", validator_name)
        module = importlib.import_module("guardrails.validators")
        validator_class = getattr(module, validator_name)
        for test_scenario in validator_test_data[validator_name]:
            if "instance_variables" in test_scenario:
                instance = validator_class(**test_scenario["instance_variables"])
            else:
                instance = validator_class()
            result = instance.validate(
                test_scenario["input_data"],
                test_scenario["metadata"],
            )
            assert isinstance(result, test_scenario["expected_result"])

            if isinstance(result, FailResult) and "fix_value" in test_scenario:
                assert result.fix_value == test_scenario["fix_value"]


@pytest.mark.parametrize("validator_test_data", [(validator_test_python_str)])
def test_validator_python_string(
    mocker, validator_test_data: Dict[str, Dict[str, str]]
):
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    for validator_name in validator_test_data:
        print("testing validator: ", validator_name)
        module = importlib.import_module("guardrails.validators")
        validator_class = getattr(module, validator_name)
        validators = [validator_class(on_fail="reask")]
        guard = Guard.from_string(
            validators,
            description=validator_test_data[validator_name]["description"],
            prompt=validator_test_data[validator_name]["prompt"],
            instructions=validator_test_data[validator_name]["instructions"],
        )
        _, final_output = guard(
            llm_api=openai.Completion.create,
            prompt_params=validator_test_data[validator_name]["prompt_params"],
            num_reasks=1,
            max_tokens=100,
        )

        assert final_output == validator_test_data[validator_name]["output"]


# TODO: Spread this object, so essentially each validator will be its own test case
@pytest.mark.parametrize("validator_test_data", [(validator_test_xml)])
def test_validator_to_xml(validator_test_data: Dict[str, Dict[str, str]]):
    for validator_name in validator_test_data:
        module = importlib.import_module("guardrails.validators")
        print("testing validator: ", validator_name)
        validator_class = getattr(module, validator_name)
        if "instance_variables" in validator_test_data[validator_name]:
            instance = validator_class(
                **validator_test_data[validator_name]["instance_variables"]
            )
        else:
            instance = validator_class()
        xml = instance.to_xml_attrib()
        assert xml == validator_test_data[validator_name]["expected_xml"]


# TODO: Spread this object, so essentially each validator will be its own test case
@pytest.mark.parametrize("validator_test_data", [(validator_test_prompt)])
def test_validator_to_prompt(validator_test_data: Dict[str, Dict[str, str]]):
    for validator_name in validator_test_data:
        module = importlib.import_module("guardrails.validators")
        print("testing validator: ", validator_name)
        validator_class = getattr(module, validator_name)
        if "instance_variables" in validator_test_data[validator_name]:
            instance = validator_class(
                **validator_test_data[validator_name]["instance_variables"]
            )
        else:
            instance = validator_class()
        prompt = instance.to_prompt()
        assert prompt == validator_test_data[validator_name]["expected_prompt"]
