import pytest
from .mock_llm_outputs import (
    MockOpenAICallable,
    MockAsyncOpenAICallable
)
from guardrails.run import Runner, AsyncRunner  # Replace 'your_module' with the actual module name
from guardrails.schema import StringSchema
from lxml import etree as ET
import guardrails as gd
from .test_assets import  string

def runner_instance(is_sync: bool):

    args = {
        "base_model": None,
        "instructions": ''' You are a helpful assistant, and you are helping me come up with a name for a pizza.
            ${gr.complete_string_suffix}
        ''',
        "prompt": gd.Prompt(string.COMPILED_PROMPT),
        "msg_history": None,
        "api": MockOpenAICallable,  # Replace with a synchronous API mock
        "input_schema": None, 
        "output_schema": StringSchema(root=ET.fromstring('''<output
    type="string"
    description="Name for the pizza"
    format="two-words"
    on-fail-two-words="reask"
/>''')) , 
        "guard_state": None 
    }
    if(is_sync): 
        return Runner(**args)
    else:
        return AsyncRunner(**args)

# Define a list of test cases with input values and expected outputs
test_cases = [
    {
        "index": 1,
        "instructions": ''' You are a helpful assistant, and you are helping me come up with a name for a pizza.
            ${gr.complete_string_suffix}
        ''',
        "prompt": '''Given the following ingredients, what would you call this pizza?
        ${ingredients}''',
        "msg_history": None,
        "api_sync": MockOpenAICallable,  # Replace with a synchronous API mock
        "api_async": MockAsyncOpenAICallable,  # Replace with an asynchronous API mock
        "output": "Tomato Cheese Pizza",  # Replace with the expected output
    },
    # Add more test cases as needed
]


# Create a parameterized test function
@pytest.mark.parametrize("test_case", test_cases)
@pytest.mark.asyncio
async def test_sync_and_async_equivalence(test_case):
    # Extract input values from the test case dictionary
    index = test_case["index"]
    instructions = test_case["instructions"]
    prompt = test_case["prompt"]
    msg_history = test_case["msg_history"]
    api_sync = test_case["api_sync"]
    api_async = test_case["api_async"]
    expected_output = test_case["output"]

    # Call the 'call' method synchronously
    result_sync = runner_instance(True).call(
        index, instructions, prompt, msg_history, api_sync, expected_output
    )

    print('result_sync: ', result_sync)

    # Call the 'async_call' method asynchronously
    result_async = await runner_instance(False).async_call(
        index, instructions, prompt, msg_history, api_async, expected_output
    )

    print('result_async: ', result_async)

    # Perform assertions to check if the results are equivalent
    assert result_sync.output == result_async.output
    # Add more assertions as needed to compare other properties

