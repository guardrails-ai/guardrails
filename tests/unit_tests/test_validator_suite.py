import os
import pytest
import openai
from pydantic import BaseModel, Field
from guardrails.guard import Guard
from typing import Dict
import importlib
from guardrails.validators import FailResult, PassResult
from integration_tests.mock_llm_outputs import MockOpenAICallable

'''
    ya I'd say at a minimum for this week try to cover happy paths for validate, parse, and fix functionality
'''

# we'll start with a dictionary that maps validator name to input and expected output
'''
valid-range  ValidRange
valid-choices ValidChoices
lower-case LowerCase
upper-case UpperCase
length ValidLength
two-words TwoWords
one-line OneLine
valid-url ValidURL
is-reachable EndpointIsReachable
bug-free-python BugFreePython
bug-free-sql BugFreeSQL
sql-column-presence SqlColumnPresence
exclude-sql-predicates ExcludeSqlPredicates
similar-to-document SimilarToDocument
is-profanity-free IsProfanityFree
is-high-quality-translation IsHighQualityTranslation
'''

validator_test_pass_fail = { 
    "BugFreeSQL": { 
        'input_data': 'select name, fro employees',
        'metadata': {}, 
        'expected_result': FailResult
    }, 
    "BugFreeSQL": { 
        'input_data': 'select name from employees;',
        'metadata': {}, 
        'expected_result': PassResult
    }, 
    "ExtractedSummarySentencesMatch": { 
        'input_data': 'It was a nice day. I went to the park. I saw a dog.', 
        'metadata': {
        "filepaths": [
            "./tests/unit_tests/test_assets/article1.txt",
            "./tests/unit_tests/test_assets/article2.txt",
        ]
        },
        'expected_result': PassResult
    }, 
    "ValidLength": {
        "input_data": 'hello there yo', 
        "metadata": {},
        "expected_result": PassResult, 
        "instance_variables": { 
            'min': 4, 
            'max': 17
        }
    }, 
    "ValidLength": {
        "input_data": 'hello there, this is getting wordyyy', 
        "metadata": {},
        "expected_result": FailResult, 
        "instance_variables": { 
            'min': 4, 
            'max': 17
        }
    }, 
    "LowerCase": { 
        "input_data": 'this is all lowercase', 
        "metadata": {},
        "expected_result": PassResult, 
    }, 
    "LowerCase": { 
        "input_data": 'OOPS, there is defintely an issue here', 
        "metadata": {},
        "expected_result": FailResult, 
    }, 
    "UpperCase": { 
        "input_data": 'this is all lowercase', 
        "metadata": {},
        "expected_result": FailResult, 
    }, 
    "UpperCase": { 
        "input_data": 'NO ISSUE HERE', 
        "metadata": {},
        "expected_result": PassResult, 
    }, 
    "TwoWords": { 
        "input_data": 'one TWO', 
        "metadata": {},
        "expected_result": PassResult, 
    },
    "TwoWords": { 
        "input_data": 'one two three four', 
        "metadata": {},
        "expected_result": FailResult, 
    }, 
    "OneLine": { 
        "input_data": 'this is a simple one liner', 
        "metadata": {},
        "expected_result": PassResult, 
    }, 
    "OneLine": { 
        "input_data": 'This should defintely fail \n since this is a new line', 
        "metadata": {},
        "expected_result": FailResult, 
    }, 
    "ValidURL": { 
        "input_data": 'http://google', 
        "metadata": {},
        "expected_result": FailResult, 
    }, 
    "ValidURL": { 
        "input_data": 'http://google.com', 
        "metadata": {},
        "expected_result": PassResult, 
    }, 
    "BugFreePython": { 
        "input_data": '''def longestPalindrome(s):\n    longest_palindrome = ''\n    for i in range(len(s)):\n        
for j in range(i, len(s)):\n            substring = s[i:j+1]\n            if substring == substring[::-1] and 
len(substring) > len(longest_palindrome):\n                longest_palindrome = substring\n    return 
longest_palindrome''', 
        "metadata": {},
        "expected_result": PassResult, 
    }, 
    "BugFreePython": { 
        "input_data": '''deff longestPalindrome(s):\n    longest_palindrome = ''\n    for i in range(len(s)):\n        
for j in range(i, len(s)):\n            substring = s[i:j+1]\n            if substring == substring[::-1] and 
len(substring) > len(longest_palindrome):\n                longest_palindrome = substring\n    return 
longest_palindrome''', 
        "metadata": {},
        "expected_result": FailResult, 
    }

}

validator_test_python_str = { 
"TwoWords": { 
        'description': 'Name for the pizza', 
        'instructions': """
You are a helpful assistant, and you are helping me come up with a name for a pizza.

${gr.complete_string_suffix}
""",
        'prompt': """
Given the following ingredients, what would you call this pizza?

${ingredients}
""", 
    'prompt_params': {"ingredients": "tomato, cheese, sour cream"}, 
    'output': 'Cheese Pizza'
    }
}

validator_test_xml = { 
    "ValidLength": {
        "expected_xml": "length: 4 17", 
        "instance_variables": { 
            'min': 4, 
            'max': 17
        }
    }, 
    "BugFreeSQL": { 
        'expected_xml': 'bug-free-sql',
    }, 
    "ExtractedSummarySentencesMatch": { 
        'expected_xml': 'extracted-summary-sentences-match',
    },
    "LowerCase": { 
        'expected_xml': 'lower-case',
    }, 
    "UpperCase": { 
        'expected_xml': 'upper-case',
    }, 
    "TwoWords": { 
        'expected_xml': 'two-words',
    }, 
    "OneLine": { 
        'expected_xml': 'one-line',
    }, 
    "ValidURL": { 
        'expected_xml': 'valid-url',
    }, 
    "BugFreePython": { 
        'expected_xml': 'bug-free-python',
    }
}

@pytest.mark.parametrize('validator_test_data', [(validator_test_pass_fail)])
def test_validate_pass_fail(validator_test_data: Dict[str, Dict[str, str]]):
    for validator_name in validator_test_data:
        print('testing validator: ', validator_name)
        module = importlib.import_module("guardrails.validators")
        validator_class = getattr(module, validator_name)
        if 'instance_variables' in validator_test_data[validator_name]: 
            instance = validator_class(**validator_test_data[validator_name]['instance_variables'])
        else:
            instance = validator_class()
        result = instance.validate(validator_test_data[validator_name]['input_data'], validator_test_data[validator_name]['metadata'])
        assert isinstance(result, validator_test_data[validator_name]['expected_result'])
        #print(instance)


'''def test_suite(validator_test_data: Dict[str, Dict[str, str]]):
    for validator_name in validator_test_data:
        #print('testing validator: ', validator_name)
        module = importlib.import_module("guardrails.validators")
        validator_class = getattr(module, validator_name)
        instance = validator_class()
        result = instance.validate(validator_test_data[validator_name]['input_data'], validator_test_data[validator_name]['metadata'])
        print('result: ', result)
        assert isinstance(result, validator_test_data[validator_name]['expected_result'])'''

@pytest.mark.parametrize('validator_test_data', [(validator_test_python_str)])
def test_validator_python_string(mocker, validator_test_data: Dict[str, Dict[str, str]]): 
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    for validator_name in validator_test_data:
        print('validator name: ', validator_name)
        module = importlib.import_module("guardrails.validators")
        validator_class = getattr(module, validator_name)
        validators = [validator_class(on_fail="reask")]
        guard = Guard.from_string(
        validators, description=validator_test_data[validator_name]['description'], prompt=validator_test_data[validator_name]['prompt'], instructions=validator_test_data[validator_name]['instructions'],
        )
        _, final_output = guard(
        llm_api=openai.Completion.create,
        prompt_params=validator_test_data[validator_name]['prompt_params'],
        num_reasks=1,
        max_tokens=100,
        )
        assert final_output == validator_test_data[validator_name]['output']

@pytest.mark.parametrize('validator_test_data', [(validator_test_xml)])
def test_validator_to_xml(validator_test_data: Dict[str, Dict[str, str]]):
        for validator_name in validator_test_data:
            module = importlib.import_module("guardrails.validators")
            validator_class = getattr(module, validator_name)
            if 'instance_variables' in validator_test_data[validator_name]: 
                instance = validator_class(**validator_test_data[validator_name]['instance_variables'])
            else:
                instance = validator_class()
            xml = instance.to_xml_attrib()
            assert xml == validator_test_data[validator_name]['expected_xml']
        


#run_test_suite_pass_fail(validator_test_pass_fail)
#test_validator_python_string(validator_test_python_str)
