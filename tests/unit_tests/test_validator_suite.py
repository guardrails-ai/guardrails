import os 
import pytest
from typing import Dict
import importlib
from guardrails.validators import FailResult, PassResult

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

validator_test_data = { 
    "ExtractedSummarySentencesMatch": { 
        'input_data': 'It was a nice day. I went to the park. I saw a dog.', 
        'metadata': {
        "filepaths": [
            "./tests/unit_tests/test_assets/article1.txt",
            "./tests/unit_tests/test_assets/article2.txt",
        ]
        },
        'expected_result': PassResult
    }
}


#@pytest.mark.parametrize('validator_test_pass_fail', (validator_test_data))
def run_test_suite_pass_fail(validator_test_data: Dict[str, Dict[str, str]]):
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


def run_test_suite(validator_test_data: Dict[str, Dict[str, str]]): 
    for validator_name in validator_test_data:
        #print('testing validator: ', validator_name)
        module = importlib.import_module("guardrails.validators")
        validator_class = getattr(module, validator_name)
        instance = validator_class()
        result = instance.validate(validator_test_data[validator_name]['input_data'], validator_test_data[validator_name]['metadata'])
        print('result: ', result)
        assert isinstance(result, validator_test_data[validator_name]['expected_result'])


run_test_suite_pass_fail(validator_test_pass_fail)
run_test_suite(validator_test_data)

'''
if 'metadata' in validator_test_data[validator_name]:
            print('yes metadata exists')

'''
