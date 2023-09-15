from guardrails.validators import FailResult, PassResult

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
