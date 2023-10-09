from guardrails.validators import FailResult, PassResult

validator_test_pass_fail = {
    "BugFreeSQL": [
        {
            "input_data": "select name, fro employees",
            "metadata": {},
            "expected_result": FailResult,
        },
        {
            "input_data": "select name from employees;",
            "metadata": {},
            "expected_result": PassResult,
        },
    ],
    "ValidLength": [
        {
            "input_data": "hello there yo",
            "metadata": {},
            "expected_result": PassResult,
            "instance_variables": {"min": 4, "max": 17},
        },
        {
            "input_data": "hello there, this is getting wordyyy",
            "metadata": {},
            "expected_result": FailResult,
            "instance_variables": {"min": 4, "max": 17},
            "fix_value": "hello there, this",
        },
    ],
    "LowerCase": [
        {
            "input_data": "this is all lowercase",
            "metadata": {},
            "expected_result": PassResult,
        },
        {
            "input_data": "OOPS, there is defintely an issue here",
            "metadata": {},
            "expected_result": FailResult,
            "fix_value": "oops, there is defintely an issue here",
        },
    ],
    "UpperCase": [
        {
            "input_data": "this is all lowercase",
            "metadata": {},
            "expected_result": FailResult,
            "fix_value": "THIS IS ALL LOWERCASE",
        },
        {
            "input_data": "NO ISSUE HERE",
            "metadata": {},
            "expected_result": PassResult,
        },
    ],
    "TwoWords": [
        {
            "input_data": "one TWO",
            "metadata": {},
            "expected_result": PassResult,
        },
        {
            "input_data": "one two three four",
            "metadata": {},
            "expected_result": FailResult,
            "fix_value": "one two",
        },
    ],
    "OneLine": [
        {
            "input_data": "this is a simple one liner",
            "metadata": {},
            "expected_result": PassResult,
        },
        {
            "input_data": "This should defintely fail \n since this is a new line",
            "metadata": {},
            "expected_result": FailResult,
            "fix_value": "This should defintely fail ",
        },
    ],
    "EndsWith": [
        {
            "input_data": ["start", "middle", "end"],
            "metadata": {},
            "expected_result": PassResult,
            "instance_variables": {
                "end": "end",
            },
        },
        {
            "input_data": ["start", "middle", "end"],
            "metadata": {},
            "expected_result": FailResult,
            "fix_value": ["start", "middle", "end", "trunk"],
            "instance_variables": {
                "end": "trunk",
            },
        },
    ],
    "ValidURL": [
        {"input_data": "http:///google", "metadata": {}, "expected_result": FailResult},
        {
            "input_data": "http://google.com",
            "metadata": {},
            "expected_result": PassResult,
        },
    ],
    "BugFreePython": [
        {
            "input_data": """def longestPalindrome(s):
            \n    longest_palindrome = ''
            \n    for i in range(len(s)):
            \n        for j in range(i, len(s)):
            \n            subs = s[i:j+1]
            \n            if subs == subs[::-1] and len(subs) > len(longest_palindrome):
            \n                longest_palindrome = subs
            \n    return longest_palindrome""",
            "metadata": {},
            "expected_result": PassResult,
        },
        {
            "input_data": """deff longestPalindrome(s):
            \n    longest_palindrome = ''
            \n    for i in range(len(s)):
            \n        for j in range(i, len(s)):
            \n            subs = s[i:j+1]
            \n            if subs == subs[::-1] and len(subs) > len(longest_palindrome):
            \n                longest_palindrome = subs
            \n    return longest_palindrome""",
            "metadata": {},
            "expected_result": FailResult,
        },
    ],
    "ReadingTime": [
        {
            "input_data": """This string is fairly short and should be
              able to be read in the given timeframe""",
            "metadata": {},
            "expected_result": PassResult,
            "instance_variables": {
                "reading_time": 5,
            },
        },
        {
            "input_data": """
            This string is fairly short and should be able to be read in the given
              timeframe but I wonder if we copy this multiple times and shortened
                the time frame to a fraction of the time
            This string is fairly short and should be able to be read in the given
              timeframe but I wonder if we copy this multiple times and shortened
                the time frame to a fraction of the time
            This string is fairly short and should be able to be read in the given
              timeframe but I wonder if we copy this multiple times and shortened
                the time frame to a fraction of the time
            This string is fairly short and should be able to be read in the given
              timeframe but I wonder if we copy this multiple times and shortened
                the time frame to a fraction of the time""",
            "metadata": {},
            "expected_result": FailResult,
            "fix_value": """
            This string is fairly short and should be able to be read in the given
              timeframe but I wonder if we copy this multiple times and shortened
                the time frame to a fraction of the time
            This string is fairly short and should be able to be read in the given
              timeframe but I wonder if we copy this multiple times and shortened
                the time frame to a fraction of the time
            This string is fairly short and should be able to be read in the given
              timeframe but I wonder if we copy this multiple times and shortened
                the time frame to a fraction of the time
            This string is fairly short and should be able to be read in the given
              timeframe but I wonder if we copy this multiple times and shortened
                the time frame to a fraction of the time""",
            "instance_variables": {
                "reading_time": 1,
            },
        },
    ],
    "ValidChoices": [
        {
            "input_data": "four",
            "metadata": {},
            "expected_result": PassResult,
            "instance_variables": {
                "choices": ["one", "two", "three", "four"],
            },
        },
        {
            "input_data": "five",
            "metadata": {},
            "expected_result": FailResult,
            "instance_variables": {
                "choices": ["one", "two", "three", "four"],
            },
        },
    ],
    "ValidRange": [
        {
            "input_data": 180,
            "metadata": {},
            "expected_result": FailResult,
            "instance_variables": {"min": 4, "max": 170},
            "fix_value": 170,
        },
        {
            "input_data": 4.5,
            "metadata": {},
            "expected_result": PassResult,
            "instance_variables": {"min": 4, "max": 170},
            "fix_value": 4,
        },
    ],
    "EndpointIsReachable": [
        {
            "input_data": "https://www.google.com/",
            "metadata": {},
            "expected_result": PassResult,
        },
        {
            "input_data": "https://wwww.google/",
            "metadata": {},
            "expected_result": FailResult,
        },
    ],
    "RegexMatch": [
        {
            "input_data": "ab12cd",
            "metadata": {},
            "expected_result": PassResult,
            "instance_variables": {"regex": "\\w+\\d\\w+", "match_type": "fullmatch"},
        },
        {
            "input_data": "abcdef",
            "metadata": {},
            "expected_result": FailResult,
            "instance_variables": {"regex": "\\w+\\d\\w+", "match_type": "fullmatch"},
        },
    ],
    "RemoveRedundantSentences": [
        {
            "input_data": "There shouldn't be any redundant sentences here \
                so this test should pass no problem.",
            "metadata": {},
            "expected_result": PassResult,
        },
        {
            "input_data": "There are redundant sentences here so this shouldn't pass. \
                Because there are redundant sentences here, this should not pass. \
                Okay this is the last sentence.",
            "metadata": {},
            "expected_result": FailResult,
            "fix_value": "There are redundant sentences here so this shouldn't pass.",
        },
    ],
    "IsProfanityFree": [
        {
            "input_data": "Keeping it clean here so this should pass \
            without any problems.",
            "metadata": {},
            "expected_result": PassResult,
        },
        {
            "input_data": "crap, is this a bad word",
            "metadata": {},
            "expected_result": FailResult,
        },
    ],
    "SimilarToList": [
        {
            "input_data": "3",
            "metadata": {"prev_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            "expected_result": PassResult,
        },
        {
            "input_data": "300",
            "metadata": {"prev_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            "expected_result": FailResult,
        },
    ],
}

validator_test_python_str = {
    "TwoWords": {
        "description": "Name for the pizza",
        "instructions": """
You are a helpful assistant, and you are helping me come up with a name for a pizza.

${gr.complete_string_suffix}
""",
        "prompt": """
Given the following ingredients, what would you call this pizza?

${ingredients}
""",
        "prompt_params": {"ingredients": "tomato, cheese, sour cream"},
        "output": "Cheese Pizza",
    }
}

validator_test_xml = {
    "ValidLength": {
        "expected_xml": "length: 4 17",
        "instance_variables": {"min": 4, "max": 17},
    },
    "BugFreeSQL": {
        "expected_xml": "bug-free-sql",
    },
    "ExtractedSummarySentencesMatch": {
        "expected_xml": "extracted-summary-sentences-match",
    },
    "LowerCase": {
        "expected_xml": "lower-case",
    },
    "UpperCase": {
        "expected_xml": "upper-case",
    },
    "TwoWords": {
        "expected_xml": "two-words",
    },
    "OneLine": {
        "expected_xml": "one-line",
    },
    "ValidURL": {
        "expected_xml": "valid-url",
    },
    "BugFreePython": {
        "expected_xml": "bug-free-python",
    },
    "EndsWith": {
        "expected_xml": "ends-with: bye",
        "instance_variables": {
            "end": "bye",
        },
    },
    "ValidChoices": {
        "expected_xml": "valid-choices: {['one', 'two', 'three', 'four']}",
        "instance_variables": {
            "choices": ["one", "two", "three", "four"],
        },
    },
    "ValidRange": {
        "expected_xml": "valid-range: 1 150",
        "instance_variables": {"min": 1, "max": 150},
    },
    "EndpointIsReachable": {
        "expected_xml": "is-reachable",
    },
    "SqlColumnPresence": {
        "expected_xml": "sql-column-presence: {['embeddings', 'guards']}",
        "instance_variables": {"cols": ["embeddings", "guards"]},
    },
    "ExcludeSqlPredicates": {
        "expected_xml": "exclude-sql-predicates: {['EXISTS', 'IN']}",
        "instance_variables": {"predicates": ["EXISTS", "IN"]},
    },
    "IsProfanityFree": {"expected_xml": "is-profanity-free"},
    "RemoveRedundantSentences": {"expected_xml": "remove-redundant-sentences"},
    "RegexMatch": {
        "expected_xml": "regex_match: \\w+\\d\\w+ fullmatch",
        "instance_variables": {"regex": "\\w+\\d\\w+", "match_type": "fullmatch"},
    },
    "SimilarToList": {
        "expected_xml": "similar-to-list: 2 0.5",
        "instance_variables": {"standard_deviations": 2, "threshold": 0.5},
    },
}

validator_test_prompt = {
    "ValidLength": {
        "expected_prompt": "length: min=4 max=17",
        "instance_variables": {"min": 4, "max": 17},
    },
    "BugFreeSQL": {
        "expected_prompt": "bug-free-sql",
    },
    "LowerCase": {
        "expected_prompt": "lower-case",
    },
    "UpperCase": {
        "expected_prompt": "upper-case",
    },
    "TwoWords": {
        "expected_prompt": "two-words",
    },
    "OneLine": {
        "expected_prompt": "one-line",
    },
    "ValidURL": {
        "expected_prompt": "valid-url",
    },
    "BugFreePython": {
        "expected_prompt": "bug-free-python",
    },
    "EndsWith": {
        "expected_prompt": "ends-with: end=bye",
        "instance_variables": {
            "end": "bye",
        },
    },
    "ValidChoices": {
        "expected_prompt": "valid-choices: choices=['one', 'two', 'three', 'four']",
        "instance_variables": {
            "choices": ["one", "two", "three", "four"],
        },
    },
    "ValidRange": {
        "expected_prompt": "valid-range: min=1 max=150",
        "instance_variables": {"min": 1, "max": 150},
    },
    "EndpointIsReachable": {
        "expected_prompt": "is-reachable",
    },
    "SqlColumnPresence": {
        "expected_prompt": "sql-column-presence: cols=['embeddings', 'guards']",
        "instance_variables": {"cols": ["embeddings", "guards"]},
    },
    "ExcludeSqlPredicates": {
        "expected_prompt": "exclude-sql-predicates: predicates=['EXISTS', 'IN']",
        "instance_variables": {"predicates": ["EXISTS", "IN"]},
    },
    "IsProfanityFree": {"expected_prompt": "is-profanity-free"},
    "RemoveRedundantSentences": {"expected_prompt": "remove-redundant-sentences"},
    "RegexMatch": {
        "expected_prompt": "results should match \\w+\\d\\w+",
        "instance_variables": {"regex": "\\w+\\d\\w+", "match_type": "fullmatch"},
    },
}
