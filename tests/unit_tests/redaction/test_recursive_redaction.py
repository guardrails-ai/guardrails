import unittest
from guardrails.telemetry.common import recursive_key_operation, redact
import ast


# Test suite for recursive_key_operation function
class TestRecursiveKeyOperation(unittest.TestCase):
    def test_list(self):
        data = '{"init_args": [], "init_kwargs": {"model": "gpt-4o-mini", \
            "api_base": "https://api.openai.com/v1", "api_key": "sk-1234"}}'
        result = recursive_key_operation(data, redact)
        assert ast.literal_eval(result)["init_kwargs"]["api_key"] == "***1234"

    def test_dict_kwargs(self):
        data = {
            "index": "0",
            "api": '{"init_args": [], "init_kwargs": {"model": "gpt-4o-mini",\
                "api_base": "https://api.openai.com/v1", "api_key": "sk-1234"}}',
            "messages": None,
            "prompt_params": "{}",
            "output_schema": '{"type": "string"}',
            "output": None,
        }
        result = recursive_key_operation(data, redact)
        assert ast.literal_eval(result["api"])["init_kwargs"]["api_key"] == "***1234"

    def test_nomatch(self):
        data = {"somekey": "soemvalue"}
        result = recursive_key_operation(data, redact)
        self.assertEqual(result, data)

    # def test_empty_dict(self):
    #     data = {}
    #     result = recursive_key_operation(data, redact)
    #     self.assertEqual(result, data)

    # def test_empty_list(self):
    #     data = []
    #     result = recursive_key_operation(data, redact)
    #     self.assertEqual(result, data)

    # def test_non_string_value(self):
    #     data = {'key': 123, 'another_key': 'value'}
    #     result = recursive_key_operation(data, redact)
    #     self.assertEqual(result, data)


if __name__ == "__main__":
    unittest.main()
