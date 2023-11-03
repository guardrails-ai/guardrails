import pytest

from guardrails.utils.parsing_utils import (
    get_code_block,
    get_template_variables,
    has_code_block,
)

json_code_block = """
```json
{
    "a": 1
}
```
"""

anonymous_code_block = """
```
{
    "a": 1
}
```
"""

no_code_block = """
{
    "a": 1
}
"""

js_code_block = """
```js
{
    "a": 1
}
```
"""


not_even_json = "This isn't even json..."


@pytest.mark.parametrize(
    "llm_ouput,expected_output",
    [
        (json_code_block, (True, 1, 24)),
        (anonymous_code_block, (True, 1, 20)),
        (js_code_block, (True, 1, 22)),
        (no_code_block, (False, None, None)),
        (not_even_json, (False, None, None)),
    ],
)
def test_has_code_block(llm_ouput, expected_output):
    actual_output = has_code_block(llm_ouput)

    assert actual_output == expected_output


json_code = """{
    "a": 1
}"""
js_code = """js
{
    "a": 1
}"""


@pytest.mark.parametrize(
    "llm_ouput,expected_output,code_type",
    [
        (json_code_block, json_code, "json"),
        (anonymous_code_block, json_code, ""),
        (js_code_block, js_code, ""),
    ],
)
def test_get_code_block(llm_ouput, expected_output, code_type):
    has, start, end = has_code_block(llm_ouput)
    actual_output = get_code_block(llm_ouput, start, end, code_type)

    assert actual_output == expected_output


def test_get_template_variables():
    string_template = "${my_var} $my_second_var {not_a_var}"
    vars = get_template_variables(string_template)

    assert vars == ["my_var", "my_second_var"]
