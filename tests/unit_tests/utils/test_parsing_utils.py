import pytest

from string import Template
from guardrails.utils.parsing_utils import get_code_block, get_template_variables, has_code_block

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

@pytest.mark.parametrize(
    "has_get_identifiers",
    [
        True,
        False
    ],
)
def test_get_template_variables(mocker, has_get_identifiers):
    orig_hasattr = hasattr
    def mock_get_identifiers(obj, key, *args, **kwargs):
        if obj == Template and key == "get_identifiers":
            return has_get_identifiers
        return orig_hasattr(obj, key, *args, **kwargs)
    
    mocker.patch('builtins.hasattr', new=mock_get_identifiers)
    get_identifiers_spy = mocker.spy(Template, 'get_identifiers')

    string_template = "${my_var} $my_second_var {not_a_var}"
    vars = get_template_variables(string_template)

    assert get_identifiers_spy.called == has_get_identifiers
    assert vars == ["my_var", "my_second_var"]
