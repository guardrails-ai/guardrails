import pytest

from guardrails.utils.json_utils import extract_json_from_ouput

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

invalid_json_code_block = """
```json
{
    a: 1
}
```
"""

not_even_json = "This isn't even json..."


@pytest.mark.parametrize(
    "llm_ouput,expected_output,expected_error",
    [
        (json_code_block, {"a": 1}, None),
        (anonymous_code_block, {"a": 1}, None),
        (no_code_block, {"a": 1}, None),
        (js_code_block, None, "Expecting value: line 1 column 1 (char 0)"),
        (
            invalid_json_code_block,
            None,
            "Expecting property name enclosed in double quotes: line 2 column 5 (char 6)",  # noqa
        ),
        (not_even_json, None, "Expecting value: line 1 column 1 (char 0)"),
    ],
)
def test_extract_json_from_ouput(llm_ouput, expected_output, expected_error):
    actual_output, actual_error = extract_json_from_ouput(llm_ouput)

    assert actual_output == expected_output
    assert str(actual_error) == str(expected_error)
