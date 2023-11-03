import json

import pytest
from lxml import etree as ET

from guardrails import Instructions, Prompt
from guardrails.datatypes import Object
from guardrails.schema import JsonSchema
from guardrails.utils import reask_utils
from guardrails.utils.logs_utils import GuardLogs
from guardrails.utils.reask_utils import (
    FieldReAsk,
    gather_reasks,
    sub_reasks_with_fixed_values,
)
from guardrails.validators import FailResult


@pytest.mark.parametrize(
    "input_dict, expected_dict",
    [
        (
            {
                "a": 1,
                "b": FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value=1,
                        )
                    ],
                ),
            },
            {"a": 1, "b": 1},
        ),
        (
            {
                "a": 1,
                "b": {
                    "c": 2,
                    "d": FieldReAsk(
                        incorrect_value=-1,
                        fail_results=[
                            FailResult(error_message="Error Msg", fix_value=2)
                        ],
                    ),
                },
            },
            {"a": 1, "b": {"c": 2, "d": 2}},
        ),
        (
            {
                "a": [
                    1,
                    2,
                    FieldReAsk(
                        incorrect_value=-1,
                        fail_results=[
                            FailResult(
                                error_message="Error Msg",
                                fix_value=3,
                            )
                        ],
                    ),
                ],
                "b": 4,
            },
            {"a": [1, 2, 3], "b": 4},
        ),
        (
            {
                "a": [
                    1,
                    2,
                    {
                        "c": FieldReAsk(
                            incorrect_value=-1,
                            fail_results=[
                                FailResult(
                                    error_message="Error Msg",
                                    fix_value=3,
                                )
                            ],
                        )
                    },
                ],
            },
            {"a": [1, 2, {"c": 3}]},
        ),
        (
            {
                "a": [
                    1,
                    2,
                    [
                        3,
                        4,
                        FieldReAsk(
                            incorrect_value=-1,
                            fail_results=[
                                FailResult(
                                    error_message="Error Msg",
                                    fix_value=5,
                                )
                            ],
                        ),
                    ],
                ]
            },
            {"a": [1, 2, [3, 4, 5]]},
        ),
        ({"a": 1}, {"a": 1}),
    ],
)
def test_sub_reasks_with_fixed_values(input_dict, expected_dict):
    """Test that sub reasks with fixed values are replaced."""
    assert sub_reasks_with_fixed_values(input_dict) == expected_dict


def test_gather_reasks():
    """Test that reasks are gathered."""
    input_dict = {
        "a": 1,
        "b": FieldReAsk(
            incorrect_value=-1,
            fail_results=[
                FailResult(
                    error_message="Error Msg",
                    fix_value=1,
                )
            ],
        ),
        "c": {
            "d": FieldReAsk(
                incorrect_value=-1,
                fail_results=[
                    FailResult(
                        error_message="Error Msg",
                        fix_value=2,
                    )
                ],
            )
        },
        "e": [
            1,
            2,
            FieldReAsk(
                incorrect_value=-1,
                fail_results=[
                    FailResult(
                        error_message="Error Msg",
                        fix_value=3,
                    )
                ],
            ),
        ],
        "f": [
            1,
            2,
            {
                "g": FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value=4,
                        )
                    ],
                )
            },
        ],
        "h": [
            1,
            2,
            [
                3,
                4,
                FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value=5,
                        )
                    ],
                ),
            ],
        ],
    }
    expected_reasks = [
        FieldReAsk(
            incorrect_value=-1,
            fail_results=[
                FailResult(
                    error_message="Error Msg",
                    fix_value=1,
                )
            ],
            path=["b"],
        ),
        FieldReAsk(
            incorrect_value=-1,
            fail_results=[
                FailResult(
                    error_message="Error Msg",
                    fix_value=2,
                )
            ],
            path=["c", "d"],
        ),
        FieldReAsk(
            incorrect_value=-1,
            fail_results=[
                FailResult(
                    error_message="Error Msg",
                    fix_value=3,
                )
            ],
            path=["e", 2],
        ),
        FieldReAsk(
            incorrect_value=-1,
            fail_results=[
                FailResult(
                    error_message="Error Msg",
                    fix_value=4,
                )
            ],
            path=["f", 2, "g"],
        ),
        FieldReAsk(
            incorrect_value=-1,
            fail_results=[
                FailResult(
                    error_message="Error Msg",
                    fix_value=5,
                )
            ],
            path=["h", 2, 2],
        ),
    ]
    assert gather_reasks(input_dict) == expected_reasks


@pytest.mark.parametrize(
    "input_dict, expected_dict",
    [
        (
            {
                "a": 1,
                "b": FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value=1,
                        )
                    ],
                ),
            },
            {
                "b": FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value=1,
                        )
                    ],
                )
            },
        ),
        (
            {
                "a": 1,
                "b": {
                    "c": 2,
                    "d": FieldReAsk(
                        incorrect_value=-1,
                        fail_results=[
                            FailResult(
                                error_message="Error Msg",
                                fix_value=2,
                            )
                        ],
                    ),
                },
            },
            {
                "b": {
                    "d": FieldReAsk(
                        incorrect_value=-1,
                        fail_results=[
                            FailResult(
                                error_message="Error Msg",
                                fix_value=2,
                            )
                        ],
                    )
                }
            },
        ),
        (
            {
                "a": [
                    1,
                    2,
                    FieldReAsk(
                        incorrect_value=-1,
                        fail_results=[
                            FailResult(
                                error_message="Error Msg",
                                fix_value=3,
                            )
                        ],
                    ),
                ],
                "b": 4,
            },
            {
                "a": [
                    FieldReAsk(
                        incorrect_value=-1,
                        fail_results=[
                            FailResult(
                                error_message="Error Msg",
                                fix_value=3,
                            )
                        ],
                    ),
                ]
            },
        ),
        ({"a": 1}, None),
    ],
)
def test_prune_json_for_reasking(input_dict, expected_dict):
    """Test that the prune_obj_for_reasking function removes ReAsk objects."""
    assert reask_utils.prune_obj_for_reasking(input_dict) == expected_dict


@pytest.mark.parametrize(
    "example_rail, expected_rail, reasks, original_response, reask_json",
    [
        (
            """
<output>
    <string name="name" required="true"/>
</output>
""",
            """
<output>
    <string name="name"/>
</output>
""",
            [
                FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value="name",
                        )
                    ],
                    path=["name"],
                )
            ],
            {
                "name": -1,
            },
            {
                "name": {
                    "incorrect_value": -1,
                    "error_message": "Error Msg",
                    "fix_value": "name",
                }
            },
        ),
        (
            """
<output>
    <string name="name" required="true"/>
    <integer name="age" required="true"/>
</output>
""",
            """
<output>
    <string name="name"/>
    <integer name="age"/>
</output>
""",
            [
                FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value="name",
                        )
                    ],
                    path=["name"],
                ),
                FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value=5,
                        )
                    ],
                    path=["age"],
                ),
            ],
            {
                "name": -1,
                "age": -1,
            },
            {
                "name": {
                    "incorrect_value": -1,
                    "error_message": "Error Msg",
                    "fix_value": "name",
                },
                "age": {
                    "incorrect_value": -1,
                    "error_message": "Error Msg",
                    "fix_value": 5,
                },
            },
        ),
    ],
)
def test_get_reask_prompt(
    example_rail, expected_rail, reasks, original_response, reask_json
):
    """Test that get_reask_prompt function returns the correct prompt."""
    expected_result_template = """
I was given the following JSON response, which had problems due to incorrect values.

%s

Help me correct the incorrect values based on the given error messages.

Given below is XML that describes the information to extract from this document and the tags to extract it into.
%s

ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`.
"""  # noqa: E501
    expected_instructions = """
You are a helpful assistant only capable of communicating with valid JSON, and no other text.

ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`.

Here are examples of simple (XML, JSON) pairs that show the expected behavior:
- `<string name='foo' format='two-words lower-case' />` => `{'foo': 'example one'}`
- `<list name='bar'><string format='upper-case' /></list>` => `{"bar": ['STRING ONE', 'STRING TWO', etc.]}`
- `<object name='baz'><string name="foo" format="capitalize two-words" /><integer name="index" format="1-indexed" /></object>` => `{'baz': {'foo': 'Some String', 'index': 1}}`
"""  # noqa: E501
    output_schema = JsonSchema(Object.from_xml(ET.fromstring(example_rail)))
    guard_logs = GuardLogs()
    validated = output_schema.validate(guard_logs, original_response, {})
    reasks = output_schema.introspect(validated)
    (
        reask_schema,
        result_prompt,
        instructions,
    ) = output_schema.get_reask_setup(reasks, reask_json, False)

    assert result_prompt == Prompt(
        expected_result_template
        % (
            json.dumps(reask_json, indent=2),
            expected_rail,
        )
    )

    assert instructions == Instructions(expected_instructions)
