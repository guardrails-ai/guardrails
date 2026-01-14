import json

import pytest

from guardrails.classes.execution.guard_execution_options import GuardExecutionOptions
from guardrails.actions.reask import (
    FieldReAsk,
    gather_reasks,
    get_reask_setup,
    prune_obj_for_reasking,
    sub_reasks_with_fixed_values,
)
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.validation.validation_result import FailResult
from guardrails.schema.rail_schema import rail_string_to_schema


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
                        fail_results=[FailResult(error_message="Error Msg", fix_value=2)],
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
    actual_reasks, _ = gather_reasks(input_dict)
    assert actual_reasks == expected_reasks


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
        (
            [
                FieldReAsk(
                    path=["$.failed_prop.child"],
                    fail_results=[
                        FailResult(error_message="child should not be None", outcome="fail")
                    ],
                    incorrect_value=None,
                ),
                "not a reask",
            ],
            [
                FieldReAsk(
                    path=["$.failed_prop.child"],
                    fail_results=[
                        FailResult(error_message="child should not be None", outcome="fail")
                    ],
                    incorrect_value=None,
                )
            ],
        ),
    ],
)
def test_prune_json_for_reasking(input_dict, expected_dict):
    """Test that the prune_obj_for_reasking function removes ReAsk objects."""
    assert prune_obj_for_reasking(input_dict) == expected_dict


@pytest.mark.parametrize(
    "example_rail, expected_rail, reasks, original_response, validation_response, json_example, expected_reask_json",  # noqa
    [
        (
            # Example RAIL
            """
<rail version="0.1" >
<output>
    <string name="name" required="true"/>
</output>
</rail>
""",
            # Expected RAIL Output Element
            """
<output>
  <string name="name" required="true"></string>
</output>
""",
            # ReAsks
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
            # Original Response
            {
                "name": -1,
            },
            # Validation Response
            {
                "name": FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value="name",
                        )
                    ],
                    path=["name"],
                )
            },
            # Json Example
            {"name": "Name"},
            # Expected Reask Json
            {"name": {"incorrect_value": -1, "error_messages": ["Error Msg"]}},
        ),
        (
            # Example RAIL
            """
<rail version="0.1" >
<output>
  <string name="name" required="true"/>
  <integer name="age" required="true"/>
</output>
</rail>
""",
            # Expected RAIL Output element
            """
<output>
  <string name="name" required="true"></string>
  <integer name="age" required="true"></integer>
</output>
""",
            # ReAsks
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
            # Original Response
            {
                "name": -1,
                "age": -1,
            },
            # Validation Response
            {
                "name": FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value="name",
                        )
                    ],
                    path=["name"],
                ),
                "age": FieldReAsk(
                    incorrect_value=-1,
                    fail_results=[
                        FailResult(
                            error_message="Error Msg",
                            fix_value=5,
                        )
                    ],
                    path=["age"],
                ),
            },
            # Json Example
            {"name": "Name", "age": 5},
            # Expected Reask Json
            {
                "name": {
                    "incorrect_value": -1,
                    "error_messages": ["Error Msg"],
                },
                "age": {
                    "incorrect_value": -1,
                    "error_messages": ["Error Msg"],
                },
            },
        ),
    ],
)
def test_get_reask_prompt(
    mocker,
    example_rail,
    expected_rail,
    reasks,
    original_response,
    validation_response,
    json_example,
    expected_reask_json,
):
    """Test that get_reask_prompt function returns the correct prompt."""
    expected_prompt_template = """
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

    mocker.patch("guardrails.actions.reask.generate_example", return_value=json_example)

    output_type = OutputTypes.DICT
    processed_schema = rail_string_to_schema(example_rail)
    output_schema = processed_schema.json_schema
    exec_options = GuardExecutionOptions(
        # Use an XML constant to make existing test cases pass
        messages=[
            {
                "role": "user",
                "content": "${gr.complete_xml_suffix_v3}",
            }
        ]
    )

    (reask_schema, reask_messages) = get_reask_setup(
        output_type,
        output_schema,
        validation_map={},
        reasks=reasks,
        parsing_response=original_response,
        validation_response=validation_response,
        exec_options=exec_options,
    )

    # NOTE: This changes once we reimplement Field Level Reasks
    assert reask_schema == output_schema

    expected_prompt = expected_prompt_template % (
        json.dumps(expected_reask_json, indent=2),
        expected_rail,
        # Examples are only included for SkeletonReAsk's
        # json.dumps(json_example, indent=2),
    )

    assert reask_messages.source[1]["content"].source == expected_prompt

    assert reask_messages.source[0]["content"].source == expected_instructions


### FIXME: Implement once Field Level ReAsk is implemented w/ JSON schema ###
# empty_root = Element("root")
# non_empty_root = Element("root")
# property = SubElement(non_empty_root, "list", name="dummy")
# property.attrib["validators"] = "length: 2"
# child = SubElement(property, "string")
# child.attrib["validators"] = "two-words"
# non_empty_output = Element("root")
# output_property = SubElement(non_empty_output, "list", name="dummy")
# output_property.attrib["validators"] = "length: 2"
# output_child = SubElement(output_property, "string")
# output_child.attrib["validators"] = "two-words"


# @pytest.mark.parametrize(
#     "root,reasks,expected_output",
#     [
#         (empty_root, None, empty_root),
#         (
#             non_empty_root,
#             [
#                 FieldReAsk(
#                     incorrect_value="",
#                     fail_results=[FailResult(error_message="child should not be None")],  # noqa
#                     path=["dummy", 0],
#                 )
#             ],
#             non_empty_output,
#         ),
#     ],
# )
# def test_get_reask_subschema(root, reasks, expected_output):
#     actual_output = get_reask_subschema(Object.from_xml(root), reasks)

#     assert actual_output == Object.from_xml(expected_output)
