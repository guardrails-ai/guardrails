import pytest
from lxml.etree import Element, SubElement

from guardrails.datatypes import Object

# from guardrails.datatypes import Object
from guardrails.utils.reask_utils import (
    FieldReAsk,
    get_pruned_tree,
    prune_obj_for_reasking,
)
from guardrails.validators import FailResult

# FIXME: These tests are not exhaustive.
# They only add missing coverage from the 0.2 release
# We really should strive for close to 100% unit test coverage
#   and use Integration tests for mimicking user flows


empty_root = Element("root")
non_empty_root = Element("root")
property = SubElement(non_empty_root, "list", name="dummy")
property.attrib["validators"] = "length: 2"
child = SubElement(property, "string")
child.attrib["validators"] = "two-words"
non_empty_output = Element("root")
output_property = SubElement(non_empty_output, "list", name="dummy")
output_property.attrib["validators"] = "length: 2"
output_child = SubElement(output_property, "string")
output_child.attrib["validators"] = "two-words"


@pytest.mark.parametrize(
    "root,reasks,expected_output",
    [
        (empty_root, None, empty_root),
        (
            non_empty_root,
            [
                FieldReAsk(
                    incorrect_value="",
                    fail_results=[FailResult(error_message="child should not be None")],
                    path=["dummy", 0],
                )
            ],
            non_empty_output,
        ),
    ],
)
def test_get_pruned_tree(root, reasks, expected_output):
    actual_output = get_pruned_tree(Object.from_xml(root), reasks)

    assert actual_output == Object.from_xml(expected_output)


def test_prune_obj_for_reasking():
    reask = FieldReAsk(
        path=["$.failed_prop.child"],
        fail_results=[
            FailResult(error_message="child should not be None", outcome="fail")
        ],
        incorrect_value=None,
    )
    reasks = [reask, "not a reask"]

    pruned_reasks = prune_obj_for_reasking(reasks)

    assert len(pruned_reasks) == 1
    assert pruned_reasks[0] == reask
