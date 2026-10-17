"""Regression tests for https://github.com/guardrails-ai/guardrails/issues/1656.

The RAIL path and the Python API disagreed on the default ``on_fail``:
Python defaulted to ``EXCEPTION`` while RAIL silently defaulted to ``NOOP``.
The RAIL default now matches the Python API default (``EXCEPTION``).
"""

from types import SimpleNamespace
from unittest.mock import patch

from lxml import etree as ET

from guardrails.schema import rail_schema
from guardrails.validator_base import OnFailAction


def _element(**attrib):
    el = ET.Element("string")
    for key, value in attrib.items():
        el.set(key, value)
    return el


def _stub_validator():
    return SimpleNamespace(rail_alias="stub_validator")


def test_rail_default_on_fail_matches_python_api():
    el = _element(validators="stub_validator")
    with patch.object(rail_schema, "get_validator", return_value=_stub_validator()):
        validators = rail_schema.get_validators(el)
    assert len(validators) == 1
    assert validators[0].on_fail_descriptor == OnFailAction.EXCEPTION


def test_explicit_on_fail_attribute_still_wins():
    el = _element(validators="stub_validator", **{"on-fail-stub_validator": "noop"})
    with patch.object(rail_schema, "get_validator", return_value=_stub_validator()):
        validators = rail_schema.get_validators(el)
    assert len(validators) == 1
    assert validators[0].on_fail_descriptor == OnFailAction.NOOP
