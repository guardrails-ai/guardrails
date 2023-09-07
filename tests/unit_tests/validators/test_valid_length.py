import pytest
from guardrails.validators import ValidLength

@pytest.mark.parametrize(
    "min,max,expected_min,expected_max",
    [
        (0,12,0,12),
        ("0","12",0,12),
        (None,12,None,12),
        (1,None,1,None),
    ]
)
def test_init(min, max, expected_min, expected_max):
    validator = ValidLength(min=min, max=max)

    assert validator._min == expected_min
    assert validator._max == expected_max

@pytest.mark.parametrize(
    "min,max,expected_xml",
    [
        (0,12,"length: 0 12"),
        ("0","12","length: 0 12"),
        (None,12,"length: None 12"),
        (1,None,"length: 1 None"),
    ]
)
def test_to_xml_attrib(min, max, expected_xml):
    validator = ValidLength(min=min, max=max)
    xml_validator = validator.to_xml_attrib()

    assert xml_validator == expected_xml