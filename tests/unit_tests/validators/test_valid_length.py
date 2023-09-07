import pytest

from guardrails.validators import ValidLength


@pytest.mark.parametrize(
    "min,max,expected_min,expected_max",
    [
        (0, 12, 0, 12),
        ("0", "12", 0, 12),
        (None, 12, None, 12),
        (1, None, 1, None),
    ],
)
def test_init(min, max, expected_min, expected_max):
    validator = ValidLength(min=min, max=max)

    assert validator._min == expected_min
    assert validator._max == expected_max
