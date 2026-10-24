import pytest
from guardrails.merge import merge
from guardrails.validator_service import SequentialValidatorService


validator_service = SequentialValidatorService()


def test_merge_does_not_raise_on_short_diff():
    """Regression: `merge` raised IndexError when an intermediate
    PRESERVED chunk had been sliced down to "" earlier in the loop.
    diff_main("", x) returns a single insertion chunk instead of the
    2-element diff the (unguarded) branch assumed, the same shape a
    sibling branch a few lines below already guards against.

    base="ab", source="b", target="baa" is a minimal reproduction found
    by fuzzing `merge` with random single/double-character edits of a
    common base -- streamed validator output under concurrent fixes.
    """
    # Against base "ab", source deletes the leading "a" and target inserts "ba"
    # at that same position (and drops the trailing "b"). The two edits touch
    # the same spot, so the merge keeps neither side's change and returns "".
    # The point of the test is that this overlap no longer raises.
    assert merge(source="b", target="baa", base="ab") == ""


def test_multi_merge_does_not_raise_on_short_diff():
    """Same defect, reached through the public multi_merge path that
    validator services actually call to reconcile two validators' fixes
    to the same streamed value."""
    # multi_merge pops "b" and then "baa", so this is merge("b", "baa", "ab").
    assert validator_service.multi_merge("ab", ["baa", "b"]) == ""


@pytest.mark.parametrize(
    "original, new_values, expected",
    [
        # test behavior on blank fixes
        ("hello world", ["", "hello nick"], "nick"),
        ("hello world", ["", "hello world"], ""),
        # test behavior on non overlapping replacements
        (
            """John is a shitty person who works at Anthropic on Claude, 
             and lives in San Francisco""",
            [
                """<PERSON> is a shitty person who works at Anthropic on <PERSON>,
              and lives in <LOCATION>""",
                """John is a ****** person who works at Anthropic on Claude,
              and lives in San Francisco""",
            ],
            """<PERSON> is a ****** person who works at Anthropic on <PERSON>,
              and lives in <LOCATION>""",
        ),
        #  test behavior with lowercase
        (
            """JOE is FUNNY and LIVES in NEW york""",
            [
                """<PERSON> is FUNNY and lives in <LOCATION>""",
                """joe is funny and lives in new york""",
            ],
            """<PERSON> is funny and lives in <LOCATION>""",
        ),
        (
            """JOHN lives IN SAN francisco""",
            [
                """<PERSON> lives in <LOCATION>""",
                """john lives in san francisco""",
            ],
            """<PERSON> lives in <LOCATION>""",
        ),
        #   (broken) test behavior with a word close to PERSON - seems to work!?
        (
            """Parson is FUNNY and LIVES in NEW york""",
            [
                """<PERSON> is FUNNY and lives in <LOCATION>""",
                """parson is funny and lives in new york""",
            ],
            """<PERSON> is funny and lives in <LOCATION>""",
        ),
    ],
)
def test_merge(original, new_values, expected):
    print("testing", original, new_values, expected)
    res = validator_service.multi_merge(original, new_values)
    print("res", res)
    assert res == expected
