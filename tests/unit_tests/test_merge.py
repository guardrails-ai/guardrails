import pytest
from guardrails.validator_service import SequentialValidatorService


validator_service = SequentialValidatorService()


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
        #   (broken) test behavior with a word close to PERSON
        # ("""Perry is FUNNY and LIVES in NEW york""",
        #  ["""<PERSON> is FUNNY and lives in <LOCATION>""",
        #   """perry is funny and lives in new york"""],
        #   """<PERSON> is funny and lives in <LOCATION>"""),
    ],
)
def test_merge(original, new_values, expected):
    print("testing", original, new_values, expected)
    res = validator_service.multi_merge(original, new_values)
    print("res", res)
    assert res == expected
