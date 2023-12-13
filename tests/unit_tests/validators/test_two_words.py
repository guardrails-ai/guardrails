import pytest

from guardrails.validators import TwoWords, PassResult, FailResult, ValidationResult


def test_two_words_happy_path():
    validator = TwoWords()

    result: PassResult = validator.validate("Hello there", {})

    assert result.outcome == "pass"

@pytest.mark.parametrize(
    "input, expected_output",
    [
        (
            "Hello there general",
            "Hello there"
        ),
        (
            "hello-there-general",
            "hello there"
        ),
        (
            "hello_there_general",
            "hello there"
        ),
        (
            "helloThereGeneral",
            "hello There"
        ),
        (
            "HelloThereGeneral",
            "Hello There"
        ),
        (
            "hello.there.general",
            "hello there"
        ),
        (
            "hello",
            "hello hello"
        )
    ]
)
def test_two_words_failures(input, expected_output):
    validator = TwoWords()

    result: FailResult = validator.validate(input, {})

    assert result.outcome == "fail"
    assert result.fix_value == expected_output