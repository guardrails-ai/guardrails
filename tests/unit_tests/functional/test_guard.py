from guardrails.functional import Guard, args, kwargs, on_fail
from guardrails.validators import EndsWith, LowerCase, OneLine, TwoWords, ValidLength


def test_add():
    guard: Guard = (
        Guard()
        .add(EndsWith("a"))
        .add(OneLine())
        .add(LowerCase)
        .add(TwoWords, on_fail="reask")
        .add(ValidLength, 0, 12, on_fail="refrain")
    )

    # print(guard.__stringify__())
    assert len(guard.validators) == 5

    assert isinstance(guard.validators[0], EndsWith)
    assert guard.validators[0]._kwargs["end"] == "a"
    assert guard.validators[0].on_fail_descriptor == "fix"  # bc this is the default

    assert isinstance(guard.validators[1], OneLine)
    assert guard.validators[1].on_fail_descriptor == "noop"  # bc this is the default

    assert isinstance(guard.validators[2], LowerCase)
    assert guard.validators[2].on_fail_descriptor == "noop"  # bc this is the default

    assert isinstance(guard.validators[3], TwoWords)
    assert guard.validators[3].on_fail_descriptor == "reask"  # bc we set it

    assert isinstance(guard.validators[4], ValidLength)
    assert guard.validators[4]._min == 0
    assert guard.validators[4]._kwargs["min"] == 0
    assert guard.validators[4]._max == 12
    assert guard.validators[4]._kwargs["max"] == 12
    assert guard.validators[4].on_fail_descriptor == "refrain"  # bc we set it


def test_integrate_instances():
    guard: Guard = Guard().integrate(
        EndsWith("a"), OneLine(), LowerCase(), TwoWords(on_fail="reask")
    )

    # print(guard.__stringify__())
    assert len(guard.validators) == 4

    assert isinstance(guard.validators[0], EndsWith)
    assert guard.validators[0]._end == "a"
    assert guard.validators[0]._kwargs["end"] == "a"
    assert guard.validators[0].on_fail_descriptor == "fix"  # bc this is the default

    assert isinstance(guard.validators[1], OneLine)
    assert guard.validators[1].on_fail_descriptor == "noop"  # bc this is the default

    assert isinstance(guard.validators[2], LowerCase)
    assert guard.validators[2].on_fail_descriptor == "noop"  # bc this is the default

    assert isinstance(guard.validators[3], TwoWords)
    assert guard.validators[3].on_fail_descriptor == "reask"  # bc we set it


def test_integrate_tuple():
    guard: Guard = Guard().integrate(
        OneLine,
        (EndsWith, ["a"], {"on_fail": "exception"}),
        (LowerCase, kwargs(on_fail="fix-reask", some_other_kwarg="kwarg")),
        (TwoWords, on_fail("reask")),
        (ValidLength, args(0, 12), kwargs(on_fail="refrain")),
    )

    # print(guard.__stringify__())
    assert len(guard.validators) == 5

    assert isinstance(guard.validators[0], OneLine)
    assert guard.validators[0].on_fail_descriptor == "noop"  # bc this is the default

    assert isinstance(guard.validators[1], EndsWith)
    assert guard.validators[1]._end == "a"
    assert guard.validators[1]._kwargs["end"] == "a"
    assert guard.validators[1].on_fail_descriptor == "exception"  # bc we set it

    assert isinstance(guard.validators[2], LowerCase)
    assert guard.validators[2]._kwargs["some_other_kwarg"] == "kwarg"
    assert (
        guard.validators[2].on_fail_descriptor == "fix-reask"
    )  # bc this is the default

    assert isinstance(guard.validators[3], TwoWords)
    assert guard.validators[3].on_fail_descriptor == "reask"  # bc we set it

    assert isinstance(guard.validators[4], ValidLength)
    assert guard.validators[4]._min == 0
    assert guard.validators[4]._kwargs["min"] == 0
    assert guard.validators[4]._max == 12
    assert guard.validators[4]._kwargs["max"] == 12
    assert guard.validators[4].on_fail_descriptor == "refrain"  # bc we set it

def test_validate():
    guard: Guard = (
        Guard()
        .add(EndsWith("a"))
        .add(OneLine)
        .add(LowerCase(on_fail="fix"))
        .add(TwoWords)
        .add(ValidLength, 0, 12, on_fail="refrain")
    )

    llm_output = "Oh Canada" # bc it meets our criteria

    response = guard.validate(llm_output)

    assert response.validation_passed == True
    assert response.validated_output == llm_output.lower()
    
    llm_output_2 = "Star Spangled Banner" # to stick with the theme

    response_2 = guard.validate(llm_output_2)

    assert response_2.validation_passed == False
    assert response_2.validated_output == None


def test_call():
    response = (
        Guard()
        .add(EndsWith("a"))
        .add(OneLine)
        .add(LowerCase(on_fail="fix"))
        .add(TwoWords)
        .add(ValidLength, 0, 12, on_fail="refrain")
    )("Oh Canada")

    assert response.validation_passed == True
    assert response.validated_output == "oh canada"