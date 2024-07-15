import pytest
import openai
import pydantic
from guardrails import Guard


def test_deprecated_properties():
    guard = Guard.from_string([])

    pytest.deprecated_call(
        match=(
            "'Guard.prompt_schema' is deprecated and will be removed in "
            "versions 0.5.x and beyond."
        ),
        func=lambda: guard.prompt_schema,
    )
    pytest.deprecated_call(
        match=(
            "'Guard.instructions_schema' is deprecated and will be removed in "
            "versions 0.5.x and beyond."
        ),
        func=lambda: guard.instructions_schema,
    )
    pytest.deprecated_call(
        match=(
            "'Guard.msg_history_schema' is deprecated and will be removed in "
            "versions 0.5.x and beyond."
        ),
        func=lambda: guard.msg_history_schema,
    )
    pytest.deprecated_call(
        match=(
            "'Guard.output_schema' is deprecated and will be removed in "
            "versions 0.5.x and beyond."
        ),
        func=lambda: guard.output_schema,
    )
    pytest.deprecated_call(
        match=(
            "'Guard.instructions' is deprecated and will be removed in "
            "versions 0.5.x and beyond. Use 'Guard.history.last.instructions' instead."
        ),
        func=lambda: guard.instructions,
    )
    pytest.deprecated_call(
        match=(
            "'Guard.prompt' is deprecated and will be removed in "
            "versions 0.5.x and beyond. Use 'Guard.history.last.prompt' instead."
        ),
        func=lambda: guard.prompt,
    )
    pytest.deprecated_call(
        match=(
            "'Guard.raw_prompt' is deprecated and will be removed in "
            "versions 0.5.x and beyond. Use 'Guard.history.last.prompt' instead."
        ),
        func=lambda: guard.raw_prompt,
    )
    pytest.deprecated_call(
        match=(
            "'Guard.base_prompt' is deprecated and will be removed in "
            "versions 0.5.x and beyond. Use 'Guard.history.last.prompt' instead."
        ),
        func=lambda: guard.base_prompt,
    )
    pytest.deprecated_call(
        match=(
            "'Guard.reask_prompt' is deprecated and will be removed in "
            "versions 0.5.x and beyond. Use 'Guard.history.last.reask_prompts' instead."
        ),
        func=lambda: guard.reask_prompt,
    )
    pytest.deprecated_call(
        match=("'Guard.reask_instructions' is deprecated and will be removed in "),
        func=lambda: guard.reask_instructions,
    )


def test_deprecated_setter_methods():
    guard = Guard.from_string([])

    pytest.deprecated_call(
        match=("'Guard.reask_prompt' is deprecated and will be removed in "),
        func=lambda: setattr(guard, "reask_prompt", "New reask prompt"),
    )
    pytest.deprecated_call(
        match=("'Guard.reask_instructions' is deprecated and will be removed in "),
        func=lambda: setattr(guard, "reask_instructions", "New reask instructions"),
    )


def test_deprecated_validation_methods():
    guard = Guard.from_string([])

    with pytest.warns(
        FutureWarning, match="The `with_prompt_validation` method is deprecated"
    ):
        guard.with_prompt_validation([])

    with pytest.warns(
        FutureWarning, match="The `with_instructions_validation` method is deprecated"
    ):
        guard.with_instructions_validation([])

    with pytest.warns(
        FutureWarning, match="The `with_msg_history_validation` method is deprecated"
    ):
        guard.with_msg_history_validation([])


def test_deprecated_validator_import():
    with pytest.warns(FutureWarning, match="Importing validators from"):
        from guardrails.validators import ValidURL  # noqa: F401


def is_openai_v0():
    return openai.__version__.startswith("0.")


def is_pydantic_v1():
    return pydantic.__version__.startswith("1.")


@pytest.mark.skipif(not is_openai_v0(), reason="OpenAI version is not v0.x")
def test_deprecated_openai_v0():
    with pytest.warns(FutureWarning, match="Support for OpenAI v0.x is deprecated"):
        from guardrails.utils.openai_utils.v0 import raise_v0_deprecation_warning

        raise_v0_deprecation_warning()


@pytest.mark.skipif(not is_pydantic_v1(), reason="Pydantic version is not v1.x")
def test_deprecated_pydantic_v1():
    with pytest.warns(FutureWarning, match="Support for Pydantic v1 is deprecated"):
        try:
            Guard.from_pydantic(object)
        except Exception:
            pass


def test_deprecated_invoke():
    guard = Guard.from_string([])

    pytest.deprecated_call(
        match=(
            "'Guard.invoke' is deprecated and will be removed in versions 0.5.x "
            "and beyond. Use Guard.to_runnable() instead."
        ),
        func=lambda: guard.invoke("test input"),
    )
