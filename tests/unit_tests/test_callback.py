import pytest

from guardrails.run import Callback


# A dummy concrete class for testing purposes
class ConcreteCallback(Callback):
    def before_prepare(
        self,
        index,
        instructions,
        prompt,
        prompt_params,
        api,
        input_schema,
        output_schema,
    ):
        return (
            index,
            instructions,
            prompt,
            prompt_params,
            api,
            input_schema,
            output_schema,
        )

    def after_prepare(
        self,
        index,
        instructions,
        prompt,
        prompt_params,
        api,
        input_schema,
        output_schema,
    ):
        return (
            index,
            instructions,
            prompt,
            prompt_params,
            api,
            input_schema,
            output_schema,
        )

    def before_call(self, index, instructions, prompt, api, output):
        return index, instructions, prompt, api, output

    def after_call(self, index, instructions, prompt, api, output):
        return index, instructions, prompt, api, output

    def before_parse(self, index, output, output_schema):
        return index, output, output_schema

    def after_parse(self, index, output, output_schema):
        return index, output, output_schema

    def before_validate(self, index, parsed_output, output_schema):
        return index, parsed_output, output_schema

    def after_validate(self, index, parsed_output, output_schema):
        return index, parsed_output, output_schema

    def before_introspect(self, index, validated_output, output_schema):
        return index, validated_output, output_schema

    def after_introspect(self, index, validated_output, output_schema):
        return index, validated_output, output_schema


def test_concrete_class_can_be_instantiated():
    try:
        c = ConcreteCallback()  # Should not raise any error

        # Check if all methods in Callback are implemented
        methods = [
            "before_prepare",
            "after_prepare",
            "before_call",
            "after_call",
            "before_parse",
            "after_parse",
            "before_validate",
            "after_validate",
            "before_introspect",
            "after_introspect",
        ]

        for method in methods:
            assert callable(
                getattr(c, method)
            ), f"Method {method} not implemented in ConcreteCallback"

    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


def test_callback_methods():
    try:
        c = ConcreteCallback()  # Should not raise any error

        # Call one of the methods and verify it returns the expected output
        result = c.before_prepare(
            0,
            "instructions",
            "prompt",
            {"param": "value"},
            "api",
            "input_schema",
            "output_schema",
        )

        expected_result = (
            0,
            "instructions",
            "prompt",
            {"param": "value"},
            "api",
            "input_schema",
            "output_schema",
        )

        assert (
            result == expected_result
        ), "Method before_prepare did not return the expected result"

    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
