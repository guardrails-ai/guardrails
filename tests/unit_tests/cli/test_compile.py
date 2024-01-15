import pytest

from guardrails.cli.compile import compile_rail, compile, logger


def test_compile_rail():
    with pytest.raises(NotImplementedError) as nie:
        compile_rail("my_spec.rail", ".rail_output")
        assert nie is not None
        assert str(nie) == "Currently compiling rail is not supported."


def test_compile(mocker):
    error_log_mock = mocker.patch.object(logger, "error")

    compile("my_spec.rail")

    error_log_mock.assert_called_once_with("Not supported yet. Use `validate` instead.")
