from tests.unit_tests.mocks.mock_file import MockFile


def test_validate(mocker):
    mock_validate_llm_output = mocker.patch("guardrails.cli.validate.validate_llm_output")
    mock_validate_llm_output.return_value = "validated output"

    mock_file = MockFile()
    mock_open = mocker.patch("guardrails.cli.validate.open")
    mock_open.return_value = mock_file
    write_spy = mocker.spy(mock_file, "write")

    mock_json_dump = mocker.patch("json.dump")

    import builtins

    print_spy = mocker.spy(builtins, "print")

    from guardrails.cli.validate import validate

    response = validate("my_spec.rail", "output", out="somewhere")

    mock_validate_llm_output.assert_called_once_with("my_spec.rail", "output")

    print_spy.assert_called_once_with("validated output")
    mock_open.assert_called_once_with("somewhere", "w")
    mock_json_dump.assert_called_once_with("validated output", mock_file)
    write_spy.assert_called_once_with("\n")

    assert response == "validated output"


def test_validate_llm_output(mocker):
    class MockGuard:
        def parse(self, *args):
            pass

    from guardrails import Guard
    from guardrails.classes import ValidationOutcome

    mock_guard = MockGuard()
    for_rail_mock = mocker.patch.object(Guard, "for_rail")
    for_rail_mock.return_value = mock_guard

    parse_mock = mocker.patch.object(mock_guard, "parse")
    parse_mock.return_value = ValidationOutcome(
        call_id="mock-call",
        raw_llm_output="output",
        validated_output="validated output",
        validation_passed=True,
    )

    from guardrails.cli.validate import validate_llm_output

    rail = "my_spec.rail"
    response = validate_llm_output(rail, "output")

    for_rail_mock.assert_called_once_with(rail)
    parse_mock.assert_called_once_with("output")

    assert response == "validated output"
