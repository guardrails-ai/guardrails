from unittest.mock import MagicMock
from guardrails.classes.history.iteration import Iteration
from guardrails.classes.validation.validation_result import FailResult, PassResult
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.types.on_fail import OnFailAction
from guardrails.validator_base import Validator, register_validator
from guardrails.validator_service.sequential_validator_service import (
    SequentialValidatorService,
)


@register_validator(name="guardrails/mock-validator", data_type="string")
class MockValidator(Validator):
    pass


class TestRunValidators:
    def test_sends_validation_session_id_to_run_validator_sync(self, mocker):
        val_svc = SequentialValidatorService(disable_tracer=True)

        validation_result = FailResult(  # type: ignore
            error_message="error",  # type: ignore
            fix_value="bar",  # type: ignore
        )

        validator_logs = ValidatorLogs(
            validator_name="MockValidator",  # type: ignore
            registered_name="guardrails/mock-validator",  # type: ignore
            value_before_validation="foo",  # type: ignore
            property_path="$",  # type: ignore
            instance_id=1,  # type: ignore
            validation_result=validation_result,  # type: ignore
        )

        rechecked_value = PassResult()

        mock_run_validator = mocker.patch.object(val_svc, "run_validator", autospec=True)
        mock_run_validator.return_value = validator_logs

        mock_run_validator_sync = mocker.patch.object(val_svc, "run_validator_sync", autospec=True)
        mock_run_validator_sync.return_value = rechecked_value

        mock_perform_correction = mocker.patch.object(val_svc, "perform_correction", autospec=True)
        mock_perform_correction.return_value = "bar"

        iteration = MagicMock(spec=Iteration)
        iteration.id = "12345"
        mock_validator = MockValidator(on_fail=OnFailAction.FIX_REASK)

        result, metadata = val_svc.run_validators(
            iteration=iteration,
            validator_map={"$": [mock_validator]},
            value="foo",
            metadata={},
            absolute_property_path="$",
            reference_property_path="$",
        )

        assert result == "bar"
        mock_run_validator.assert_called_once_with(iteration, mock_validator, "foo", {}, "$", False)
        mock_run_validator_sync.assert_called_once_with(
            mock_validator,
            "bar",
            {},
            validator_logs,
            False,
            validation_session_id="12345",
        )
        mock_perform_correction.assert_called_once_with(
            validation_result,
            "foo",
            mock_validator,
            rechecked_value=rechecked_value,
        )
