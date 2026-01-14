from asyncio import Future
import pytest
from unittest.mock import MagicMock

from guardrails.guard import Guard
from guardrails.async_guard import AsyncGuard
from guardrails.classes.history.call import Call
from guardrails.classes.history.iteration import Iteration
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.run.async_runner import AsyncRunner
from guardrails.run.async_stream_runner import AsyncStreamRunner
from guardrails.run.runner import Runner
from guardrails.run.stream_runner import StreamRunner
from guardrails.version import GUARDRAILS_VERSION
from tests.unit_tests.mocks.mock_span import MockSpan

try:
    import mlflow
except ImportError:
    mlflow = None


@pytest.mark.skipif(
    mlflow is None,
    reason="mlflow not installed.",
)
class TestMlFlowInstrumentor:
    def test__init__(self):
        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        from guardrails import settings

        assert m.experiment_name == "mock experiment"
        assert settings.disable_tracing is True

    def test_instrument(self, mocker):
        mock_enable = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.tracing.enable"
        )
        mock_set_experiment = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.set_experiment"
        )

        from tests.unit_tests.mocks import mock_hub

        mocker.patch("guardrails.hub", return_value=mock_hub)

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        # Prevent real methods from being wrapped and persistint into other tests
        mocker.patch("guardrails.integrations.databricks.ml_flow_instrumentor.Guard._execute")
        guard_execute = Guard._execute
        mock_instrument_guard = mocker.patch.object(m, "_instrument_guard")

        mocker.patch("guardrails.integrations.databricks.ml_flow_instrumentor.AsyncGuard._execute")
        async_guard_execute = AsyncGuard._execute
        mock_instrument_async_guard = mocker.patch.object(m, "_instrument_async_guard")

        mocker.patch("guardrails.integrations.databricks.ml_flow_instrumentor.Runner.step")
        runner_step = Runner.step
        mock_instrument_runner_step = mocker.patch.object(m, "_instrument_runner_step")

        mocker.patch("guardrails.integrations.databricks.ml_flow_instrumentor.StreamRunner.step")
        stream_runner_step = StreamRunner.step
        mock_instrument_stream_runner_step = mocker.patch.object(
            m, "_instrument_stream_runner_step"
        )

        mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.AsyncRunner.async_step"
        )
        async_runner_step = AsyncRunner.async_step
        mock_instrument_async_runner_step = mocker.patch.object(m, "_instrument_async_runner_step")

        mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.AsyncStreamRunner.async_step"
        )
        async_stream_runner_step = AsyncStreamRunner.async_step
        mock_instrument_async_stream_runner_step = mocker.patch.object(
            m, "_instrument_async_stream_runner_step"
        )

        mocker.patch("guardrails.integrations.databricks.ml_flow_instrumentor.Runner.call")
        runner_call = Runner.call
        mock_instrument_runner_call = mocker.patch.object(m, "_instrument_runner_call")

        mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.AsyncRunner.async_call"
        )
        async_runner_call = AsyncRunner.async_call
        mock_instrument_async_runner_call = mocker.patch.object(m, "_instrument_async_runner_call")

        m.instrument()

        mock_enable.assert_called_once()
        mock_set_experiment.assert_called_once_with("mock experiment")

        mock_instrument_guard.assert_called_once_with(guard_execute)
        mock_instrument_async_guard.assert_called_once_with(async_guard_execute)
        mock_instrument_runner_step.assert_called_once_with(runner_step)
        mock_instrument_stream_runner_step.assert_called_once_with(stream_runner_step)
        mock_instrument_async_runner_step.assert_called_once_with(async_runner_step)
        mock_instrument_async_stream_runner_step.assert_called_once_with(async_stream_runner_step)
        mock_instrument_runner_call.assert_called_once_with(runner_call)
        mock_instrument_async_runner_call.assert_called_once_with(async_runner_call)

    def test__instrument_guard(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_guard_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_guard_attributes"
        )
        mock_trace_stream_guard = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.trace_stream_guard"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        mock_result = ValidationOutcome(call_id="mock call id", validation_passed=True)
        mock_execute = MagicMock()
        mock_execute.return_value = mock_result
        mock_guard = MagicMock(spec=Guard)
        mock_guard._execute = mock_execute
        mock_guard.name = "mock guard"
        mock_guard.history = []

        wrapped_execute = m._instrument_guard(mock_guard._execute)

        wrapped_execute(mock_guard)

        mock_start_span.assert_called_once_with(
            name="guardrails/guard",
            span_type="guard",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard",
            },
        )

        # Internally called, not the wrapped call above
        mock_guard._execute.assert_called_once()
        mock_span.set_attribute.assert_called_once_with("guard.name", "mock guard")
        mock_add_guard_attributes.assert_called_once_with(mock_span, [], mock_result)

        mock_trace_stream_guard.assert_not_called()

    def test__instrument_guard_stream(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_guard_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_guard_attributes"
        )
        mock_trace_stream_guard = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.trace_stream_guard"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        mock_result = iter([ValidationOutcome(call_id="mock call id", validation_passed=True)])
        mock_execute = MagicMock()
        mock_execute.return_value = mock_result
        mock_guard = MagicMock(spec=Guard)
        mock_guard._execute = mock_execute
        mock_guard.name = "mock guard"
        mock_guard.history = []

        wrapped_execute = m._instrument_guard(mock_guard._execute)

        wrapped_execute(mock_guard, stream=True)

        mock_start_span.assert_called_once_with(
            name="guardrails/guard",
            span_type="guard",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard",
            },
        )

        # Internally called, not the wrapped call above
        mock_guard._execute.assert_called_once()
        mock_span.set_attribute.assert_called_once_with("guard.name", "mock guard")
        mock_trace_stream_guard.assert_called_once_with(mock_span, mock_result, [])

        mock_add_guard_attributes.assert_not_called()

    @pytest.mark.asyncio
    async def test__instrument_async_guard(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_guard_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_guard_attributes"
        )
        mock_trace_async_stream_guard = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.trace_async_stream_guard"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        validation_outcome = ValidationOutcome(call_id="mock call id", validation_passed=True)
        mock_result = Future()
        mock_result.set_result(validation_outcome)
        mock_execute = MagicMock()
        mock_execute.return_value = mock_result
        mock_guard = MagicMock(spec=AsyncGuard)
        mock_guard._execute = mock_execute
        mock_guard.name = "mock guard"
        mock_guard.history = []

        wrapped_execute = m._instrument_async_guard(mock_guard._execute)

        await wrapped_execute(mock_guard)

        mock_start_span.assert_called_once_with(
            name="guardrails/guard",
            span_type="guard",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard",
                "async": True,
            },
        )

        # Internally called, not the wrapped call above
        mock_guard._execute.assert_called_once()
        mock_span.set_attribute.assert_called_once_with("guard.name", "mock guard")
        mock_add_guard_attributes.assert_called_once_with(mock_span, [], validation_outcome)

        mock_trace_async_stream_guard.assert_not_called()

    @pytest.mark.asyncio
    async def test__instrument_async_guard_stream(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_guard_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_guard_attributes"
        )
        mock_trace_async_stream_guard = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.trace_async_stream_guard"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        async def async_iterable():
            yield ValidationOutcome(call_id="mock call id", validation_passed=True)

        async_gen = async_iterable()

        async def mock_execute(*args, **kwargs):
            return async_gen

        mock_guard = MagicMock(spec=AsyncGuard)
        mock_guard._execute = mock_execute
        mock_guard.name = "mock guard"
        mock_guard.history = []

        wrapped_execute = m._instrument_async_guard(mock_guard._execute)

        await wrapped_execute(mock_guard)

        mock_start_span.assert_called_once_with(
            name="guardrails/guard",
            span_type="guard",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard",
                "async": True,
            },
        )

        mock_span.set_attribute.assert_called_once_with("guard.name", "mock guard")
        mock_trace_async_stream_guard.assert_called_once_with(mock_span, async_gen, [])

        mock_add_guard_attributes.assert_not_called()

    def test__instrument_runner_step(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_step_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_step_attributes"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        iteration = Iteration(call_id="mock call id", index=0)
        mock_step = MagicMock(return_value=iteration)
        mock_runner = MagicMock(spec=Runner)
        mock_runner.step = mock_step

        wrapped_step = m._instrument_runner_step(mock_runner.step)

        wrapped_step(mock_runner)

        mock_start_span.assert_called_once_with(
            name="guardrails/guard/step",
            span_type="step",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard/step",
            },
        )

        # Internally called, not the wrapped call above
        mock_runner.step.assert_called_once()
        mock_add_step_attributes.assert_called_once_with(mock_span, iteration, mock_runner)

    def test__instrument_stream_runner_step(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_step_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_step_attributes"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        iteration = Iteration(call_id="mock call id", index=0)
        call = Call()
        call.iterations.push(iteration)

        def step_iterable():
            yield ValidationOutcome(call_id="mock call id", validation_passed=True)

        step_gen = step_iterable()
        mock_runner = MagicMock(spec=StreamRunner)
        mock_runner.step = MagicMock(return_value=step_gen)

        wrapped_step = m._instrument_stream_runner_step(mock_runner.step)

        wrapped_gen = wrapped_step(mock_runner, call_log=call)
        for gen in wrapped_gen:
            pass

        mock_start_span.assert_called_once_with(
            name="guardrails/guard/step",
            span_type="step",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard/step",
                "stream": True,
            },
        )

        # Internally called, not the wrapped call above
        mock_runner.step.assert_called_once()
        mock_add_step_attributes.assert_called_once_with(
            mock_span, iteration, mock_runner, call_log=call
        )

    @pytest.mark.asyncio
    async def test__instrument_async_runner_step(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_step_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_step_attributes"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        iteration = Iteration(call_id="mock call id", index=0)
        response = Future()
        response.set_result(iteration)
        mock_step = MagicMock(return_value=response)
        mock_runner = MagicMock(spec=AsyncRunner)
        mock_runner.async_step = mock_step

        wrapped_step = m._instrument_async_runner_step(mock_runner.async_step)

        await wrapped_step(mock_runner)

        mock_start_span.assert_called_once_with(
            name="guardrails/guard/step",
            span_type="step",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard/step",
                "async": True,
            },
        )

        # Internally called, not the wrapped call above
        mock_runner.async_step.assert_called_once()
        mock_add_step_attributes.assert_called_once_with(mock_span, iteration, mock_runner)

    @pytest.mark.asyncio
    async def test__instrument_async_stream_runner_step(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_step_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_step_attributes"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        iteration = Iteration(call_id="mock call id", index=0)
        call = Call()
        call.iterations.push(iteration)

        async def step_iterable():
            yield ValidationOutcome(call_id="mock call id", validation_passed=True)

        step_gen = step_iterable()
        mock_runner = MagicMock(spec=AsyncStreamRunner)
        mock_runner.async_step = MagicMock(return_value=step_gen)

        wrapped_step = m._instrument_async_stream_runner_step(mock_runner.async_step)

        wrapped_gen = wrapped_step(mock_runner, call_log=call)
        async for gen in wrapped_gen:
            pass

        mock_start_span.assert_called_once_with(
            name="guardrails/guard/step",
            span_type="step",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard/step",
                "async": True,
                "stream": True,
            },
        )

        # Internally called, not the wrapped call above
        mock_runner.async_step.assert_called_once()
        mock_add_step_attributes.assert_called_once_with(
            mock_span, iteration, mock_runner, call_log=call
        )

    def test__instrument_runner_call(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_call_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_call_attributes"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        llmResponse = LLMResponse(output="mock output")
        mock_call = MagicMock(return_value=llmResponse)
        mock_runner = MagicMock(spec=Runner)
        mock_runner.call = mock_call

        wrapped_call = m._instrument_runner_call(mock_runner.call)

        wrapped_call(mock_runner)

        mock_start_span.assert_called_once_with(
            name="guardrails/guard/step/call",
            span_type="LLM",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard/step/call",
            },
        )

        # Internally called, not the wrapped call above
        mock_runner.call.assert_called_once()
        mock_add_call_attributes.assert_called_once_with(mock_span, llmResponse, mock_runner)

    @pytest.mark.asyncio
    async def test__instrument_async_runner_call(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_call_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_call_attributes"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor

        m = MlFlowInstrumentor("mock experiment")

        llmResponse = LLMResponse(output="mock output")
        response = Future()
        response.set_result(llmResponse)
        mock_call = MagicMock(return_value=response)
        mock_runner = MagicMock(spec=AsyncRunner)
        mock_runner.async_call = mock_call

        wrapped_call = m._instrument_async_runner_call(mock_runner.async_call)

        await wrapped_call(mock_runner)

        mock_start_span.assert_called_once_with(
            name="guardrails/guard/step/call",
            span_type="LLM",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard/step/call",
                "async": True,
            },
        )

        # Internally called, not the wrapped call above
        mock_runner.async_call.assert_called_once()
        mock_add_call_attributes.assert_called_once_with(mock_span, llmResponse, mock_runner)

    def test__instrument_validator_validate(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.get_current_active_span",
            return_value=mock_span,
        )
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_validator_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_validator_attributes"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor
        from tests.unit_tests.mocks.mock_hub import MockValidator

        m = MlFlowInstrumentor("mock experiment")

        wrapped_validate = m._instrument_validator_validate(MockValidator.validate)

        mock_validator = MockValidator()

        resp = wrapped_validate(mock_validator, True, {})

        mock_start_span.assert_called_once_with(
            name="mock-validator.validate",
            span_type="validator",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard/step/validator",
            },
        )

        # Internally called, not the wrapped call above
        mock_add_validator_attributes.assert_called_once_with(
            mock_validator,
            True,
            {},
            validator_span=mock_span,  # type: ignore
            validator_name="mock-validator",
            obj_id=id(mock_validator),
            on_fail_descriptor="exception",
            result=resp,
            init_kwargs={},
            validation_session_id="unknown",
        )

    @pytest.mark.asyncio
    async def test__instrument_validator_async_validate(self, mocker):
        mock_span = MockSpan()
        mock_start_span = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.mlflow.start_span",
            return_value=mock_span,
        )

        mock_add_validator_attributes = mocker.patch(
            "guardrails.integrations.databricks.ml_flow_instrumentor.add_validator_attributes"
        )

        from guardrails.integrations.databricks import MlFlowInstrumentor
        from tests.unit_tests.mocks.mock_hub import MockValidator

        m = MlFlowInstrumentor("mock experiment")

        wrapped_async_validate = m._instrument_validator_async_validate(
            MockValidator.async_validate
        )

        mock_validator = MockValidator()

        resp = await wrapped_async_validate(mock_validator, True, {})

        mock_start_span.assert_called_once_with(
            name="mock-validator.validate",
            span_type="validator",
            attributes={
                "guardrails.version": GUARDRAILS_VERSION,
                "type": "guardrails/guard/step/validator",
                "async": True,
            },
        )

        # Internally called, not the wrapped call above
        mock_add_validator_attributes.assert_called_once_with(
            mock_validator,
            True,
            {},
            validator_span=mock_span,  # type: ignore
            validator_name="mock-validator",
            obj_id=id(mock_validator),
            on_fail_descriptor="exception",
            result=resp,
            init_kwargs={},
            validation_session_id="unknown",
        )
