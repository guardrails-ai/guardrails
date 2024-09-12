import asyncio
from typing import Any, Awaitable, Coroutine, Dict, List, Optional, Tuple, Union

from guardrails.actions.filter import Filter
from guardrails.actions.refrain import Refrain
from guardrails.classes.history import Iteration
from guardrails.classes.validation.validation_result import (
    FailResult,
    PassResult,
    ValidationResult,
)
from guardrails.telemetry.validator_tracing import trace_async_validator
from guardrails.types import ValidatorMap, OnFailAction
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.actions.reask import FieldReAsk
from guardrails.validator_base import Validator
from guardrails.validator_service.validator_service_base import (
    ValidatorRun,
    ValidatorServiceBase,
)

ValidatorResult = Optional[Union[ValidationResult, Awaitable[ValidationResult]]]


class AsyncValidatorService(ValidatorServiceBase):
    async def execute_validator(
        self,
        validator: Validator,
        value: Any,
        metadata: Optional[Dict],
        stream: Optional[bool] = False,
        *,
        validation_session_id: str,
        **kwargs,
    ) -> Optional[ValidationResult]:
        validate_func = (
            validator.async_validate_stream if stream else validator.async_validate
        )
        traced_validator = trace_async_validator(
            validator_name=validator.rail_alias,
            obj_id=id(validator),
            on_fail_descriptor=validator.on_fail_descriptor,
            validation_session_id=validation_session_id,
            **validator._kwargs,
        )(validate_func)
        if stream:
            result = await traced_validator(value, metadata, **kwargs)
        else:
            result = await traced_validator(value, metadata)
        return result

    async def run_validator_async(
        self,
        validator: Validator,
        value: Any,
        metadata: Dict,
        stream: Optional[bool] = False,
        *,
        validation_session_id: str,
        **kwargs,
    ) -> ValidationResult:
        result = await self.execute_validator(
            validator,
            value,
            metadata,
            stream,
            validation_session_id=validation_session_id,
            **kwargs,
        )

        if result is None:
            result = PassResult()
        return result

    async def run_validator(
        self,
        iteration: Iteration,
        validator: Validator,
        value: Any,
        metadata: Dict,
        absolute_property_path: str,
        stream: Optional[bool] = False,
        **kwargs,
    ) -> ValidatorRun:
        validator_logs = self.before_run_validator(
            iteration, validator, value, absolute_property_path
        )

        result = await self.run_validator_async(
            validator,
            value,
            metadata,
            stream,
            validation_session_id=iteration.id,
            **kwargs,
        )

        validator_logs = self.after_run_validator(validator, validator_logs, result)

        if isinstance(result, FailResult):
            rechecked_value = None
            if validator.on_fail_descriptor == OnFailAction.FIX_REASK:
                fixed_value = result.fix_value
                rechecked_value = await self.run_validator_async(
                    validator,
                    fixed_value,
                    result.metadata or {},
                    stream,
                    validation_session_id=iteration.id,
                    **kwargs,
                )
            value = self.perform_correction(
                result,
                value,
                validator,
                rechecked_value=rechecked_value,
            )

        # handle overrides
        # QUESTION: Should this consider the rechecked_value as well?
        elif (
            isinstance(result, PassResult)
            and result.value_override is not PassResult.ValueOverrideSentinel
        ):
            value = result.value_override

        validator_logs.value_after_validation = value

        return ValidatorRun(
            value=value,
            metadata=metadata,
            on_fail_action=validator.on_fail_descriptor,
            validator_logs=validator_logs,
        )

    async def run_validators(
        self,
        iteration: Iteration,
        validator_map: ValidatorMap,
        value: Any,
        metadata: Dict,
        absolute_property_path: str,
        reference_property_path: str,
        stream: Optional[bool] = False,
        **kwargs,
    ):
        validators = validator_map.get(reference_property_path, [])
        coroutines: List[Coroutine[Any, Any, ValidatorRun]] = []
        validators_logs: List[ValidatorLogs] = []
        for validator in validators:
            coroutines.append(
                self.run_validator(
                    iteration,
                    validator,
                    value,
                    metadata,
                    absolute_property_path,
                    stream=stream,
                    **kwargs,
                )
            )

        results = await asyncio.gather(*coroutines)
        reasks: List[FieldReAsk] = []
        for res in results:
            validators_logs.append(res.validator_logs)
            # QUESTION: Do we still want to do this here or handle it during the merge?
            # return early if we have a filter, refrain, or reask
            if isinstance(res.value, (Filter, Refrain)):
                return res.value, metadata
            elif isinstance(res.value, FieldReAsk):
                reasks.append(res.value)

        # handle reasks
        if len(reasks) > 0:
            first_reask = reasks[0]
            fail_results = []
            for reask in reasks:
                fail_results.extend(reask.fail_results)
            first_reask.fail_results = fail_results
            return first_reask, metadata

        # merge the results
        fix_values = [
            res.value
            for res in results
            if (
                isinstance(res.validator_logs.validation_result, FailResult)
                and (
                    res.on_fail_action == OnFailAction.FIX
                    or res.on_fail_action == OnFailAction.FIX_REASK
                    or res.on_fail_action == OnFailAction.CUSTOM
                )
            )
        ]
        if len(fix_values) > 0:
            value = self.merge_results(value, fix_values)

        return value, metadata

    async def validate_children(
        self,
        value: Any,
        metadata: Dict,
        validator_map: ValidatorMap,
        iteration: Iteration,
        abs_parent_path: str,
        ref_parent_path: str,
        stream: Optional[bool] = False,
        **kwargs,
    ):
        async def validate_child(
            child_value: Any, *, key: Optional[str] = None, index: Optional[int] = None
        ):
            child_key = key or index
            abs_child_path = f"{abs_parent_path}.{child_key}"
            ref_child_path = ref_parent_path
            if key is not None:
                ref_child_path = f"{ref_child_path}.{key}"
            elif index is not None:
                ref_child_path = f"{ref_child_path}.*"
            new_child_value, new_metadata = await self.async_validate(
                child_value,
                metadata,
                validator_map,
                iteration,
                abs_child_path,
                ref_child_path,
                stream=stream,
                **kwargs,
            )
            return child_key, new_child_value, new_metadata

        coroutines = []
        if isinstance(value, List):
            for index, child in enumerate(value):
                coroutines.append(validate_child(child, index=index))
        elif isinstance(value, Dict):
            for key in value:
                child = value.get(key)
                coroutines.append(validate_child(child, key=key))

        results = await asyncio.gather(*coroutines)

        for key, child_value, child_metadata in results:
            value[key] = child_value
            # TODO address conflicting metadata entries
            metadata = {**metadata, **child_metadata}

        return value, metadata

    async def async_validate(
        self,
        value: Any,
        metadata: dict,
        validator_map: ValidatorMap,
        iteration: Iteration,
        absolute_path: str,
        reference_path: str,
        stream: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Any, dict]:
        child_ref_path = reference_path.replace(".*", "")
        # Validate children first
        if isinstance(value, List) or isinstance(value, Dict):
            await self.validate_children(
                value,
                metadata,
                validator_map,
                iteration,
                absolute_path,
                child_ref_path,
                stream=stream,
                **kwargs,
            )

        # Then validate the parent value
        value, metadata = await self.run_validators(
            iteration,
            validator_map,
            value,
            metadata,
            absolute_path,
            reference_path,
            stream=stream,
            **kwargs,
        )

        return value, metadata

    def validate(
        self,
        value: Any,
        metadata: dict,
        validator_map: ValidatorMap,
        iteration: Iteration,
        absolute_path: str,
        reference_path: str,
        loop: asyncio.AbstractEventLoop,
        stream: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Any, dict]:
        value, metadata = loop.run_until_complete(
            self.async_validate(
                value,
                metadata,
                validator_map,
                iteration,
                absolute_path,
                reference_path,
                stream=stream,
                **kwargs,
            )
        )
        return value, metadata
