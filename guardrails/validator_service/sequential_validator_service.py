import asyncio
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from guardrails.actions.filter import Filter
from guardrails.actions.refrain import Refrain
from guardrails.classes.history import Iteration
from guardrails.classes.validation.validation_result import (
    FailResult,
    PassResult,
    StreamValidationResult,
    ValidationResult,
)
from guardrails.merge import merge
from guardrails.types import ValidatorMap, OnFailAction
from guardrails.utils.exception_utils import UserFacingException
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.actions.reask import ReAsk
from guardrails.validator_base import Validator
from guardrails.validator_service.validator_service_base import ValidatorServiceBase


class SequentialValidatorService(ValidatorServiceBase):
    def run_validator_sync(
        self,
        validator: Validator,
        value: Any,
        metadata: Dict,
        validator_logs: ValidatorLogs,
        stream: Optional[bool] = False,
        *,
        validation_session_id: str,
        **kwargs,
    ) -> Optional[ValidationResult]:
        result = self.execute_validator(
            validator,
            value,
            metadata,
            stream,
            validation_session_id=validation_session_id,
            **kwargs,
        )
        if asyncio.iscoroutine(result):
            raise UserFacingException(
                ValueError(
                    "Cannot use async validators with a synchronous Guard! "
                    f"Either use AsyncGuard or remove {validator_logs.validator_name}."
                )
            )
        if result is None:
            return result
        return cast(ValidationResult, result)

    def run_validator(
        self,
        iteration: Iteration,
        validator: Validator,
        value: Any,
        metadata: Dict,
        property_path: str,
        stream: Optional[bool] = False,
        **kwargs,
    ) -> ValidatorLogs:
        validator_logs = self.before_run_validator(
            iteration, validator, value, property_path
        )

        result = self.run_validator_sync(
            validator,
            value,
            metadata,
            validator_logs,
            stream,
            validation_session_id=iteration.id,
            **kwargs,
        )

        return self.after_run_validator(validator, validator_logs, result)

    def run_validators_stream(
        self,
        iteration: Iteration,
        validator_map: ValidatorMap,
        value_stream: Iterable[Tuple[Any, bool]],
        metadata: Dict[str, Any],
        absolute_property_path: str,
        reference_property_path: str,
        **kwargs,
    ) -> Iterable[StreamValidationResult]:
        validators = validator_map.get(reference_property_path, [])
        for validator in validators:
            if validator.on_fail_descriptor == OnFailAction.FIX:
                return self.run_validators_stream_fix(
                    iteration,
                    validator_map,
                    value_stream,
                    metadata,
                    absolute_property_path,
                    reference_property_path,
                    **kwargs,
                )
        return self.run_validators_stream_noop(
            iteration,
            validator_map,
            value_stream,
            metadata,
            absolute_property_path,
            reference_property_path,
            **kwargs,
        )

    # requires at least 2 validators
    def multi_merge(self, original: str, new_values: list[str]) -> Optional[str]:
        current = new_values.pop()
        print("Fmerging these:", new_values)
        while len(new_values) > 0:
            nextval = new_values.pop()
            current = merge(current, nextval, original)
        print("\nFmerge result:", current)
        return current

    def run_validators_stream_fix(
        self,
        iteration: Iteration,
        validator_map: ValidatorMap,
        value_stream: Iterable[Tuple[Any, bool]],
        metadata: Dict[str, Any],
        absolute_property_path: str,
        reference_property_path: str,
        **kwargs,
    ) -> Iterable[StreamValidationResult]:
        validators = validator_map.get(reference_property_path, [])
        acc_output = ""
        validator_partial_acc: dict[int, str] = {}
        for validator in validators:
            validator_partial_acc[id(validator)] = ""
        last_chunk = None
        last_chunk_validated = False
        last_chunk_missing_validators = []
        refrain_triggered = False
        for chunk, finished in value_stream:
            original_text = chunk
            acc_output += chunk
            fixed_values = []
            last_chunk = chunk
            last_chunk_missing_validators = []
            if refrain_triggered:
                break
            for validator in validators:
                # reset chunk to original text
                chunk = original_text
                validator_logs = self.run_validator(
                    iteration,
                    validator,
                    chunk,
                    metadata,
                    absolute_property_path,
                    True,
                    remainder=finished,
                    **kwargs,
                )
                result = validator_logs.validation_result
                if result is None:
                    last_chunk_missing_validators.append(validator)
                result = cast(ValidationResult, result)
                # if we have a concrete result, log it in the validation map
                if isinstance(result, FailResult):
                    is_filter = validator.on_fail_descriptor is OnFailAction.FILTER
                    is_refrain = validator.on_fail_descriptor is OnFailAction.REFRAIN
                    if is_filter or is_refrain:
                        refrain_triggered = True
                        break
                    rechecked_value = None
                    chunk = self.perform_correction(
                        result,
                        chunk,
                        validator,
                        rechecked_value=rechecked_value,
                    )
                    fixed_values.append(chunk)
                    validator_partial_acc[id(validator)] += chunk  # type: ignore
                elif isinstance(result, PassResult):
                    if (
                        validator.override_value_on_pass
                        and result.value_override is not result.ValueOverrideSentinel
                    ):
                        chunk = result.value_override
                    else:
                        chunk = result.validated_chunk
                    fixed_values.append(chunk)
                    validator_partial_acc[id(validator)] += chunk  # type: ignore
                validator_logs.value_after_validation = chunk
                if result and result.metadata is not None:
                    metadata = result.metadata

            if refrain_triggered:
                # if we have a failresult from a refrain/filter validator, yield empty
                yield StreamValidationResult(
                    chunk="", original_text=acc_output, metadata=metadata
                )
            else:
                # if every validator has yielded a concrete value, merge and yield
                # only merge and yield if all validators have run
                # TODO: check if only 1 validator - then skip merging
                if len(fixed_values) == len(validators):
                    last_chunk_validated = True
                    values_to_merge = []
                    for validator in validators:
                        values_to_merge.append(validator_partial_acc[id(validator)])
                    merged_value = self.multi_merge(acc_output, values_to_merge)
                    # merged_value = self.multi_merge(acc_output, values_to_merge)
                    # reset validator_partial_acc
                    for validator in validators:
                        validator_partial_acc[id(validator)] = ""
                    yield StreamValidationResult(
                        chunk=merged_value, original_text=acc_output, metadata=metadata
                    )
                    acc_output = ""
                else:
                    last_chunk_validated = False
        # handle case where LLM doesn't yield finished flag
        # we need to validate remainder of accumulated chunks
        if not last_chunk_validated and not refrain_triggered:
            original_text = last_chunk
            for validator in last_chunk_missing_validators:
                last_log = self.run_validator(
                    iteration,
                    validator,
                    # use empty chunk
                    # validator has already accumulated the chunk from the first loop
                    "",
                    metadata,
                    absolute_property_path,
                    True,
                    remainder=True,
                    **kwargs,
                )
                result = last_log.validation_result
                if isinstance(result, FailResult):
                    rechecked_value = None
                    last_chunk = self.perform_correction(
                        result,
                        last_chunk,
                        validator,
                        rechecked_value=rechecked_value,
                    )
                    validator_partial_acc[id(validator)] += last_chunk  # type: ignore
                elif isinstance(result, PassResult):
                    if (
                        validator.override_value_on_pass
                        and result.value_override is not result.ValueOverrideSentinel
                    ):
                        last_chunk = result.value_override
                    else:
                        last_chunk = result.validated_chunk
                    validator_partial_acc[id(validator)] += last_chunk  # type: ignore
                last_log.value_after_validation = last_chunk
                if result and result.metadata is not None:
                    metadata = result.metadata
            values_to_merge = []
            for validator in validators:
                values_to_merge.append(validator_partial_acc[id(validator)])
            merged_value = self.multi_merge(acc_output, values_to_merge)
            yield StreamValidationResult(
                chunk=merged_value,
                original_text=original_text,  # type: ignore
                metadata=metadata,  # type: ignore
            )
            # yield merged value

    def run_validators_stream_noop(
        self,
        iteration: Iteration,
        validator_map: ValidatorMap,
        value_stream: Iterable[Tuple[Any, bool]],
        metadata: Dict[str, Any],
        absolute_property_path: str,
        reference_property_path: str,
        **kwargs,
    ) -> Iterable[StreamValidationResult]:
        validators = validator_map.get(reference_property_path, [])
        # Validate the field
        # TODO: Under what conditions do we yield?
        # When we have at least one non-None value?
        # When we have all non-None values?
        # Does this depend on whether we are fix or not?
        for chunk, finished in value_stream:
            original_text = chunk
            for validator in validators:
                validator_logs = self.run_validator(
                    iteration,
                    validator,
                    chunk,
                    metadata,
                    absolute_property_path,
                    True,
                    **kwargs,
                )
                result = validator_logs.validation_result
                result = cast(ValidationResult, result)

                if isinstance(result, FailResult):
                    rechecked_value = None
                    chunk = self.perform_correction(
                        result,
                        chunk,
                        validator,
                        rechecked_value=rechecked_value,
                    )
                elif isinstance(result, PassResult):
                    if (
                        validator.override_value_on_pass
                        and result.value_override is not result.ValueOverrideSentinel
                    ):
                        chunk = result.value_override

                validator_logs.value_after_validation = chunk
                if result and result.metadata is not None:
                    metadata = result.metadata
                # # TODO: Filter is no longer terminal, so we shouldn't yield, right?
                # if isinstance(chunk, (Refrain, Filter, ReAsk)):
                #     yield chunk, metadata
            yield StreamValidationResult(
                chunk=chunk, original_text=original_text, metadata=metadata
            )

    def run_validators(
        self,
        iteration: Iteration,
        validator_map: ValidatorMap,
        value: Any,
        metadata: Dict[str, Any],
        absolute_property_path: str,
        reference_property_path: str,
        stream: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
        # Validate the field
        validators = validator_map.get(reference_property_path, [])
        for validator in validators:
            if stream:
                if validator.on_fail_descriptor is OnFailAction.REASK:
                    raise ValueError(
                        """Reask is not supported for stream validation, 
                        only noop and exception are supported."""
                    )
                if validator.on_fail_descriptor is OnFailAction.FIX:
                    raise ValueError(
                        """Fix is not supported for stream validation, 
                        only noop and exception are supported."""
                    )
                if validator.on_fail_descriptor is OnFailAction.FIX_REASK:
                    raise ValueError(
                        """Fix reask is not supported for stream validation, 
                        only noop and exception are supported."""
                    )
                if validator.on_fail_descriptor is OnFailAction.FILTER:
                    raise ValueError(
                        """Filter is not supported for stream validation, 
                        only noop and exception are supported."""
                    )
                if validator.on_fail_descriptor is OnFailAction.REFRAIN:
                    raise ValueError(
                        """Refrain is not supported for stream validation, 
                        only noop and exception are supported."""
                    )
            validator_logs = self.run_validator(
                iteration,
                validator,
                value,
                metadata,
                absolute_property_path,
                stream,
                **kwargs,
            )
            result = validator_logs.validation_result

            result = cast(ValidationResult, result)
            if isinstance(result, FailResult):
                rechecked_value = None
                if validator.on_fail_descriptor == OnFailAction.FIX_REASK:
                    fixed_value = result.fix_value
                    rechecked_value = self.run_validator_sync(
                        validator,
                        fixed_value,
                        metadata,
                        validator_logs,
                        stream,
                        **kwargs,
                    )
                value = self.perform_correction(
                    result,
                    value,
                    validator,
                    rechecked_value=rechecked_value,
                )
            elif isinstance(result, PassResult):
                if (
                    validator.override_value_on_pass
                    and result.value_override is not result.ValueOverrideSentinel
                ):
                    value = result.value_override
            elif not stream:
                raise RuntimeError(f"Unexpected result type {type(result)}")

            validator_logs.value_after_validation = value
            if result and result.metadata is not None:
                metadata = result.metadata

            if isinstance(value, (Refrain, Filter, ReAsk)):
                return value, metadata
        return value, metadata

    def validate(
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
        ###
        # NOTE: The way validation can be executed now is fundamentally wide open.
        #   Since validators are tracked against the JSONPaths for the
        #       properties they should be applied to, we have the following options:
        #       1. Keep performing a Deep-First-Search
        #           - This is useful for backwards compatibility.
        #           - Is there something we gain by validating inside out?
        #       2. Swith to a Breadth-First-Search
        #           - Possible, no obvious advantages
        #       3. Run un-ordered
        #           - This would allow for true parallelism
        #           - Also means we're not unnecessarily iterating down through
        #               the object if there aren't any validations applied there.
        ###

        child_ref_path = reference_path.replace(".*", "")
        # Validate children first
        if isinstance(value, List):
            for index, child in enumerate(value):
                abs_child_path = f"{absolute_path}.{index}"
                ref_child_path = f"{child_ref_path}.*"
                child_value, metadata = self.validate(
                    child,
                    metadata,
                    validator_map,
                    iteration,
                    abs_child_path,
                    ref_child_path,
                )
                value[index] = child_value
        elif isinstance(value, Dict):
            for key in value:
                child = value.get(key)
                abs_child_path = f"{absolute_path}.{key}"
                ref_child_path = f"{child_ref_path}.{key}"
                child_value, metadata = self.validate(
                    child,
                    metadata,
                    validator_map,
                    iteration,
                    abs_child_path,
                    ref_child_path,
                )
                value[key] = child_value

        # Then validate the parent value
        value, metadata = self.run_validators(
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

    def validate_stream(
        self,
        value_stream: Iterable[Tuple[Any, bool]],
        metadata: dict,
        validator_map: ValidatorMap,
        iteration: Iteration,
        absolute_path: str,
        reference_path: str,
        **kwargs,
    ) -> Iterable[StreamValidationResult]:
        # I assume validate stream doesn't need validate_dependents
        # because right now we're only handling StringSchema

        # Validate the field
        gen = self.run_validators_stream(
            iteration,
            validator_map,
            value_stream,
            metadata,
            absolute_path,
            reference_path,
            **kwargs,
        )
        return gen
