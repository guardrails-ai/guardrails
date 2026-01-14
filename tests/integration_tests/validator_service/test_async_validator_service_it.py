import asyncio
import pytest
from time import sleep
from guardrails.validator_base import Validator, register_validator
from guardrails.classes.validation.validation_result import PassResult


@register_validator(name="test/validator1", data_type="string")
class Validator1(Validator):
    def validate(self, value, metadata):
        # This seems more realistic but is unreliable
        # counter = 0
        # for i in range(100000000):
        #     counter += 1
        # This seems suspicious, but is consistent
        sleep(0.3)
        metadata["order"].append("test/validator1")
        return PassResult()


@register_validator(name="test/validator2", data_type="string")
class Validator2(Validator):
    def validate(self, value, metadata):
        # counter = 0
        # for i in range(1):
        #     counter += 1
        sleep(0.1)
        metadata["order"].append("test/validator2")
        return PassResult()


@register_validator(name="test/validator3", data_type="string")
class Validator3(Validator):
    def validate(self, value, metadata):
        # counter = 0
        # for i in range(100000):
        #     counter += 1
        sleep(0.2)
        metadata["order"].append("test/validator3")
        return PassResult()


@register_validator(name="test/async_validator1", data_type="string")
class AsyncValidator1(Validator):
    async def async_validate(self, value, metadata):
        await asyncio.sleep(0.3)
        metadata["order"].append("test/async_validator1")
        return PassResult()


@register_validator(name="test/async_validator2", data_type="string")
class AsyncValidator2(Validator):
    async def async_validate(self, value, metadata):
        await asyncio.sleep(0.1)
        metadata["order"].append("test/async_validator2")
        return PassResult()


@register_validator(name="test/async_validator3", data_type="string")
class AsyncValidator3(Validator):
    async def async_validate(self, value, metadata):
        await asyncio.sleep(0.2)
        metadata["order"].append("test/async_validator3")
        return PassResult()


class TestValidatorConcurrency:
    @pytest.mark.asyncio
    async def test_async_validate_with_sync_validators(self):
        from guardrails.validator_service import AsyncValidatorService
        from guardrails.classes.history import Iteration

        iteration = Iteration(
            call_id="mock_call_id",
            index=0,
        )

        async_validator_service = AsyncValidatorService()

        value, metadata = await async_validator_service.async_validate(
            value="value",
            metadata={"order": []},
            validator_map={
                "$": [
                    # Note the order
                    Validator1(),
                    Validator2(),
                    Validator3(),
                ]
            },
            iteration=iteration,
            absolute_path="$",
            reference_path="$",
        )

        assert value == "value"
        assert metadata == {"order": ["test/validator2", "test/validator3", "test/validator1"]}

    def test_validate_with_sync_validators(self):
        from guardrails.validator_service import AsyncValidatorService
        from guardrails.classes.history import Iteration

        iteration = Iteration(
            call_id="mock_call_id",
            index=0,
        )

        async_validator_service = AsyncValidatorService()

        loop = asyncio.get_event_loop()
        value, metadata = async_validator_service.validate(
            value="value",
            metadata={"order": []},
            validator_map={
                "$": [
                    # Note the order
                    Validator1(),
                    Validator2(),
                    Validator3(),
                ]
            },
            iteration=iteration,
            absolute_path="$",
            reference_path="$",
            loop=loop,
        )

        assert value == "value"
        assert metadata == {"order": ["test/validator2", "test/validator3", "test/validator1"]}

    @pytest.mark.asyncio
    async def test_async_validate_with_async_validators(self):
        from guardrails.validator_service import AsyncValidatorService
        from guardrails.classes.history import Iteration

        iteration = Iteration(
            call_id="mock_call_id",
            index=0,
        )

        async_validator_service = AsyncValidatorService()

        value, metadata = await async_validator_service.async_validate(
            value="value",
            metadata={"order": []},
            validator_map={
                "$": [
                    # Note the order
                    AsyncValidator1(),
                    AsyncValidator2(),
                    AsyncValidator3(),
                ]
            },
            iteration=iteration,
            absolute_path="$",
            reference_path="$",
        )

        assert value == "value"
        assert metadata == {
            "order": [
                "test/async_validator2",
                "test/async_validator3",
                "test/async_validator1",
            ]
        }

    def test_validate_with_async_validators(self):
        from guardrails.validator_service import AsyncValidatorService
        from guardrails.classes.history import Iteration

        iteration = Iteration(
            call_id="mock_call_id",
            index=0,
        )

        async_validator_service = AsyncValidatorService()

        loop = asyncio.get_event_loop()
        value, metadata = async_validator_service.validate(
            value="value",
            metadata={"order": []},
            validator_map={
                "$": [
                    # Note the order
                    AsyncValidator1(),
                    AsyncValidator2(),
                    AsyncValidator3(),
                ]
            },
            iteration=iteration,
            absolute_path="$",
            reference_path="$",
            loop=loop,
        )

        assert value == "value"
        assert metadata == {
            "order": [
                "test/async_validator2",
                "test/async_validator3",
                "test/async_validator1",
            ]
        }

    @pytest.mark.asyncio
    async def test_async_validate_with_mixed_validators(self):
        from guardrails.validator_service import AsyncValidatorService
        from guardrails.classes.history import Iteration

        iteration = Iteration(
            call_id="mock_call_id",
            index=0,
        )

        async_validator_service = AsyncValidatorService()

        value, metadata = await async_validator_service.async_validate(
            value="value",
            metadata={"order": []},
            validator_map={
                "$": [
                    # Note the order
                    Validator1(),
                    Validator2(),
                    AsyncValidator3(),
                ]
            },
            iteration=iteration,
            absolute_path="$",
            reference_path="$",
        )

        assert value == "value"
        assert metadata == {
            "order": ["test/validator2", "test/async_validator3", "test/validator1"]
        }

    def test_validate_with_mixed_validators(self):
        from guardrails.validator_service import AsyncValidatorService
        from guardrails.classes.history import Iteration

        iteration = Iteration(
            call_id="mock_call_id",
            index=0,
        )

        async_validator_service = AsyncValidatorService()

        loop = asyncio.get_event_loop()
        value, metadata = async_validator_service.validate(
            value="value",
            metadata={"order": []},
            validator_map={
                "$": [
                    # Note the order
                    Validator1(),
                    Validator2(),
                    AsyncValidator3(),
                ]
            },
            iteration=iteration,
            absolute_path="$",
            reference_path="$",
            loop=loop,
        )

        assert value == "value"
        assert metadata == {
            "order": ["test/validator2", "test/async_validator3", "test/validator1"]
        }
