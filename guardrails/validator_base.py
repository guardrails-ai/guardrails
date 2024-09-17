# TODO:
#   - [ ] Rename this to just validator.py 0.5.x
#   - [ ] Maintain validator_base.py for exports but deprecate them
#   - [ ] Remove validator_base.py in 0.6.x

import asyncio
from functools import partial
import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from string import Template
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from warnings import warn
import warnings

import nltk
import requests
from langchain_core.runnables import Runnable

from guardrails.classes import ErrorSpan  # noqa
from guardrails.classes import PassResult  # noqa
from guardrails.classes import FailResult, ValidationResult
from guardrails.classes.credentials import Credentials
from guardrails.constants import hub
from guardrails.hub_token.token import VALIDATOR_HUB_SERVICE, get_jwt_token
from guardrails.logger import logger
from guardrails.remote_inference import remote_inference
from guardrails.types.on_fail import OnFailAction
from guardrails.utils.safe_get import safe_get
from guardrails.utils.hub_telemetry_utils import HubTelemetry

#   See: https://github.com/guardrails-ai/guardrails/issues/829
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


### functions to get chunks ###
def split_sentence_str(chunk: str):
    """A naive sentence splitter that splits on periods."""
    if "." not in chunk:
        return []
    fragments = chunk.split(".")
    return [fragments[0] + ".", ".".join(fragments[1:])]


def split_sentence_nltk(chunk: str):
    """
    NOTE: this approach currently does not work
    Use a sentence tokenizer to split the chunk into sentences.

    Because using the tokenizer is expensive, we only use it if there
    is a period present in the chunk.
    """
    # using the sentence tokenizer is expensive
    # we check for a . to avoid wastefully calling the tokenizer
    if "." not in chunk:
        return []
    sentences = nltk.sent_tokenize(chunk)
    if len(sentences) == 0:
        return []
    # return the sentence
    # then the remaining chunks that aren't finished accumulating
    return [sentences[0], "".join(sentences[1:])]


# TODO: Can we remove dataclass? It was originally added to support pydantic 1.*
@dataclass  # type: ignore
class Validator:
    """Base class for validators."""

    rail_alias: str = ""

    run_in_separate_process = False
    override_value_on_pass = False
    required_metadata_keys = []
    _metadata = {}

    def __init__(
        self,
        on_fail: Optional[Union[Callable[[Any, FailResult], Any], OnFailAction]] = None,
        **kwargs,
    ):
        self.creds = Credentials.from_rc_file()
        self._disable_telemetry = self.creds.enable_metrics is not True
        if not self._disable_telemetry:
            self._hub_telemetry = HubTelemetry()

        self.use_local = kwargs.get("use_local", None)
        self.validation_endpoint = kwargs.get("validation_endpoint", None)
        if not self.creds:
            raise ValueError(
                "No credentials found. Please run `guardrails configure` and try again."
            )
        self.hub_jwt_token = get_jwt_token(self.creds)

        # If use_local is not set, we can fall back to the setting determined in CLI
        if self.use_local is None:
            self.use_local = not remote_inference.get_use_remote_inference(self.creds)

        if not self.validation_endpoint:
            validator_id = self.rail_alias.split("/")[-1]
            submission_url = (
                f"{VALIDATOR_HUB_SERVICE}/validator/{validator_id}/inference"
            )
            self.validation_endpoint = submission_url
        self.on_fail_descriptor: Union[str, OnFailAction] = "custom"

        # chunking function returns empty list or list of 2 chunks
        # first chunk is the chunk to validate
        # second chunk is incomplete chunk that needs further accumulation
        self.accumulated_chunks: List[str] = []

        if on_fail is None:
            on_fail = OnFailAction.NOOP
        if isinstance(on_fail, OnFailAction):
            self.on_fail_descriptor = on_fail
            self.on_fail_method = None
        elif (
            isinstance(on_fail, str)
            and OnFailAction.__members__.get(on_fail.upper()) is not None
        ):
            self.on_fail_descriptor = (
                OnFailAction.__members__.get(on_fail.upper())
                or ""  # this default isn't needed, it's just for pyright
            )
            self.on_fail_method = None
        else:
            self.on_fail_descriptor = OnFailAction.CUSTOM
            self._set_on_fail_method(on_fail)

        # Store the kwargs for the validator.
        self._kwargs = kwargs

        assert (
            self.rail_alias in validators_registry
        ), f"Validator {self.__class__.__name__} is not registered. "

    def _set_on_fail_method(self, on_fail: Callable[[Any, FailResult], Any]):
        """Set the on_fail method for the validator."""
        on_fail_args = inspect.getfullargspec(on_fail)
        second_arg = safe_get(on_fail_args.args, 1)
        if second_arg is None:
            raise ValueError(
                "The on_fail method must take two arguments: "
                "the value being validated and the FailResult."
            )
        second_arg_type = on_fail_args.annotations.get(second_arg)
        if second_arg_type == List[FailResult]:
            warnings.warn(
                "Specifying a List[FailResult] as the second argument"
                " for a custom on_fail handler is deprecated. "
                "Please use FailResult instead.",
                DeprecationWarning,
            )

            def on_fail_wrapper(value: Any, fail_result: FailResult) -> Any:
                return on_fail(value, [fail_result])  # type: ignore

            self.on_fail_method = on_fail_wrapper
        else:
            self.on_fail_method = on_fail

    def _validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        """User implementable function.

        Validates a value and return a validation result. This method
        should call _inference() in the implementation to perform
        inference on some input value.
        """
        raise NotImplementedError

    def _inference_local(self, model_input: Any) -> Any:
        """User implementable function.

        Runs a machine learning pipeline on some input on the local
        machine. This function should receive the expected input to the
        ML model, and output the results from the ml model.
        """
        raise NotImplementedError

    def _inference_remote(self, model_input: Any) -> Any:
        """User implementable function.

        Runs a machine learning pipeline on some input on a remote
        machine. This function should receive the expected input to the
        ML model, and output the results from the ml model.

        Can call _hub_inference_request() if request is routed through
        the hub.
        """
        raise NotImplementedError

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        """Do not override this function, instead implement _validate().

        External facing validate function. This function acts as a
        wrapper for _validate() and is intended to apply any meta-
        validation requirements, logic, or pre/post processing.
        """
        validation_result = self._validate(value, metadata)
        self._log_telemetry()
        return validation_result

    async def async_validate(
        self, value: Any, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Use this function if your validation logic requires asyncio.

        Guaranteed to work with AsyncGuard

        May not work with synchronous Guards if they are used within an
        async context     due to lack of available event loops.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.validate, value, metadata)

    def _inference(self, model_input: Any) -> Any:
        """Calls either a local or remote inference engine for use in the
        validation call.

        Args:
            model_input (Any): Receives the input to be passed to your ML model.

        Returns:
            Any: Returns the output from the ML model inference.
        """
        # Only use if both are set, otherwise fall back to local inference
        if self.use_local:
            return self._inference_local(model_input)
        if not self.use_local and self.validation_endpoint:
            return self._inference_remote(model_input)

        raise RuntimeError(
            "No inference endpoint set, but use_local was false. "
            "Please set either use_local=True or "
            "set an validation_endpoint to perform inference in the validator."
        )

    def _chunking_function(self, chunk: str) -> List[str]:
        """The strategy used for chunking accumulated text input into
        validation sets.

        Args:
            chunk (str): The text to chunk into some subset.

        Returns:
            list[str]: The text chunked into some subset.
        """
        return split_sentence_str(chunk)

    def validate_stream(
        self, chunk: Any, metadata: Dict[str, Any], **kwargs
    ) -> Optional[ValidationResult]:
        """Validates a chunk emitted by an LLM. If the LLM chunk is smaller
        than the validator's chunking strategy, it will be accumulated until it
        reaches the desired size. In the meantime, the validator will return
        None.

        If the LLM chunk is larger than the validator's chunking
        strategy, it will split it into validator-sized chunks and
        validate each one, returning an array of validation results.

        Otherwise, the validator will validate the chunk and return the
        result.
        """
        # combine accumulated chunks and new [:-1]chunk
        self.accumulated_chunks.append(chunk)
        accumulated_text = "".join(self.accumulated_chunks)
        # check if enough chunks have accumulated for validation
        split_contents = self._chunking_function(accumulated_text)

        # if remainder kwargs is passed, validate remainder regardless
        remainder = kwargs.get("remainder", False)
        if remainder:
            split_contents = [accumulated_text, ""]
        # if no chunks are returned, we haven't accumulated enough
        if len(split_contents) == 0:
            return None
        [chunk_to_validate, new_accumulated_chunks] = split_contents
        self.accumulated_chunks = [new_accumulated_chunks]
        # exclude last chunk, because it may not be a complete chunk
        validation_result = self.validate(chunk_to_validate, metadata)
        # if validate doesn't set validated chunk, we set it
        if validation_result.validated_chunk is None:
            validation_result.validated_chunk = chunk_to_validate
        if isinstance(validation_result, FailResult):
            if validation_result.error_spans is None:
                validation_result.error_spans = [
                    ErrorSpan(
                        start=0,
                        end=len(chunk_to_validate),
                        reason="The input failed validation.",
                    )
                ]

        return validation_result

    async def async_validate_stream(
        self, chunk: Any, metadata: Dict[str, Any], **kwargs
    ) -> Optional[ValidationResult]:
        loop = asyncio.get_event_loop()
        validate_stream_partial = partial(
            self.validate_stream, chunk, metadata, **kwargs
        )
        return await loop.run_in_executor(None, validate_stream_partial)

    def _hub_inference_request(
        self, request_body: Union[dict, str], validation_endpoint: str
    ) -> Any:
        """Makes a request to the Validator Hub to run a ML based validation
        model. This request is authed through the hub and rerouted to a hosted
        ML model. The reply from the hosted endpoint is returned and sent to
        this client.

        Args:
            request_body (dict): A dictionary containing the required info for the final
            validation_endpoint (str): The url to request as an endpoint
            inference endpoint to run.

        Raises:
            HttpError: If the recieved reply was not ok.

        Returns:
            Any: Post request response from the ML based validation model.
        """
        headers = {
            "Authorization": f"Bearer {self.hub_jwt_token}",
            "Content-Type": "application/json",
        }
        req = requests.post(validation_endpoint, data=request_body, headers=headers)
        if not req.ok:
            if req.status_code == 401:
                raise Exception(
                    "401: Remote Inference Unauthorized. Please run "
                    "`guardrails configure`. You can find a new"
                    " token at https://hub.guardrailsai.com/keys"
                )
            else:
                logging.error(req.status_code)

        return req.json()

    def to_prompt(self, with_keywords: bool = True) -> str:
        """Convert the validator to a prompt.

        E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
        ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

        Args:
            with_keywords: Whether to include the keyword arguments in the prompt.

        Returns:
            A string representation of the validator.
        """
        if not len(self._kwargs):
            return self.rail_alias

        kwargs = self._kwargs.copy()
        for k, v in kwargs.items():
            if not isinstance(v, str):
                kwargs[k] = str(v)

        params = " ".join(list(kwargs.values()))
        if with_keywords:
            params = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        return f"{self.rail_alias}: {params}"

    # TODO: Is this still used anywhere?
    def to_xml_attrib(self):
        """Convert the validator to an XML attribute."""

        if not len(self._kwargs):
            return self.rail_alias

        validator_args = []
        init_args = inspect.getfullargspec(self.__init__)
        for arg in init_args.args[1:]:
            if arg not in ("on_fail", "args", "kwargs"):
                arg_value = self._kwargs.get(arg)
                str_arg = str(arg_value)
                if str_arg is not None:
                    str_arg = "{" + str_arg + "}" if " " in str_arg else str_arg
                    validator_args.append(str_arg)

        params = " ".join(validator_args)
        return f"{self.rail_alias}: {params}"

    def get_args(self):
        """Get the arguments for the validator."""
        return self._kwargs

    def __call__(self, value):
        result = self.validate(value, {})
        if isinstance(result, FailResult):
            from guardrails.validator_service.validator_service_base import (
                ValidatorServiceBase,
            )

            validator_service = ValidatorServiceBase()
            return validator_service.perform_correction(result, value, self)
        return value

    def __eq__(self, other):
        if not isinstance(other, Validator):
            return False
        return self.to_prompt() == other.to_prompt()

    # TODO: Make this a generic method on an abstract class
    def __stringify__(self):
        template = Template(
            """
            ${class_name} {
                rail_alias: ${rail_alias},
                on_fail: ${on_fail_descriptor},
                run_in_separate_process: ${run_in_separate_process},
                override_value_on_pass: ${override_value_on_pass},
                required_metadata_keys: ${required_metadata_keys},
                kwargs: ${kwargs}
            }"""
        )
        return template.safe_substitute(
            {
                "class_name": self.__class__.__name__,
                "rail_alias": self.rail_alias,
                "on_fail_descriptor": self.on_fail_descriptor,
                "run_in_separate_process": self.run_in_separate_process,
                "override_value_on_pass": self.override_value_on_pass,
                "required_metadata_keys": self.required_metadata_keys,
                "kwargs": self._kwargs,
            }
        )

    """
    This method allows the user to provide metadata to validators used in an LCEL chain.
    This is necessary because they can't pass metadata directly to `validate` in a chain
        because is called internally during `invoke`.

    Usage
    ---
    my_validator = Validator(args).with_metadata({ "key": "value" })

    chain = prompt | model | my_validator | output_parser
    chain.invoke({...})

    When called multiple times on the same validator instance,
        the metadata value will be override.
    This allows the user to change the metadata programmatically
        for different chains or calls.
    """

    def with_metadata(self, metadata: Dict[str, Any]):
        """Assigns metadata to this validator to use during validation."""
        self._metadata = metadata
        return self

    def to_runnable(self) -> Runnable:
        from guardrails.integrations.langchain.validator_runnable import (
            ValidatorRunnable,
        )

        return ValidatorRunnable(self)

    def _log_telemetry(self) -> None:
        """Logs telemetry after the validator is called."""

        if not self._disable_telemetry:
            # Get HubTelemetry singleton and create a new span to
            # log the validator inference
            used_guardrails_endpoint = (
                VALIDATOR_HUB_SERVICE in self.validation_endpoint and not self.use_local
            )
            used_custom_endpoint = not self.use_local and not used_guardrails_endpoint
            self._hub_telemetry.create_new_span(
                span_name="/validator_inference",
                attributes=[
                    ("validator_name", self.rail_alias),
                    ("used_remote_inference", not self.use_local),
                    ("used_local_inference", self.use_local),
                    ("used_guardrails_endpoint", used_guardrails_endpoint),
                    ("used_custom_endpoint", used_custom_endpoint),
                ],
                is_parent=False,  # This span will have no children
                has_parent=True,  # This span has a parent
            )


V = TypeVar("V", bound=Validator, covariant=True)
validators_registry: Dict[str, Type[Validator]] = {}
types_to_validators = defaultdict(list)


def validator_factory(name: str, validate: Callable) -> Type[Validator]:
    def validate_wrapper(self, *args, **kwargs):
        return validate(*args, **kwargs)

    validator = type(
        name,
        (Validator,),
        {"validate": validate_wrapper, "rail_alias": name},
    )
    return validator


def register_validator(
    name: str, data_type: Union[str, List[str]], has_guardrails_endpoint: bool = False
) -> Callable[[Union[Type[V], Callable]], Union[Type[V], Type[Validator]]]:
    """Register a validator for a data type."""
    from guardrails.datatypes import types_registry

    if isinstance(data_type, str):
        data_type = types_registry if data_type == "all" else [data_type]
    # Make sure that the data type string exists in the data types registry.
    for dt in data_type:
        if dt not in types_registry:
            raise ValueError(f"Data type {dt} is not registered.")

        types_to_validators[dt].append(name)

    def decorator(
        cls_or_func: Union[Type[V], Callable],
    ) -> Union[Type[V], Type[Validator]]:
        """Register a validator for a data type."""
        if isinstance(cls_or_func, type) and issubclass(cls_or_func, Validator):
            cls = cls_or_func
            cls.rail_alias = name
        elif callable(cls_or_func) and not isinstance(cls_or_func, type):
            func = cls_or_func
            func.rail_alias = name  # type: ignore
            # ensure function takes two args
            if not func.__code__.co_argcount == 2:
                raise ValueError(
                    f"Validator function {func.__name__} must take two arguments."
                )
            # dynamically create Validator subclass with `validate` method as `func`
            cls = validator_factory(name, func)
        else:
            raise ValueError(
                "Only functions and Validator subclasses "
                "can be registered as validators."
            )
        validators_registry[name] = cls
        return cls

    return decorator


def try_to_import_hub():
    try:
        # This should import everything and trigger registration
        # So it should only have to happen once
        # in lieu of completely unregistered validators
        import guardrails.hub  # noqa
    except ImportError:
        logger.error("Could not import hub. Validators may not work properly.")


# TODO: Move this to validator_utils.py
def get_validator_class(name: Optional[str]) -> Optional[Type[Validator]]:
    if not name:
        return None
    is_hub_validator = name.startswith(hub)
    validator_key = name.replace(hub, "") if is_hub_validator else name

    registration = validators_registry.get(validator_key)
    if not registration:
        try_to_import_hub()
        registration = validators_registry.get(validator_key)

    if not registration:
        warn(f"Validator with id {name} was not found in the registry!  Ignoring...")
        return None

    return registration
