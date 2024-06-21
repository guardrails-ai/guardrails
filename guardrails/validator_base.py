# TODO:
#   - [ ] Rename this to just validator.py 0.5.x
#   - [ ] Maintain validator_base.py for exports but deprecate them
#   - [ ] Remove validator_base.py in 0.6.x

import inspect
import nltk
from collections import defaultdict
from string import Template
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from warnings import warn

from langchain_core.runnables import Runnable

from guardrails.classes import (
    ValidationResult,
    PassResult,  # noqa
    FailResult,
    ErrorSpan,  # noqa
)
from guardrails.constants import hub
from guardrails.types.on_fail import OnFailAction
from dataclasses import dataclass


# TODO: Use a different, lighter weight tokenizer
#   that doesn't require downloads during runtime
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


validators_registry: Dict[str, Type["Validator"]] = {}
types_to_validators = defaultdict(list)


def validator_factory(name: str, validate: Callable) -> Type["Validator"]:
    def validate_wrapper(self, *args, **kwargs):
        return validate(*args, **kwargs)

    validator = type(
        name,
        (Validator,),
        {"validate": validate_wrapper, "rail_alias": name},
    )
    return validator


def register_validator(name: str, data_type: Union[str, List[str]]):
    """Register a validator for a data type."""
    from guardrails.datatypes import types_registry

    if isinstance(data_type, str):
        data_type = types_registry if data_type == "all" else [data_type]
    # Make sure that the data type string exists in the data types registry.
    for dt in data_type:
        if dt not in types_registry:
            raise ValueError(f"Data type {dt} is not registered.")

        types_to_validators[dt].append(name)

    def decorator(cls_or_func: Union[Type[Validator], Callable]):
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


# TODO: Move this to validator_utils.py
def get_validator_class(name: Optional[str]) -> Optional[Type["Validator"]]:
    if not name:
        return None
    is_hub_validator = name.startswith(hub)
    validator_key = name.replace(hub, "") if is_hub_validator else name
    registration = validators_registry.get(validator_key)
    if not registration and name.startswith(hub):
        # This should import everything and trigger registration
        # So it should only have to happen once
        # in lieu of completely unregistered validators
        import guardrails.hub  # noqa

        return validators_registry.get(validator_key)

    if not registration:
        warn(f"Validator with id {name} was not found in the registry!  Ignoring...")
        return None

    return registration


@dataclass  # type: ignore
class Validator:
    """Base class for validators."""

    rail_alias: str = ""

    run_in_separate_process = False
    override_value_on_pass = False
    required_metadata_keys = []
    _metadata = {}

    def __init__(
        self, on_fail: Optional[Union[Callable, OnFailAction]] = None, **kwargs
    ):
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
            self.on_fail_method = on_fail

        # Store the kwargs for the validator.
        self._kwargs = kwargs

        assert (
            self.rail_alias in validators_registry
        ), f"Validator {self.__class__.__name__} is not registered. "

    def chunking_function(self, chunk: str):
        return split_sentence_str(chunk)

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        """Validates a value and return a validation result."""
        raise NotImplementedError

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
        splitcontents = self.chunking_function(accumulated_text)

        # if remainder kwargs is passed, validate remainder regardless
        remainder = kwargs.get("remainder", False)
        if remainder:
            splitcontents = [accumulated_text, ""]
        if len(splitcontents) == 0:
            return PassResult()
        [chunk_to_validate, new_accumulated_chunks] = splitcontents
        self.accumulated_chunks = [new_accumulated_chunks]
        # exclude last chunk, because it may not be a complete chunk
        validation_result = self.validate(chunk_to_validate, metadata)
        # if validate doesn't set validated chunk, we set it
        if validation_result.validated_chunk is None:
            validation_result.validated_chunk = chunk_to_validate
        return validation_result

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
            from guardrails.validator_service import ValidatorServiceBase

            validator_service = ValidatorServiceBase()
            return validator_service.perform_correction(
                [result], value, self, self.on_fail_descriptor
            )
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


# Superseded by guardrails/types/validator.py::PydanticValidatorSpec
ValidatorSpec = Union[Validator, Tuple[Union[Validator, str, Callable], str]]
