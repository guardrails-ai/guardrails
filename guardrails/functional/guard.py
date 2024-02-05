from string import Template
from typing import Any, Dict, List, Optional, Tuple, Type, Union, overload

from guardrails.classes.generic.stack import Stack
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.guard import Guard as OGuard
from guardrails.utils.safe_get import safe_get
from guardrails.validator_base import Validator


class Guard:
    validators: List[Validator]
    guard: Optional[OGuard[str]]

    def __init__(self):
        self.validators = []
        self.guard = None

    def __repr__(self):
        return self.guard.__repr__() if self.guard else ""

    def __rich_repr__(self):
        yield self.guard.__rich_repr__() if self.guard else ""

    def __stringify__(self):
        template = Template(
            """
            Guard {
                 validators: [
                    ${validators}
                ]
            }
                 """
        )
        return template.safe_substitute(
            {"validators": ",\n".join([v.__stringify__() for v in self.validators])}
        )

    def _get_validator(
        self,
        validator: Union[
            Validator,
            Type[Validator],
            Tuple[
                Type[Validator],
                Optional[Union[List[Any], Dict[str, Any]]],
                Optional[Dict[str, Any]],
            ],
        ],
        *args,
        **kwargs,
    ) -> Validator:
        invalid_error = ValueError(f"Invalid arguments! {validator}")
        if isinstance(validator, Validator):
            return validator
        elif isinstance(validator, Type):
            return validator(*args, **kwargs)
        elif isinstance(validator, Tuple):
            validator_cls = safe_get(validator, 0)
            args = safe_get(validator, 1, [])
            kwargs = {}
            if isinstance(args, Dict):
                kwargs = args
                args = []
            kwargs = safe_get(validator, 2, kwargs)
            if validator_cls:
                validator_inst = validator_cls(*args, **kwargs)
                return validator_inst
            raise invalid_error
        else:
            raise invalid_error

    @overload
    def add(self, validator: Validator) -> "Guard":
        ...

    @overload
    def add(self, validator: Type[Validator], *args, **kwargs) -> "Guard":
        ...

    def add(
        self, validator: Union[Validator, Type[Validator]], *args, **kwargs
    ) -> "Guard":
        if validator:
            self.validators.append(self._get_validator(validator, *args, **kwargs))

        return self

    # Or add_many; open to suggestions
    @overload
    def integrate(self, *validators: Validator) -> "Guard":
        ...

    @overload
    def integrate(
        self,
        *validators: Tuple[
            Type[Validator],
            Optional[Union[List[Any], Dict[str, Any]]],
            Optional[Dict[str, Any]],
        ],
    ) -> "Guard":
        ...

    def integrate(
        self,
        *validators: Union[
            Validator,
            Tuple[
                Type[Validator],
                Optional[Union[List[Any], Dict[str, Any]]],
                Optional[Dict[str, Any]],
            ],
        ],
    ) -> "Guard":
        for v in validators:
            self.validators.append(self._get_validator(v))

        return self

    def validate(self, llm_output: str, *args, **kwargs) -> ValidationOutcome[str]:
        if not self.guard:
            self.guard = OGuard.from_string(validators=self.validators)

        return self.guard.parse(llm_output=llm_output, *args, **kwargs)

    def __call__(self, llm_output: str, *args, **kwargs) -> ValidationOutcome[str]:
        return self.validate(llm_output, *args, **kwargs)

    @property
    def history(self):
        return self.guard.history if self.guard else Stack()
