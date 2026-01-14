from typing import Any, Dict
from guardrails_api_client import ValidatorReference as IValidatorReference

from guardrails.utils.serialization_utils import to_dict


# Docs only
class ValidatorReference(IValidatorReference):
    """ValidatorReference is a serialized reference for constructing a
    Validator.

    Attributes:
        id (Optional[str]): The unique identifier for this Validator.
            Often the hub id; e.g. guardrails/regex_match.  Default None.
        on (Optional[str]): A reference to the property this validator should be
            applied against.  Can be a valid JSON path or a meta-property
            such as `prompt` or `output`. Default None.
        on_fail (Optional[str]): The OnFailAction to apply during validation.
            Default None.
        args (Optional[List[Any]]): Positional arguments. Default None.
        kwargs (Optional[Dict[str, Any]]): Keyword arguments. Default None.
    """

    @classmethod
    def from_interface(cls, interface: IValidatorReference) -> "ValidatorReference":
        """Create a ValidatorReference from an interface."""
        return cls(
            id=interface.id,
            on=interface.on,
            on_fail=interface.on_fail,  # type: ignore
            args=interface.args,
            kwargs=interface.kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        ref_dict = super().to_dict()

        # serialize args and kwargs
        if self.args:
            ref_dict["args"] = [to_dict(a) for a in self.args]

        if self.kwargs:
            ref_dict["kwargs"] = {k: to_dict(v) for k, v in self.kwargs.items()}

        return ref_dict
