from guardrails_api_client import ValidatorReference as IValidatorReference


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
