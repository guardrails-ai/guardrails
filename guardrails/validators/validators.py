try:
    import nltk  # type: ignore
except ImportError:
    nltk = None  # type: ignore

if nltk is not None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


# @register_validator('required', 'all')
# class Required(Validator):
#     """Validates that a value is not None."""

#     def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> bool:
#         """Validates that a value is not None."""

#         return value is not None


# @register_validator('description', 'all')
# class Description(Validator):
#     """Validates that a value is not None."""

#     def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> bool:
#         """Validates that a value is not None."""

#         return value is not None
