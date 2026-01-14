import re
import string
from typing import Any, Callable, Dict, Optional

import rstr

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="regex_match", data_type="string")
class RegexMatch(Validator):
    """Validates that a value matches a regular expression.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `regex_match`                     |
    | Supported data types          | `string`                          |
    | Programmatic fix              | Generate a string that matches the regular expression |

    Args:
        regex: Str regex pattern
        match_type: Str in {"search", "fullmatch"} for a regex search or full-match option
    """  # noqa

    def __init__(
        self,
        regex: str,
        match_type: Optional[str] = None,
        on_fail: Optional[Callable] = None,
    ):
        # todo -> something forces this to be passed as kwargs and therefore xml-ized.
        # match_types = ["fullmatch", "search"]
        if match_type is None:
            match_type = "fullmatch"
        assert match_type in [
            "fullmatch",
            "search",
        ], 'match_type must be in ["fullmatch", "search"]'

        super().__init__(
            on_fail=on_fail,
            match_type=match_type,
            regex=regex,
        )
        self._regex = regex
        self._match_type = match_type

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        p = re.compile(self._regex)
        """Validates that value matches the provided regular expression."""
        # Pad matching string on either side for fix
        # example if we are performing a regex search
        str_padding = "" if self._match_type == "fullmatch" else rstr.rstr(string.ascii_lowercase)
        self._fix_str = str_padding + rstr.xeger(self._regex) + str_padding

        if not getattr(p, self._match_type)(value):
            return FailResult(
                error_message=f"Result must match {self._regex}",
                fix_value=self._fix_str,
            )
        return PassResult()

    def to_prompt(self, with_keywords: bool = True) -> str:
        return "results should match " + self._regex
