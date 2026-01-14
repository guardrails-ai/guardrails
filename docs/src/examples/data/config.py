"""
All guards defined here will be initialized, if and only if
the application is using in memory guards.

The application will use in memory guards if pg_host is left
undefined. Otherwise, a postgres instance will be started
and guards will be persisted into postgres. In that case,
these guards will not be initialized.
"""

from typing import Any, Callable, Dict, Optional, Union
from guardrails import Guard, OnFailAction
from guardrails.validators import (
    Validator,
    register_validator,
    FailResult,
    PassResult,
    ValidationResult,
)
from guardrails.hub import RegexMatch

name_case = Guard(name="name-case", description="Checks that a string is in Name Case format.").use(
    RegexMatch(regex="^(?:[A-Z][^\s]*\s?)+$", on_fail=OnFailAction.NOOP)
)

all_caps = Guard(name="all-caps", description="Checks that a string is all capital.").use(
    RegexMatch(regex="^[A-Z\\s]*$", on_fail=OnFailAction.NOOP)
)


@register_validator(name="custom/dynamic-enum", data_type="all")
class DynamicEnum(Validator):
    def __init__(
        self,
        enum_fetcher: Callable,
        on_fail: Optional[Union[Callable, OnFailAction]] = None,
    ):
        super().__init__(on_fail=on_fail, enum_fetcher=enum_fetcher)
        self.enum_fetcher = enum_fetcher

    def validate(self, value: Any, metdata: Optional[Dict] = {}) -> ValidationResult:
        enum_fetcher_args = metdata.get("enum_fetcher_args", [])
        dynamic_enum = self.enum_fetcher(*enum_fetcher_args)

        if value not in dynamic_enum:
            return FailResult(
                error_message="Value must be in the dynamically chosen enum!",
                fix_value=dynamic_enum[0],
            )
        return PassResult()


valid_topics = ["music", "cooking", "camping", "outdoors"]
invalid_topics = ["sports", "work", "ai"]
all_topics = [*valid_topics, *invalid_topics]


def custom_enum_fetcher(*args):
    topic_type = args[0]
    if topic_type == "valid":
        return valid_topics
    elif topic_type == "invalid":
        return invalid_topics
    return all_topics


custom_code_guard = Guard(
    name="custom",
    description="Uses a custom callable init argument for dynamic enum checks",
).use(DynamicEnum(custom_enum_fetcher, on_fail=OnFailAction.NOOP))
