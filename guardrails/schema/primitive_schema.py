from typing import List, Optional

from guardrails_ai.types import Validator as ValidatorReference, JSONSchema

from guardrails.classes.output_type import OutputTypes
from guardrails.classes.schema.processed_schema import ProcessedSchema
from guardrails.types.simple import SimpleTypes
from guardrails.validator_base import Validator


def primitive_to_schema(
    validators: List[Validator],
    *,
    type: SimpleTypes = SimpleTypes.STRING,
    description: Optional[str] = None,
) -> ProcessedSchema:
    processed_schema = ProcessedSchema(validators=[], validator_map={})

    # TODO: Update when we support other primitive types
    processed_schema.output_type = OutputTypes.STRING

    processed_schema.validators = [
        ValidatorReference(
            id=v.rail_alias,
            on="$",
            on_fail=v.on_fail_descriptor,  # type: ignore
            kwargs=v.get_args(),
        )
        for v in validators
    ]
    processed_schema.validator_map = {"$": validators}
    processed_schema.json_schema = JSONSchema(
        type=type, description=description
    ).model_dump(exclude_none=True, by_alias=True)

    return processed_schema
