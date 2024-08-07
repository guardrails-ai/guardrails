from typing import List, Optional

from guardrails_api_client.models.model_schema import ModelSchema
from guardrails_api_client.models.simple_types import SimpleTypes
from guardrails_api_client.models.validation_type import ValidationType
from guardrails_api_client.models.validator_reference import ValidatorReference

from guardrails.classes.output_type import OutputTypes
from guardrails.classes.schema.processed_schema import ProcessedSchema
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
    processed_schema.json_schema = ModelSchema(
        type=ValidationType(type), description=description
    ).to_dict()

    return processed_schema
