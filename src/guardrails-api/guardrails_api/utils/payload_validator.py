from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, jsonschema as jsonschema_ref
from guardrails_api.open_api_spec import get_open_api_spec
from guardrails_api.classes.http_error import HttpError
from guardrails_api.utils.remove_nones import remove_nones

api_spec = get_open_api_spec()
registry = Registry().with_resources(
    [
        (
            "urn:guardrails-api-spec",
            jsonschema_ref.DRAFT202012.create_resource(api_spec),
        )
    ]
)

guard_validator = Draft202012Validator(
    {
        "$ref": "urn:guardrails-api-spec#/components/schemas/Guard",
    },
    registry=registry,
)


def validate_payload(payload: dict):
    filtered_payload = remove_nones(payload)
    fields = {}
    error: ValidationError
    for error in guard_validator.iter_errors(filtered_payload):
        fields[error.json_path] = fields.get(error.json_path, [])
        fields[error.json_path].append(error.message)

    if fields:
        raise HttpError(
            400,
            "BadRequest",
            "The request payload did not match the required schema.",
            fields,
        )
