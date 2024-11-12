from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import AliasChoices, AliasGenerator, AliasPath, BaseModel
from pydantic.fields import FieldInfo
from guardrails_api_client import ValidatorReference
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.schema.processed_schema import ProcessedSchema
from guardrails.logger import logger
from guardrails.types import (
    ModelOrListOfModels,
    ModelOrListOrDict,
    ModelOrModelUnion,
)
from guardrails.utils.safe_get import safe_get
from guardrails.utils.validator_utils import safe_get_validator
from guardrails.validator_base import Validator


def _resolve_alias(alias: Union[str, AliasPath, AliasChoices]) -> List[str]:
    aliases = []
    if isinstance(alias, str):
        aliases.append(alias)
    elif isinstance(alias, AliasPath):
        aliases.append(".".join(str(alias.path)))
    elif isinstance(alias, AliasChoices):
        for choice in alias.choices:
            aliases.extend(_resolve_alias(choice))

    return aliases


def _collect_aliases(
    field: Union[FieldInfo, AliasGenerator], field_name: str, model: Type[BaseModel]
) -> List[str]:
    aliases = []

    if field.alias:
        if isinstance(field.alias, str):
            aliases.append(field.alias)
        elif isinstance(field.alias, Callable):
            aliases.append(field.alias(field_name))

    if field.serialization_alias:
        if isinstance(field.serialization_alias, str):
            aliases.append(field.serialization_alias)
        elif isinstance(field.serialization_alias, Callable):
            aliases.append(field.serialization_alias(field_name))

    if field.validation_alias:
        if isinstance(field.validation_alias, Callable):
            aliases.extend(_resolve_alias(field.validation_alias(field_name)))
        else:
            aliases.extend(_resolve_alias(field.validation_alias))

    alias_generator = model.model_config.get("alias_generator")
    if alias_generator:
        if isinstance(alias_generator, Callable):
            aliases.append(alias_generator(field_name))
        elif isinstance(alias_generator, AliasGenerator):
            return _collect_aliases(alias_generator, field_name, model)

    return aliases


def is_base_model_type(any_type: Any) -> bool:
    try:
        inherits_from_base_model = issubclass(any_type, BaseModel)
        return inherits_from_base_model
    except TypeError:
        return False


def get_base_model(
    pydantic_class: ModelOrListOrDict,
) -> Tuple[ModelOrModelUnion, Any, Optional[Any]]:
    schema_model = pydantic_class
    type_origin = get_origin(pydantic_class)
    key_type_origin = None

    if type_origin is list:
        item_types = get_args(pydantic_class)
        if len(item_types) > 1:
            raise ValueError("List data type must have exactly one child.")
        item_type = safe_get(item_types, 0)
        if not item_type or not issubclass(item_type, BaseModel):
            raise ValueError("List item type must be a Pydantic model.")
        schema_model = item_type
    elif type_origin is dict:
        key_value_types = get_args(pydantic_class)
        value_type = safe_get(key_value_types, 1)
        key_type_origin = safe_get(key_value_types, 0)
        if not value_type or not issubclass(value_type, BaseModel):
            raise ValueError("Dict value type must be a Pydantic model.")
        schema_model = value_type
    elif type_origin is Union:
        union_members = get_args(pydantic_class)
        model_members = list(filter(is_base_model_type, union_members))
        if len(model_members) > 0:
            schema_model = Union[tuple(union_members)]  # type: ignore
            return (schema_model, type_origin, key_type_origin)

    if not is_base_model_type(schema_model):
        raise ValueError(
            "'output_class' must be of Type[pydantic.BaseModel]"
            " or List[Type[pydantic.BaseModel]]!"
        )

    return (schema_model, type_origin, key_type_origin)


def try_get_base_model(
    pydantic_class: ModelOrListOrDict,
) -> Tuple[Optional[Type[BaseModel]], Optional[Any], Optional[Any]]:
    try:
        model, type_origin, key_type_origin = get_base_model(pydantic_class)
        return (model, type_origin, key_type_origin)
    except ValueError:
        return (None, None, None)
    except TypeError:
        return (None, None, None)


def extract_union_member(
    member: Type,
    processed_schema: ProcessedSchema,
    json_path: str,
    aliases: List[str],
) -> Type:
    aliases = aliases or []
    field_model, field_type_origin, key_type_origin = try_get_base_model(member)
    if not field_model:
        return member
    if field_type_origin is Union:
        union_members = get_args(field_model)
        extracted_union_members = []
        for m in union_members:
            extracted_union_members.append(
                extract_union_member(m, processed_schema, json_path, aliases)
            )
        return Union[tuple(extracted_union_members)]  # type: ignore

    else:
        extracted_field_model = extract_validators(
            model=field_model,
            processed_schema=processed_schema,
            json_path=json_path,
            aliases=aliases,
        )
        if field_type_origin is list:
            return List[extracted_field_model]
        elif field_type_origin is dict:
            return Dict[key_type_origin, extracted_field_model]  # type: ignore
        return extracted_field_model


def extract_validators(
    model: Type[BaseModel],
    processed_schema: ProcessedSchema,
    json_path: str,
    aliases: Optional[List[str]] = None,
) -> Type[BaseModel]:
    aliases = aliases or []
    for field_name in model.model_fields:
        alias_paths = []
        field_path = f"{json_path}.{field_name}"
        # alias_paths.append(field_path)
        for alias_path in aliases:
            alias_paths.append(f"{alias_path}.{field_name}")
        field: FieldInfo = model.model_fields[field_name]
        for alias in _collect_aliases(field, field_name, model):
            alias_paths.append(f"{json_path}.{alias}")
            for alias_path in aliases:
                alias_paths.append(f"{alias_path}.{alias}")

        if field.json_schema_extra is not None and isinstance(
            field.json_schema_extra, dict
        ):
            # NOTE: It's impossible to copy a class type so using
            #   'pop' here mutates the original Pydantic Model.
            # Using 'get' adds a pointless 'validators' field to the
            #   json schema but that doesn't break anything.
            validators = field.json_schema_extra.get("validators", [])

            if not isinstance(validators, list) and not isinstance(
                validators, Validator
            ):
                logger.warning(
                    f"Invalid value assigned to {field_name}.validators! {validators}"
                )
                continue
            validator_instances: List[Validator] = []

            # Only for backwards compatibility
            if isinstance(validators, Validator):
                validator_instances.append(validators)
            else:
                validator_list = [
                    safe_get_validator(v)  # type: ignore
                    for v in validators
                ]
                validator_instances.extend([v for v in validator_list if v is not None])
            all_paths = [field_path]
            all_paths.extend(alias_paths)
            for path in all_paths:
                entry = processed_schema.validator_map.get(path, [])
                entry.extend(validator_instances)
                processed_schema.validator_map[path] = entry
                validator_references = [
                    ValidatorReference(
                        id=v.rail_alias,
                        on=path,
                        on_fail=v.on_fail_descriptor,  # type: ignore
                        kwargs=v.get_args(),
                    )
                    for v in validator_instances
                ]
                processed_schema.validators.extend(validator_references)
        if field.annotation:
            field_model, field_type_origin, key_type_origin = try_get_base_model(
                field.annotation
            )
            if field_model:
                if field_type_origin is Union:
                    union_members = list(get_args(field_model))
                    extracted_union_members = []
                    for m in union_members:
                        extracted_union_members.append(
                            extract_union_member(
                                m,
                                processed_schema=processed_schema,
                                json_path=field_path,
                                aliases=alias_paths,
                            )
                        )

                    model.model_fields[field_name].annotation = Union[  # type: ignore
                        tuple(extracted_union_members)  # type: ignore
                    ]
                else:
                    extracted_field_model = extract_validators(
                        model=field_model,
                        processed_schema=processed_schema,
                        json_path=field_path,
                        aliases=alias_paths,
                    )
                    if field_type_origin is list:
                        model.model_fields[field_name].annotation = List[
                            extracted_field_model
                        ]
                    elif field_type_origin is dict:
                        model.model_fields[field_name].annotation = Dict[
                            key_type_origin, extracted_field_model  # type: ignore
                        ]
                    else:
                        model.model_fields[
                            field_name
                        ].annotation = extracted_field_model  # noqa
    return model


def pydantic_to_json_schema(
    pydantic_class: Type[BaseModel], type_origin: Optional[Any] = None
) -> Dict[str, Any]:
    # Convert Pydantic model to JSON schema
    json_schema = pydantic_class.model_json_schema()
    json_schema["title"] = pydantic_class.__name__

    if type_origin is list:
        json_schema = {
            "title": f"Array<{json_schema.get('title')}>",
            "type": "array",
            "items": json_schema,
        }

    return json_schema


def pydantic_model_to_schema(
    pydantic_class: ModelOrListOfModels,
) -> ProcessedSchema:
    processed_schema = ProcessedSchema(validators=[], validator_map={})

    schema_model, type_origin, _key_type_origin = get_base_model(pydantic_class)

    processed_schema.output_type = (
        OutputTypes.LIST if type_origin is list else OutputTypes.DICT
    )

    model = extract_validators(schema_model, processed_schema, "$")
    json_schema = pydantic_to_json_schema(model, type_origin)
    processed_schema.json_schema = json_schema

    return processed_schema
