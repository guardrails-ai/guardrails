import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, Union

import regex

from guardrails.datatypes import (
    URL,
    Boolean,
    Case,
    Choice,
    DataType,
    Date,
    Email,
    Enum,
    Float,
    Integer,
)
from guardrails.datatypes import List as ListDataType
from guardrails.datatypes import (
    Object,
    PythonCode,
    String,
    Time,
    deprecated_string_types,
)
from guardrails.utils.parsing_utils import get_code_block, has_code_block


@dataclass
class Placeholder:
    optional: bool

    def verify(
        self,
        json_value,
        prune_extra_keys: bool,
        coerce_types: bool,
        validate_subschema: bool,
    ):
        if self.optional and json_value is None:
            return True
        return None


type_map: Dict[Type[DataType], Type] = {
    String: str,
    Integer: int,
    Float: float,
    Boolean: bool,
    Object: dict,
    ListDataType: list,
    Date: str,
    Time: str,
    Enum: str,
}

ignore_types = [
    Email,  # email and url should become string validators
    URL,
    PythonCode,
]


@dataclass
class ValuePlaceholder(Placeholder):
    datatype_type: Type[DataType]

    @property
    def type_object(self):
        if self.datatype_type in ignore_types:
            return Any
        return type_map[self.datatype_type]

    class VerificationFailed:
        # Sentinel value
        pass

    def verify(
        self,
        json_value,
        prune_extra_keys: bool,
        coerce_types: bool,
        validate_subschema: bool = False,
    ) -> Union[Type[VerificationFailed], Any]:
        super_result = super().verify(
            json_value,
            prune_extra_keys=prune_extra_keys,
            coerce_types=coerce_types,
            validate_subschema=validate_subschema,
        )
        if super_result is not None:
            return super_result
        expected_type = self.type_object
        if expected_type == Any:
            return json_value
        if not isinstance(json_value, expected_type):
            if not coerce_types:
                return self.VerificationFailed

            # don't coerce lists or objects to strings
            if isinstance(json_value, (list, dict)) and expected_type == str:
                return self.VerificationFailed

            try:
                return expected_type(json_value)
            except (ValueError, TypeError):
                return self.VerificationFailed
        return json_value


@dataclass
class DictPlaceholder(Placeholder):
    children: Dict[str, Any]

    def verify(
        self,
        json_value,
        prune_extra_keys: bool,
        coerce_types: bool,
        validate_subschema: bool = False,
    ) -> bool:
        # If json value is None, and the placeholder is optional, return True
        super_result = super().verify(
            json_value,
            prune_extra_keys=prune_extra_keys,
            coerce_types=coerce_types,
            validate_subschema=validate_subschema,
        )
        if super_result is not None:
            return super_result

        # If the json value is not a dict, return False
        if not isinstance(json_value, dict):
            return False

        # If expected dictionary does not specify any keys, then varification passes.
        if not self.children.keys():
            return True

        # Compare the keys in the json value to the keys in the schema.
        json_keys = set(json_value.keys())
        schema_keys = set(self.children.keys())

        # Prune extra keys if necessary.
        extra_keys = json_keys - schema_keys
        if prune_extra_keys and extra_keys:
            for key in extra_keys:
                del json_value[key]

        if not validate_subschema:
            # If the json value does not contain all the required keys, return False.
            if any(
                key not in json_keys and not self.children[key].optional
                for key in schema_keys
            ):
                return False
        # Else, when validating sub-schema, some keys may be missing, hence
        # we skip checking for all required keys.

        # Verify each key in the json value.
        for key, placeholder in self.children.items():
            if not validate_subschema:
                if placeholder.optional and key not in json_value:
                    continue
            else:
                # In sub-schema validation, some non-optional keys may be missing
                if key not in json_value:
                    continue

            if isinstance(placeholder, ValuePlaceholder):
                value = placeholder.verify(
                    json_value[key],
                    prune_extra_keys=prune_extra_keys,
                    coerce_types=coerce_types,
                )
                if value is ValuePlaceholder.VerificationFailed:
                    return False
                json_value[key] = value
            else:
                if not placeholder.verify(
                    json_value[key],
                    prune_extra_keys=prune_extra_keys,
                    coerce_types=coerce_types,
                    validate_subschema=validate_subschema,
                ):
                    return False

        return True


@dataclass
class ListPlaceholder(Placeholder):
    child: Optional[Placeholder]

    def verify(
        self,
        json_value,
        prune_extra_keys: bool,
        coerce_types: bool,
        validate_subschema: bool = False,
    ) -> bool:
        super_result = super().verify(
            json_value,
            prune_extra_keys=prune_extra_keys,
            coerce_types=coerce_types,
            validate_subschema=validate_subschema,
        )
        if super_result is not None:
            return super_result

        if not isinstance(json_value, list):
            return False

        if self.child is None:
            return True

        if isinstance(self.child, ValuePlaceholder):
            for i, item in enumerate(json_value):
                value = self.child.verify(
                    item,
                    prune_extra_keys=prune_extra_keys,
                    coerce_types=coerce_types,
                )
                if value is ValuePlaceholder.VerificationFailed:
                    return False
                json_value[i] = value
            return True
        else:
            for item in json_value:
                if not self.child.verify(
                    item,
                    prune_extra_keys=prune_extra_keys,
                    coerce_types=coerce_types,
                    validate_subschema=validate_subschema,
                ):
                    return False

        return True


@dataclass
class ChoicePlaceholder(Placeholder):
    discriminator: str
    cases: Dict[str, Any]

    def verify(
        self,
        json_value,
        prune_extra_keys: bool,
        coerce_types: bool,
        validate_subschema: bool = False,
    ) -> bool:
        super_result = super().verify(
            json_value,
            prune_extra_keys=prune_extra_keys,
            coerce_types=coerce_types,
            validate_subschema=validate_subschema,
        )
        if super_result is not None:
            return super_result

        if not isinstance(json_value, dict):
            return False
        if self.discriminator not in json_value:
            return False

        discriminator_value = json_value[self.discriminator]
        if discriminator_value not in self.cases:
            return False

        discriminator_schema = self.cases[discriminator_value]
        value = {k: v for k, v in json_value.items() if k != self.discriminator}

        if not isinstance(discriminator_schema, DictPlaceholder):
            raise ValueError("Choice cases must be objects")
        if self.discriminator in discriminator_schema.children:
            raise ValueError("Found name collision between discriminator and object")
        return discriminator_schema.verify(
            value,
            prune_extra_keys=prune_extra_keys,
            coerce_types=coerce_types,
            validate_subschema=validate_subschema,
        )


def generate_type_skeleton_from_schema(schema: Object) -> Placeholder:
    """Generate a JSON skeleton from an XML schema."""

    def _recurse_schema(schema: DataType):
        if isinstance(schema, Object):
            return DictPlaceholder(
                children={
                    name_: _recurse_schema(child_)
                    for name_, child_ in vars(schema.children).items()
                },
                optional=schema.optional,
            )
        if isinstance(schema, ListDataType):
            child_len = len(vars(schema.children).values())
            if not child_len:
                child_ = None
            elif child_len == 1:
                child_ = _recurse_schema(list(vars(schema.children).values())[0])
            else:
                raise ValueError("List must have exactly zero or one child")
            return ListPlaceholder(
                child=child_,
                optional=schema.optional,
            )
        if isinstance(schema, Choice):
            return ChoicePlaceholder(
                cases={
                    name_: DictPlaceholder(
                        children={
                            name__: _recurse_schema(child__)
                            for name__, child__ in vars(case.children).items()
                        },
                        optional=schema.optional,
                    )
                    for name_, case in vars(schema.children).items()
                    if isinstance(case, Case)
                },
                optional=schema.optional,
                discriminator=schema.discriminator_key,
            )
        else:
            datatype_type = type(schema)
            if schema.tag in deprecated_string_types:
                datatype_type = String
                warnings.warn(
                    f"""The '{schema.tag}' type is deprecated. Use the \
string type instead. Support for this type will \
be dropped in version 0.3.0 and beyond.""",
                    DeprecationWarning,
                )

            return ValuePlaceholder(
                datatype_type=datatype_type,
                optional=schema.optional,
            )

    return _recurse_schema(schema)


def verify_schema_against_json(
    schema: Object,
    generated_json: Dict[str, Any],
    prune_extra_keys: bool = False,
    coerce_types: bool = False,
    validate_subschema: bool = False,
):
    """Verify that a JSON schema is valid for a given XML."""

    type_skeleton = generate_type_skeleton_from_schema(schema)
    return type_skeleton.verify(
        generated_json,
        prune_extra_keys=prune_extra_keys,
        coerce_types=coerce_types,
        validate_subschema=validate_subschema,
    )


def extract_json_from_ouput(output: str) -> Tuple[Optional[Dict], Optional[Exception]]:
    # Find and extract json from code blocks
    extracted_code_block = output
    has_json_block, json_start, json_end = has_code_block(output, "json")
    if has_json_block and json_start is not None and json_end is not None:
        extracted_code_block = get_code_block(output, json_start, json_end, "json")
    else:
        has_block, block_start, block_end = has_code_block(output)
        if has_block and block_start is not None and block_end is not None:
            extracted_code_block = get_code_block(output, block_start, block_end)
        else:
            json_pattern = regex.compile(r"\{(?:[^{}]+|\{(?:(?R)|[^{}]+)*\})*\}")
            json_groups = json_pattern.findall(output)
            json_start, json_end = output.find("{"), output.rfind("}")
            if len(json_groups) > 0 and len(json_groups[0]) == (
                json_end - json_start + 1
            ):
                extracted_code_block = json_groups[0]

    # Treat the output as a JSON string, and load it into a dict.
    error = None
    try:
        output_as_dict = json.loads(extracted_code_block, strict=False)
    except json.decoder.JSONDecodeError as e:
        output_as_dict = None
        error = e
    return output_as_dict, error
