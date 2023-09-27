import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, Union

import lxml.etree as ET

from guardrails.utils.parsing_utils import get_code_block, has_code_block

logger = logging.getLogger(__name__)


@dataclass
class Placeholder:
    optional: bool

    def verify(
        self,
        json_value,
        prune_extra_keys: bool,
        coerce_types: bool,
    ):
        if self.optional and json_value is None:
            return True
        return None


# TODO - deprecate these altogether
deprecated_string_types = {"sql", "email", "url", "pythoncode"}


@dataclass
class ValuePlaceholder(Placeholder):
    type_map = {
        "string": str,
        "integer": int,
        "float": float,
        "bool": bool,
        "object": dict,
        "list": list,
        "date": str,
        "time": str,
        "email": str,  # email and url should become string validators
        "url": str,
        "pythoncode": str,
        "sql": str,
    }
    ignore_types = [
        "pydantic",
    ]

    type_string: str

    @property
    def type_object(self):
        if self.type_string in self.ignore_types:
            return Any
        return self.type_map[self.type_string]

    class VerificationFailed:
        # Sentinel value
        pass

    def verify(
        self,
        json_value,
        prune_extra_keys: bool,
        coerce_types: bool,
    ) -> Union[Type[VerificationFailed], Any]:
        super_result = super().verify(
            json_value,
            prune_extra_keys=prune_extra_keys,
            coerce_types=coerce_types,
        )
        if super_result is not None:
            return super_result
        expected_type = self.type_object
        if expected_type == Any:
            return json_value
        if not isinstance(json_value, expected_type):
            if not coerce_types:
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
    ) -> bool:
        # If json value is None, and the placeholder is optional, return True
        super_result = super().verify(
            json_value,
            prune_extra_keys=prune_extra_keys,
            coerce_types=coerce_types,
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

        # If the json value does not contain all the required keys, return False.
        if any(
            key not in json_keys and not self.children[key].optional
            for key in schema_keys
        ):
            return False

        # Verify each key in the json value.
        for key, placeholder in self.children.items():
            if placeholder.optional and key not in json_value:
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
    ) -> bool:
        super_result = super().verify(
            json_value,
            prune_extra_keys=prune_extra_keys,
            coerce_types=coerce_types,
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
    ) -> bool:
        super_result = super().verify(
            json_value,
            prune_extra_keys=prune_extra_keys,
            coerce_types=coerce_types,
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
        )


def generate_type_skeleton_from_schema(schema: ET._Element) -> Placeholder:
    """Generate a JSON skeleton from an XML schema."""

    def _recurse_schema(schema):
        is_optional = schema.attrib.get("required", "true") == "false"
        if schema.tag == "object":
            return DictPlaceholder(
                children={
                    child.attrib["name"]: _recurse_schema(child) for child in schema
                },
                optional=is_optional,
            )
        elif schema.tag == "list":
            if len(schema) == 0:
                child = None
            elif len(schema) == 1:
                child = _recurse_schema(schema[0])
            else:
                raise ValueError("List must have exactly zero or one child")
            return ListPlaceholder(
                child=child,
                optional=is_optional,
            )
        elif schema.tag == "choice":
            return ChoicePlaceholder(
                cases={
                    case.attrib["name"]: DictPlaceholder(
                        children={
                            child.attrib["name"]: _recurse_schema(child)
                            for child in case
                        },
                        optional=is_optional,
                    )
                    for case in schema
                    if case.tag == "case"
                },
                optional=is_optional,
                discriminator=schema.attrib["discriminator"],
            )
        else:
            type_string = schema.tag
            if schema.tag in deprecated_string_types:
                warnings.warn(
                    f"""The '{schema.tag}' type is deprecated. Use the \
string type instead. Support for this type will \
be dropped in version 0.3.0 and beyond.""",
                    DeprecationWarning,
                )
                type_string = "string"

            return ValuePlaceholder(
                type_string=type_string,
                optional=is_optional,
            )

    return DictPlaceholder(
        children={child.attrib["name"]: _recurse_schema(child) for child in schema},
        optional=False,
    )


def verify_schema_against_json(
    xml_schema: ET._Element,
    generated_json: Dict[str, Any],
    prune_extra_keys: bool = False,
    coerce_types: bool = False,
):
    """Verify that a JSON schema is valid for a given XML."""

    type_skeleton = generate_type_skeleton_from_schema(xml_schema)
    return type_skeleton.verify(
        generated_json,
        prune_extra_keys=prune_extra_keys,
        coerce_types=coerce_types,
    )


def extract_json_from_ouput(output: str) -> Tuple[Dict, Optional[Exception]]:
    # Find and extract json from code blocks
    extracted_code_block = output
    has_json_block, json_start, json_end = has_code_block(output, "json")
    if has_json_block:
        extracted_code_block = get_code_block(output, json_start, json_end, "json")
    else:
        has_block, block_start, block_end = has_code_block(output)
        if has_block:
            extracted_code_block = get_code_block(output, block_start, block_end)

    # Treat the output as a JSON string, and load it into a dict.
    error = None
    try:
        output_as_dict = json.loads(extracted_code_block, strict=False)
    except json.decoder.JSONDecodeError as e:
        output_as_dict = None
        error = e
    return output_as_dict, error
