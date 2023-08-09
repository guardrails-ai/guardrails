from dataclasses import dataclass
from typing import Any, Dict, Optional

import lxml.etree as ET


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
    }
    ignore_types = [
        "pydantic",
        "email",  # email and url should become string validators
        "url",
    ]

    type_string: str

    @property
    def type_object(self):
        if self.type_string in self.ignore_types:
            return Any
        return self.type_map[self.type_string]

    @staticmethod
    def verification_failed():
        # Sentinel value
        pass

    def verify(
        self,
        json_value,
        prune_extra_keys: bool,
        coerce_types: bool,
    ):
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
                return self.verification_failed
            try:
                return expected_type(json_value)
            except (ValueError, TypeError):
                return self.verification_failed
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

        # Separate out the choice keys from the schema keys.
        choice_keys = set()
        for key in schema_keys:
            if isinstance(self.children[key], ChoicePlaceholder):
                choice_keys.update(self.children[key].cases.keys())

        # Prune extra keys if necessary.
        extra_keys = json_keys - schema_keys - choice_keys
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
                if value is ValuePlaceholder.verification_failed:
                    return False
                json_value[key] = value
            elif isinstance(placeholder, ChoicePlaceholder):
                if not placeholder.verify(
                    json_value,
                    prune_extra_keys=prune_extra_keys,
                    coerce_types=coerce_types,
                ):
                    return False
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
                if value is ValuePlaceholder.verification_failed:
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
    name: str
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
        if self.name not in json_value:
            return False

        value_name = json_value[self.name]
        if value_name not in self.cases:
            return False
        if value_name not in json_value:
            return False
        if any(
            key in json_value and json_value[key] is not None
            for key in self.cases.keys()
            if key != value_name
        ):
            return False

        value_schema = self.cases[value_name]
        value = json_value[value_name]
        if isinstance(value_schema, ValuePlaceholder):
            value = value_schema.verify(
                value,
                prune_extra_keys=prune_extra_keys,
                coerce_types=coerce_types,
            )
            if value is ValuePlaceholder.verification_failed:
                return False
            json_value[value_name] = value
        else:
            if not value_schema.verify(
                value,
                prune_extra_keys=prune_extra_keys,
                coerce_types=coerce_types,
            ):
                return False

        return True


def generate_type_skeleton_from_schema(schema: ET._Element) -> DictPlaceholder:
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
                name=schema.attrib["name"],
                cases={
                    child.attrib["name"]: _recurse_schema(child[0])
                    for child in schema
                    if child.tag == "case"
                },
                optional=is_optional,
            )
        else:
            return ValuePlaceholder(
                type_string=schema.tag,
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
