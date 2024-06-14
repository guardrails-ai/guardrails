import json
from typing import Union

from jsonformer import Jsonformer

from guardrails.formatters.base_formatter import BaseFormatter
from guardrails.llm_providers import (
    ArbitraryCallable,
    HuggingFacePipelineCallable,
    HuggingFaceModelCallable,
)


def _deref_schema_path(schema: dict, path: Union[list, str]):
    # schema is assumed to be the "$defs" field from JSONSchema, {"$defs": ...}.
    if isinstance(path, str):
        path = path.split("/")
    if path[0] == "#":
        assert "$defs" in schema
        return _deref_schema_path(schema, path[1:])
    if len(path) == 1:
        return schema[path[0]]
    else:
        return _deref_schema_path(schema[path[0]], path[1:])


def _jsonschema_to_jsonformer(
    schema: dict, path: list = None, objdefs: dict = None
) -> dict:
    """Converts the large-ish JSONSchema standard into the JSONFormer schema format.
    These are mostly identical, but the jsonschema supports '$defs' and '$ref'.
    There's an additional inconsistency in the use 'integer' versus 'number'.
    ```
    jsonschema_style =
    {
        '$defs': {
            'Obs': {
                'type': 'object',
                'properties': {'blah': {'type': 'integer'}}}
            },
        'type': 'object'
        'properties': {
            's': {'type': 'string'},
            'i': {'type': 'integer'},
            'b': {'type': 'boolean'},
            'a': {'type': 'array', 'items': {'type': 'integer'}},
            'o': {'$ref': '#/$defs/Obs'}
        },
    }

    jsonformer_style =
    {
        "type": "object",
        "properties": {
            "s": {"type": "string"},
            "i": {"type": "number"},
            "b": {"type": "boolean"},
            "a": {"type": "array", "items": {"type": "string"}}
            "o": {"type": "object", "properties": ...},
        }
    }
    ```
    """
    if path is None:
        path = []
    if objdefs is None:
        objdefs = {"$defs": dict()}

    # We may get something we don't expect...
    if not isinstance(schema, dict):
        raise Exception(
            f"Error: could not convert/parse base schema. Encountered `{schema}`"
        )

    if "$defs" in schema:
        # We have some sub-schemas defined here.  We need to convert them.
        # We may also need to handle sub-schema defs.
        # For now, build a quick tree in the defs.
        current = objdefs["$defs"]
        for step in path:
            if step not in current:
                current[step] = dict()
            current = current[step]
        current.update(schema["$defs"])

    result = dict()
    for k, v in schema.items():
        # Convert {"type": "integer"} to {"type": "number"} float is already 'number'.
        if k == "type" and v == "integer":
            result["type"] = "number"
        elif k == "type" and v == "object":
            result["type"] = "object"
            result["properties"] = dict()
            for subkey, subvalue in schema["properties"].items():  # Must be present.
                path.append(subkey)
                result["properties"][subkey] = _jsonschema_to_jsonformer(
                    subvalue,
                    path,
                    objdefs,
                )
                assert path.pop() == subkey
        elif k == "type" and v == "array":
            result["type"] = "array"
            result["items"] = _jsonschema_to_jsonformer(schema["items"], path, objdefs)
        elif k == "$ref":
            result = _jsonschema_to_jsonformer(
                _deref_schema_path(objdefs, v), path, objdefs
            )
        else:
            result[k] = v
    return result


class JsonFormatter(BaseFormatter):
    def __init__(self, schema: dict):
        self.output_schema = _jsonschema_to_jsonformer(schema)

    def wrap_callable(self, llm_callable) -> ArbitraryCallable:
        # JSON Schema enforcement experiment.
        if isinstance(llm_callable, HuggingFacePipelineCallable):
            model = llm_callable.init_kwargs["pipeline"]
            return ArbitraryCallable(
                lambda p: json.dumps(
                    Jsonformer(
                        model=model.model,
                        tokenizer=model.tokenizer,
                        json_schema=self.output_schema,
                        prompt=p,
                    )()
                )
            )
        elif isinstance(llm_callable, HuggingFaceModelCallable):
            # This will not work because 'model_generate' is the .gen method.
            # model = self.api.init_kwargs["model_generate"]
            # Use the __self__ to grab the base mode for passing into JF.
            model = llm_callable.init_kwargs["model_generate"].__self__
            tokenizer = llm_callable.init_kwargs["tokenizer"]
            return ArbitraryCallable(
                lambda p: json.dumps(
                    Jsonformer(
                        model=model,
                        tokenizer=tokenizer,
                        json_schema=self.output_schema,
                        prompt=p,
                    )()
                )
            )
        else:
            raise ValueError(
                "JsonFormatter can only be used with HuggingFace*Callable."
            )
