from typing import Any, Dict, Optional
from guardrails_api_client import ModelSchema as IModelSchema, ValidationType


# Because pydantic insists on including None values in the serialized dictionary
class ModelSchema(IModelSchema):
    def to_dict(self) -> Dict[str, Any]:
        super_dict = super().to_dict()
        return {k: v for k, v in super_dict.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional["ModelSchema"]:
        i_model_schema = super().from_dict(d)

        if not i_model_schema:
            return i_model_schema

        trimmed = {k: v for k, v in i_model_schema.to_dict().items() if v is not None}

        output_schema_type = trimmed.get("type")
        if output_schema_type:
            trimmed["type"] = ValidationType.from_dict(output_schema_type)

        return cls(**trimmed)
