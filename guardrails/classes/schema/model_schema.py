from typing import Any, Dict, Optional
from guardrails_api_client import ModelSchema as IModelSchema, ValidationType


# Because pydantic insists on including None values in the serialized dictionary
class ModelSchema(IModelSchema):
    def to_dict(self) -> Dict[str, Any]:
        super_dict = super().to_dict()
        return {k: v for k, v in super_dict.items() if v is not None}

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> "ModelSchema":
        if not obj:
            obj = {"type": "string"}

        i_model_schema = super().from_dict(obj)

        i_model_schema_dict = (
            i_model_schema.to_dict() if i_model_schema else {"type": "string"}
        )

        trimmed = {k: v for k, v in i_model_schema_dict.items() if v is not None}

        output_schema_type = trimmed.get("type")
        if output_schema_type:
            trimmed["type"] = ValidationType.from_dict(output_schema_type)  # type: ignore

        return cls(**trimmed)  # type: ignore
