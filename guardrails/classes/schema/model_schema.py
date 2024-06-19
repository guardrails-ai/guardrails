from typing import Any, Dict, Optional
from guardrails_api_client import ModelSchema as IModelSchema


# Because pydantic insists on including None values in the serialized dictionary
class ModelSchema(IModelSchema):
    def to_dict(self) -> Dict[str, Any]:
        super_dict = super().to_dict()
        return {k: v for k, v in super_dict.items() if v is not None}

    def from_dict(self, d: Dict[str, Any]) -> Optional["ModelSchema"]:
        i_model_schema = super().from_dict(d)

        if not i_model_schema:
            return i_model_schema

        trimmed = {k: v for k, v in i_model_schema.to_dict().items() if v is not None}

        return ModelSchema(**trimmed)
