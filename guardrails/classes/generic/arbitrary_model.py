from pydantic import BaseModel


class ArbitraryModel(BaseModel):
    """Empty Pydantic model with a config that allows arbitrary types and
    aliases."""

    model_config = {
        "validate_by_alias": True,
        "validate_by_name": True,
        "arbitrary_types_allowed": True,
    }
