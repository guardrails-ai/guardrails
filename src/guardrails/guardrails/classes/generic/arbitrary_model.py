from pydantic import BaseModel, ConfigDict


class ArbitraryModel(BaseModel):
    """Empty Pydantic model with a config that allows arbitrary types."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
