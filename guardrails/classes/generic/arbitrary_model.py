from pydantic import BaseModel, ConfigDict


class ArbitraryModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
