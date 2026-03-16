from typing import Optional

from pydantic import BaseModel, Field


class ValidatorRegistryEntry(BaseModel):
    import_path: Optional[str] = Field(default=None)
    exports: list[str] = Field(default_factory=list)
    installed_at: Optional[str] = Field(default=None)
    package_name: Optional[str] = Field(default=None)


class ValidatorRegistry(BaseModel):
    version: int = Field(default=1)
    validators: dict[str, ValidatorRegistryEntry] = Field(default_factory=dict)
