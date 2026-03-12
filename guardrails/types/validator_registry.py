from pydantic import BaseModel


class ValidatorRegistryEntry(BaseModel):
    import_path: str
    exports: list[str]
    installed_at: str
    package_name: str


class ValidatorRegistry(BaseModel):
    version: int
    validators: dict[str, ValidatorRegistryEntry]
