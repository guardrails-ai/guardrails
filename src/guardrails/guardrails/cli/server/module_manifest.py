from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydash.strings import snake_case

from guardrails.classes.generic.serializeable import (
    Serializeable,
    SerializeableJSONEncoder,
)


@dataclass
class Contributor(Serializeable):
    name: str
    email: str


@dataclass
class Repository(Serializeable):
    url: str
    branch: Optional[str] = None


@dataclass
class ModuleTags(Serializeable):
    content_type: Optional[List[str]] = field(default_factory=list)
    validation_category: Optional[List[str]] = field(default_factory=list)
    process_requirements: Optional[List[str]] = field(default_factory=list)
    has_guardrails_endpoint: Optional[bool] = field(default_factory=bool)


@dataclass
class ModelAuth(Serializeable):
    type: str
    name: str
    displayName: Optional[str] = None


@dataclass
class ModuleManifest(Serializeable):
    id: str
    name: str
    author: Contributor
    maintainers: List[Contributor]
    repository: Repository
    namespace: str
    package_name: str
    module_name: str
    exports: List[str]
    tags: Optional[ModuleTags] = None
    requires_auth: Optional[bool] = True
    post_install: Optional[str] = None
    index: Optional[str] = None
    required_model_auth: Optional[List[ModelAuth]] = field(default_factory=list)

    # @override
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        init_kwargs = {snake_case(k): data.get(k) for k in data}
        init_kwargs["encoder"] = init_kwargs.get("encoder", SerializeableJSONEncoder)
        author = init_kwargs.pop("author", {})
        maintainers = init_kwargs.pop("maintainers", [])
        repository = init_kwargs.pop("repository", {})
        tags = init_kwargs.pop("tags", {})
        return cls(
            **init_kwargs,
            author=Contributor.from_dict(author),  # type: ignore
            maintainers=[Contributor.from_dict(m) for m in maintainers],  # type: ignore
            repository=Repository.from_dict(repository),  # type: ignore
            tags=ModuleTags.from_dict(tags),  # type: ignore
        )
