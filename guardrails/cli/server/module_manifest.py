from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from guardrails.cli.server.serializeable import Serializeable


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


@dataclass
class ModuleManifest(Serializeable):
    author: Contributor
    maintainers: List[Contributor]
    repository: Repository
    namespace: str
    package_name: str
    module_name: str
    exports: List[str]
    tags: ModuleTags
    post_install: Optional[str] = None
    index: Optional[str] = None

    # @override
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            Contributor.from_dict(data.get("author", {})),
            [Contributor.from_dict(m) for m in data.get("maintainers", [])],
            Repository.from_dict(data.get("repository", {})),
            data.get("namespace"),  # type: ignore
            data.get("packageName"),  # type: ignore
            data.get("moduleName"),  # type: ignore
            data.get("exports"),  # type: ignore
            ModuleTags.from_dict(data.get("tags", {})),
            data.get("postInstall"),
            data.get("index"),
        )
