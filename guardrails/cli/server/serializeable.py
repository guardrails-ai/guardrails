import json
import inspect
from dataclasses import InitVar, asdict, dataclass, field, is_dataclass
from json import JSONEncoder
from typing import Any, Dict


class SerializeableJSONEncoder(JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


@dataclass
class Serializeable:
    encoder: InitVar[JSONEncoder] = field(kw_only=True, default=SerializeableJSONEncoder)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        annotations = inspect.get_annotations(cls)
        attributes = dict.keys(annotations)
        kwargs = { k: data.get(k) for k in data if k in attributes }
        return cls(**kwargs)
    
    @property
    def __dict__(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self, cls=self.encoder)