import inspect
import json
import sys
from dataclasses import InitVar, asdict, dataclass, field, is_dataclass
from json import JSONEncoder
from typing import Any, Dict

from pydash.strings import snake_case


def get_annotations(obj):
    if sys.version_info.minor >= 10 and hasattr(inspect, "get_annotations"):
        return inspect.get_annotations(obj)  # type: ignore
    else:
        return obj.__annotations__


class SerializeableJSONEncoder(JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


encoder_kwargs = {}
if sys.version_info.minor >= 10:
    encoder_kwargs["kw_only"] = True
    encoder_kwargs["default"] = SerializeableJSONEncoder


@dataclass
class Serializeable:
    encoder: InitVar[JSONEncoder] = field(**encoder_kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        annotations = get_annotations(cls)
        attributes = dict.keys(annotations)
        snake_case_kwargs = {
            snake_case(k): data.get(k) for k in data if snake_case(k) in attributes
        }
        snake_case_kwargs["encoder"] = snake_case_kwargs.get(
            "encoder", SerializeableJSONEncoder
        )
        return cls(**snake_case_kwargs)  # type: ignore

    @property
    def __dict__(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self):
        return json.dumps(self, cls=self.encoder)  # type: ignore
