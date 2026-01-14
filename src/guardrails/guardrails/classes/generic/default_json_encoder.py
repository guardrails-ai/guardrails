from datetime import datetime
from dataclasses import asdict, is_dataclass
from pydantic import BaseModel
from json import JSONEncoder


class DefaultJSONEncoder(JSONEncoder):
    def default(self, o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        elif isinstance(o, BaseModel):
            return o.model_dump()
        elif is_dataclass(o):
            return asdict(o)
        elif isinstance(o, set):
            return list(o)
        elif isinstance(o, datetime):
            return o.isoformat()
        elif hasattr(o, "__dict__"):
            return o.__dict__
        return super().default(o)
