from datetime import datetime
import json
from typing import Any, Optional
import warnings

from guardrails.classes.generic.default_json_encoder import DefaultJSONEncoder


# TODO: What other common cases we should consider?
def serialize(val: Any) -> Optional[str]:
    try:
        return json.dumps(val, cls=DefaultJSONEncoder)
    except Exception as e:
        warnings.warn(str(e))
        return None


# We want to do the oppisite of what we did in the DefaultJSONEncoder
# TODO: What's a good way to expose a configurable API for this?
#       Do we wrap JSONDecoder with an extra layer to supply the original object?
def deserialize(original: Optional[Any], serialized: Optional[str]) -> Any:
    try:
        if original is None or serialized is None:
            return None

        loaded_val = json.loads(serialized)
        if isinstance(original, datetime):
            return datetime.fromisoformat(loaded_val)
        elif isinstance(original, set):
            return set(original)
        elif hasattr(original, "__class__"):
            # TODO: Handle nested classes
            # NOTE: nested pydantic classes already work
            if isinstance(loaded_val, dict):
                return original.__class__(**loaded_val)
            elif isinstance(loaded_val, list):
                return original.__class__(loaded_val)
        return loaded_val
    except Exception as e:
        warnings.warn(str(e))
        return None
