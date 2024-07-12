import json
from typing import Any, Dict


def try_to_json(value: Any):
    try:
        json.dumps(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def extract_serializeable_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {k: metadata[k] for k in metadata if try_to_json(metadata[k])}
