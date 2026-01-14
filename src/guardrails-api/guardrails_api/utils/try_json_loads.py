import json


def try_json_loads(val):
    try:
        json_val = json.loads(val)
        return json_val
    except Exception:
        return val
