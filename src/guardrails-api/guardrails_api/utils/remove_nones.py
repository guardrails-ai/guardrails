from typing import Any, Dict, List


def remove_nones(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    filtered = {}
    for key, value in list(dictionary.items()):
        if isinstance(value, Dict):
            filtered[key] = remove_nones(value)
        elif isinstance(value, List):
            filtered_list = []
            for v in value:
                list_value = v
                if isinstance(v, Dict):
                    list_value = remove_nones(v)
                if list_value is not None:
                    filtered_list.append(list_value)
            filtered[key] = filtered_list
        elif value is not None:
            filtered[key] = dictionary[key]
    return filtered
