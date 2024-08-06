from typing import Any, Dict, List


class Filter:
    pass


def apply_filters(value: Any) -> Any:
    """Recursively filter out any values that are instances of Filter."""
    if isinstance(value, Filter):
        pass
    elif isinstance(value, List):
        # Cleaner syntax but requires two iterations
        # filtered_list = list(filter(None, map(apply_filters, value)))
        filtered_list = []
        for item in value:
            filtered_item = apply_filters(item)
            if filtered_item is not None:
                filtered_list.append(filtered_item)

        return filtered_list
    elif isinstance(value, Dict):
        # Cleaner syntax but requires two iterations
        # filtered_dict = {
        #     k: apply_filters(v)
        #     for k, v in value.items()
        #     if apply_filters(v)
        # }
        filtered_dict = {}
        for k, v in value.items():
            # Should we omit the key or just the value?
            filtered_value = apply_filters(v)
            if filtered_value is not None:
                filtered_dict[k] = filtered_value

        return filtered_dict
    else:
        return value
