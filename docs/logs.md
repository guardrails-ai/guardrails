# Inspecting logs

All `gd.Guard` calls are logged internally, and can be accessed via two methods, `gd.Guard.guard_state` or `guardrails.log`.

## 🪵 Accessing logs via `guardrails.log`

This is the simplest way to access logs. It returns a list of all `gd.Guard` calls, in the order they were made.

In order to access logs, run:

```bash

eliot-tree --output-format=ascii guardrails.log

```

## 🇻🇦 Accessing logs via `gd.Guard.guard_state`

`guard_state` is an attribute of the `gd.Guard` class. It contains:

1. A list of all `gd.Guard` calls, in the order they were made.
2. For each call, reasks needed and their results.

To pretty print logs, run:

```python
from rich import print

print(guard.state.most_recent_call.tree)
```

![guard_state](img/guard_history.png)

To access fine-grained logs on field validation, see the FieldValidationLogs object:

```python
from pydantic import BaseModel
import dataclasses
import json


def to_dict(obj):
    if isinstance(obj, BaseModel):
        return obj.dict()
    elif isinstance(obj, list):
        return [to_dict(e) for e in obj]
    elif isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    else:
        return obj


validation_logs = guard.guard_state.all_histories[0].history[0].field_validation_logs
print(json.dumps(to_dict(validation_logs), indent=2))
```
```json
{
  "validator_logs": [],
  "children": {
    "name": {
      "validator_logs": [
        {
          "validator_name": "TwoWords",
          "value_before_validation": "peter parker the second",
          "validation_result": {
            "outcome": "fail",
            "metadata": null,
            "error_message": "must be exactly two words",
            "fix_value": "peter parker"
          },
          "value_after_validation": {
            "incorrect_value": "peter parker the second",
            "fail_results": [
              {
                "outcome": "fail",
                "metadata": null,
                "error_message": "must be exactly two words",
                "fix_value": "peter parker"
              }
            ],
            "path": [
              "name"
            ]
          }
        }
      ],
      "children": {}
    }
  }
}


```