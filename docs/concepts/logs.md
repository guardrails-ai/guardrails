# Inspecting logs

All `Guard` calls are logged internally, and can be accessed via the guard history.

## 🇻🇦 Accessing logs via `Guard.history`

`history` is an attribute of the `Guard` class. It implements a standard `Stack` interface with a few extra helper methods and properties.  For more information on our `Stack` implementation see the [Helper Classes](/api_reference/helper_classes) page.

Each entry in the history stack is a `Call` log which will contain information specific to a particular `Guard.__call__` or `Guard.parse` call in the order that they were executed within the current session.

For example, if you have a guard:

```py
my_guard = Guard.from_rail(...)
```

and you call it multiple times:

```py
response_1 = my_guard(...)

response_2 = my_guard.parse(...)
```

Then `guard.history` will have two call logs with the first representing the first call `response_1 = my_guard(...)` and the second representing the following `parse` call `response_2 = my_guard.parse(...)`.

To pretty print logs for the latest call, run:

```python
from rich import print

print(guard.history.last.tree)
```

![guard_state](../img/guard_history.png)

The `Call` log will contain initial and final information about a particular guard call.

```py
first_call = my_guard.history.first
```

For example, it tracks the initial inputs as provided:
```py
print(first_call.prompt)
```
```bash
INSERT SAMPLE HERE
```

as well as the final outputs:
```py
print(first_call.status) # The final status of this guard call
print(first_call.validated_output) # The final valid output of this guard call
```
```bash
INSERT SAMPLES HERE
```


The `Call` log also tracks cumulative values from any iterations that happen within the call.

For example, if the first response from the LLM fails validation and a reask occurs, the `Call` log can provide total tokens consumed (*currently only for OpenAI models), as well as access to all of the raw outputs from the LLM:
```py
print(first_call.prompt_tokens_consumed) # Total number of prompt tokens consumed across iterations within this call
print(first_call.completion_tokens_consumed) # Total number of completion tokens consumed across iterations within this call
print(first_call.tokens_consumed) # Total number of tokens consumed; equal to the sum of the two values above
print(first_call.raw_outputs) # An Stack of the LLM responses in order that they were received
```
```bash
INSERT SAMPLES HERE
```

For more information on `Call`, see the [History & Logs](/api_reference/history_and_logs/#guardrails.classes.history.Call) page.

## 🇻🇦 Accessing logs from individual steps
In addition to the cumulative values available directly on the `Call` log, it also contains a `Stack` of `Iteration`'s.  Each `Iteration` represent the logs from within a step in the guardrails process.  This includes the call to the LLM as well as parsing and validating the LLM's response.

Each `Iteration` is treated as a stateless entity so it will only contain information about the inputs and outputs of the particular step it represents.

For example, in order to see the raw LLM response as well as the logs for the specific validations that failed during the first step of a call, we can access this information via that steps `Iteration`:

```py
first_step = first_call.iterations.first
first_llm_output first_step.raw_output
print(first_llm_output)
validation_logs = first_step.validator_logs
print(validation_logs.json(indent=2))
```
```bash
INSERT SAMPLE HERE
```

```json
[
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
]
```

For more information on the properties available on `Iteration`, ee the [History & Logs](/api_reference/history_and_logs/#guardrails.classes.history.Iteration) page.