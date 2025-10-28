# Migrating to 0.3.0

The 0.3.0 release contains a handful of breaking changes compared to the previous 0.2.x releases.  This guide will list out these changes as well as how to migrate to the new features that encompass them in 0.3.x.

## Guardrails Response Object
_Validation Outcome_

Previous when calling `__call__` or `parse` on a Guard, the Guard would return a tuple of the raw llm output and the validated output or just the validated output respecitvely.

Now, in order to communicate more information, we respond with a `ValidationOutcome` class that contains the above information and more. See [ValidationOutcome](/docs/api_reference_markdown/guards#validationoutcome) in the API Reference for more information on these additioanl properties.

In order to limit how much this changes breaks the current experience, we made this class iterable so you can still deconstruct its properties.

Whereas in `0.2.x` you would receive the response like so:
```py
guard = Guard(...)

raw_output, validated_output = guard(...)
# or
validated_output = guard.parse(...)
```

You can now keep a similar experience by adding a rest argument at the end:
```py
guard = Guard(...)

raw_output, validated_output, *rest = guard(...)
# or
raw_output, validated_output, *rest = guard.parse(...)
```

You can also simple use the response in its object form:
```py
guard = Guard(...)

response = guard(...)
# or
response = guard.parse(...)

validated_output = response.validated_output
```

One new property that we want to highlight on this return structure is `validation_passed` field.  This field is a boolean that will be `True` if the guard process resulted in valid output and `False` otherwise.  In conjunction with this, `ValidationOutcome.validated_output` will only have a value if validation succeeded.  If the end result was not valid and either was or still contained reasks, `validated_output` will be none and the invalid output will be captured on `ValidationOutcome.reask`.

## History & Logs Improvements
If you're familiar with Guardrails, then you might have used the `Guard.state` property to inspect how the Guard process behaved over time.  In order to make the Guard process more transparent, as part of `v0.3.0` we redesigned how you access this information.  

Now, on a Guard, you can access logs related to any `__call__` or `parse` call within the current session via `Guard.history`.  We documented this new structure and how it works [here](/docs/concepts/logs), but to give a quick example of the differences, where before if you needed to check if validation of a particualr step succeeded or not you might need to do something like this:
```py
guard_history = guard.state.most_recent_call
last_step_logs: GuardLogs = guard_history.history[-1]
validation_logs = last_step_logs.field_validation_logs.validator_logs
failed_validations = list([log for log in validation_logs if log.validation_result.outcome == 'fail'])
validation_succeeded = len(failed_validations) == 0
```

now you can check via:
```py
guard.history.last.status
```

Another quick example, in order to check your total token consumption in `v0.2.0`:
```py
latest_call = guard.state.most_recent_call

total_token_count = 0
for log in latest_call.history:
    total_token_count = total_token_count + log.llm_response.prompt_token_count
    total_token_count = total_token_count + log.llm_response.response_token_count
```

Now, in `v0.3.0` you can simply:
```py
guard.history.last.tokens_consumed
```

Besides the examples above, if you dive deeper into the new history structure you can find more insights into exactly how the LLM was called in each step of the process.  See [here](/docs/concepts/logs) for more details.