# Inspect Guard Run History and Logs

All `Guard` calls are logged internally, and can be accessed via the guard history.

## Accessing logs via `Guard.history`

`history` is an attribute of the `Guard` class. It implements a standard `Stack` interface with a few extra helper methods and properties.  For more information on our `Stack` implementation see the [Helper Classes](/docs/api_reference/helper_classes.md) page.

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
--8<--

docs/html/single-step-history.html

--8<--

The `Call` log will contain initial and final information about a particular guard call.

```py
first_call = my_guard.history.first
```

For example, it tracks the initial inputs as provided:
```py
print("prompt\n-----")
print(first_call.prompt)
print("prompt params\n------------- ")
print(first_call.prompt_params)
```
```log
prompt
-----

You are a human in an enchanted forest. You come across opponents of different types. You should fight smaller opponents, run away from bigger ones, and freeze if the opponent is a bear.

You run into a ${opp_type}. What do you do?

${gr.complete_json_suffix_v2}


Here are a few examples

goblin: {"action": {"chosen_action": "fight", "weapon": "crossbow"}}
troll: {"action": {"chosen_action": "fight", "weapon": "sword"}}
giant: {"action": {"chosen_action": "flight", "flight_direction": "north", "distance": 1}}
dragon: {"action": {"chosen_action": "flight", "flight_direction": "south", "distance": 4}}
black bear: {"action": {"chosen_action": "freeze", "duration": 3}}
beets: {"action": {"chosen_action": "fight", "weapon": "fork"}}

prompt params
------------- 
{'opp_type': 'grizzly'}
```

as well as the final outputs:
```py
print("status: ", first_call.status) # The final status of this guard call
print("validated response:", first_call.validated_output) # The final valid output of this guard call
```
```log
status:  pass
validated response: {'action': {'chosen_action': 'freeze', 'duration': 3}}
```


The `Call` log also tracks cumulative values from any iterations that happen within the call.

For example, if the first response from the LLM fails validation and a reask occurs, the `Call` log can provide total tokens consumed (*currently only for OpenAI models), as well as access to all of the raw outputs from the LLM:
```py
print("prompt token usage: ", first_call.prompt_tokens_consumed) # Total number of prompt tokens consumed across iterations within this call
print("completion token usage: ", first_call.completion_tokens_consumed) # Total number of completion tokens consumed across iterations within this call
print("total token usage: ",first_call.tokens_consumed) # Total number of tokens consumed; equal to the sum of the two values above
print("llm responses\n-------------") # An Stack of the LLM responses in order that they were received
for r in first_call.raw_outputs:
  print(r)
```
```log
prompt token usage:  909
completion token usage:  57
total token usage:  966

llm responses
-------------
{"action": {"chosen_action": "freeze"}}
{
  "action": {
    "chosen_action": "freeze",
    "duration": null
  }
}
{
  "action": {
    "chosen_action": "freeze",
    "duration": 1
  }
}
```

For more information on `Call`, see the [History & Logs](/docs/api_reference/history_and_logs.md) page.

## 🇻🇦 Accessing logs from individual steps
In addition to the cumulative values available directly on the `Call` log, it also contains a `Stack` of `Iteration`'s.  Each `Iteration` represent the logs from within a step in the guardrails process.  This includes the call to the LLM, as well as parsing and validating the LLM's response.

Each `Iteration` is treated as a stateless entity so it will only contain information about the inputs and outputs of the particular step it represents.

For example, in order to see the raw LLM response as well as the logs for the specific validations that failed during the first step of a call, we can access this information via that steps `Iteration`:

```py
first_step = first_call.iterations.first

first_llm_output = first_step.raw_output
print("First LLM response\n------------------")
print(first_llm_output)
print(" ")

validation_logs = first_step.validator_logs
print("\nValidator Logs\n--------------")
for log in validation_logs:
    print(log.json(indent=2))
```
```log
First LLM response
------------------
{"action": {"chosen_action": "fight", "weapon": "spoon"}}
 

Validator Logs
--------------
{
  "validator_name": "ValidChoices",
  "value_before_validation": "spoon",
  "validation_result": {
    "outcome": "fail",
    "metadata": null,
    "error_message": "Value spoon is not in choices ['crossbow', 'axe', 'sword', 'fork'].",
    "fix_value": null
  },
  "value_after_validation": {
    "incorrect_value": "spoon",
    "fail_results": [
      {
        "outcome": "fail",
        "metadata": null,
        "error_message": "Value spoon is not in choices ['crossbow', 'axe', 'sword', 'fork'].",
        "fix_value": null
      }
    ],
    "path": [
      "action",
      "weapon"
    ]
  }
}
```

Similar to the `Call` log, we can also see the token usage for just this step:
```py
print("prompt token usage: ", first_step.prompt_tokens_consumed)
print("completion token usage: ", first_step.completion_tokens_consumed)
print("token usage for this step: ",first_step.tokens_consumed)
```
```log
prompt token usage:  617
completion token usage:  16
token usage for this step:  633
```

For more information on the properties available on `Iteration`, see the [History & Logs](/docs/api_reference_markdown/history_and_logs/#guardrails.classes.history.Iteration) page.