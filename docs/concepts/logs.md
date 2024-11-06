#  Logs and History

All `Guard` calls are logged internally, and can be accessed via the guard history.

Whenever `Guard.__call__` or `Guard.parse` is called, a new `Call` entry is added to a stack in sequence of execution. This `Call` stack can be accessed through `Guard.history`.

Calls can be further decomposed into a stack of `Iteration` objects. These are stateless and represent the interactions within a `Call` between llms, validators, inputs and outputs. The `Iteration` stack can be accessed through `call.iterations`.

## General Access
Given:
```py
my_guard = Guard.for_pydantic(...)

response_1 = my_guard(...)

response_2 = my_guard.parse(...)
```

`my_guard.history`'s first `Call` entry will represent the guard execution corresponding to response_1 and the second will correspond to response_2's execution. 

To pretty print logs for the latest call, run:

```python
from rich import print

print(guard.history.last.tree)
```
--8<--

docs/html/single-step-history.html

--8<--

## Calls
### Initial Input
Initial inputs like messages from a call are available on each call. 

```py
first_call = my_guard.history.first
print("message\n-----")
print(first_call.messages[0]["content"])
print("prompt params\n------------- ")
print(first_call.prompt_params)
```
```log
message
-----

You are a human in an enchanted forest. You come across opponents of different types. You should fight smaller opponents, run away from bigger ones, and freeze if the opponent is a bear.

You run into a ${opp_type}. What do you do?

${gr.complete_xml_suffix_v2}


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

### Final Output
Final output of call is accessible on a call. 
```py
print("status: ", first_call.status) # The final status of this guard call
print("validated response:", first_call.validated_output) # The final valid output of this guard call
```
```log
status:  pass
validated response: {'action': {'chosen_action': 'freeze', 'duration': 3}}
```

### Cumulative Raw LLM outputs
`Call` log also the raw returns of llms before validation
```py
print("llm responses\n-------------") # An Stack of the LLM responses in order that they were received
for r in first_call.raw_outputs:
  print(r)
```
```log
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

### Cumulative Token usage
`Call` log also tracks llm token usage (*currently only for OpenAI models)
```py
print("prompt token usage: ", first_call.prompt_tokens_consumed) # Total number of prompt tokens consumed across iterations within this call
print("completion token usage: ", first_call.completion_tokens_consumed) # Total number of completion tokens consumed across iterations within this call
print("total token usage: ",first_call.tokens_consumed) # Total number of tokens consumed; equal to the sum of the two values above
```
```log
prompt token usage:  909
completion token usage:  57
total token usage:  966
```

## Iterations
### Validator logs
Detailed validator logs including outcomes and error spans can be accessed on interations.
```py
first_step = first_call.iterations.first

validation_logs = first_step.validator_logs
print("\nValidator Logs\n--------------")
for log in validation_logs:
    print(log.json(indent=2))
```
```log
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

Failed validations can be conveniently accessed via `iteration.failed_validations`

### Raw LLM output
If multiple llm calls are made like in the case of the reask. Iterations contain the return of each call to an llm.
```py
first_step = first_call.iterations.first

first_llm_output = first_step.raw_output
print("First LLM response\n------------------")
print(first_llm_output)
```
```log
First LLM response
------------------
{"action": {"chosen_action": "fight", "weapon": "spoon"}}
```

### Token Usage
Token usage on a per step basis can be accessed on an Iteration.
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

For more information on `Call`, see the [History & Logs](/docs/api_reference_markdown/history_and_logs) page.
For more information on the properties available on `Iteration`, see the [History & Logs](/docs/api_reference_markdown/history_and_logs/#guardrails.classes.history.Iteration) page.