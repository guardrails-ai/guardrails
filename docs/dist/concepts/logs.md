import CodeOutputBlock from '../../docusaurus/code-output-block.jsx';

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


<CodeOutputBlock dangerouslySetInnerHTML={{ __html: "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Logs<br />└── ╭────────────────────────────────────────────────── Step 0 ───────────────────────────────────────────────────╮<br />    │ <span style=\"background-color: #f0f8ff\">╭──────────────────────────────────────────────── Prompt ─────────────────────────────────────────────────╮</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ You are a human in an enchanted forest. You come across opponents of different types. You should fight  │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ smaller opponents, run away from bigger ones, and freeze if the opponent is a bear.                     │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ You run into a grizzly. What do you do?                                                                 │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ Given below is XML that describes the information to extract from this document and the tags to extract │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ it into.                                                                                                │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ &lt;output&gt;                                                                                                │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│     &lt;choice name=\"action\" discriminator=\"chosen_action\"&gt;                                                │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│         &lt;case name=\"fight\"&gt;                                                                             │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│             &lt;string name=\"weapon\" format=\"valid-choices: choices=['crossbow', 'axe', 'sword',           │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ 'fork']\"/&gt;                                                                                              │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│         &lt;/case&gt;                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│         &lt;case name=\"flight\"&gt;                                                                            │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│             &lt;string name=\"flight_direction\" format=\"valid-choices: choices=['north', 'south', 'east',   │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ 'west']\"/&gt;                                                                                              │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│             &lt;integer name=\"distance\" format=\"valid-choices: choices=[1, 2, 3, 4]\"/&gt;                     │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│         &lt;/case&gt;                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│         &lt;case name=\"freeze\"&gt;                                                                            │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│             &lt;integer name=\"duration\" format=\"valid-choices: choices=[1, 2, 3, 4]\"/&gt;                     │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│         &lt;/case&gt;                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│     &lt;/choice&gt;                                                                                           │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ &lt;/output&gt;                                                                                               │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding  │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g.        │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ requests for lists, objects and specific types. Be correct and concise.                                 │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ Here are examples of simple (XML, JSON) pairs that show the expected behavior:                          │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ - `&lt;string name='foo' format='two-words lower-case' /&gt;` =&gt; `{'foo': 'example one'}`                     │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ - `&lt;list name='bar'&gt;&lt;string format='upper-case' /&gt;&lt;/list&gt;` =&gt; `{\"bar\": ['STRING ONE', 'STRING TWO',     │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ etc.]}`                                                                                                 │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ - `&lt;object name='baz'&gt;&lt;string name=\"foo\" format=\"capitalize two-words\" /&gt;&lt;integer name=\"index\"          │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ format=\"1-indexed\" /&gt;&lt;/object&gt;` =&gt; `{'baz': {'foo': 'Some String', 'index': 1}}`                        │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ Here are a few examples                                                                                 │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ goblin: {\"action\": {\"chosen_action\": \"fight\", \"weapon\": \"crossbow\"}}                                    │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ giant: {\"action\": {\"chosen_action\": \"flight\", \"flight_direction\": \"north\", \"distance\": 1}}              │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ dragon: {\"action\": {\"chosen_action\": \"flight\", \"flight_direction\": \"south\", \"distance\": 4}}             │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ troll: {\"action\": {\"chosen_action\": \"fight\", \"weapon\": \"sword\"}}                                        │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ black bear: {\"action\": {\"chosen_action\": \"freeze\", \"duration\": 3}}                                      │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│ beets: {\"action\": {\"chosen_action\": \"fight\", \"weapon\": \"fork\"}}                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">│                                                                                                         │</span> │<br />    │ <span style=\"background-color: #f0f8ff\">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span> │<br />    │ <span style=\"background-color: #fff0f2\">╭───────────────────────────────────────────── Instructions ──────────────────────────────────────────────╮</span> │<br />    │ <span style=\"background-color: #fff0f2\">│ You are a helpful assistant, able to express yourself purely through JSON, strictly and precisely       │</span> │<br />    │ <span style=\"background-color: #fff0f2\">│ adhering to the provided XML schemas.                                                                   │</span> │<br />    │ <span style=\"background-color: #fff0f2\">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span> │<br />    │ <span style=\"background-color: #e7dfeb\">╭──────────────────────────────────────────── Message History ────────────────────────────────────────────╮</span> │<br />    │ <span style=\"background-color: #e7dfeb\">│ No message history.                                                                                     │</span> │<br />    │ <span style=\"background-color: #e7dfeb\">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span> │<br />    │ <span style=\"background-color: #f5f5dc\">╭──────────────────────────────────────────── Raw LLM Output ─────────────────────────────────────────────╮</span> │<br />    │ <span style=\"background-color: #f5f5dc\">│ {\"action\": {\"chosen_action\": \"freeze\", \"duration\": 4}}                                                  │</span> │<br />    │ <span style=\"background-color: #f5f5dc\">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span> │<br />    │ <span style=\"background-color: #f0fff0\">╭─────────────────────────────────────────── Validated Output ────────────────────────────────────────────╮</span> │<br />    │ <span style=\"background-color: #f0fff0\">│ {'action': {'chosen_action': 'freeze', 'duration': 4}}                                                  │</span> │<br />    │ <span style=\"background-color: #f0fff0\">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span> │<br />    ╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯<br /></pre><br /><br /><script><br />const htmlToImage = require('html-to-image');<br /><br />htmlToImage.toPng(document.getElementById('body'))<br />  .then(function (dataUrl) {<br />    download(dataUrl, 'my-node.png');<br />  });<br /></script>"}} />

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
For more information on the properties available on `Iteration`, see the [History & Logs](/docs/api_reference_markdown/history_and_logs#iteration) page.