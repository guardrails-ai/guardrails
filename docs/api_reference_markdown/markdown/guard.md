# Guard











































































































uardrails. It is








































































































ntic
- from_string

The \_\_call_\_
method functions as a wrapper around LLM APIs. It takes in an LLM
API, and optional prompt parameters, and returns the raw output from
the LLM and the validated output.

#### *classmethod* \_\_call_\_(llm_api, prompt_params=None, num_reasks=None, prompt=None, instructions=None, msg_history=None, metadata=None, full_schema_reask=None, \*args, \*\*kwargs)

Call the LLM and validate the output. Pass an async LLM API to
return a coroutine.

* **Parameters:**
* **llm_api** (*Callable* *|* *Callable**[**[**Any**]**,* *Awaitable**[**Any**]**]*) – The LLM API to call
(e.g. openai.Completion.create or openai.Completion.acreate)
* **prompt_params** (*Dict* *|* *None*) – The parameters to pass to the prompt.format() method.
* **num_reasks** (*int* *|* *None*) – The max times to re-ask the LLM for invalid output.
* **prompt** (*str* *|* *None*) – The prompt to use for the LLM.
* **instructions** (*str* *|* *None*) – Instructions for chat models.
* **msg_history** (*List**[**Dict**]* *|* *None*) – The message history to pass to the LLM.
* **metadata** (*Dict* *|* *None*) – Metadata to pass to the validators.
* **full_schema_reask** (*bool* *|* *None*) – When reasking, whether to regenerate the full schema
or just the incorrect values.
Defaults to True if a base model is provided,
False otherwise.
* **Returns:**
The raw text output from the LLM and the validated output.
* **Return type:**
*Tuple*[str, *Dict*] | *Awaitable*[*Tuple*[str, *Dict*]]

#### \_\_init_\_(rail, num_reasks=None, base_model=None)

Initialize the Guard.

* **Parameters:**
* **rail** ([*Rail*](rail.md#guardrails.rail.Rail)) – 
* **num_reasks** (*int* *|* *None*) – 
* **base_model** (*BaseModel* *|* *None*) – 

#### \_\_new_\_(\*\*kwargs)

#### configure(num_reasks=None)

Configure the Guard.

* **Parameters:**
**num_reasks** (*int* *|* *None*) – 

#### *classmethod* from_pydantic(output_class, prompt=None, instructions=None, num_reasks=None)

Create a Guard instance from a Pydantic model and prompt.

* **Parameters:**
* **output_class** (*BaseModel*) – 
* **prompt** (*str* *|* *None*) – 
* **instructions** (*str* *|* *None*) – 
* **num_reasks** (*int* *|* *None*) – 
* **Return type:**
[*Guard*](#guardrails.guard.Guard)

#### *classmethod* from_rail(rail_file, num_reasks=None)

Create a Schema from a .rail file.

* **Parameters:**
* **rail_file** (*str*) – The path to the .rail file.
* **num_reasks** (*int* *|* *None*) – The max times to re-ask the LLM for invalid output.
* **Returns:**
An instance of the Guard class.
* **Return type:**
[*Guard*](#guardrails.guard.Guard)

#### *classmethod* from_rail_string(rail_string, num_reasks=None)

Create a Schema from a .rail string.

* **Parameters:**
* **rail_string** (*str*) – The .rail string.
* **num_reasks** (*int* *|* *None*) – The max times to re-ask the LLM for invalid output.
* **Returns:**
An instance of the Guard class.
* **Return type:**
[*Guard*](#guardrails.guard.Guard)

#### *classmethod* from_string(validators, description=None, prompt=None, instructions=None, reask_prompt=None, reask_instructions=None, num_reasks=None)

Create a Guard instance for a string response with prompt,
instructions, and validations.

Parameters: Arguments
: validators: (List[Validator]): The list of validators to apply to the string output.
description (str, optional): A description for the string to be generated. Defaults to None.
prompt (str, optional): The prompt used to generate the string. Defaults to None.
instructions (str, optional): Instructions for chat models. Defaults to None.
reask_prompt (str, optional): An alternative prompt to use during reasks. Defaults to None.
reask_instructions (str, optional): Alternative instructions to use during reasks. Defaults to None.
num_reasks (int, optional): The max times to re-ask the LLM for invalid output.

* **Parameters:**
* **validators** (*List**[**Validator**]*) – 
* **description** (*str* *|* *None*) – 
* **prompt** (*str* *|* *None*) – 
* **instructions** (*str* *|* *None*) – 
* **reask_prompt** (*str* *|* *None*) – 
* **reask_instructions** (*str* *|* *None*) – 
* **num_reasks** (*int* *|* *None*) – 
* **Return type:**
[*Guard*](#guardrails.guard.Guard)

#### parse(llm_output, metadata=None, llm_api=None, num_reasks=None, prompt_params=None, full_schema_reask=None, \*args, \*\*kwargs)

Alternate flow to using Guard where the llm_output is known.

* **Parameters:**
* **llm_api** (*Callable* *|* *Callable**[**[**Any**]**,* *Awaitable**[**Any**]**]* *|* *None*) – The LLM API to call
(e.g. openai.Completion.create or openai.Completion.acreate)
* **num_reasks** (*int* *|* *None*) – The max times to re-ask the LLM for invalid output.
* **llm_output** (*str*) – 
* **metadata** (*Dict* *|* *None*) – 
* **prompt_params** (*Dict* *|* *None*) – 
* **full_schema_reask** (*bool* *|* *None*) – 
* **Returns:**
The validated response.
* **Return type:**
*Tuple*[str, *Dict*] | *Awaitable*[*Tuple*[str, *Dict*]]

#### *property* state*: GuardState*

Return the state.
a** (*Dict* *|* *None*) – 
>   * **prompt_params** (*Dict* *|* *None*) – 
>   * **full_schema_reask** (*bool* *|* *None*) – 
> * **Returns:**
>   The validated response.
> * **Return type:**
>   *Tuple*[str, *Dict*] | *Awaitable*[*Tuple*[str, *Dict*]]

> #### *property* state*: GuardState*

> Return the state.
