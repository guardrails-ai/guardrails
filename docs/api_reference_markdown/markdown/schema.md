# Schema


































































































































































































































































































































































































rated by a colon with a




























































































































































































































































































































































































mat** (*str* *|* *None*) – 
* **element** (*\_Element* *|* *None*) – 
* **Return type:**
None

#### element*: \_Element | None* *= None*

#### *property* empty*: bool*

Return True if the format attribute is empty, False otherwise.

#### format*: str | None* *= None*

#### *classmethod* from_element(element)

Create a FormatAttr object from an XML element.

* **Parameters:**
**element** (`ET._Element`) – The XML element.
* **Returns:**
A FormatAttr object.
* **Return type:**
[*FormatAttr*](#guardrails.schema.FormatAttr)

#### get_validators(strict=False)

Get the list of validators from the format attribute. Only the
validators that are registered for this element will be returned.

For example, if the format attribute is “valid-url; is-reachable”, and
“is-reachable” is not registered for this element, then only the ValidUrl
validator will be returned, after instantiating it with the arguments
specified in the format attribute (if any).

* **Parameters:**
**strict** (*bool*) – If True, raise an error if a validator is not registered for
this element. If False, ignore the validator and print a warning.
* **Returns:**
A list of validators.
* **Return type:**
*List*[*Validator*]

#### parse()

Parse the format attribute into a dictionary of validators.

* **Returns:**
A dictionary of validators, where the key is the validator name, and
the value is a list of arguments.
* **Return type:**
*Dict*

#### *classmethod* parse_token(token)

Parse a single token in the format attribute, and return the
validator name and the list of arguments.

* **Parameters:**
**token** (`str`) – The token to parse, one of the tokens returned by
self.tokens.
* **Returns:**
A tuple of the validator name and the list of arguments.
* **Return type:**
*Tuple*[str, *List*[*Any*]]

#### to_prompt(with_keywords=True)

Convert the format string to another string representation for use
in prompting. Uses the validators’ to_prompt method in order to
construct the string to use in prompting.

For example, the format string “valid-url; other-validator: 1.0
{1 + 2}” will be converted to “valid-url other-validator:
arg1=1.0 arg2=3”.

* **Parameters:**
**with_keywords** (*bool*) – 
* **Return type:**
str

#### *property* tokens*: List[str]*

Split the format attribute into tokens.

For example, the format attribute “valid-url; is-reachable” will
be split into [“valid-url”, “is-reachable”]. The semicolon is
used as a delimiter, but not if it is inside curly braces,
because the format string can contain Python expressions that
contain semicolons.

#### *property* unregistered_validators*: List[str]*

Get the list of validators from the format attribute that are not
registered for this element.

#### *property* validators*: List[Validator]*

Get the list of validators from the format attribute.

Only the validators that are registered for this element will be
returned.

### *class* guardrails.schema.JsonSchema

#### *async* async_validate(guard_logs, data, metadata)

Validate a dictionary of data against the schema.

* **Parameters:**
* **data** (*Dict**[**str**,* *Any**]* *|* *None*) – The data to validate.
* **guard_logs** (*GuardLogs*) – 
* **metadata** (*Dict*) – 
* **Returns:**
The validated data.
* **Return type:**
*Any*

#### get_reask_setup(reasks, original_response, use_full_schema, prompt_params=None)

Construct a schema for reasking, and a prompt for reasking.

* **Parameters:**
* **reasks** (*List**[**FieldReAsk**]*) – List of tuples, where each tuple contains the path to the
reasked element, and the ReAsk object (which contains the error
message describing why the reask is necessary).
* **original_response** (*Any*) – The value that was returned from the API, with reasks.
* **use_full_schema** (*bool*) – Whether to use the full schema, or only the schema
for the reasked elements.
* **prompt_params** (*Dict**[**str**,* *Any**]* *|* *None*) – 
* **Returns:**
The schema for reasking, and the prompt for reasking.
* **Return type:**
*Tuple*[[*Schema*](#guardrails.schema.Schema), *Prompt*, *Instructions*]

#### introspect(data)

Inspect the data for reasks.

* **Parameters:**
**data** (*Any*) – The data to introspect.
* **Returns:**
A list of ReAsk objects.
* **Return type:**
list

#### parse(output)

Parse the output from the large language model.

* **Parameters:**
**output** (*str*) – The output from the large language model.
* **Returns:**
The parsed output, and the exception that was raised (if any).
* **Return type:**
*Tuple*[*Dict*, Exception | None]

#### preprocess_prompt(prompt_callable, instructions, prompt)

Preprocess the instructions and prompt before sending it to the
model.

* **Parameters:**
* **prompt_callable** (*PromptCallableBase*) – The callable to be used to prompt the model.
* **instructions** (*Instructions* *|* *None*) – The instructions to preprocess.
* **prompt** (*Prompt*) – The prompt to preprocess.

#### reask_prompt_vars *= {'output_schema', 'previous_response'}*

#### setup_schema(root)

Parse the schema specification.

* **Parameters:**
**root** (*\_Element*) – The root element of the schema specification.
* **Return type:**
None

#### transpile(method='default')

Convert the XML schema to a string that is used for prompting a
large language model.

* **Returns:**
The prompt.
* **Parameters:**
**method** (*str*) – 
* **Return type:**
str

#### validate(guard_logs, data, metadata)

Validate a dictionary of data against the schema.

* **Parameters:**
* **data** (*Dict**[**str**,* *Any**]* *|* *None*) – The data to validate.
* **guard_logs** (*GuardLogs*) – 
* **metadata** (*Dict*) – 
* **Returns:**
The validated data.
* **Return type:**
*Any*

### *class* guardrails.schema.Schema

Schema class that holds a \_schema attribute.

#### \_\_init_\_(root=None, schema=None, reask_prompt_template=None, reask_instructions_template=None)

* **Parameters:**
* **root** (*\_Element* *|* *None*) – 
* **schema** (*Dict**[**str**,* *DataType**]* *|* *None*) – 
* **reask_prompt_template** (*str* *|* *None*) – 
* **reask_instructions_template** (*str* *|* *None*) – 
* **Return type:**
None

#### *async* async_validate(guard_logs, data, metadata)

Asynchronously validate a dictionary of data against the schema.

* **Parameters:**
* **data** (*Any*) – The data to validate.
* **guard_logs** (*GuardLogs*) – 
* **metadata** (*Dict*) – 
* **Returns:**
The validated data.
* **Return type:**
*Any*

#### check_valid_reask_prompt(reask_prompt)

* **Parameters:**
**reask_prompt** (*str* *|* *None*) – 
* **Return type:**
None

#### get_reask_setup(reasks, original_response, use_full_schema, prompt_params=None)

Construct a schema for reasking, and a prompt for reasking.

* **Parameters:**
* **reasks** (*List**[**FieldReAsk**]*) – List of tuples, where each tuple contains the path to the
reasked element, and the ReAsk object (which contains the error
message describing why the reask is necessary).
* **original_response** (*Any*) – The value that was returned from the API, with reasks.
* **use_full_schema** (*bool*) – Whether to use the full schema, or only the schema
for the reasked elements.
* **prompt_params** (*Dict**[**str**,* *Any**]* *|* *None*) – 
* **Returns:**
The schema for reasking, and the prompt for reasking.
* **Return type:**
*Tuple*[[*Schema*](#guardrails.schema.Schema), *Prompt*, *Instructions*]

#### introspect(data)

Inspect the data for reasks.

* **Parameters:**
**data** (*Any*) – The data to introspect.
* **Returns:**
A list of ReAsk objects.
* **Return type:**
*List*[*FieldReAsk*]

#### items()

* **Return type:**
*Dict*[str, *DataType*]

#### parse(output)

Parse the output from the large language model.

* **Parameters:**
**output** (*str*) – The output from the large language model.
* **Returns:**
The parsed output, and the exception that was raised (if any).
* **Return type:**
*Tuple*[*Any*, Exception | None]

#### *property* parsed_rail*: \_Element | None*

#### preprocess_prompt(prompt_callable, instructions, prompt)

Preprocess the instructions and prompt before sending it to the
model.

* **Parameters:**
* **prompt_callable** (*PromptCallableBase*) – The callable to be used to prompt the model.
* **instructions** (*Instructions* *|* *None*) – The instructions to preprocess.
* **prompt** (*Prompt*) – The prompt to preprocess.

#### *property* reask_prompt_template*: Prompt | None*

#### setup_schema(root)

Parse the schema specification.

* **Parameters:**
**root** (*\_Element*) – The root element of the schema specification.
* **Return type:**
None

#### to_dict()

Convert the schema to a dictionary.

* **Return type:**
*Dict*[str, *Any*]

#### transpile(method='default')

Convert the XML schema to a string that is used for prompting a
large language model.

* **Returns:**
The prompt.
* **Parameters:**
**method** (*str*) – 
* **Return type:**
str

#### validate(guard_logs, data, metadata)

Validate a dictionary of data against the schema.

* **Parameters:**
* **data** (*Any*) – The data to validate.
* **guard_logs** (*GuardLogs*) – 
* **metadata** (*Dict*) – 
* **Returns:**
The validated data.
* **Return type:**
*Any*

### *class* guardrails.schema.Schema2Prompt

Class that contains transpilers to go from a schema to its
representation in a prompt.

This is important for communicating the schema to a large language
model, and this class will provide multiple alternatives to do so.

#### *classmethod* default(schema)

Default transpiler.

Converts the XML schema to a string directly after removing:
: - Comments
- Action attributes like ‘on-fail-
&lt;br/>
```
*
```
&lt;br/>
’

* **Parameters:**
**schema** ([*Schema*](#guardrails.schema.Schema)) – The schema to transpile.
* **Returns:**
The prompt.
* **Return type:**
str

#### *static* pydantic_to_object(root, schema_dict)

Recursively replace all pydantic elements with object elements.

* **Parameters:**
* **root** (*Element*) – 
* **schema_dict** (*Dict**[**str**,* *DataType**]*) – 
* **Return type:**
None

#### *static* remove_comments(element)

Recursively remove all comments.

* **Parameters:**
**element** (*\_Element*) – 
* **Return type:**
None

#### *static* remove_on_fail_attributes(element)

Recursively remove all attributes that start with ‘on-fail-‘.

* **Parameters:**
**element** (*\_Element*) – 
* **Return type:**
None

#### *static* validator_to_prompt(root, schema_dict)

Recursively remove all validator arguments in the format
attribute.

* **Parameters:**
* **root** (*Element*) – 
* **schema_dict** (*Dict**[**str**,* *DataType**]*) – 
* **Return type:**
None

### *class* guardrails.schema.StringSchema

#### \_\_init_\_(root, reask_prompt_template=None, reask_instructions_template=None)

* **Parameters:**
* **root** (*\_Element*) – 
* **reask_prompt_template** (*str* *|* *None*) – 
* **reask_instructions_template** (*str* *|* *None*) – 
* **Return type:**
None

#### *async* async_validate(guard_logs, data, metadata)

Validate a dictionary of data against the schema.

* **Parameters:**
* **data** (*Any*) – The data to validate.
* **guard_logs** (*GuardLogs*) – 
* **metadata** (*Dict*) – 
* **Returns:**
The validated data.
* **Return type:**
*Any*

#### get_reask_setup(reasks, original_response, use_full_schema, prompt_params=None)

Construct a schema for reasking, and a prompt for reasking.

* **Parameters:**
* **reasks** (*List**[**FieldReAsk**]*) – List of tuples, where each tuple contains the path to the
reasked element, and the ReAsk object (which contains the error
message describing why the reask is necessary).
* **original_response** (*FieldReAsk*) – The value that was returned from the API, with reasks.
* **use_full_schema** (*bool*) – Whether to use the full schema, or only the schema
for the reasked elements.
* **prompt_params** (*Dict**[**str**,* *Any**]* *|* *None*) – 
* **Returns:**
The schema for reasking, and the prompt for reasking.
* **Return type:**
*Tuple*[[*Schema*](#guardrails.schema.Schema), *Prompt*, *Instructions*]

#### introspect(data)

Inspect the data for reasks.

* **Parameters:**
**data** (*Any*) – The data to introspect.
* **Returns:**
A list of ReAsk objects.
* **Return type:**
*List*[*FieldReAsk*]

#### parse(output)

Parse the output from the large language model.

* **Parameters:**
**output** (*str*) – The output from the large language model.
* **Returns:**
The parsed output, and the exception that was raised (if any).
* **Return type:**
*Tuple*[*Any*, Exception | None]

#### preprocess_prompt(prompt_callable, instructions, prompt)

Preprocess the instructions and prompt before sending it to the
model.

* **Parameters:**
* **prompt_callable** (*PromptCallableBase*) – The callable to be used to prompt the model.
* **instructions** (*Instructions* *|* *None*) – The instructions to preprocess.
* **prompt** (*Prompt*) – The prompt to preprocess.

#### reask_prompt_vars *= {'error_messages', 'output_schema', 'previous_response'}*

#### setup_schema(root)

Parse the schema specification.

* **Parameters:**
**root** (*\_Element*) – The root element of the schema specification.
* **Return type:**
None

#### transpile(method='default')

Convert the XML schema to a string that is used for prompting a
large language model.

* **Returns:**
The prompt.
* **Parameters:**
**method** (*str*) – 
* **Return type:**
str

#### validate(guard_logs, data, metadata)

Validate a dictionary of data against the schema.

* **Parameters:**
* **data** (*Any*) – The data to validate.
* **guard_logs** (*GuardLogs*) – 
* **metadata** (*Dict*) – 
* **Returns:**
The validated data.
* **Return type:**
*Any*
rompt_callable** (*PromptCallableBase*) – The callable to be used to prompt the model.
>   * **instructions** (*Instructions* *|* *None*) – The instructions to preprocess.
>   * **prompt** (*Prompt*) – The prompt to preprocess.

> #### reask_prompt_vars *= {'error_messages', 'output_schema', 'previous_response'}*

> #### setup_schema(root)

> Parse the schema specification.

> * **Parameters:**
>   **root** (*\_Element*) – The root element of the schema specification.
> * **Return type:**
>   None

> #### transpile(method='default')

> Convert the XML schema to a string that is used for prompting a
> large language model.

> * **Returns:**
>   The prompt.
> * **Parameters:**
>   **method** (*str*) – 
> * **Return type:**
>   str

> #### validate(guard_logs, data, metadata)

> Validate a dictionary of data against the schema.

> * **Parameters:**
>   * **data** (*Any*) – The data to validate.
>   * **guard_logs** (*GuardLogs*) – 
>   * **metadata** (*Dict*) – 
> * **Returns:**
>   The validated data.
> * **Return type:**
>   *Any*
