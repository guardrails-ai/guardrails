# Rail





















































 AI Language) is a dialect of XML that allows users to



















































s).

A RAIL file contains a root element called
: &lt;rail version=”x.y”>

that contains the following elements as children:
: 1. &lt;input strict=True/False>, which contains the input schema
2. &lt;output strict=True/False>, which contains the output schema
3. &lt;prompt>, which contains the prompt to be passed to the LLM
4. &lt;instructions>, which contains the instructions to be passed to the LLM

#### \_\_init_\_(input_schema=(None,), output_schema=(None,), instructions=(None,), prompt=(None,), version=('0.1',))

* **Parameters:**
* **input_schema** ([*Schema*](schema.md#guardrails.schema.Schema) *|* *None*) – 
* **output_schema** ([*Schema*](schema.md#guardrails.schema.Schema) *|* *None*) – 
* **instructions** (*Instructions* *|* *None*) – 
* **prompt** (*Prompt* *|* *None*) – 
* **version** (*str* *|* *None*) – 
* **Return type:**
None

#### \_\_new_\_(\*\*kwargs)

#### *classmethod* from_file(file_path)

* **Parameters:**
**file_path** (*str*) – 
* **Return type:**
[*Rail*](#guardrails.rail.Rail)

#### *classmethod* from_pydantic(output_class, prompt=None, instructions=None, reask_prompt=None, reask_instructions=None)

* **Parameters:**
* **output_class** (*BaseModel*) – 
* **prompt** (*str* *|* *None*) – 
* **instructions** (*str* *|* *None*) – 
* **reask_prompt** (*str* *|* *None*) – 
* **reask_instructions** (*str* *|* *None*) – 

#### *classmethod* from_string(string)

* **Parameters:**
**string** (*str*) – 
* **Return type:**
[*Rail*](#guardrails.rail.Rail)

#### *classmethod* from_string_validators(validators, description=None, prompt=None, instructions=None, reask_prompt=None, reask_instructions=None)

* **Parameters:**
* **validators** (*List**[**Validator**]*) – 
* **description** (*str* *|* *None*) – 
* **prompt** (*str* *|* *None*) – 
* **instructions** (*str* *|* *None*) – 
* **reask_prompt** (*str* *|* *None*) – 
* **reask_instructions** (*str* *|* *None*) – 

#### *classmethod* from_xml(xml)

* **Parameters:**
**xml** (*\_Element*) – 

#### input_schema*: [Schema](schema.md#guardrails.schema.Schema) | None* *= (None,)*

#### instructions*: Instructions | None* *= (None,)*

#### output_schema*: [Schema](schema.md#guardrails.schema.Schema) | None* *= (None,)*

#### prompt*: Prompt | None* *= (None,)*

#### version*: str | None* *= ('0.1',)*
d#guardrails.schema.Schema) | None* *= (None,)*

> #### prompt*: Prompt | None* *= (None,)*

> #### version*: str | None* *= ('0.1',)*
