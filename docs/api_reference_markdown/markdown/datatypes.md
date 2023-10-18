# Data Types













































































































































Element tag: &lt;case>








































































































































t_attr** ([*FormatAttr*](schema.md#guardrails.schema.FormatAttr)) – 
* **element** (*\_Element*) – 
* **Return type:**
None

#### collect_validation(key, value, schema)

Gather validators on a value.

* **Parameters:**
* **key** (*str*) – 
* **value** (*Any*) – 
* **schema** (*Dict*) – 
* **Return type:**
[*FieldValidation*](#guardrails.datatypes.FieldValidation)

#### rail_alias *= 'case'*

### *class* guardrails.datatypes.Choice

Element tag: &lt;object>

#### \_\_init_\_(children, format_attr, element)

* **Parameters:**
* **children** (*Dict**[**str**,* *Any**]*) – 
* **format_attr** ([*FormatAttr*](schema.md#guardrails.schema.FormatAttr)) – 
* **element** (*\_Element*) – 
* **Return type:**
None

#### collect_validation(key, value, schema)

Gather validators on a value.

* **Parameters:**
* **key** (*str*) – 
* **value** (*Any*) – 
* **schema** (*Dict*) – 
* **Return type:**
[*FieldValidation*](#guardrails.datatypes.FieldValidation)

#### rail_alias *= 'choice'*

### *class* guardrails.datatypes.Date

Element tag: &lt;date>

To configure the date format, create a date-format attribute on the
element. E.g. &lt;date name=”…” … date-format=”%Y-%m-%d” />

#### \_\_init_\_(children, format_attr, element)

* **Parameters:**
* **children** (*Dict**[**str**,* *Any**]*) – 
* **format_attr** ([*FormatAttr*](schema.md#guardrails.schema.FormatAttr)) – 
* **element** (*\_Element*) – 
* **Return type:**
None

#### rail_alias *= 'date'*

### *class* guardrails.datatypes.Email

Element tag: &lt;email>

#### rail_alias *= 'email'*

### *class* guardrails.datatypes.FieldValidation

FieldValidation(key: Any, value: Any, validators: List[guardrails.validators.Validator], children: List[ForwardRef(‘FieldValidation’)])

#### \_\_init_\_(key, value, validators, children)

* **Parameters:**
* **key** (*Any*) – 
* **value** (*Any*) – 
* **validators** (*List**[**Validator**]*) – 
* **children** (*List**[*[*FieldValidation*](#guardrails.datatypes.FieldValidation)*]*) – 
* **Return type:**
None

#### children*: List[[FieldValidation](#guardrails.datatypes.FieldValidation)]*

#### key*: Any*

#### value*: Any*

### *class* guardrails.datatypes.Float

Element tag: &lt;float>

#### rail_alias *= 'float'*

### *class* guardrails.datatypes.Integer

Element tag: &lt;integer>

#### rail_alias *= 'integer'*

### *class* guardrails.datatypes.List

Element tag: &lt;list>

#### collect_validation(key, value, schema)

Gather validators on a value.

* **Parameters:**
* **key** (*str*) – 
* **value** (*Any*) – 
* **schema** (*Dict*) – 
* **Return type:**
[*FieldValidation*](#guardrails.datatypes.FieldValidation)

#### rail_alias *= 'list'*

### *class* guardrails.datatypes.NonScalarType

### *class* guardrails.datatypes.Object

Element tag: &lt;object>

#### collect_validation(key, value, schema)

Gather validators on a value.

* **Parameters:**
* **key** (*str*) – 
* **value** (*Any*) – 
* **schema** (*Dict*) – 
* **Return type:**
[*FieldValidation*](#guardrails.datatypes.FieldValidation)

#### rail_alias *= 'object'*

### *class* guardrails.datatypes.Percentage

Element tag: &lt;percentage>

#### rail_alias *= 'percentage'*

### *class* guardrails.datatypes.Pydantic

Element tag: &lt;pydantic>

#### \_\_init_\_(model, children, format_attr, element)

* **Parameters:**
* **model** (*Type**[**BaseModel**]*) – 
* **children** (*Dict**[**str**,* *Any**]*) – 
* **format_attr** ([*FormatAttr*](schema.md#guardrails.schema.FormatAttr)) – 
* **element** (*\_Element*) – 
* **Return type:**
None

#### rail_alias *= 'pydantic'*

### *class* guardrails.datatypes.PythonCode

Element tag: &lt;pythoncode>

#### rail_alias *= 'pythoncode'*

### *class* guardrails.datatypes.SQLCode

Element tag: &lt;sql>

#### rail_alias *= 'sql'*

### *class* guardrails.datatypes.ScalarType

### *class* guardrails.datatypes.String

Element tag: &lt;string>

#### rail_alias *= 'string'*

### *class* guardrails.datatypes.Time

Element tag: &lt;time>

To configure the date format, create a date-format attribute on the
element. E.g. &lt;time name=”…” … time-format=”%H:%M:%S” />

#### \_\_init_\_(children, format_attr, element)

* **Parameters:**
* **children** (*Dict**[**str**,* *Any**]*) – 
* **format_attr** ([*FormatAttr*](schema.md#guardrails.schema.FormatAttr)) – 
* **element** (*\_Element*) – 
* **Return type:**
None

#### rail_alias *= 'time'*

### *class* guardrails.datatypes.URL

Element tag: &lt;url>

#### rail_alias *= 'url'*

### guardrails.datatypes.verify_metadata_requirements(metadata, datatypes)

* **Parameters:**
* **metadata** (*dict*) – 
* **datatypes** (*Iterable**[**DataType**]*) – 
* **Return type:**
*List*[str]
s* guardrails.datatypes.URL

> Element tag: <url>

> #### rail_alias *= 'url'*

> ### guardrails.datatypes.verify_metadata_requirements(metadata, datatypes)

> * **Parameters:**
>   * **metadata** (*dict*) – 
>   * **datatypes** (*Iterable**[**DataType**]*) – 
> * **Return type:**
>   *List*[str]
