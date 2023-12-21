# Migrating to 0.2.0

The 0.2.0 release contains a handful of breaking changes compared to the previous 0.1.x releases.  This guide will list out these changes as well as how to migrate to the new features that encompass them in 0.2.x.

## Pydantic Support

In 0.1.x, Guardrails supported pydantic models with a `register_pydantic` decorator.  This decorator has been removed in 0.2.x and replaced with the `Guard.from_pydantic` method.  This method takes in a pydantic model and returns a `Guard` instance that can be used to validate the model.

To migrate from the `register_pydantic` decorator to the `Guard.from_pydantic` method, instantiate the model outside of the decorator and pass it into the `Guard.from_pydantic` method.  If you wish to continue using validators defined on the model with Guardrails, you must define them separately and register them with the `register_validator` decorator.

For example, see the following example migration from a 0.1.x-style rail spec defines a model with the `register_pydantic` decorator to a 0.2.x-style Guard constructed from a pydantic model:


=== "0.1.x"
    ```xml

    <rail version="0.1">
    <script language="python">
        from guardrails.utils.pydantic_utils import register_pydantic
        from pydantic import BaseModel, validator

        @register_pydantic
        class Person(BaseModel):
            '''
            Information about a person.

            Args:
                name (str): The name of the person.
                age (int): The age of the person.
                zip_code (str): The zip code of the person.
            '''
            name: str
            age: int
            zip_code: str

            @validator("zip_code")
            def zip_code_must_be_numeric(cls, v):
                if not v.isnumeric():
                    raise ValueError("Zip code must be numeric.")
                return v

            @validator("age")
            def age_must_be_between_0_and_150(cls, v):
                if not 0 >= v >= 150:
                    raise ValueError("Age must be between 0 and 150.")
                return v

            @validator("zip_code")
            def zip_code_in_california(cls, v):
                if not v.startswith("9"):
                    raise ValueError("Zip code must be in California, and start with 9.")
                if v == "90210":
                    raise ValueError("Zip code must not be Beverly Hills.")
                return v

    </script>

    <output>
        <list name="people" description="A list of 3 people.">
            <pydantic description="Information about a person." model="Person" on-fail-pydantic="reask" />
        </list>
    </output>
    <prompt>
        Generate data for possible users in accordance with the specification below.
        @xml_prefix_prompt
        {output_schema}
        @complete_json_suffix_v2
    </prompt>
    </rail>
    ```

=== "0.2.x"
    ```py
    from guardrails import Guard
    from typing import Any, Dict, List

    from pydantic import BaseModel, Field

    from guardrails.validators import (
        FailResult,
        PassResult,
        ValidationResult,
        Validator,
        register_validator,
    )


    @register_validator(name="zip_code_must_be_numeric", data_type="string")
    class ZipCodeMustBeNumeric(Validator):
        def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
            if not value.isnumeric():
                return FailResult(error_message="Zip code must be numeric.")
            return PassResult()


    @register_validator(name="age_must_be_between_0_and_150", data_type="integer")
    class AgeMustBeBetween0And150(Validator):
        def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
            if not 0 <= value <= 150:
                return FailResult(error_message="Age must be between 0 and 150.")
            return PassResult()


    @register_validator(name="zip_code_in_california", data_type="string")
    class ZipCodeInCalifornia(Validator):
        def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
            if not value.startswith("9"):
                return FailResult(
                    error_message="Zip code must be in California, and start with 9."
                )
            if value == "90210":
                return FailResult(error_message="Zip code must not be Beverly Hills.")
            return PassResult()


    class Person(BaseModel):
        """Information about a person.
        Args:
            name (str): The name of the person.
            age (int): The age of the person.
            zip_code (str): The zip code of the person.
        """

        name: str
        age: int = Field(..., validators=[AgeMustBeBetween0And150(on_fail="reask")])
        zip_code: str = Field(
            ...,
            validators=[
                ZipCodeMustBeNumeric(on_fail="reask"),
                ZipCodeInCalifornia(on_fail="reask"),
            ],
        )


    class ListOfPeople(BaseModel):
        """A list of people.
        Args:
            people (List[Person]): A list of people.
        """

        people: List[Person]


    prompt = """
    Generate data for possible users in accordance with the specification below.
    ${gr.xml_prefix_prompt}
    ${output_schema}
    ${gr.complete_json_suffix_v2}
    """

    guard = Guard.from_pydantic(ListOfPeople, prompt=prompt)
    ```
    
## Choice

In 0.1.x, the `choice` tag was defined in the following way, and its output placed a choice discriminator on the top level, with another element whose key is the value of the choice, containing the case output:


=== "0.1.x spec"
    ```xml
    <output>
        <choice name="action">
            <case name="fight">
                <string name="fight_move"/>
            </case>
            <case name="flight">
                <object name="flight">
                    <string name="flight_direction"/>
                    <integer name="flight_speed"/>
                </object>
            </case>
        </choice>
    </output>
    ```
=== "0.1.x output"
    ```json
    {
        "action": "fight",
        "fight": {
            "fight_move": "punch"
        }
    }
    ```


In 0.2.x, the `choice` tag follows the OpenAPI discriminated union pattern, wherein its output is a single object with a discriminator field nested inside the generated object:

=== "0.2.x rail spec"
    ```xml
    <output>
        <choice name="choice" discriminator="action">
            <case name="fight">
                <string name="fight_move"/>
            </case>
            <case name="flight">
                <string name="flight_direction"/>
                <integer name="flight_speed"/>
            </case>
        </choice>
    </output>
    ```
=== "0.2.x pydantic guard"
    ```py
    from typing import Literal, Union
    from pydantic import BaseModel, Field
    from guardrails import Guard


    class Fight(BaseModel):
        action: Literal["fight"]
        fight_move: str


    class Flight(BaseModel):
        action: Literal["flight"]
        flight_direction: str
        flight_speed: int


    class Choice(BaseModel):
        choice: Union[Fight, Flight] = Field(..., discriminator="action")


    guard = Guard.from_pydantic(Choice)
    ```
=== "0.2.x output"
    ```json
    {
        "choice": {
            "action": "fight",
            "fight_move": "punch"
        }
    }
    ```
    

Notice that the `choice` tag now has a `discriminator` attribute, which is the name of the field that will be used to determine which case is used.  This field is required, and must be a string. Also, the inside of the `case` tag is implicitly wrapped in an object, so the `flight` case is no longer wrapped in an `object` tag.

## Changes To Validators
Previously, Validators had a `validate` method that accepted the key, value, and entire schema to validate a specific property on that schema.  It was also expected to return the full schema after performing validation.  In 0.2.x this has been simplified.  Now `Validator.validate` need only accept the value for the property being validated and any metdata necessary for the validation to run.  Also rather than returning the schema, `validate` methods should return a `ValidationResult`.  See the below example to see these differences in practice.

=== "0.1.x"
    ```py
    @register_validator(name="length", data_type=["string", "list"])
    class ValidLength(Validator):
        """Validate that the length of value is within the expected range.

        - Name for `format` attribute: `length`
        - Supported data types: `string`, `list`, `object`
        - Programmatic fix: If shorter than the minimum, pad with empty last elements.
            If longer than the maximum, truncate.
        """

        def __init__(
            self, min: int = None, max: int = None, on_fail: Optional[Callable] = None
        ):
            super().__init__(on_fail=on_fail, min=min, max=max)
            self._min = int(min) if min is not None else None
            self._max = int(max) if max is not None else None

        def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
            """Validate that a value is within a range."""
            logger.debug(
                f"Validating {value} is in length range {self._min} - {self._max}..."
            )

            if self._min is not None and len(value) < self._min:
                logger.debug(f"Value {value} is less than {self._min}.")

                # Repeat the last character to make the value the correct length.
                if isinstance(value, str):
                    last_val = value[-1]
                else:
                    last_val = [value[-1]]

                corrected_value = value + last_val * (self._min - len(value))
                raise EventDetail(
                    key,
                    value,
                    schema,
                    f"Value has length less than {self._min}. "
                    f"Please return a longer output, "
                    f"that is shorter than {self._max} characters.",
                    corrected_value,
                )

            if self._max is not None and len(value) > self._max:
                logger.debug(f"Value {value} is greater than {self._max}.")
                raise EventDetail(
                    key,
                    value,
                    schema,
                    f"Value has length greater than {self._max}. "
                    f"Please return a shorter output, "
                    f"that is shorter than {self._max} characters.",
                    value[: self._max],
                )

            return schema
    ```

=== "0.2.x"
    ```py
    @register_validator(name="length", data_type=["string", "list"])
    class ValidLength(Validator):
        """Validates that the length of value is within the expected range.

        **Key Properties**

        | Property                      | Description                       |
        | ----------------------------- | --------------------------------- |
        | Name for `format` attribute   | `length`                          |
        | Supported data types          | `string`, `list`, `object`        |
        | Programmatic fix              | If shorter than the minimum, pad with empty last elements. If longer than the maximum, truncate. |

        Args:
            min: The inclusive minimum length.
            max: The inclusive maximum length.
        """

        def __init__(
            self, min: int = None, max: int = None, on_fail: Optional[Callable] = None
        ):
            super().__init__(on_fail=on_fail, min=min, max=max)
            self._min = int(min) if min is not None else None
            self._max = int(max) if max is not None else None

        def validate(self, value: Any, metadata: Dict) -> ValidationResult:
            """Validates that the length of value is within the expected range."""
            logger.debug(
                f"Validating {value} is in length range {self._min} - {self._max}..."
            )

            if self._min is not None and len(value) < self._min:
                logger.debug(f"Value {value} is less than {self._min}.")

                # Repeat the last character to make the value the correct length.
                if isinstance(value, str):
                    last_val = value[-1]
                else:
                    last_val = [value[-1]]

                corrected_value = value + last_val * (self._min - len(value))
                return FailResult(
                    error_message=f"Value has length less than {self._min}. "
                    f"Please return a longer output, "
                    f"that is shorter than {self._max} characters.",
                    fix_value=corrected_value,
                )

            if self._max is not None and len(value) > self._max:
                logger.debug(f"Value {value} is greater than {self._max}.")
                return FailResult(
                    error_message=f"Value has length greater than {self._max}. "
                    f"Please return a shorter output, "
                    f"that is shorter than {self._max} characters.",
                    fix_value=value[: self._max],
                )

            return PassResult()
    ```

## Removal Of Script Support

With additional first class support for Pydantic models as described above, support for custom code via the `<script>` tag is being dropped.

If you were using scripts to make runtime changes to your output schemas, you can simply move this logic to your own code and format it accordingly:


=== "0.1.x"
    ```py
    from guardrails import Guard

    rail_str = """
    <rail version="0.1">

    <output
        type="string"
        description="{script_var}"
    />

    <script language="python">
    script_var = "I'm the script variable!"
    </script>

    </rail>
    """

    guard = Guard.from_rail_string(rail_string=rail_str)
    ```
=== "0.2.x"
    ```py
    from guardrails import Guard

    script_var = "I'm the script variable!"

    rail_str = """
    <rail version="0.1">

    <output
        type="string"
        description="{script_var}"
    />

    </rail>
    """.format(script_var=script_var)

    guard = Guard.from_rail_string(rail_string=rail_str)
    ```


Or you can use prompt parameters to pass these values in when executing the Guard via `__call__` or `parse`:


=== "0.1.x"
    ```py
    from guardrails import Guard

    rail_str = """
    <rail version="0.1">

    <output
        type="string"
        description="{script_var}"
    />

    <script language="python">
    script_var = "I'm the script variable!"
    </script>

    </rail>
    """

    guard = Guard.from_rail_string(rail_string=rail_str)
    ```
=== "0.2.x"
    ```py
    from guardrails import Guard

    script_var = "I'm the script variable!"

    rail_str = """
    <rail version="0.1">

    <output
        type="string"
        description="${script_var}"
    />

    </rail>
    """

    guard = Guard.from_rail_string(rail_string=rail_str)

    guard(
        ...,
        prompt_params={ "script_var": script_var }
    )
    ```


Alternatively if you were using scripts for custom validators, these can now be registered with the `@register_validator`  in your own code as shown above in the [Pydantic Support](#pydantic-support) section.  Custom validators registered in your own code will be picked up by Guardrails when you execute the `Guard` via `__call__` or `parse`.

## String Formatting In Prompts
Previously Guardrails used `f-string`'s and `str.format` to include prompt variables, prompt primitives, and other partials in the prompt sent to the LLM.  This necessitated any braces, aka curly brackets, to be escaped with additional braces to prevent its contents from being considered as variable that should be substituted during string formatting.

We now use `Template.safe_substitute` to avoid this issue completely.  This does, however, require a change to how variables are expressed in string arguments such as prompts and instructions.  Now instead of expressing a variable for the prompt as `{{my_var}}`, it should be expressed as `${my_var}`.

In addition to this change in variable syntax, we have also started namespacing our prompt primitives.  Whereas before you would use one of our built in primitives like so: `@complete_json_suffix_v2`, they should now be specified as `${gr.complete_json_suffix_v2}`.

To highlight these changes in practice, below is the prompt section of a RAIL spec where each tab shows the respective before and after format.

=== "0.1.x"
    ```xml
    <prompt>
    Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 'None'.

    {document}

    @xml_prefix_prompt

    {{output_schema}}

    @json_suffix_prompt
    </prompt>
    ```
=== "0.2.x"
    ```xml
    <prompt>
    Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 'None'.

    ${document}

    ${gr.xml_prefix_prompt}

    ${output_schema}

    ${gr.json_suffix_prompt}
    </prompt>
    ```

### Replacing The Old Format With The New
#### String Variables
To easily find usage of the old format of string variables (i.e. `{document}` or `{{output_schema}}`) in your code, you can utilize the below regex:

##### Find
`([{]+)([^}"']*)([}]+)`

##### Replace
`${$2}`

#### Prompt Primitives
To easily find usage of the old format of prompt primitives (i.e. `@json_suffix_prompt`) in your code, you can utilize the below regex:

##### Find
`(^[@])(\w+)`

##### Replace
`${gr.$2}`

### Using Guardrails With LangChain
The changes to string format also effect the Guardrails AI integration with LangChain.  Since LangChain supports `f-string`'s and `jinja2` for templating, and the `GuardrailsOutputParser` opts for the `f-string` option, any braces in the prompt must be escaped before instantiating the LangChain `PromptTemplate`.  To make this easy, we added an `escape` method on our prompt class to handle this for you; you simply need to invoke it when passing the prompt to `PromptTemplate` like so:
```py
from langchain.output_parsers import GuardrailsOutputParser
from langchain.prompts import PromptTemplate

output_parser = GuardrailsOutputParser.from_rail_string(rail_spec, api=openai.ChatCompletion.create)

prompt = PromptTemplate(
    template=output_parser.guard.prompt.escape(),
    input_variables=output_parser.guard.prompt.variable_names,
)
```

See the [LangChain integration docs](/docs/integrations/langchain) for more details.