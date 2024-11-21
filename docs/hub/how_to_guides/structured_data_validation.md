# Use Validators for Structured Data (JSON) Validation

Each validator can be applied to a specific field in the structured data. For example, if you have a structured data object like this:

```py
data = {
    "name": "John Doe",
    "email": "john@doe.com"
}
```

You can independently validate the `name` and `email` fields using the `Name` and `Email` validators, respectively, as well as the entire `data` object.

## Install the Validator

First, install the validator using the `guardrails hub install` command. For example, to install the `email` validator, you can run:

```bash
guardrails hub install hub://guardrails/regex_match

```

## Create a Pydantic Model with Validation for the Structured Data

First, create a Pydantic model for the structured data. The example below creates a Pydantic model for the `data` object above. In order to add the `RegexMatch` validator to the `name` and `email` fields, you can use the `Field` class from Pydantic and pass the `validators` argument to it.

```py
from pydantic import BaseModel
from guardrails.hub import RegexMatch

NAME_REGEX = "^[A-Z][a-z]+\s[A-Z][a-z]+$"
EMAIL_REGEX = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"


class Data(BaseModel):
    name: str = Field(validators=[RegexMatch(regex=NAME_REGEX)])
    email: str = Field(validators=[RegexMatch(regex=EMAIL_REGEX)])
```

## Create a Pydantic Guard

Create a Pydantic `Guard` object and pass the `Data` model to it.

```py
from guardrails import Guard

guard = Guard.for_pydantic(Data)
```

## Validate the Structured Data

You can now validate the structured data using the `guard` object. For example:

```py
data = {
    "name": "John Doe",
    "email": "john@doe.com"
}


guard.validate(data)
```
