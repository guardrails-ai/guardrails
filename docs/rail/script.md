# `Script` Element

!!! note
    This is a beta feature, and serves more advanced use cases. If you're just getting started with Guardrails, you can skip this section for now.

The `<script></script>` element contains any custom code that a developer wants to use. Common use cases include:

1. Custom `Validators`: Here's a few examples of adding custom validators via the `<script>` tag:
      1. [adding a validator to filter secret keys](../examples/no_secrets_in_generated_text.ipynb),
      2. [adding a validator to check if an ingredient is vegan](../examples/recipe_generation.ipynb),
      3. [adding a validator to check if a chess move is valid](../examples/valid_chess_moves.ipynb).
2. Custom `DataTypes`: Examples coming soon!

Here's the syntax for the `<script>` element:

```xml
<rail version="0.1">
<script language="python">

# your code here

</script>
</rail>
```

## 🔐 Adding a custom `Validator`

Here's an example of adding a custom validator to check if a generated text contains any secret keys. The validator is added via the `<script>` element.

```xml
<rail version="0.1">

<script>
from guardrails.validators import Validator, EventDetail, register_validator <!-- (1)! -->


@register_validator(name="custom-validator", data_type="string") <!-- (2)! -->
class CustomValidatorName(Validator): <!-- (3)! -->

    def validate(self, key, value, schema) -> Dict: <!-- (4)! -->
        # Check if value meets the criteria.

        valid_condition = ...
        descriptive_error_message = ...

        if not valid_condition:
            # Create a programatically corrected value.
            correct_value = ...
            raise EventDetail(   <!-- (5)! -->
                key=key,
                value=value,
                schema=schema,
                error_message=descriptive_error_message,
                fix_value=correct_value,
            )

        return schema  <!-- (6)! -->

</script>
</rail>
```

1. In order to add a custom validator, you need to import the `Validator` class, `EventDetail` class, and `register_validator` decorator.
2. Add the `register_validator` decorator to your custom validator class. The `name` argument is the name of the validator (this will be used in `RAIL` as the formatter name), and the `data_type` argument is the data type that the validator is applicable to. In this case, the validator is applicable to strings.
3. Subclass the `Validator` class.
4. You only need to implement the `validate` method. The `validate` method takes in the `key`, `value`, and `schema` as arguments. The `key` is the key of the value in the JSON object, the `value` is the value itself, and the `schema` is the schema of the value.
5. The `validate` method raises an `EventDetail` object if the value is invalid. This object is then used to take corrective action specified in the `RAIL` spec.
6. The `validate` method should return the `schema` if the value is valid.

The custom validator defined in above can be used in the `RAIL` spec as follows:

```xml
<rail version="0.1">
<output>
    <string .... format="custom-validator" on-fail-custom-validator="fix">
</output>
</rail>
```

## 🧭 Adding a custom `DataType`

Coming soon!
