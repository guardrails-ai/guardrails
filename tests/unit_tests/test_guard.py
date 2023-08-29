import pytest

import guardrails
from guardrails import Validator
from guardrails.datatypes import verify_metadata_requirements
from guardrails.validators import PassResult, register_validator


@register_validator("myrequiringvalidator", data_type="string")
class RequiringValidator(Validator):
    required_metadata_keys = ["required_key"]

    def validate(self, value, metadata):
        return PassResult()


@register_validator("myrequiringvalidator2", data_type="string")
class RequiringValidator2(Validator):
    required_metadata_keys = ["required_key2"]

    def validate(self, value, metadata):
        return PassResult()


@pytest.mark.parametrize(
    "spec,metadata",
    [
        (
            """
<rail version="0.1">
<output>
    <string name="string_name" format="myrequiringvalidator" />
</output>
</rail>
        """,
            {"required_key": "a"},
        ),
        (
            """
<rail version="0.1">
<output>
    <object name="temp_name">
        <string name="string_name" format="myrequiringvalidator" />
    </object>
    <list name="list_name">
        <string name="string_name" format="myrequiringvalidator2" />
    </list>
</output>
</rail>
        """,
            {"required_key": "a", "required_key2": "b"},
        ),
        (
            """
<rail version="0.1">
<output>
    <object name="temp_name">
    <list name="list_name">
    <choice name="choice_name" discriminator="hi">
    <case name="hello">
        <string name="string_name" />
    </case>
    <case name="hiya">
        <string name="string_name" format="myrequiringvalidator" />
    </case>
    </choice>
    </list>
    </object>
</output>
</rail>
""",
            {"required_key": "a"},
        ),
    ],
)
def test_required_metadata(spec, metadata):
    guard = guardrails.Guard.from_rail_string(spec)

    with pytest.raises(ValueError):
        guard.parse("{}")
    missing_keys = verify_metadata_requirements(
        {}, guard.output_schema.to_dict().values()
    )
    assert set(missing_keys) == set(metadata)

    guard.parse("{}", metadata=metadata, num_reasks=0)
    not_missing_keys = verify_metadata_requirements(
        metadata, guard.output_schema.to_dict().values()
    )
    assert not_missing_keys == []
