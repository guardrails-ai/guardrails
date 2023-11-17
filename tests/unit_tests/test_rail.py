import pytest

from guardrails.rail import Rail


def test_rail_scalar_string():
    rail_spec = """
<rail version="0.1">
<output>
  <string name="string_name" />
</output>

<instructions>
Hello world
</instructions>

<prompt>
Hello world
</prompt>

</rail>
"""
    Rail.from_string(rail_spec)


def test_rail_object_with_scalar():
    rail_spec = """
<rail version="0.1">

<output>
    <object name="temp_name">
        <string name="string_name" />
        <integer name="int_name" />
    </object>
</output>

<instructions>
Hello world
</instructions>

<prompt>
Hello world
</prompt>
</rail>
"""
    Rail.from_string(rail_spec)


def test_rail_object_with_list():
    rail_spec = """
<rail version="0.1">

<output>
    <object name="temp_name">
        <integer name="int_name" />
        <list name="list_name">
            <string name="string_name" />
        </list>
    </object>
</output>

<prompt>
Hello world
</prompt>

</rail>
"""
    Rail.from_string(rail_spec)


def test_rail_object_with_object():
    rail_spec = """
<rail version="0.1">

<output>
    <object name="temp_name">
        <integer name="int_name" />
        <object name="object_name">
            <string name="string_name" />
        </object>
    </object>
</output>

<prompt>
Hello world
</prompt>

</rail>
"""
    Rail.from_string(rail_spec)


def test_rail_list_with_scalar():
    rail_spec = """
<rail version="0.1">

<output>
    <list name="temp_name">
        <string name="string_name" />
    </list>
</output>

<prompt>
Hello world
</prompt>

</rail>
"""
    Rail.from_string(rail_spec)


def test_rail_list_with_list():
    rail_spec = """
<rail version="0.1">

<output>
    <list name="temp_name">
        <list name="list_name">
            <string name="string_name" />
        </list>
    </list>
</output>

<prompt>
Hello world
</prompt>

</rail>
"""
    Rail.from_string(rail_spec)


def test_rail_list_with_object():
    rail_spec = """
<rail version="0.1">

<output>
    <list name="temp_name">
        <object name="object_name">
            <string name="string_name" />
        </object>
    </list>
</output>

<prompt>
Hello world
</prompt>

</rail>
"""
    Rail.from_string(rail_spec)


def test_format_deprecated():
    rail_spec = """
    <rail version="0.1">
    <output>
      <string name="string_name" format="two-words"/>
    </output>

    <instructions>
    Hello world
    </instructions>

    <prompt>
    Hello world
    </prompt>

    </rail>
    """
    with pytest.warns(DeprecationWarning):
        rail = Rail.from_string(rail_spec)
    validator = rail.output_schema.root_datatype.children.string_name.validators[0]
    assert validator.rail_alias == "two-words"
