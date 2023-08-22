import warnings

from lxml.etree import _Element, tostring
from pydantic import BaseModel, Field

from guardrails.rail import Rail, generate_xml_code
from guardrails.validators import OneLine


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


def test_generate_xml_code_pydantic():
    class Joke(BaseModel):
        joke: str = Field(validators=[OneLine()])

    prompt = "Tell me a joke."
    instructions = "Make sure it's funny."
    reask_prompt = "That wasn't very funny.  Tell me a different joke."
    reask_instructions = "Make sure it's funny this time."

    xml: _Element = generate_xml_code(
        prompt=prompt,
        output_class=Joke,
        instructions=instructions,
        reask_prompt=reask_prompt,
        reask_instructions=reask_instructions,
    )

    actual_xml = tostring(xml, encoding="unicode", pretty_print=True)

    expected_xml = """<rail version="0.1">
  <output>
    <string name="joke" format="one-line" on-fail-one-line="noop"/>
  </output>
  <prompt>Tell me a joke.</prompt>
  <instructions>Make sure it's funny.</instructions>
  <reask_prompt>That wasn't very funny.  Tell me a different joke.</reask_prompt>
  <reask_instructions>Make sure it's funny this time.</reask_instructions>
</rail>
"""

    assert actual_xml == expected_xml


def test_generate_xml_code_pydantic_with_validations_warning(mocker):
    warn_spy = mocker.spy(warnings, "warn")

    class Joke(BaseModel):
        joke: str = Field(validators=[OneLine()])

    prompt = "Tell me a joke."
    instructions = "Make sure it's funny."
    reask_prompt = "That wasn't very funny.  Tell me a different joke."
    reask_instructions = "Make sure it's funny this time."
    validations = [OneLine()]

    xml: _Element = generate_xml_code(
        prompt=prompt,
        output_class=Joke,
        instructions=instructions,
        reask_prompt=reask_prompt,
        reask_instructions=reask_instructions,
        validators=validations,
    )

    actual_xml = tostring(xml, encoding="unicode", pretty_print=True)

    expected_xml = """<rail version="0.1">
  <output>
    <string name="joke" format="one-line" on-fail-one-line="noop"/>
  </output>
  <prompt>Tell me a joke.</prompt>
  <instructions>Make sure it's funny.</instructions>
  <reask_prompt>That wasn't very funny.  Tell me a different joke.</reask_prompt>
  <reask_instructions>Make sure it's funny this time.</reask_instructions>
</rail>
"""

    assert actual_xml == expected_xml

    warn_spy.assert_called_once_with(
        "Do not specify root level validators on a Pydantic model.  These validators will be ignored."  # noqa
    )


def test_generate_xml_code_string():
    prompt = "Tell me a joke."
    instructions = "Make sure it's funny."
    reask_prompt = "That wasn't very funny.  Tell me a different joke."
    reask_instructions = "Make sure it's funny this time."
    validations = [OneLine()]
    description = "Tell me a joke."

    xml: _Element = generate_xml_code(
        prompt=prompt,
        instructions=instructions,
        reask_prompt=reask_prompt,
        reask_instructions=reask_instructions,
        validators=validations,
        description=description,
    )

    actual_xml = tostring(xml, encoding="unicode", pretty_print=True)

    expected_xml = """<rail version="0.1">
  <output format="one-line" on-fail-one-line="noop" description="Tell me a joke." type="string"/>
  <prompt>Tell me a joke.</prompt>
  <instructions>Make sure it's funny.</instructions>
  <reask_prompt>That wasn't very funny.  Tell me a different joke.</reask_prompt>
  <reask_instructions>Make sure it's funny this time.</reask_instructions>
</rail>
"""  # noqa

    assert actual_xml == expected_xml
