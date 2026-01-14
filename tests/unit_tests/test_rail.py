from guardrails.schema.rail_schema import rail_string_to_schema


def test_rail_scalar_string():
    rail_spec = """
<rail version="0.1">
<output>
  <string name="string_name" />
</output>

<messages>
<message role="system">
Hello world
</message>
<message role="user">
Hello world
</message>
</messages>
</rail>
"""
    rail_string_to_schema(rail_spec)


def test_rail_object_with_scalar():
    rail_spec = """
<rail version="0.1">

<output>
    <object name="temp_name">
        <string name="string_name" />
        <integer name="int_name" />
    </object>
</output>

<messages>
<message role="system">
Hello world
</message>
<message role="user">
Hello world
</message>
</messages>
</rail>
"""
    rail_string_to_schema(rail_spec)


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
    rail_string_to_schema(rail_spec)


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
    rail_string_to_schema(rail_spec)


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
    rail_string_to_schema(rail_spec)


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
    rail_string_to_schema(rail_spec)


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
    rail_string_to_schema(rail_spec)


def test_rail_outermost_list():
    rail_spec = """
<rail version="0.1">

<output type="list">
    <object>
        <string name="string_name" description="Any random string value" />
    </object>
</output>

<prompt>
Hello world
</prompt>

</rail>
"""
    rail_string_to_schema(rail_spec)


def test_format_not_read_as_validators():
    rail_spec = """
    <rail version="0.1">
    <output>
      <string name="string_name" format="two-words"/>
    </output>
    <messages>
    <message role="system">
    Hello world
    </message>
    <message role="user">
    Hello world
    </message>
    </messages>
    </rail>
    """
    # Declaring Validators in the format field was dropped in 0.5.x
    #   as stated in the previous DeprecationWarning.
    processed_schema = rail_string_to_schema(rail_spec)
    assert len(processed_schema.validators) == 0


def test_rail_with_messages():
    rail_spec = """
<rail version="0.1">
<output>
  <string name="string_name" />
</output>
<messages>
    <message role="system">You are a helpful assistant.</message>
    <message role="user">Tell me a joke.</message>
    <message role="system">Be brief and concise. 
    It should not be longer than 1-2 sentences.</message>
</messages>
</rail>
"""
    schema = rail_string_to_schema(rail_spec)
    assert len(schema.exec_opts.messages) == 3
    assert schema.exec_opts.messages == [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Tell me a joke.",
        },
        {
            "role": "system",
            "content": "Be brief and concise. \n    It should not be longer than 1-2 sentences.",
        },
    ]
