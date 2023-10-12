from guardrails.guard import Guard


def test_passed_date_format():
    rail_spec = """
<rail version="0.1">

<output>
    <string name="name"/>
    <date name="dob" date-format="%Y-%m-%d"/>
</output>


<prompt>
Dummy prompt.
</prompt>

</rail>
"""

    guard = Guard.from_rail_string(rail_spec)
    guard.parse(llm_output='{"name": "John Doe", "dob": "2021-01-01"}', num_reasks=0)

def test_defaulted_date_parser():
    rail_spec = """
<rail version="0.1">

<output>
    <string name="name"/>
    <date name="dob"/>
</output>


<prompt>
Dummy prompt.
</prompt>

</rail>
"""

    guard = Guard.from_rail_string(rail_spec)
    guard.parse(llm_output='{"name": "John Doe", "dob": "2021-01-01"}', num_reasks=0)

def test_defaulted_date_parser_cohere_style_datestring():
    rail_spec = """
<rail version="0.1">

<output>
    <string name="name"/>
    <date name="dob"/>
</output>


<prompt>
Dummy prompt.
</prompt>

</rail>
"""

    guard = Guard.from_rail_string(rail_spec)
    guard.parse(llm_output='{"name": "John Doe", "dob": "2021-01-01T11:10:00+01:00"}', num_reasks=0)