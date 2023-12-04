import json

import pytest

from guardrails import Guard

llm_output = {
    "code": "print(1)",
    "some_url": "https://docs.guardrailsai.com",
    "some_email": "info@guardrailsai.com",
    "query": "select 1;",
}


def test_deprecated_xml_types_backwards_compatability():
    guard = Guard.from_rail_string(
        """
<rail version="0.1">
<output>
    <pythoncode
        name="code"
        description="some random python code"
        format="bug-free-python"
    />
    <url
        name="some_url"
        description="a random web url."
        format="valid-url; is-reachable"
    />
    <email
        name="some_email"
        description="some email"
    />
    <sql
        name="query"
        description="a query"
        format="bug-free-sql; sql-column-presence: cols=['user']"
    />
</output>

<prompt>
noop
</prompt>
</rail>
"""
    )
    with pytest.warns(DeprecationWarning):
        response = guard.parse(llm_output=json.dumps(llm_output))
        assert response.validated_output == llm_output
