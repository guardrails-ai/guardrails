
import pytest

from tests.integration_tests.test_assets.fixtures import (  # noqa
    fixture_llm_output,
    fixture_rail_spec,
    fixture_validated_output,
)


@pytest.fixture
def string_rail_spec():
    return """
<rail version="0.1">
<output
  type="string"
  validators="two-words"
  on-fail-two-words="fix"
/>
<prompt>
Hi please make me a string
</prompt>
</rail>
"""

@pytest.fixture
def string_llm_output():
    return "string output yes"


@pytest.fixture
def validated_string_output():
    return "string output"
