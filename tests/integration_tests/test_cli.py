# ruff: noqa: E501

import json
import os
import subprocess
from tempfile import TemporaryDirectory

import pytest

RAIL_SPEC = """
<rail version="0.1">

<output>
    <object name="patient_info">
        <string name="gender" description="Patient's gender" />
        <integer name="age" validators="hub://guardrails/valid_range: 0 100" />
        <string name="symptoms" description="Symptoms that the patient is currently experiencing" />
    </object>
</output>

<prompt>

Given the following doctor's notes about a patient, please extract a dictionary that contains the patient's information.

${doctors_notes}

${gr.complete_json_suffix_v2}
</prompt>
</rail>
"""

LLM_OUTPUT = """
{
    "patient_info": {
        "gender": "Male",
        "age": 49,
        "symptoms": "Chronic macular rash to face & hair, worse in beard, eyebrows & nares. Itchy, flaky, slightly scaly. Moderate response to OTC steroid cream"
    }
}
"""


@pytest.mark.skip(
    "This test doesn't work once we remove validators from the main repo."
    "The hub install is actually working, but the running code is still in context of the local repo"
    "so when get_validator_class tries to import from guardrails.hub, it only sees the empty local repository."
)
def test_cli():
    with TemporaryDirectory() as tmpdir:
        # Write the rail spec to a file
        rail_spec_path = os.path.join(tmpdir, "dummy_spec.rail")
        with open(rail_spec_path, "w") as f:
            f.write(RAIL_SPEC)

        validated_output_path = os.path.join(tmpdir, "validated_output")

        subprocess.run(["guardrails", "hub", "install", "hub://guardrails/valid_range", "--quiet"])

        # Run the cli command
        result = subprocess.run(
            [
                "guardrails",
                "validate",
                rail_spec_path,
                LLM_OUTPUT,
                "--out",
                validated_output_path,
            ],
            capture_output=True,
            text=True,
        )

        print(result.stdout)

        assert result.returncode == 0

        # Check that the output file is correct
        with open(validated_output_path, "r") as f:
            validated_output = json.load(f)
            assert validated_output == json.loads(LLM_OUTPUT)
