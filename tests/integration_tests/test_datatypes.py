from pydantic import BaseModel, Field
import guardrails as gd


def test_pydantic_str_list_incompatability():
    class Structure(BaseModel):
        element: str = Field(description="an element")

    guard = gd.Guard.from_pydantic(output_class=Structure)

    invalid_output = '{"element": ["test"]}'
    # invalid_output = '{"element": {"test": "hello"}}'
    # invalid_output = '{"element": "1234"}'

    validated_outcome = guard.parse(llm_output=invalid_output)
    assert (
        validated_outcome.fail_results[0].error_message == "JSON does not match schema"
    )
