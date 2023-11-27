from guardrails.classes.inputs import Inputs
from guardrails.llm_providers import OpenAICallable
from guardrails.prompt.instructions import Instructions
from guardrails.prompt.prompt import Prompt


# Guard against regressions in pydantic BaseModel
def test_empty_initialization():
    inputs = Inputs()

    assert inputs.llm_api is None
    assert inputs.llm_output is None
    assert inputs.instructions is None
    assert inputs.prompt is None
    assert inputs.msg_history is None
    assert inputs.prompt_params is None
    assert inputs.num_reasks is None
    assert inputs.metadata is None
    assert inputs.full_schema_reask is None


def test_non_empty_initialization():
    llm_api = OpenAICallable(text="Respond with a greeting.")
    llm_output = "Hello there!"
    instructions = Instructions(source="You are a greeting bot.")
    prompt = Prompt(source="Respond with a ${greeting_type} greeting.")
    msg_history = [
        {"some_key": "doesn't actually matter because this isn't that strongly typed"}
    ]
    prompt_params = {"greeting_type": "friendly"}
    num_reasks = 0
    metadata = {"some_meta_data": "doesn't actually matter"}
    full_schema_reask = False

    inputs = Inputs(
        llm_api=llm_api,
        llm_output=llm_output,
        instructions=instructions,
        prompt=prompt,
        msg_history=msg_history,
        prompt_params=prompt_params,
        num_reasks=num_reasks,
        metadata=metadata,
        full_schema_reask=full_schema_reask,
    )

    assert inputs.llm_api is not None
    assert inputs.llm_api == llm_api
    assert inputs.llm_output is not None
    assert inputs.llm_output == llm_output
    assert inputs.instructions is not None
    assert inputs.instructions == instructions
    assert inputs.prompt is not None
    assert inputs.prompt == prompt
    assert inputs.msg_history is not None
    assert inputs.msg_history == msg_history
    assert inputs.prompt_params is not None
    assert inputs.prompt_params == prompt_params
    assert inputs.num_reasks is not None
    assert inputs.num_reasks == num_reasks
    assert inputs.metadata is not None
    assert inputs.metadata == metadata
    assert inputs.full_schema_reask is not None
    assert inputs.full_schema_reask == full_schema_reask
