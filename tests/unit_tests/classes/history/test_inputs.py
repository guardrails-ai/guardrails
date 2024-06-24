from guardrails.classes.history.inputs import Inputs
from guardrails.llm_providers import OpenAICallable
from guardrails.prompt.instructions import Instructions
from guardrails.prompt.prompt import Prompt
from guardrails.prompt.messages import Messages


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
    messages = Messages(source=[
        {"role": "system", "content": "You are a greeting bot."},
        {"role": "user", "content": "Respond with a ${greeting_type} greeting."}
    ])
    prompt_params = {"greeting_type": "friendly"}
    num_reasks = 0
    metadata = {"some_meta_data": "doesn't actually matter"}
    full_schema_reask = False

    inputs = Inputs(
        llm_api=llm_api,
        llm_output=llm_output,
        messages=messages,
        prompt_params=prompt_params,
        num_reasks=num_reasks,
        metadata=metadata,
        full_schema_reask=full_schema_reask,
    )

    assert inputs.llm_api is not None
    assert inputs.llm_api == llm_api
    assert inputs.llm_output is not None
    assert inputs.llm_output == llm_output
    assert inputs.messages is not None
    assert inputs.messages == messages
    assert inputs.prompt_params is not None
    assert inputs.prompt_params == prompt_params
    assert inputs.num_reasks is not None
    assert inputs.num_reasks == num_reasks
    assert inputs.metadata is not None
    assert inputs.metadata == metadata
    assert inputs.full_schema_reask is not None
    assert inputs.full_schema_reask == full_schema_reask
