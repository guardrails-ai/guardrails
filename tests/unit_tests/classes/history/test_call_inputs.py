from guardrails.classes.history.call_inputs import CallInputs


def test_empty_initialization():
    call_inputs = CallInputs()

    # Overrides and additional properties
    assert call_inputs.llm_api is None
    assert call_inputs.args == []
    assert call_inputs.kwargs == {}

    # Inherited properties
    assert call_inputs.llm_output is None
    assert call_inputs.messages is None
    assert call_inputs.prompt_params is None
    assert call_inputs.num_reasks is None
    assert call_inputs.metadata is None
    assert call_inputs.full_schema_reask is None


def test_non_empty_initialization():
    def custom_llm():
        return "Hello there!"

    llm_api = custom_llm
    prompt = "Respond with a friendly greeting."
    instructions = "You are a greeting bot."
    args = ["arg1"]
    kwargs = {"kwarg1": 1}

    call_inputs = CallInputs(
        llm_api=llm_api,
        prompt=prompt,
        instructions=instructions,
        args=args,
        kwargs=kwargs,
    )

    # We only care about overrides and additional props
    # because the others were tested in test_inputs.py
    assert call_inputs.llm_api == llm_api
    assert call_inputs.prompt == prompt
    assert call_inputs.instructions == instructions
    assert call_inputs.args == args
    assert call_inputs.kwargs == kwargs
