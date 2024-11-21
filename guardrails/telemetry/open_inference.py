from typing import Any, Dict, List, Optional

from guardrails.telemetry.common import get_span, to_dict, serialize


def trace_operation(
    *,
    input_mime_type: Optional[str] = None,
    input_value: Optional[Any] = None,
    output_mime_type: Optional[str] = None,
    output_value: Optional[Any] = None,
):
    """Traces an operation (any function call) using OpenInference semantic
    conventions."""
    current_span = get_span()

    if current_span is None:
        return

    ser_input_mime_type = serialize(input_mime_type)
    if ser_input_mime_type:
        current_span.set_attribute("input.mime_type", ser_input_mime_type)

    ser_input_value = serialize(input_value)
    if ser_input_value:
        current_span.set_attribute("input.value", ser_input_value)

    ser_output_mime_type = serialize(output_mime_type)
    if ser_output_mime_type:
        current_span.set_attribute("output.mime_type", ser_output_mime_type)

    ser_output_value = serialize(output_value)
    if ser_output_value:
        current_span.set_attribute("output.value", ser_output_value)


def trace_llm_call(
    *,
    function_call: Optional[
        Dict[str, Any]
    ] = None,  # JSON String	"{function_name: 'add', args: [1, 2]}"	Object recording details of a function call in models or APIs  # noqa
    input_messages: Optional[
        List[Dict[str, Any]]
    ] = None,  # List of objects†	[{"message.role": "user", "message.content": "hello"}]	List of messages sent to the LLM in a chat API request  # noqa
    invocation_parameters: Optional[
        Dict[str, Any]
    ] = None,  # JSON string	"{model_name: 'gpt-3', temperature: 0.7}"	Parameters used during the invocation of an LLM or API  # noqa
    model_name: Optional[
        str
    ] = None,  # String	"gpt-3.5-turbo"	The name of the language model being utilized  # noqa
    output_messages: Optional[
        List[Dict[str, Any]]
    ] = None,  # List of objects	[{"message.role": "user", "message.content": "hello"}]	List of messages received from the LLM in a chat API request  # noqa
    prompt_template_template: Optional[
        str
    ] = None,  # String	"Weather forecast for {city} on {date}"	Template used to generate prompts as Python f-strings  # noqa
    prompt_template_variables: Optional[
        Dict[str, Any]
    ] = None,  # JSON String	{ context: "<context from retrieval>", subject: "math" }	JSON of key value pairs applied to the prompt template  # noqa
    prompt_template_version: Optional[
        str
    ] = None,  # String	"v1.0"	The version of the prompt template  # noqa
    token_count_completion: Optional[
        int
    ] = None,  # Integer	15	The number of tokens in the completion  # noqa
    token_count_prompt: Optional[
        int
    ] = None,  # Integer	5	The number of tokens in the prompt  # noqa
    token_count_total: Optional[
        int
    ] = None,  # Integer	20	Total number of tokens, including prompt and completion  # noqa
):
    """Traces an LLM call using OpenInference semantic conventions."""
    current_span = get_span()

    if current_span is None:
        return

    ser_function_call = serialize(function_call)
    if ser_function_call:
        current_span.set_attribute("llm.function_call", ser_function_call)

    if input_messages and isinstance(input_messages, list):
        for i, message in enumerate(input_messages):
            msg_obj = to_dict(message)
            for key, value in msg_obj.items():
                if value is not None:
                    standardized_key = f"message.{key}" if "message" not in key else key
                    current_span.set_attribute(
                        f"llm.input_messages.{i}.{standardized_key}",
                        serialize(value),  # type: ignore
                    )

    ser_invocation_parameters = serialize(invocation_parameters)
    if ser_invocation_parameters:
        current_span.set_attribute(
            "llm.invocation_parameters", ser_invocation_parameters
        )

    ser_model_name = serialize(model_name)
    if ser_model_name:
        current_span.set_attribute("llm.model_name", ser_model_name)

    if output_messages and isinstance(output_messages, list):
        for i, message in enumerate(output_messages):
            # Most responses are either dictionaries or Pydantic models
            msg_obj = to_dict(message)
            for key, value in msg_obj.items():
                if value is not None:
                    standardized_key = f"message.{key}" if "message" not in key else key
                    current_span.set_attribute(
                        f"llm.output_messages.{i}.{standardized_key}",
                        serialize(value),  # type: ignore
                    )

    ser_prompt_template_template = serialize(prompt_template_template)
    if ser_prompt_template_template:
        current_span.set_attribute(
            "llm.prompt_template.template", ser_prompt_template_template
        )

    ser_prompt_template_variables = serialize(prompt_template_variables)
    if ser_prompt_template_variables:
        current_span.set_attribute(
            "llm.prompt_template.variables", ser_prompt_template_variables
        )

    ser_prompt_template_version = serialize(prompt_template_version)
    if ser_prompt_template_version:
        current_span.set_attribute(
            "llm.prompt_template.version", ser_prompt_template_version
        )

    if token_count_completion:
        current_span.set_attribute("llm.token_count.completion", token_count_completion)

    if token_count_prompt:
        current_span.set_attribute("llm.token_count.prompt", token_count_prompt)

    if token_count_total:
        current_span.set_attribute("llm.token_count.total", token_count_total)
