# Why use Guardrails AI?

Guardrails AI is a trusted framework for developing Generative AI applications, with thousands of weekly downloads and a dedicated team constantly refining its capabilities. 

While users may find various reasons to integrate Guardrails AI into their projects, we believe its core strengths lie in simplifying LLM response validation, enhancing reusability, and providing robust operational features. These benefits can significantly reduce development time and improve the consistency of AI applications.


## [A Standard for LLM Response Validation](/docs/concepts/validators)

Guardrails AI provides a framework for creating reusable validators to check LLM outputs. This approach reduces code duplication and improves maintainability by allowing developers to create validators that can be integrated into multiple LLM calls. Using this approach, we're able to uplevel performance, LLM feature compatability, and LLM app reliability.

Here's an example of validation with and without Guardrails AI:

```python
# Without Guardrails AI

def is_haiku(value):
    if not value or len(value.split("\n")) != 3:
        return "This is not a haiku"
    return value

response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
)
print(is_haiku(response.choices[0].message.content))

## With Guardrails AI
@register_validator(name="is-haiku", data_type="string")
def is_haiku(value, metadata):
    if not value or len(value.split("\n")) != 3:
        return FailResult(error_message="This is not a haiku")
    return PassResult()

response = Guard().use(is_haiku)(
    model='gpt-3.5-turbo',
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
)
print(response.validated_output)
```

## Performance
Guardrails AI includes built-in support for asynchronous calls, parallelization, and even has an out-of-the-box validation server. These features contribute to the scalability of AI applications by allowing efficient handling of multiple LLM interactions and real-time processing of responses.

Guardrails AI implements automatic retries and exponential backoff for common LLM failure conditions. This built-in error handling improves the overall reliability of AI applications without requiring additional error-handling code. By automatically managing issues such as network failures or API rate limits, Guardrails AI helps ensure consistent performance of LLM-based applications.

Providing a comprehensive set of tools for working with LLMs streamlines the development process and promotes the creation of more robust and reliable AI applications.


## [Streaming](/docs/concepts/streaming)

Guardrails AI supports [streaming validation](/docs/how_to_guides/enable_streaming), and it's the only library to our knowledge that can *fix LLM responses in real-time*. This feature is particularly useful for applications that require immediate feedback or correction of LLM outputs, like chat bots.

## [The Biggest LLM Validation Library](/docs/concepts/hub)

[Guardrails Hub](https://hub.guardrailsai.com) is our centralized location for uploading validators that we and members of our community make available for other developers and companies. 

Validators are written using a few different methods:
1. Simple, function-based validators
2. Classifier based validators
3. LLM based validators

Some of these validators require additional infrastructure, and Guardrails provides the patterns and tools to make it easy to deploy and use them.

The Guardrails Hub is open for submissions, and we encourage you to contribute your own validators to help the community.


## [Supports All LLMs](/docs/how_to_guides/using_llms)

Guardrails AI supports many major LLMs directly, as well as a host of other LLMs via our integrations with LangChain and Hugging Face. This means that you can use the same validators across multiple LLMs, making it easy to swap out LLMs based on performance and quality of responses.

Supported models can be found in our [LiteLLM partner doc](https://docs.litellm.ai/docs/providers).

Don't see your LLM? You can always write a thin wrapper using the [instructions in our docs](/docs/how_to_guides/using_llms#custom-llm-wrappers).

## [Monitoring](/docs/concepts/telemetry)

Guardrails AI automatically keeps a log of all LLM calls and steps taken during processing, which you can access programmatically via a guard’s history. Additionally, Guardrails AI [supports OpenTelemetry for capturing metrics](/docs/concepts/telemetry), enabling easy integration with Grafana, Arize AI, iudex,  OpenInference, and all major Application Performance Monitoring (APM) services.

## [Structured Data](/docs/how_to_guides/generate_structured_data)
Guardrails AI excels at [validating structured output](/docs/how_to_guides/generate_structured_data), returning data through a JSON-formatted response or generating synthetic structured data. Used in conjunction with Pydantic, you can define reusable models in Guardrails AI for verifying structured responses that you can then reuse across apps and teams.


## [Used Widely in the Open-Source Community](/docs/getting_started/contributing)

We’re honored and humbled that open-source projects that support AI application development are choosing to integrate Guardrails AI. Supporting guards provides open-source projects an easy way to ensure they’re processing the highest-quality LLM output possible.
