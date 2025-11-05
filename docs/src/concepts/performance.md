# Performance

Performance for Gen AI apps can mean two things: 

* Application performance: The total time taken to return a response to a user request
* Accuracy: How often a given LLM returns an accurate answer

This document addresses application performance and strategies to minimize latency in responses. For tracking accuracy, see our [Telemetry](/docs/concepts/telemetry) page.

## Basic application performance

Guardrails consist of a guard and a series of validators that the guard uses to validate LLM responses. Generally, a guard runs in sub-10ms performance. Validators should only add around 100ms of additional latency when configured correctly. 

The largest latency and performance issues will come from your selection of LLM. It's important to capture metrics around LLM usage and assess how different LLMs handle different workloads in terms of both performance and result accuracy. [Guardrails AI's LiteLLM support](https://www.guardrailsai.com/blog/guardrails-litellm-validate-llm-output) makes it easy to switch out LLMs with minor changes to your guard calls. 

## Performance tips

Here are a few tips to get the best performance out of your Guardrails-enabled applications.

**Use async guards for the best performance**. Use the `AsyncGuard` class to make concurrent calls to multiple LLMs and process the response chunks as they arrive. For more information, see [Async stream-validate LLM responses](/docs/concepts/async_streaming).

**Use a remote server for heavy workloads**. More compute-intensive workloads, such as remote inference endpoints, work best when run with dedicated memory and CPU. For example, guards that use a single Machine Learning (ML) model for validation can run in milliseconds on GPU-equipped machines, while they may take tens of seconds on normal CPUs. However, guardrailing orchestration itself performs better on general compute.

To account for this, offload performance-critical validation work by: 

* Using [Guardrails Server](/docs/concepts/deploying) to run certain guard executions on a dedicated server
* Leverage [remote validation inference](/docs/concepts/remote_validation_inference) to configure validators to call a REST API for inference results instead of running them locally

The Guardrails client/server model is hosted via Flask. For best performance, [follow our guidelines on configuring your WSGI servers properly](/docs/concepts/deploying) for production.

**Use purpose-built LLMs for re-validators**. When a guard fails, you can decide how to handle it by setting the appropriate OnFail action. The `OnFailAction.REASK` and `OnFailAction.FIX_REASK` action will ask the LLM to correct its output, with `OnFailAction.FIX_REASK` running re-validation on the revised output. In general, re-validation works best when using a small, purpose-built LLM fine-tuned to your use case. 

## Measure performance using telemetry

Guardrails supports OpenTelemetry (OTEL) and a number of OTEL-compatible telemetry providers. You can use telemetry to measure the performance and accuracy of Guardrails AI-enabled applications, as well as the performance of your LLM calls. 

For more, read our [Telemetry](/docs/concepts/telemetry) documentation.