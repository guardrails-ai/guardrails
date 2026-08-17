# Tracing to Langfuse

[Langfuse](https://langfuse.com) is an open-source LLM observability platform.
Guardrails already emits an OpenTelemetry trace for every guard invocation; this
integration exports those traces to Langfuse and records validation results as
Langfuse scores.

## Install

```bash
pip install "guardrails-ai[langfuse]"
```

## Configure

Credentials come from the environment:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"   # EU region (default)
```

Other regions are `https://us.cloud.langfuse.com`, `https://jp.cloud.langfuse.com`
and `https://hipaa.cloud.langfuse.com`. Self-hosted Langfuse needs v3.22.0 or
newer, which is when the OpenTelemetry endpoint was introduced.

## Use

```python
from guardrails import Guard
from guardrails.integrations.langfuse import LangfuseInstrumentor
from guardrails_ai.regex_match import RegexMatch  # pip install guardrails-ai-regex-match

langfuse = LangfuseInstrumentor().instrument()

guard = Guard().use(RegexMatch, regex="^[A-Z][a-z]*$")
guard(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is my name?"}],
)

langfuse.flush()  # only needed in short-lived processes
```

`instrument()` returns the `Langfuse` client, which you can use for anything else
Langfuse offers. Call it once, at startup.

## What you get

A trace named after your guard, with the hierarchy Guardrails already produces:

```
guard                    (span)       validation_passed, number_of_reasks, ... as trace metadata
└── step                 (span)
    ├── call             (generation) model, token usage, cost
    └── <validator>.validate (guardrail)  WARNING level when validation fails
```

Validators are typed as Langfuse `guardrail` observations, so they are filterable
by type and render in the Langfuse **Graph** view alongside the rest of the run.

Only the failing validator span is marked `WARNING`; the guard span is left at
the default level deliberately, since under `on_fail="fix"` or `"reask"` a
validator failing and being corrected is normal operation, and flagging every
trace would make the level meaningless. Failed runs are still findable via the
`validation_passed` metadata below and the score.

Plus one Langfuse **score** per validator (`guardrails.<validator-name>`, with the
error message as the comment when it fails) and one for the run overall
(`guardrails.validation_passed`). Scores are what let you chart validator pass
rates over time.

`validation_passed`, `number_of_reasks`, `number_of_llm_calls` and `execution_id`
are promoted to top-level trace metadata, because Langfuse only supports
filtering on top-level metadata keys. Every trace is tagged `guardrails`, so you
can segment guard traffic from the rest of your application in the Langfuse UI.

## Attaching a user or session

Use Langfuse's own helper. Guardrails spans are ordinary children of the ambient
OpenTelemetry context, so no extra wiring is needed:

```python
from langfuse import propagate_attributes

with propagate_attributes(user_id="user_123", session_id="session_abc"):
    guard(model="gpt-4o-mini", messages=[...])
```

Guardrails also reads OpenTelemetry baggage (`user.id`, `organization`, `app`),
which Langfuse maps natively. Baggage propagates across service boundaries, so
never put secrets in it.

## Things to know

**Initialize before any other Langfuse client.** Langfuse caches one client per
public key and returns the cached one on later calls, discarding the export
filter and mapping hook this integration installs. `LangfuseInstrumentor` raises
if a client already exists rather than silently dropping your spans. One Langfuse
project per process is supported.

**Do not configure another OTLP tracer alongside this.**
`default_otlp_tracer()` and `default_otel_collector_tracer()` both call
`trace.set_tracer_provider()`, which OpenTelemetry only honours once — whichever
loses the race is silently ignored.

**Do not combine with `MlFlowInstrumentor`.** It sets
`settings.disable_tracing = True`, which stops Guardrails emitting the spans this
integration exports.

**Streaming traces have no token usage or cost.** Guardrails' LLM providers
return the stream before final-usage telemetry runs, so streaming generations
carry the model and the correct trace shape but no usage. Non-streaming calls are
unaffected.

**Redaction is partial.** Guardrails redacts keys containing `key`, `token` or
`password` in step and call inputs and in `llm.invocation_parameters`, but *not*
in validator inputs and outputs, nor in the guard span's input. If your prompts
or validated values carry sensitive data, pass Langfuse's `mask` function:

```python
def mask(data):
    ...  # return redacted data

LangfuseInstrumentor(mask=mask).instrument()
```

Any keyword argument accepted by `Langfuse(...)` can be passed to
`LangfuseInstrumentor(...)`, except an isolated `tracer_provider` — Guardrails
takes its tracers from the global provider, so an isolated one would receive no
Guardrails spans, and the instrumentor rejects it.

**Guardrails' own hub telemetry is unrelated** and unaffected; it is a separate
pipeline controlled by `guardrails configure`.

## Turning off scores

```python
LangfuseInstrumentor(emit_validation_scores=False).instrument()
```

## Debugging

If traces do not appear, set `LANGFUSE_DEBUG=True` and look for dropped-span
messages naming the instrumentation scope. Guardrails spans use the scopes
`guardrails-ai` and `guardrails.telemetry.guard_tracing`; both are admitted by
this integration's export filter. In the Langfuse UI a span's scope appears under
`metadata.scope.name`.
