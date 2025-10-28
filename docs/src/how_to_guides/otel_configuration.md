# Open Telemetry (OTEL) Configuration

Guardrails enables you to collect key metrics and traces for full observability over your guard executions. This allows you to monitor successful and failed validations along with key data regarding them.

We utilize Open Telemetry (OTEL) for this purpose. This guide will help you configure Guardrails as a source of telemetry data for any collectors you may already have.

> If you have issues with the Grafana integration ensure you read this guide to make sure you configure telemetry as expected.

For more details for how to set up collectors using managed services refer to:
- [Grafana Integration](/docs/integrations/telemetry/grafana)
- [Arize AI Integration](https://docs.arize.com/arize/large-language-models/guardrails)


## Guardrails SDK

### Environment Configuration

Once you have an OTEL collector set up ensure you have the following environment variables defined.

- `OTEL_EXPORTER_OTLP_ENDPOINT=...` (Set to your collector endpoint e.g `http://localhost:4317`)
- `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`

> (For self-hosted or local OpenTelemetry Collector setups) If your grpc endpoint port in `OTEL_EXPORTER_OTLP_ENDPOINT` has issues try the http port.

If your collector uses authentication ensure you have the following configuration set:

- `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20...`

> ⚠️ Python requires `Basic%20` instead of `Basic ` before your token, if you are not using Python simply use `Basic `.

For to avoid issues with reporting telemetry it is important these environment variables are always present, we recommend that they are placed on an `.env` file in your project with dotenv `python-dotenv`.

*Example* `.env`

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20... # Optional for auth
```

### Usage

Once the above env vars have been set, you can configure the tracer and traceprovider and init the guard:

```python
# load .env file & place on top of file
from dotenv import load_dotenv
load_dotenv()

from guardrails.telemetry import default_otlp_tracer

# Configure the TracerPovider
default_otlp_tracer("my-schema-guard")

# Define guard
guard = Guard().use(...)
```


## Guardrails Server

### Environment Configuration

Once you have an OTEL collector set up ensure you have the following environment variables defined.

- `OTEL_METRICS_EXPORTER=otlp`
- `OTEL_TRACES_EXPORTER=otlp`
- `OTEL_PYTHON_TRACER_PROVIDER=gr-api-tracer-provider`
- `OTEL_SERVICE_NAME=gr-api-service`
- `OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST=Accept-Encoding,User-Agent,Referer`
- `OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_RESPONSE=Last-Modified,Content-Type`
- `OTEL_EXPORTER_OTLP_ENDPOINT=...` (Set to your collector endpoint e.g `http://localhost:4317`)
- `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`

> (For self-hosted or local OpenTelemetry Collector setups) If your grpc endpoint port in `OTEL_EXPORTER_OTLP_ENDPOINT` has issues try the http port.

If your collector uses authentication ensure you have the following configuration set:

- `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20...`

> ⚠️ Python requires `Basic%20` instead of `Basic ` before your token, if you are not using Python simply use `Basic `.


Here is a full list of the environment variables which can be placed in an `.env` file:

```
OTEL_METRICS_EXPORTER=otlp
OTEL_TRACES_EXPORTER=otlp
OTEL_PYTHON_TRACER_PROVIDER=gr-api-tracer-provider
OTEL_SERVICE_NAME=gr-api-service
OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST=Accept-Encoding,User-Agent,Referer
OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_RESPONSE=Last-Modified,Content-Type
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20... # Optional for auth
```

> ⚠️ Important: Ensure `OTEL_PYTHON_TRACER_PROVIDER` is set otherwise the Guardrails API may fail to start when either `OTEL_TRACES_EXPORTER` or `OTEL_METRICS_EXPORTER` is set to anything but `none`

### Usage

In Guardrails SDK `>=0.5.0` you can start a standalone server to run guard executions while continuing to use the SDK for requesting validations.

*By setting the environment variables as defined above* which can be exported to the current shell with:

```bash
export $(grep -v '^#' .env | xargs) 
```

> It is important to have the environment variables exported to current shell when running `guardrails start`.

Once you ensure the environment variables are set (`env | grep -i otel`). One can start the Guardrails server with:

```bash
guardrails start --config=./config.py
```

Then run validations against it:

```python
from guardrails import Guard
guard = Guard(name="RegexGuard")

guard.validate("123-456-7890")  # Guardrail passes

try:
    guard.validate("1234-789-0000")  # Guardrail fails
except Exception as e:
    print(e)
```

At which point you should see traces relating to endpoints being hit on the Guardrails Server.

*To ensure you collect traces of both server & sdk* you should ensure the `config.py` file has been configured in the same way as the SDK and continue using the same `.env` as the one prepared for the server.


`config.py`

```python
from guardrails.telemetry import default_otlp_tracer

# Configure the TracerPovider
default_otlp_tracer("my-schema-guard")

# Define guard
guard = Guard().use(...)
```