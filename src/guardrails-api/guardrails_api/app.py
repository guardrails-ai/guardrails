from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from guardrails import configure_logging
from guardrails_api.clients.cache_client import CacheClient
from guardrails_api.clients.postgres_client import postgres_is_enabled
from guardrails_api.otel import otel_is_disabled, initialize
from guardrails_api.utils.trace_server_start_if_enabled import (
    trace_server_start_if_enabled,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry import trace, context, baggage

from rich.console import Console
from rich.rule import Rule
from typing import Optional
import importlib.util
import json
import os

from starlette.middleware.base import BaseHTTPMiddleware


class RequestInfoMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tracer = trace.get_tracer(__name__)
        # Get the current context and attach it to this task
        with tracer.start_as_current_span("request_info") as span:
            client_ip = request.client.host
            user_agent = request.headers.get("user-agent", "unknown")
            referrer = request.headers.get("referrer", "unknown")
            user_id = request.headers.get("x-user-id", "unknown")
            organization = request.headers.get("x-organization", "unknown")
            app = request.headers.get("x-app", "unknown")

            context.attach(baggage.set_baggage("client.ip", client_ip))
            context.attach(baggage.set_baggage("http.user_agent", user_agent))
            context.attach(baggage.set_baggage("http.referrer", referrer))
            context.attach(baggage.set_baggage("user.id", user_id))
            context.attach(baggage.set_baggage("organization", organization))
            context.attach(baggage.set_baggage("app", app))

            span.set_attribute("client.ip", client_ip)
            span.set_attribute("http.user_agent", user_agent)
            span.set_attribute("http.referrer", referrer)
            span.set_attribute("user.id", user_id)
            span.set_attribute("organization", organization)
            span.set_attribute("app", app)

            response = await call_next(request)
            return response


# Custom JSON encoder
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            return list(o)
        if callable(o):
            return str(o)
        return super().default(o)


def register_config(config: Optional[str] = None):
    default_config_file = os.path.join(os.getcwd(), "./config.py")
    config_file = config or default_config_file
    config_file_path = os.path.abspath(config_file)
    if os.path.isfile(config_file_path):
        spec = importlib.util.spec_from_file_location("config", config_file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)


def create_app(env: Optional[str] = None, config: Optional[str] = None, port: Optional[int] = None):
    trace_server_start_if_enabled()
    # used to print user-facing messages during server startup
    console = Console()

    if os.environ.get("APP_ENVIRONMENT") != "production":
        from dotenv import load_dotenv

        # Always load default env file, but let user specified file override it.
        default_env_file = os.path.join(os.path.dirname(__file__), "default.env")
        load_dotenv(default_env_file, override=True)

        if env:
            env_file_path = os.path.abspath(env)
            load_dotenv(env_file_path, override=True)

    set_port = port or os.environ.get("PORT", 8000)
    host = os.environ.get("HOST", "http://localhost")
    self_endpoint = os.environ.get("SELF_ENDPOINT", f"{host}:{set_port}")
    os.environ["SELF_ENDPOINT"] = self_endpoint

    register_config(config)

    app = FastAPI(openapi_url="")

    # Add the custom middleware
    app.add_middleware(RequestInfoMiddleware)

    # Initialize FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    guardrails_log_level = os.environ.get("GUARDRAILS_LOG_LEVEL", "INFO")
    configure_logging(log_level=guardrails_log_level)

    if not otel_is_disabled():
        initialize()

    # if no pg_host is set, don't set up postgres
    if postgres_is_enabled():
        from guardrails_api.clients.postgres_client import PostgresClient

        pg_client = PostgresClient()
        pg_client.initialize(app)

    cache_client = CacheClient()
    cache_client.initialize()

    from guardrails_api.api.root import router as root_router
    from guardrails_api.api.guards import router as guards_router, guard_client

    app.include_router(root_router)
    app.include_router(guards_router)

    # Custom JSON encoder
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"message": str(exc)},
        )

    console.print(f"\n:rocket: Guardrails API is available at {self_endpoint}")
    console.print(f":book: Visit {self_endpoint}/docs to see available API endpoints.\n")

    console.print(":green_circle: Active guards and OpenAI compatible endpoints:")

    guards = guard_client.get_guards()

    for g in guards:
        g_dict = g.to_dict()
        console.print(
            (
                f"- Guard: [bold white]{g_dict.get('name')}[/bold white] "
                f"{self_endpoint}/guards/{g_dict.get('name')}/openai/v1"
            )
        )

    console.print("")
    console.print(Rule("[bold grey]Server Logs[/bold grey]", characters="=", style="white"))

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
