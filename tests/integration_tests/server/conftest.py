"""Testcontainers fixtures for server integration tests.

Provides a complete e2e environment with:
- Guardrails server container (pre-built image)
- Ollama container for LLM inference (no external API keys needed)

Environment variables:
- GUARDRAILS_TEST_IMAGE: Docker image to test (default: guardrails-server:ci)
- GUARDRAILS_TOKEN: Hub token for extension installation
- GUARDRAILS_EXTENSIONS: Comma-separated list of hub extensions to install
- OLLAMA_HOST: If set, use host Ollama instead of container (e.g., "http://localhost:11434")
- OLLAMA_MODEL: LLM model to use (default: "smollm:360m")
"""

import os

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.core.waiting_utils import wait_for_logs


# Configuration via environment variables
GUARDRAILS_IMAGE = os.getenv("GUARDRAILS_TEST_IMAGE", "guardrails-server:ci")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")  # Use host Ollama if set
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smollm:135m")


class OllamaContainer(DockerContainer):
    """Ollama container for local LLM inference."""

    def __init__(self, model: str = OLLAMA_MODEL, **kwargs):
        super().__init__("ollama/ollama:latest", **kwargs)
        self.with_exposed_ports(11434)
        self.model = model

    def get_base_url(self) -> str:
        """Get the Ollama API URL."""
        host = self.get_container_host_ip()
        port = self.get_exposed_port(11434)
        return f"http://{host}:{port}"

    def get_internal_url(self) -> str:
        """Get the internal URL for container-to-container communication."""
        return "http://ollama:11434"

    def start(self):
        """Start container and pull the model."""
        super().start()
        # Wait for Ollama to be ready
        wait_for_logs(self, "Listening on", timeout=60)

        # Pull the model
        print(f"Pulling Ollama model: {self.model}...")
        exit_code, output = self.exec(["ollama", "pull", self.model])
        if exit_code != 0:
            raise RuntimeError(f"Failed to pull model {self.model}: {output}")
        print(f"Model {self.model} ready")

        return self


class GuardrailsContainer(DockerContainer):
    """Guardrails server container with health check support."""

    def __init__(
        self,
        image: str = GUARDRAILS_IMAGE,
        ollama_base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(image, **kwargs)
        self.with_exposed_ports(8000)

        # Pass through Guardrails Hub credentials for runtime extension installation
        if guardrails_token := os.getenv("GUARDRAILS_TOKEN"):
            self.with_env("GUARDRAILS_TOKEN", guardrails_token)
        if guardrails_extensions := os.getenv("GUARDRAILS_EXTENSIONS"):
            self.with_env("GUARDRAILS_EXTENSIONS", guardrails_extensions)

        # Configure Ollama as LLM backend
        if ollama_base_url:
            self.with_env("OLLAMA_API_BASE", ollama_base_url)

        # Pass through OpenAI key if explicitly set (fallback for non-Ollama tests)
        if openai_key := os.getenv("OPENAI_API_KEY"):
            self.with_env("OPENAI_API_KEY", openai_key)

    def get_base_url(self) -> str:
        """Get the base URL for the running container."""
        host = self.get_container_host_ip()
        port = self.get_exposed_port(8000)
        return f"http://{host}:{port}"

    def start(self):
        """Start container and wait for health check."""
        super().start()
        # Wait for uvicorn startup message (longer timeout for extension installation)
        wait_for_logs(self, "Uvicorn running on", timeout=180)
        return self


@pytest.fixture(scope="session")
def docker_network():
    """Create a Docker network for container communication."""
    network = Network()
    network.create()
    yield network
    network.remove()


@pytest.fixture(scope="session")
def ollama_container(docker_network):
    """Session-scoped fixture that starts Ollama container.

    If OLLAMA_HOST is set, returns None and tests use host Ollama instead.
    """
    if OLLAMA_HOST:
        print(f"Using host Ollama at {OLLAMA_HOST}")
        yield None
        return

    print("Starting Ollama container...")
    container = OllamaContainer(model=OLLAMA_MODEL)
    container.with_network(docker_network)
    container.with_network_aliases("ollama")
    container.start()

    yield container

    container.stop()


@pytest.fixture(scope="session")
def ollama_url(ollama_container) -> str:
    """Get the Ollama URL (host or container)."""
    if OLLAMA_HOST:
        return OLLAMA_HOST
    return ollama_container.get_base_url()


@pytest.fixture(scope="session")
def ollama_internal_url(ollama_container) -> str:
    """Get the internal Ollama URL for container-to-container communication."""
    if OLLAMA_HOST:
        # When using host Ollama, containers need host.docker.internal
        return "http://host.docker.internal:11434"
    return ollama_container.get_internal_url()


@pytest.fixture(scope="session")
def guardrails_container(docker_network, ollama_internal_url):
    """Session-scoped fixture that starts the Guardrails server container."""
    container = GuardrailsContainer(ollama_base_url=ollama_internal_url)
    container.with_network(docker_network)
    container.with_network_aliases("guardrails")
    container.start()

    yield container

    container.stop()


@pytest.fixture(scope="session")
def server_url(guardrails_container):
    """Get the server URL for the running container."""
    return guardrails_container.get_base_url()


@pytest.fixture(scope="session")
def ollama_model() -> str:
    """Get the Ollama model name for use in tests."""
    return f"ollama/{OLLAMA_MODEL}"


@pytest.fixture(autouse=True)
def configure_guardrails_client(server_url):
    """Configure guardrails client to use the test server."""
    from guardrails import settings

    # Store original values
    original_base_url = os.environ.get("GUARDRAILS_BASE_URL")
    original_use_server = settings.use_server

    # Configure for test - base_url is read from env var by GuardrailsApiClient
    os.environ["GUARDRAILS_BASE_URL"] = server_url
    settings.use_server = True

    yield

    # Restore
    settings.use_server = original_use_server
    if original_base_url is not None:
        os.environ["GUARDRAILS_BASE_URL"] = original_base_url
    else:
        os.environ.pop("GUARDRAILS_BASE_URL", None)
