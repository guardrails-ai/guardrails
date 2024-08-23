import enum
import os
import threading
from typing import Optional


class AuthSchemeRemoteInferencing(str, enum.Enum):
    HUB = "hub"
    SIGV4 = "sigv4"


class Settings:
    AuthSchemeRemoteInferencing = AuthSchemeRemoteInferencing

    _instance = None
    _lock = threading.Lock()
    """Whether to use a local server for running Guardrails."""
    use_server: Optional[bool]

    """The authentication scheme to use for remote inferencing."""
    remote_inferencing_auth_scheme: Optional[str] = os.getenv(
        "GR_AUTH_SCHEME_REMOTE_INFERENCING", AuthSchemeRemoteInferencing.HUB.value
    )

    """The aws host to specify for sigv4 request signing. 
    You can use {region} as a placeholder for the region.
    """
    auth_scheme_sigv4_host: Optional[str] = os.getenv(
        "GR_AUTH_SCHEME_SIGV4_HOST", "runtime.sagemaker.{region}.amazonaws.com"
    )

    """The aws service to specify for sigv4 request signing."""
    auth_scheme_sigv4_service: Optional[str] = os.getenv(
        "GR_AUTH_SCHEME_SIGV4_SERVICE", "sagemaker"
    )

    """The aws region to specify for sigv4 request signing."""
    auth_scheme_sigv4_region: Optional[str] = (
        os.getenv("GR_AUTH_SCHEME_SIGV4_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
    )

    """Whether to disable tracing.

    Traces are only ever sent to a telemetry sink you specify via
    environment variables or by instantiating a TracerProvider.
    """
    disable_tracing: Optional[bool]

    def __new__(cls) -> "Settings":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Settings, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.use_server = None
        self.disable_tracing = None


settings = Settings()
