# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
ARG VARIANT=slim
ARG UV_VERSION=latest

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown

ARG OCI_SOURCE="https://github.com/guardrails-ai/guardrails"
ARG OCI_URL="https://www.guardrailsai.com/"
ARG OCI_VENDOR="Guardrails"
ARG OCI_AUTHORS=""

ARG GUARDRAILS_TEMPLATE="guard-template.json"

# Named stage for uv binary (ARGs don't work in COPY --from)
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-${VARIANT} AS builder

# Redeclare ARGs needed in this stage (global ARGs reset after FROM)
ARG GUARDRAILS_TEMPLATE

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl gcc g++ jq && \
    rm -rf /var/lib/apt/lists/*

# Copy everything needed for uv sync
COPY pyproject.toml uv.lock ./
COPY src/guardrails/ ./src/guardrails/
COPY src/guardrails-api/ ./src/guardrails-api/
COPY docker/$GUARDRAILS_TEMPLATE ./$GUARDRAILS_TEMPLATE

# Install workspace packages with server extras
# pip is needed for runtime hub extension installation
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --all-packages --frozen --no-editable --extra server && \
    uv pip install pip

ENV HOME=/app

# Create config files (template will be applied at runtime via entrypoint)
RUN mkdir -p /app/.guardrails && \
    touch /app/config.py /app/.guardrailsrc

FROM python:${PYTHON_VERSION}-${VARIANT} AS runtime

ARG PYTHON_VERSION
ARG VARIANT
ARG VERSION
ARG COMMIT
ARG BUILD_DATE
ARG OCI_SOURCE
ARG OCI_URL
ARG OCI_VENDOR
ARG OCI_AUTHORS
ARG GUARDRAILS_TEMPLATE

LABEL org.opencontainers.image.title="guardrails-server" \
      org.opencontainers.image.description="Guardrails FastAPI server" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.url="${OCI_URL}" \
      org.opencontainers.image.vendor="${OCI_VENDOR}" \
      org.opencontainers.image.authors="${OCI_AUTHORS}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="python:${PYTHON_VERSION}-${VARIANT}"

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 65532 nonroot && \
    useradd --uid 65532 --gid nonroot --shell /bin/false --create-home nonroot

WORKDIR /app

COPY --from=builder --chown=nonroot:nonroot /app/.venv /app/.venv
COPY --from=builder --chown=nonroot:nonroot /app/${GUARDRAILS_TEMPLATE} /app/${GUARDRAILS_TEMPLATE}
COPY --chown=nonroot:nonroot docker/entrypoint.sh /app/entrypoint.sh

# Create writable directories for runtime extension installation
RUN mkdir -p /app/.guardrails && \
    touch /app/config.py /app/.guardrailsrc && \
    chown -R nonroot:nonroot /app /app/.guardrails /app/config.py /app/.guardrailsrc && \
    chmod +x /app/entrypoint.sh

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOGLEVEL="INFO" \
    GUARDRAILS_LOG_LEVEL="INFO" \
    APP_ENVIRONMENT="production" \
    GUARDRAILS_TEMPLATE="${GUARDRAILS_TEMPLATE}" \
    PATH="/app/.venv/bin:$PATH" \
    GUARDRAILS_VERSION="${VERSION}" \
    GUARDRAILS_COMMIT="${COMMIT}"

USER nonroot

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/ || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "guardrails_api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
