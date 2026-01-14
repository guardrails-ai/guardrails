FROM public.ecr.aws/docker/library/python:3.12-slim AS builder

# Accept a build arg for the Guardrails token
# We'll add this to the config using the configure command below
# ARG GUARDRAILS_TOKEN

# Create app directory
WORKDIR /app

# print the version just to verify
RUN python3 --version

# Install some utilities; you may not need all of these
RUN apt-get update
RUN apt-get install -y git

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./
COPY README.md LICENSE MANIFEST.in setup.cfg setup.py ./
COPY guardrails_api ./guardrails_api

RUN uv sync --frozen --no-dev --no-editable


FROM public.ecr.aws/docker/library/python:3.12-slim AS runtime

# Accept a build arg for the Guardrails token
# We'll add this to the config using the configure command below
# ARG GUARDRAILS_TOKEN

# Create app directory
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 65532 nonroot && \
    useradd --uid 65532 --gid nonroot --shell /bin/false --create-home nonroot

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

# Set the directory for nltk data
ENV NLTK_DATA=/opt/nltk_data

# Download punkt data
RUN python -m nltk.downloader -d /opt/nltk_data punkt

# Run the Guardrails configure command to create a .guardrailsrc file
# RUN guardrails configure --enable-metrics --enable-remote-inferencing  --token $GUARDRAILS_TOKEN

# Install any validators from the hub you want
RUN guardrails hub install hub://guardrails/valid_length

# Copy the rest over
# We use a .dockerignore to keep unwanted files exluded
COPY . .

RUN chown -R nonroot:nonroot /app

USER nonroot

EXPOSE 8000

# This is our start command; yours might be different.
# The guardrails-api is a standard FastAPI application.
# You can use whatever production server you want that support FastAPI.
# Here we use gunicorn
CMD gunicorn --bind 0.0.0.0:8000 --timeout=90 --workers=2 'guardrails_api.app:create_app(".env", "sample-config.py")'