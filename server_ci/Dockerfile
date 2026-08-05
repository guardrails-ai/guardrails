# New LiteLLM version has a dependency on madoka which requires g++ to build the wheel
FROM python:3.13

# Set environment variables to avoid writing .pyc files and to unbuffer Python output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOGLEVEL="DEBUG"
ENV GUARDRAILS_LOG_LEVEL="DEBUG"
ENV APP_ENVIRONMENT="production"

WORKDIR /app


ENV PATH="/root/.local/bin:$PATH"

# Copy the entrypoint script
COPY /server_ci/fastapi-entry.sh /app/fastapi-entry.sh
COPY ../ /app/guardrails
COPY /server_ci/requirements.lock /app/requirements.lock
COPY /server_ci/config.py /app/config.py

# Install locked versions of all dependencies
RUN pip install -r /app/requirements.lock

# Install local version of guardrails-ai
RUN pip install -U "/app/guardrails[api]"


# Expose port 8000 for the application
EXPOSE 8000

# Command to start the Gunicorn server with specified settings
CMD ["/bin/bash", "/app/fastapi-entry.sh"]