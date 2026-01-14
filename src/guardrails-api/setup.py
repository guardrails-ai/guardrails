import codecs
import os
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


setup(
    name="guardrails-api",
    description="Guardrails API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8, <4",
    install_requires=[
        "guardrails-ai>=0.5.10",
        "jsonschema>=4.22.0,<5",
        "referencing>=0.35.1,<1",
        "boto3>=1.34.115,<2",
        "psycopg2-binary>=2.9.9,<3",
        "litellm>=1.39.3,<2",
        "typer>=0.9.4,<1",
        "opentelemetry-api>=1.0.0,<2",
        "opentelemetry-sdk>=1.0.0,<2",
        "opentelemetry-exporter-otlp-proto-grpc>=1.0.0,<2",
        "opentelemetry-exporter-otlp-proto-http>=1.0.0,<2",
        "opentelemetry-instrumentation-fastapi>=0.48b0",
        "requests>=2.32.3",
        "aiocache>=0.11.1",
        "fastapi>=0.114.1",
        "SQLAlchemy>=2.0.34",
        "uvicorn>=0.30.6",
    ],
    package_data={"guardrails_api": ["py.typed", "open-api-spec.json"]},
)
