import os

from openai.version import VERSION as OPENAI_VERSION

if OPENAI_VERSION.startswith("1"):
    os.environ["OPENAI_API_KEY"] = "mocked"
