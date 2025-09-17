#!/bin/bash

python -m venv server_ci/.venv && \
source server_ci/.venv/bin/activate && \
pip install . && \
pip install -r server_ci/requirements.txt && \
server_ci/.venv/bin/pytest server_ci/tests || exit 1