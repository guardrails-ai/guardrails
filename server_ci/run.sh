#!/bin/bash

tag=${1:-fastapi}

docker stop guardrails-container || true
docker rm guardrails-container || true
docker run -d --name guardrails-container -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY guardrails:$tag || exit 1