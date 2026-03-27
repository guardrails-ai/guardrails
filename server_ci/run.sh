#!/bin/bash


docker stop guardrails-container || true
docker rm guardrails-container || true
docker run -d --name guardrails-container -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY guardrails:server-ci || exit 1