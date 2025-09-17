#!/bin/bash

tag=${1:-fastapi}

docker buildx build \
    --platform linux/amd64 \
    -f "./server_ci/Dockerfile.$tag" \
    -t "guardrails:$tag" \
    --build-arg GUARDRAILS_TOKEN="$GUARDRAILS_TOKEN" \
    --progress plain \
    --load . \
    || exit 1