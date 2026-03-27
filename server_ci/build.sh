#!/bin/bash


docker buildx build \
    --platform linux/amd64 \
    -f "./server_ci/Dockerfile" \
    -t "guardrails:server-ci" \
    --build-arg GUARDRAILS_TOKEN="$GUARDRAILS_TOKEN" \
    --progress plain \
    --load . \
    || exit 1