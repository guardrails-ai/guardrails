#!/bin/bash
docker stop guardrails-container || true
docker rm guardrails-container || true