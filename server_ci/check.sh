#!/bin/bash

for i in {1..30}; do
if docker exec guardrails-container curl -s http://localhost:8000/; then
    echo "Server is up!"
    break
fi
echo "Waiting for server..."
sleep 5
done