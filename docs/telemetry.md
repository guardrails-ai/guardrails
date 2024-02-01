1. Go to grafana
2. search for opentelemetry in data sources
3. Create a token, paste in the resulting file to otel-collector-config.yml
4. docker-compose up -f telemetry/docker-compose.yml up
5. run a guard and look for traces