gunicorn \
    --workers 3 \
    --threads 2 \
    --bind 0.0.0.0:8000 \
    --worker-class gthread \
    --timeout 30 \
    --keep-alive 20 \
    --preload \
    --graceful-timeout 60 \
    "guardrails_api.app:create_app()"