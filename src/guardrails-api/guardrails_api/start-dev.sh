gunicorn --bind 0.0.0.0:8000 \
         --timeout 120 \
         --workers 2 \
         --threads 2 \
         --worker-class=uvicorn.workers.UvicornWorker \
         "guardrails_api.app:create_app()" \
         --reload \
         --capture-output \
         --enable-stdio-inheritance \
         --access-logfile - \
         --error-logfile - \
         --access-logformat '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" pid=%(p)s'