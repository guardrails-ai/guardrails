import os
from string import Template

from fastapi import HTTPException, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from guardrails_api.open_api_spec import get_open_api_spec
from sqlalchemy import text
from guardrails_api.classes.health_check import HealthCheck
from guardrails_api.clients.postgres_client import PostgresClient, postgres_is_enabled
from guardrails_api.utils.logger import logger


class HealthCheckResponse(BaseModel):
    status: int
    message: str


router = APIRouter()


@router.get("/")
async def home():
    return "Hello, world!"


@router.get("/health-check", response_model=HealthCheckResponse)
async def health_check():
    try:
        if not postgres_is_enabled():
            return HealthCheck(200, "Ok").to_dict()

        pg_client = PostgresClient()
        query = text("SELECT count(datid) FROM pg_stat_activity;")
        with pg_client.SessionLocal() as session:
            response = session.execute(query).all()

        logger.info("response: %s", response)

        return HealthCheck(200, "Ok").to_dict()
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/api-docs", response_class=JSONResponse)
async def api_docs():
    api_spec = get_open_api_spec()
    return JSONResponse(content=api_spec)


@router.get("/docs", response_class=HTMLResponse)
async def docs():
    host = os.environ.get("SELF_ENDPOINT", "http://localhost:8000")
    swagger_ui = Template("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="description" content="SwaggerUI" />
  <title>SwaggerUI</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js" crossorigin></script>
<script>
  window.onload = () => {
    window.ui = SwaggerUIBundle({
      url: '${apiDocUrl}',
      dom_id: '#swagger-ui',
    });
  };
</script>
</body>
</html>""").safe_substitute(apiDocUrl=f"{host}/api-docs")

    return HTMLResponse(content=swagger_ui)
