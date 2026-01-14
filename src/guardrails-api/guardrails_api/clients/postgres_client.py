import boto3
import json
import os
import threading
from fastapi import FastAPI
from typing import Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


def postgres_is_enabled() -> bool:
    return os.environ.get("PGHOST", None) is not None


# Global variables for database session
postgres_client = None
SessionLocal = None


class PostgresClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                cls._instance = super(PostgresClient, cls).__new__(cls)
        return cls._instance

    def fetch_pg_secret(self, secret_arn: str) -> dict:
        client = boto3.client("secretsmanager")
        response: dict = client.get_secret_value(SecretId=secret_arn)
        secret_string = response.get("SecretString")
        try:
            secret = json.loads(secret_string)
            return secret
        except Exception:
            pass

    def get_pg_creds(self) -> Tuple[str, str]:
        pg_user = None
        pg_password = None
        pg_password_secret = os.environ.get("PGPASSWORD_SECRET_ARN")
        if pg_password_secret is not None:
            pg_secret = self.fetch_pg_secret(pg_password_secret) or {}
            pg_user = pg_secret.get("username")
            pg_password = pg_secret.get("password")

        pg_user = pg_user or os.environ.get("PGUSER", "postgres")
        pg_password = pg_password or os.environ.get("PGPASSWORD")
        return pg_user, pg_password

    def get_db(self):
        if postgres_is_enabled():
            db = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()
        else:
            yield None

    def initialize(self, app: FastAPI):
        pg_user, pg_password = self.get_pg_creds()
        pg_host = os.environ.get("PGHOST", "localhost")
        pg_port = os.environ.get("PGPORT", "5432")
        pg_database = os.environ.get("PGDATABASE", "postgres")

        pg_endpoint = (
            pg_host
            if pg_host.endswith(
                f":{pg_port}"
            )  # FIXME: This is a cheap check; maybe use a regex instead?
            else f"{pg_host}:{pg_port}"
        )

        conf = f"postgresql://{pg_user}:{pg_password}@{pg_endpoint}/{pg_database}"

        if os.environ.get("NODE_ENV") == "production":
            conf = f"{conf}?sslmode=verify-ca&sslrootcert=global-bundle.pem"

        engine = create_engine(conf)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        self.app = app
        self.engine = engine
        self.SessionLocal = SessionLocal
        # Create tables
        from guardrails_api.models import GuardItem, GuardItemAudit  # noqa

        Base.metadata.create_all(bind=engine)

        # Execute custom SQL
        with engine.connect() as connection:
            connection.execute(text(INIT_EXTENSIONS))
            connection.execute(text(AUDIT_FUNCTION))
            connection.execute(text(AUDIT_TRIGGER))
            connection.commit()


# Define INIT_EXTENSIONS, AUDIT_FUNCTION, and AUDIT_TRIGGER here as they were in your original code
INIT_EXTENSIONS = """
-- Your SQL for initializing extensions
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp') THEN
        CREATE EXTENSION "uuid-ossp";
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE EXTENSION "vector";
    END IF;
END $$;
"""

AUDIT_FUNCTION = """
CREATE OR REPLACE FUNCTION guard_audit_function() RETURNS TRIGGER AS $guard_audit$
BEGIN
    IF (TG_OP = 'DELETE') THEN
    INSERT INTO guards_audit SELECT uuid_generate_v4(), OLD.*, now(), 'D';
    ELSIF (TG_OP = 'UPDATE') THEN
    INSERT INTO guards_audit SELECT uuid_generate_v4(), OLD.*, now(), 'U';
    ELSIF (TG_OP = 'INSERT') THEN
    INSERT INTO guards_audit SELECT uuid_generate_v4(), NEW.*, now(), 'I';
    END IF;
    RETURN null;
END;
$guard_audit$
LANGUAGE plpgsql;
"""

AUDIT_TRIGGER = """
DROP TRIGGER IF EXISTS guard_audit_trigger
  ON guards;
CREATE TRIGGER guard_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON guards
    FOR EACH ROW
    EXECUTE PROCEDURE guard_audit_function();
"""
