# guardrails-poc
Docker compose stub of Guardrails as a Service

## Setting Up
We strongly encourage you to use a virtual environment when developing in python.
To set one up for this project run the following:
```bash
python3 -m venv ./.venv
source ./.venv/bin/activate
```
Your terminal should now show that you are working from within the virtual environment.
Now you can install the dependencies:
```bash
make install
```

And start the dev server:
```bash
make start
```

Once the service has launched, you should be able to navigate to the Swagger documenation for the guardrails-api at http://localhost:8000


### Local Infrastructure
By default, the server will start with an in-memory store for Guards.  As of June 4th, 2024 this store does not support write operations via the API.  In order to utilize all CRUD operations you will need a postgres database running locally and you will need to provide the following environment variables (sane defaults included for demonstration purposes):

```sh
export PGPORT=5432
export PGDATABASE=postgres
export PGHOST=localhost
export PGUSER=${PGUSER:-postgres}
export PGPASSWORD=${PGPASSWORD:-changeme}
```

You can create this database however you wish, but we do have a docker-compose configuration to stand up the database as well as a local opentelemetry stack.
To use this make sure you have docker installed, then run:

`docker compose --profile db up --build`
to run just the database.

`docker compose --profile infra up --build`
to run the database and opentelemetry infrastructure

or
`docker compose --profile all up --build`
to run everything including the guardrails-api


The last option is useful when checking that everything will work as planned in a more productionized environment.  When developing, it's generally faster to just run the minimum infrastructure you need via docker and run the api on a bare process with the `make start` command.
