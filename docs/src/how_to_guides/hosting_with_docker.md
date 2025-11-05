# Hosting Guardrails with Docker

While you can call Guardrails from your own Python applications, you can also set it up as a service hosted under Flask that you can call via REST API. This simplifies the development and deployment of Guardrails-powered apps by factoring it out into its own scalable endpoint. 

In this page, we’ll discuss how to set up the Guardrails server locally as well as how to deploy it as a production service using Docker. 

## Setup the Guardrails Server Locally

:::note
This guide is condensed to get you up and running quickly.

To see the full guide on using the Guardrails Server, see the [usage doc](/docs/getting_started/guardrails_server).

To learn more about the Guardrails Server, see the [concepts doc](/docs/concepts/deploying).
:::

[Install guardrails](https://www.guardrailsai.com/docs/guardrails_ai/installation) using the "[api]" extra and configure the client. You can find a token on [the Guardrails Hub token page](https://hub.guardrailsai.com/tokens) after making a free account with Guardrails AI.  


```
pip install "guardrails-ai[api]"
guardrails configure
```

Now let’s create a server config file that holds our Guards.

First, we need to install any validators we want to use from the hub. In this example we’ll use the `RegexMatch` validator. Second, we  create our Guards in the `config.py`. We can perform both of these steps at the same time using the `guardrails create` command.


```
guardrails create --validators=hub://guardrails/regex_match --guard-name=title-case
```


Note that this created a `config.py` file for us. If we look inside, we should find some templated code like this:


```
from guardrails import Guard
from guardrails.hub import RegexMatch

guard = Guard()
guard.name = 'title-case'

print("GUARD PARAMETERS UNFILLED! UPDATE THIS FILE!")  # TODO: Remove this when parameters are filled.
guard.use(RegexMatch())  # TODO: Add parameters.
```


Our next step is to update the config with any parameters we want to use and remove the `TODO`s. We'll use a regex that matches title case strings.


```
from guardrails import Guard
from guardrails.hub import RegexMatch
guard = Guard()
guard.name = 'title-case'
guard.use(RegexMatch(regex="^(?:[A-Z][^\\s]*\\s?)+$"))
```


Now that we have Guardrails installed and defined in our config file, we just need to start the dev server.

`guardrails start --config=./config.py`


You can then access documentation for the Guardrails API by navigating in your browser to [http://localhost:8000/docs](http://localhost:8000/docs). 

![openapi server docs](./assets/openapi_server_docs.png "image_tooltip")


You can use the interactive Swagger page to test calls to Guardrails Server, which you can then implement as REST API calls in your own app.


## Best practices
Running the Guardrails server like this is useful for rapid development and testing out guardrails, but it is not suitable for production use. It's a simple Flask server that is not optimized for stability, performance, or security. Flask recommends [different options](https://flask.palletsprojects.com/en/2.3.x/deploying/) for deploying a Flask app in production, such as using a WSGI server like Gunicorn or uWSGI.

Furthermore, the dev server isn't as portable or deployable broadly as many cloud ecosystems would require.

Our solution for this is to build a Docker container for the Guardrails server, using Gunicorn to server the Flask app.

## Hosting Guardrails with Docker

The Guardrails server can be packaged into a Docker container for production deployments. Once containerized, you can host it in any Docker-compatible hosting service, such as Kubernetes, AWS Fargate, Google Cloud Run, etc.


### Building and running the Docker container 

You can build the Docker container as you would any other Docker application.
We need a dockerfile to get started. You can download the file from [our Github repo](https://github.com/guardrails-ai/guardrails-lite-server/blob/main/Dockerfile). Save it in the top level of your project directory as a file called `Dockerfile`.


```bash
docker build -f Dockerfile -t "guardrails-server:latest" --build-arg GUARDRAILS_TOKEN=YOUR_TOKEN_HERE .
```


You can then start the container and access the docs via [https://localhost:8000/docs](https://localhost:8000/docs).


```
docker run -p 8000:8000 -d guardrails-server:latest
```



### Understanding the Guardrails server Dockerfile 
Below we walk through the most significant lines in case you need to customize the server to fit your needs. 


```
ARG GUARDRAILS_TOKEN
```


Declare a build arg for specifying a Guardrails token to be used during configuration.


```
RUN python3 -m venv /opt/venv
```


This runs Guardrails server in a virtual environment, enabling you to run other Python applications on the image without encountering version conflicts between Python packages. 


```
RUN apt-get update
RUN apt-get install -y git
```


Guardrails Hub installs validators using Git, so we include it as a dependency here. 


```
COPY requirements*.txt .
```


Copies over any requirements files containing dependencies  specific to your container. This could just as easily be a `pyproject.toml` or `poetry.lock` depending on how you manage your dependencies.


```
RUN pip install -r requirements-lock.txt
```


Installs all requirements for the container. If you have additional Python packages you need to leverage on the container, or if you want to run a specific version of a dependency (e.g., a specific version of Guardrails), you can add them to another requirements*.txt file and insert a second line. 


```
RUN python -m nltk.downloader -d /opt/nltk_data punkt
```


Download data for the punkt module in the Natural Language Toolkit (NLTK), which Guardrails leverages for tokenization during stream validation.


```
RUN guardrails configure --enable-metrics --enable-remote-inferencing  --token $GUARDRAILS_TOKEN
```


Configure guardrails to allow anonymous metrics collection, use remote inference endpoints instead of local models for any support validators, and provide the Guardrails token for calling those endpoints.


```
RUN guardrails hub install hub://guardrails/regex_match
```


Installs validators from Guardrails Hub. Add additional validators you want to install from Guardrails Hub here. 


```
COPY . .
```


Copy over any other files you might need into the container’s work directory. You can add files to the `.dockerignore` file to prevent unwanted files and folders from copying.


```
EXPOSE 8000
```


Exposes the Web server for Guardrails Server on port 8000. You can bind this to a different local port when running your Docker container [using the -p argument to docker run](https://docs.docker.com/network/).  Just be sure to pass the custom port as a third argument to `guardrails_api.app:create_app() `in your startup command.


```
CMD gunicorn --bind 0.0.0.0:8000 --timeout=90 --workers=4 'guardrails_api.app:create_app(None, "config.py")'
```


Since the Flask development server is not production-ready, we run it behind [Gunicorn](https://gunicorn.org/) to provide concurrent request processing. In this configuration, the Gunicorn server on any given container instance will spawn up 4 concurrent workers to handle many requests at a time with a 90 second timeout for silent (unresponsive) workers. 

Note that you can replace Gunicorn here with another WSGI server.

### Using OpenAI with the Guardrails server

Just like the open source library, the Guardrails server comes with first class support for OpenAI, and other LLM APIs. By setting your Guardrails Docker Image up to talk to OpenAI, you can pass around guarded OpenAI sdk compatible endpoints to your team, ensuring that all uses of the API are validated and secure.

As with any other use case with OpenAI, you must authenticate with an API key in order to make requests to its services. You have two options for providing an OpenAI API key to the Guardrails server:


1. Specify the API key at build or deploy time as an environment variable on the container.  This approach allows you to avoid including your key in requests and abstract it away into the container runtime.  Note, though, that if and when your key changes, you would need to redeploy the server with the updated key.

2. Specify the API key at runtime by including it as a named header `x-openai-api-key`. This approach allows you to utilize different API keys with different scopes if necessary as well as performing key rotation without the need to restart the server.

Below we’ll show a quick example of setting the OpenAI API key as an environment variable on the container.

First declare this new environment variable in the Dockerfile right below the Guardrails token.  These new lines would look like this:


```
ARG GUARDRAILS_TOKEN
ENV OPENAI_API_KEY
```


Second, when running the docker container, include this new environment variable with an `--env` flag:


```
docker run --env OPENAI_API_KEY=sk-my_openai_api_key -d guardrails-server:latest
```


This is the simplest and most explicit way to set the environment variable in the container.  Note also that both Docker and the Guardrails server support `.env` files.  Should you wish to use this option instead, you can do so in the docker run command:


```
docker run --env-file ./.env -d guardrails-server:latest
```


Or when starting the Guardrails server:


```
CMD gunicorn --bind 0.0.0.0:8000 --timeout=90 --workers=4 'guardrails_api.app:create_app("./.env", "config.py")'
```

### Using Other LLMs with the Guardrails server

One final note, the same approaches above can be used to set other environment variables including other OpenAI env vars like `OPENAI_API_BASE`, as well as for other libraries or LLMs like` ANTHROPIC_API_KEY`, `TOGETHERAI_API_KEY`, etc..
