# Set API Keys and other EnvVars

Guardrails recognizes a handful of environment variables that can be used at runtime.  Most of these correlate to environment variables used or expected by the various LLM clients.  Below you can find a list of these and their uses.

### `OPENAI_API_KEY`
This environment variable can be used to set your api key credentials for Open AI models.  It will be used wherever Open AI is called if an api_key kwarg is not passed to `__call__` or `parse`.

### `GUARDRAILS_PROCESS_COUNT`
This environment variable can be used to set the process count for the multiprocessing executor.  The multiprocessing executor is used to run validations in parallel where possible.  To disable this behaviour and force synchronous validation, you can set this environment variable to `'1'`.  The default is `'10'`.

### `INSPIREDCO_API_KEY`
This environment variable can be used to set your api key credentials for the Inspired Cognition API Client.  It will be used wherever the Inspired Cognition API is called.  Currently this is only used in the `is-high-quality-translation` validator.