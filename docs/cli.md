# Using Guardrails from a CLI

Guardrails can be used from the command line to validate the output of an LLM. Currently, the guardrails CLI doesn't support reasking.


## Usage

```bash
guardrails validate <path to rail spec> <llm output as string> --out <output path for validated JSON>
```

## Validator Hub
In addition to providing a command line interface for validation, the guardrails cli also supports interacting with the Validator Hub.

### Configuration
In order to access any Validators from the Hub that require authentication, you will need to set up your environment through the `guardrails configure` command.  Before running `guardrails configure`, go to the [Validator Hub]() to generate your access tokens.  [Add more detail on this process].

Once you have your tokens, run `guardrails configure` and enter the client id and client secret you retrieved above.

Also, if you want to opt out of anonymous metrics collection, you can do this now by specifying `--no-metrics=true`.

### Installing Validators
In order to install a validator from the hub, you can find its identifier on its respective page in the [Validator Hub]().  This identifier is shaped as a namespace and a name: `{namespace}/{validator_name}`.  Once you have this you can go to any project where you have guardrails installed and run `guardrails hub install hub://{namespace}/{validator_name}`.