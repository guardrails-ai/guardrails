# Guardrails CLI

**Usage**:

```console
$ [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `configure`
* `create`
* `hub`: Manage validators installed from the...
* `start`
* `validate`: Validate the output of an LLM against a...
* `watch`

## `configure`

**Usage**:

```console
$ configure [OPTIONS]
```

**Options**:

* `--enable-metrics / --disable-metrics`: Opt out of anonymous metrics collection.  [default: enable-metrics]
* `--token TEXT`: API Key for Guardrails. If not provided, you will be prompted for it.
* `--enable-remote-inferencing / --disable-remote-inferencing`: Opt in to remote inferencing. If not provided, you will be prompted for it.  [default: enable-remote-inferencing]
* `--clear-token`: Clear the existing token from the configuration file.
* `--help`: Show this message and exit.

## `create`

**Usage**:

```console
$ create [OPTIONS]
```

**Options**:

* `--validators TEXT`: A comma-separated list of validator hub URIs.   [required]
* `--name TEXT`: The name of the guard to define in the file.
* `--filepath TEXT`: The path to which the configuration file should be saved.  [default: config.py]
* `--dry-run / --no-dry-run`: Print out the validators to be installed without making any changes.  [default: no-dry-run]
* `--help`: Show this message and exit.

## `hub`

Manage validators installed from the Guardrails Hub.

**Usage**:

```console
$ hub [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `create-validator`: Lightweight method for creating simple...
* `install`
* `submit`: Submit a validator to the Guardrails AI...
* `uninstall`: Uninstall a validator from the Hub.

### `hub create-validator`

Lightweight method for creating simple validators.

For more complex submissions see here:
https://github.com/guardrails-ai/validator-template?tab=readme-ov-file#how-to-create-a-guardrails-validator

**Usage**:

```console
$ hub create-validator [OPTIONS] NAME [FILEPATH]
```

**Arguments**:

* `NAME`: The name for your validator.  [required]
* `[FILEPATH]`: The location to write your validator template to  `[default: ./{validator_name}.py]`

**Options**:

* `--help`: Show this message and exit.

### `hub install`

**Usage**:

```console
$ hub install [OPTIONS] PACKAGE_URI
```

**Arguments**:

* `PACKAGE_URI`: URI to the package to install.Example: hub://guardrails/regex_match.  [required]

**Options**:

* `--install-local-models / --no-install-local-models`: Install local models
* `--quiet`: Run the command in quiet mode to reduce output verbosity.
* `--help`: Show this message and exit.

### `hub submit`

Submit a validator to the Guardrails AI team for review and
publishing.

**Usage**:

```console
$ hub submit [OPTIONS] PACKAGE_NAME [FILEPATH]
```

**Arguments**:

* `PACKAGE_NAME`: The package name for your validator.  [required]
* `[FILEPATH]`: The location to your validator file.  `[default: ./{package_name}.py]`

**Options**:

* `--help`: Show this message and exit.

### `hub uninstall`

Uninstall a validator from the Hub.

**Usage**:

```console
$ hub uninstall [OPTIONS] PACKAGE_URI
```

**Arguments**:

* `PACKAGE_URI`: URI to the package to uninstall. Example: hub://guardrails/regex_match.  [required]

**Options**:

* `--help`: Show this message and exit.

## `start`

**Usage**:

```console
$ start [OPTIONS]
```

**Options**:

* `--env TEXT`: An env file to load environment variables from.
* `--config TEXT`: A config file to load Guards from.
* `--port INTEGER`: The port to run the server on.  [default: 8000]
* `--help`: Show this message and exit.

## `validate`

Validate the output of an LLM against a `rail` spec.

**Usage**:

```console
$ validate [OPTIONS] RAIL LLM_OUTPUT
```

**Arguments**:

* `RAIL`: Path to the rail spec.  [required]
* `LLM_OUTPUT`: String of llm output.  [required]

**Options**:

* `--out TEXT`: Path to the compiled output directory.  [default: .rail_output]
* `--help`: Show this message and exit.

## `watch`

**Usage**:

```console
$ watch [OPTIONS]
```

**Options**:

* `--plain / --no-plain`: Do not use any rich formatting, instead printing each entry on a line.  [default: no-plain]
* `--num-lines INTEGER`: Print the last n most recent lines. If omitted, will print all history.  [default: 0]
* `--follow / --no-follow`: Continuously read the last output commands  [default: follow]
* `--log-path-override TEXT`: Specify a path to the log output file.
* `--help`: Show this message and exit.