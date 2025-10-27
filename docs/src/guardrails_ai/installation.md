# Installing Guardrails AI 

Guardrails AI runs anywhere your python app runs. It is a simple pip install away.

```bash
pip install guardrails-ai
```

## Releases

Currently in beta, Guardrails AI maintains both stable and pre-release versions. 

Different versions can be found in the PyPi Release History:
https://pypi.org/project/guardrails-ai/#history


### Install Pre-Release Version
To install the latest, experimental pre-released version, run:

```bash
pip install --pre guardrails-ai
```

### Install specific version
To install a specific version, run:

```bash
# pip install guardrails-ai==[version-number]

# Example:
pip install guardrails-ai==0.5.0a10
```

## Install from GitHub

Installing directly from GitHub is useful when a release has not yet been cut with the changes pushed to a branch that you need. Non-released versions may include breaking changes, and may not yet have full test coverage. We recommend using a released version whenever possible.

```bash
# pip install git+https://github.com/guardrails-ai/guardrails.git@[branch/commit/tag]
# Examples:
pip install git+https://github.com/guardrails-ai/guardrails.git@main
pip install git+https://github.com/guardrails-ai/guardrails.git@0.5.0-dev
```


## Install Guardrails-JS

Guardrails AI also has a JavaScript version. To install the JavaScript version, run:

```bash
npm i git+https://github.com/guardrails-ai/guardrails-js.git
```
