# Installing GuardRails AI 

Guardrails AI runs anywhere your python app runs. It is a simple pip install away.

```bash
pip install guardrails-ai
```

## Releases

Currently in beta, GuardRails AI maintains both stable and pre-release versions. 

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
pip install guardrails-ai==0.2.0a6
```

## Install from GitHub

Installing directly from GitHub is useful when a release has not yet been cut with the changes pushed to a branch that you need. Non-released versions may include breaking changes, and may not yet have full test coverage. We recommend using a released version whenever possible.

```bash
# pip install git+https://github.com/ShreyaR/guardrails.git@[branch/commit/tag]
# Examples:
pip install git+https://github.com/ShreyaR/guardrails.git@main
pip install git+https://github.com/ShreyaR/guardrails.git@dev
```

## Recommended Python Dependency Versions

The GuardRails package doesn't support pydantic versions 2.0.0 and above. We recommend using pydantic version 1.10.9.

```bash
pip install pydantic==1.10.9
```