# Installing GuardRails AI 

Guardrails AI runs anywhere your python app runs. It is a simple pip install away.

```bash
pip install guardrails-ai
```

## Releases

Currently in beta, GuardRails AI maintains both stable and pre-release versions. 

Different versions can be found in the PyPi Release History:
https://pypi.org/project/guardrails-ai/#history


To install the latest, experimental pre-released version, run:

```bash
pip install --pre guardrails-ai
```

To install a specific version, run:

```bash
# pip install guardrails-ai==[version-number]

# Example:
pip install guardrails-ai==0.2.0a1
```

## Recommended Python Dependency Versions

The GuardRails package doesn't support pydantic versions 2.0.0 and above. We recommend using pydantic version 1.10.9.

```bash
pip install pydantic==1.10.9
```