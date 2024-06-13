# Structure

Guardrails docs are served as a docusaurus site. The docs are compiled from various sources

1. Manually written markdown files in the `docs` directory
2. Python notebooks in the `docs` directory translated to markdown using nb-docs
3. Automatically generated python docs from the `guardrails` directory

These sources need to be built and compiled before the site can be served. 


## Installation

```bash
# First, make sure you're in a venv
python -m venv .venv
source .venv/bin/activate

# Make the project
make full

# Serve the docs
npm run start
```
