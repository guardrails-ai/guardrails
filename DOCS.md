# Structure

Guardrails docs are served as a docusaurus site. The docs are compiled from various sources

1. Manually written markdown files in the `docs` directory
2. Python notebooks in the `docs` directory translated to markdown using nb-docs
3. Automatically generated python docs from the `guardrails` directory

These sources need to be built and compiled before the site can be served. 


## Installation

```bash
# Install poetry
pip install poetry

# Make sure you're in a venv (Recommended)
# Use conda or other venv management tools if you'd like
python -m venv .venv
source .venv/bin/activate

# Make the project
make full

# Serve the docs
npm run start
```

## How the build process works

1. pydocs is used to create python docs in the 'docs/' directory
1. a new folder called 'docs/build' is created
1. docs are copied from 'docs/src' to 'docs/build'
1. nbdocs is used on all notebooks in the 'docs/build' directory. This creates md files parallel to the notebooks in the dir structure.
1. md files are iterated and converted to mdx files. We import some custom components at the top of each mdx file.

## Troubleshooting/common problems

1. On first run, the docs build does not complete and the site is not served
    - This is usually an intermittent failure with nb-docs. Try running `npm run start` again
    - If this doesn't work, try running `rm -rf docs/build; npm run start`
    - If even that doesn't work, please file an issue. Something may be wrong with docs builds on the branch
1. I updated a notebook and it didn't update in the docs
    - This is likely because the notebook wasn't converted to markdown, or files were not overwritten
    - To fix this, run `rm -rf docs/build; npm run start`
