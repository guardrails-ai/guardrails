#!/bin/bash
export NLTK_DATA=/tmp/nltk_data;

# Remove the local guardrails directory and use the installed version
rm -rf ./guardrails

# Navigate to notebooks
cd docs/examples

# Get the notebook name from the matrix variable
notebook="$1"

# Check if the notebook should be processed
invalid_notebooks=("llamaindex-output-parsing.ipynb" "competitors_check.ipynb" "guardrails_server.ipynb" "valid_chess_moves.ipynb")
if [[ ! " ${invalid_notebooks[@]} " =~ " ${notebook} " ]]; then
  echo "Processing $notebook..."

  echo "Guardrails Hub Init File Contents: "
  cat /home/runner/work/guardrails/guardrails/.venv/lib/python3.11/site-packages/guardrails/hub/__init__.py

  # Example install
  export GUARDRAILS_TOKEN=$(cat ~/.guardrailsrc| awk -F 'token=' '{print $2}' | awk '{print $1}' | tr -d '\n')
  pip install -vvv --index-url=https://__token__:$GUARDRAILS_TOKEN@pypi.guardrailsai.com/simple --extra-index-url=https://pypi.org/simple \
    guardrails-grhub-toxic-language

  # poetry run jupyter nbconvert --to notebook --execute "$notebook"
  jupyter nbconvert --to notebook --execute "$notebook"
  if [ $? -ne 0 ]; then
    echo "Error found in $notebook"
    echo "Error in $notebook. See logs for details." >> errors.txt
    exit 1
  fi
fi

exit 0