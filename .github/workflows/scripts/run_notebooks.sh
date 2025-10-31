#!/bin/bash
export NLTK_DATA=/tmp/nltk_data;

# Remove the local guardrails directory and use the installed version
rm -rf ./guardrails

# Navigate to notebooks
cd docs/src/examples

# Get the notebook name from the matrix variable
notebook="$1"

# Check if the notebook should be processed
invalid_notebooks=("llamaindex-output-parsing.ipynb" "competitors_check.ipynb" "guardrails_server.ipynb" "valid_chess_moves.ipynb")
if [[ ! " ${invalid_notebooks[@]} " =~ " ${notebook} " ]]; then
  echo "Processing $notebook..."
  # poetry run jupyter nbconvert --to notebook --execute "$notebook"
  jupyter nbconvert --to notebook --execute "$notebook"
  if [ $? -ne 0 ]; then
    echo "Error found in $notebook"
    echo "Error in $notebook. See logs for details." >> errors.txt
    exit 1
  fi
fi

exit 0