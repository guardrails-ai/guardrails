#!/bin/bash
export NLTK_DATA=/tmp/nltk_data;

cd docs/examples

# Get the notebook name from the matrix variable
notebook="$1"

# Check if the notebook should be processed
invalid_notebooks=("valid_chess_moves.ipynb" "llamaindex-output-parsing.ipynb" "competitors_check.ipynb")
if [[ ! " ${invalid_notebooks[@]} " =~ " ${notebook} " ]]; then
  echo "Processing $notebook..."
  poetry run jupyter nbconvert --to notebook --execute "$notebook"
  if [ $? -ne 0 ]; then
    echo "Error found in $notebook"
    echo "Error in $notebook. See logs for details." >> errors.txt
  fi
fi

exit 0