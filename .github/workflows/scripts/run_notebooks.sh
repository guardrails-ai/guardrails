#!/bin/bash

mkdir /tmp/nltk_data;
poetry run python -m nltk.downloader -d /tmp/nltk_data punkt;
export NLTK_DATA=/tmp/nltk_data;

cd docs/examples

# Function to process a notebook
process_notebook() {
    notebook="$1"
<<<<<<< HEAD
    if [ "$notebook" != "valid_chess_moves.ipynb" ] && [ "$notebook" != "translation_with_quality_check.ipynb" ] && [ "$notebook" != "competitors_check.ipynb" ]; then
=======
    invalid_notebooks=("valid_chess_moves.ipynb" "translation_with_quality_check.ipynb" "llamaindex-output-parsing.ipynb")
    if [[ ! " ${invalid_notebooks[@]} " =~ " ${notebook} " ]]; then
>>>>>>> 0.3.0
        echo "Processing $notebook..."
        poetry run jupyter nbconvert --to notebook --execute "$notebook"
        if [ $? -ne 0 ]; then
            echo "Error found in $notebook"
            echo "Error in $notebook. See logs for details." >> errors.txt
        fi
    fi
}

export -f process_notebook  # Export the function for parallel execution

# Create a file to collect errors
> errors.txt

# Run in parallel
ls *.ipynb | parallel process_notebook

# Check if there were any errors
if [ -s errors.txt ]; then
    echo "Some notebooks had errors"
    cat errors.txt
    exit 1
else
    echo "All notebooks ran successfully."
fi