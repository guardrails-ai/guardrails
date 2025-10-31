#!/bin/bash
# Array to store notebook names
notebook_names="["

# Compile list of file names
for file in $(ls docs/src/examples/*.ipynb); do
  # Add the full filename with extension
  filename=$(basename "$file")

  notebook_names+="\"$filename\","
done
notebook_names="${notebook_names%,}]"

# echo $notebook_names


# find line that begins with "notebook:" and replace it with notebook: $notebook_names
sed "s/notebook: \[.*\]/notebook: $notebook_names/" .github/workflows/examples_check.yml > .github/workflows/examples_check.yml.tmp
mv .github/workflows/examples_check.yml.tmp .github/workflows/examples_check.yml