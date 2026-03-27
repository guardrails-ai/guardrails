#!/bin/bash
ignore=(chatbot.ipynb translation_with_quality_check.ipynb guardrails_server.ipynb)
# Array to store notebook names
notebook_names="["

# Compile list of file names
for file in $(ls docs/examples/*.ipynb); do
  # Add the full filename with extension
  filename=$(basename "$file")
  if ! [[ ${ignore[*]} =~ "$filename" ]]
    then
      notebook_names+="\"$filename\","
  fi

done
notebook_names="${notebook_names%,}]"

# echo $notebook_names


# find line that begins with "notebook:" and replace it with notebook: $notebook_names
sed "s/notebook: \[.*\]/notebook: $notebook_names/" .github/workflows/examples_check.yml > .github/workflows/examples_check.yml.tmp
mv .github/workflows/examples_check.yml.tmp .github/workflows/examples_check.yml