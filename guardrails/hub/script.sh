#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <pip-package-name> <relative-directory-path>"
    exit 1
fi

# Assign arguments to variables
PACKAGE_NAME=$1
RELATIVE_DIR_PATH=$2

# Get the site-packages directory for the installed pip package
SITE_PACKAGES_DIR=$(python3 -c "import os, site; print(next(p for p in site.getsitepackages() if p.endswith('site-packages')))")

# Get the last folder name from the relative directory path
LAST_FOLDER_NAME=$(basename "$RELATIVE_DIR_PATH")


TARGET_DIR="$SITE_PACKAGES_DIR/$PACKAGE_NAME/$RELATIVE_DIR_PATH"

# echo "Target directory: $TARGET_DIR"

# Create a symlink in the current directory to the target directory

if [ -L "./$LAST_FOLDER_NAME" ]; then
    rm "./$LAST_FOLDER_NAME"
    echo "Removed existing symlink: ./$LAST_FOLDER_NAME"
fi

ln -s "$TARGET_DIR" "./$LAST_FOLDER_NAME"

echo "Symlink created: ./$LAST_FOLDER_NAME -> $TARGET_DIR"

echo "__________________________Contents of the __init__.py file__________________________"

cat $SITE_PACKAGES_DIR/$PACKAGE_NAME/hub/__init__.py
