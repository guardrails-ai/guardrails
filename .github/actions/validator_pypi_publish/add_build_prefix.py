import re
import sys


def add_package_name_prefix(pyproject_path, pep_503_new_package_name):
    # Read the existing pyproject.toml file
    with open(pyproject_path, "r") as f:
        content = f.read()

    try:
        # Check for the presence of the project.name key using a regular expression
        project_name_match = re.search(
            r'^name\s*=\s*"(.*?)"', content, flags=re.MULTILINE
        )
        if not project_name_match:
            print(f"Could not find the 'project.name' in {pyproject_path}.")
            sys.exit(1)
        existing_name = project_name_match.group(1)
    except Exception as e:
        print(f"Failed to parse project name in {pyproject_path}: {e}")
        sys.exit(1)

    # Update the project name to the new PEP 503-compliant name
    updated_content = re.sub(
        rf'(^name\s*=\s*")({re.escape(existing_name)})(")',
        rf"\1{pep_503_new_package_name}\3",
        content,
        flags=re.MULTILINE,
    )

    # Now we manually add the [tool.setuptools] section with the new folder name
    # If the section already exists, we append the correct package name
    setuptools_section = f"""

[tool.setuptools]
packages = ["{pep_503_new_package_name}"]

"""

    # Check if the [tool.setuptools] section already exists
    if "[tool.setuptools]" in updated_content:
        # If it exists, update the packages value
        updated_content = re.sub(
            r"(^\[tool\.setuptools\].*?^packages\s*=\s*\[.*?\])",
            f'[tool.setuptools]\npackages = ["{pep_503_new_package_name}"]',
            updated_content,
            flags=re.DOTALL | re.MULTILINE,
        )
    else:
        # If the section doesn't exist, append it at the end of the file
        updated_content += setuptools_section

    # Write the modified content back to the pyproject.toml file
    with open(pyproject_path, "w") as f:
        f.write(updated_content)

    print(
        "Updated project name to "
        "'{pep_503_new_package_name}' and added package folder in {pyproject_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <pyproject_path> <pep_503_new_package_name>")
        sys.exit(1)

    pyproject_path = sys.argv[1]
    pep_503_new_package_name = sys.argv[2]

    add_package_name_prefix(pyproject_path, pep_503_new_package_name)
