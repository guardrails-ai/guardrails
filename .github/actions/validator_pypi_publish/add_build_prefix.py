import re
import sys
import toml


def add_package_name_prefix(pyproject_path, pep_503_new_package_name, validator_folder_name):
    # Read the existing pyproject.toml file
    with open(pyproject_path, "r") as f:
        content = f.read()
        parsed_toml = toml.loads(content)

    # get the existing package name
    existing_name = parsed_toml.get("project", {}).get("name")

    # Update the project name to the new PEP 503-compliant name
    # The package name would've been converted to PEP 503-compliant anyways
    # But we use this name since it's been concatenated with the seperator
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
packages = ["{validator_folder_name}"]

"""

    # Check if the [tool.setuptools] section already exists
    if "[tool.setuptools]" in updated_content:
        # If it exists, update the packages value
        updated_content = re.sub(
            r"(^\[tool\.setuptools\].*?^packages\s*=\s*\[.*?\])",
            f'[tool.setuptools]\npackages = ["{validator_folder_name}"]',
            updated_content,
            flags=re.DOTALL | re.MULTILINE,
        )
    else:
        # If the section doesn't exist, append it at the end of the file
        updated_content += setuptools_section

    # Write the modified content back to the pyproject.toml file
    with open(pyproject_path, "w") as f:
        f.write(updated_content)

    print(f"Updated project name to '{pep_503_new_package_name}'.")
    print(f"Added package folder '{validator_folder_name}' in {pyproject_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python script.py <pyproject_path>"
            " <pep_503_new_package_name> <validator-folder-name>"
        )
        sys.exit(1)

    pyproject_path = sys.argv[1]
    pep_503_new_package_name = sys.argv[2]
    validator_folder_name = sys.argv[3]

    add_package_name_prefix(pyproject_path, pep_503_new_package_name, validator_folder_name)
