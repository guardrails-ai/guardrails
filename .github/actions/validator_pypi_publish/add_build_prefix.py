import toml
import re
import sys


def add_package_name_prefix(pyproject_path, validator_name):
    with open(pyproject_path, "r") as f:
        content = f.read()

    try:
        toml_data = toml.loads(content)
    except Exception as e:
        print(f"Failed to parse {pyproject_path}: {e}")
        sys.exit(1)

    existing_name = toml_data.get("project", {}).get("name")
    if not existing_name:
        print(f"Could not find the 'project.name' in {pyproject_path}.")
        sys.exit(1)

    validator_name = validator_name.split("/")
    new_name = f"{validator_name[0]}-grhub-{validator_name[1]}"

    updated_content = re.sub(
        rf'(^name\s*=\s*")({re.escape(existing_name)})(")',
        rf"\1{new_name}\3",
        content,
        flags=re.MULTILINE,
    )

    with open(pyproject_path, "w") as f:
        f.write(updated_content)

    print(f"Updated project name to '{new_name}' in {pyproject_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <pyproject_path> <validator-name>")
        sys.exit(1)

    pyproject_path = sys.argv[1]
    validator_name = sys.argv[2]

    add_package_name_prefix(pyproject_path, validator_name)
