import toml
import re
import sys


def add_package_name_prefix(pyproject_path):
    with open(pyproject_path, "r") as f:
        content = f.read()

    try:
        toml_data = toml.loads(content)
    except Exception as e:
        print(f"Failed to parse {pyproject_path}: {e}")
        sys.exit(1)

    existing_name = toml_data.get("project", {}).get("name")
    if not existing_name:
        print("Could not find the 'project.name' in pyproject.toml.")
        sys.exit(1)

    new_name = f"guardrails-ai-validator-{existing_name}"

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
    add_package_name_prefix("pyproject.toml")
