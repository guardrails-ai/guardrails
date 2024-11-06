def concat_name(validator_id):
    validator_id_parts = validator_id.split("/")
    namespace = validator_id_parts[0]
    package_name = validator_id_parts[1]
    return f"{namespace}-grhub-{package_name}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python concat_name.py <validator-id>")
        sys.exit(1)

    package_name = sys.argv[1]
    print(concat_name(package_name))
