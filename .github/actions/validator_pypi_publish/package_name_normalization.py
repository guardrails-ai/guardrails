from packaging.utils import canonicalize_name  # PEP 503


def normalize_package_name(concatanated_name: str) -> str:
    return canonicalize_name(concatanated_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python package_name_normalization.py <concat-name>")
        sys.exit(1)

    concatenated_name = sys.argv[1]
    print(normalize_package_name(concatenated_name))
