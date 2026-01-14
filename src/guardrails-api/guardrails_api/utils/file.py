def get_file_contents(file_path: str):
    try:
        contents = open(file_path)
        return contents
    except Exception:
        return None
