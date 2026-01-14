import json
import os

open_api_spec = None


def get_open_api_spec():
    global open_api_spec
    if not open_api_spec:
        api_spec_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "open-api-spec.json")
        )

        with open(api_spec_file_path) as api_spec_file:
            open_api_spec = json.loads(api_spec_file.read())

    return open_api_spec
