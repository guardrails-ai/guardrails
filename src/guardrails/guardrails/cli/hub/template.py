import json
import os

from guardrails.cli.server.hub_client import get_guard_template


def get_template(template_name: str) -> tuple[dict, str]:
    # if template ends in .json load file from disk relative to the execution directory
    if template_name.endswith(".json"):
        template_file_name = template_name
        try:
            file_path = os.path.join(os.getcwd(), template_name)
            with open(file_path, "r") as fin:
                return json.load(fin), template_file_name
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file {template_name} not found.")

    template_file_name = f"{template_name.split('/')[-1]}.json"

    template = get_guard_template(template_name)

    # write template to file
    out_path = os.path.join(os.getcwd(), template_file_name)
    with open(out_path, "wt") as file_out:
        file_out.write(json.dumps(template, indent=4))

    return template, template_file_name
