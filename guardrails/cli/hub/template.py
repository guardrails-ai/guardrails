import json
import os

from guardrails.cli.server.hub_client import get_guard_template


def get_template(template_name: str) -> tuple[dict, str]:
    # if template ends in .json load file from disk relative to the execution directory

    if template_name.endswith(".json"):
        template_file_name = template_name
        file_path = os.path.join(os.getcwd(), template_name)
        with open(file_path, "r") as fin:
            return json.load(fin), template_file_name

    # todo - load this from the hub and create an appropriate file
    template_file_name = f"{template_name.split('/')[-1]}.json"
    template = get_guard_template(template_name)
    # write template to file
    out_path = os.path.join(os.getcwd(), template_file_name)
    print(f"Writing template to {out_path}")
    with open(out_path, "wt") as fout:
        fout.write(json.dumps(template, indent=4))

    return template, template_file_name
