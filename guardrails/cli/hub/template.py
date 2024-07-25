import json
import os

TEMPLATES = {
    "chatbot": {
        "name": "chat_pack",
        "description": "a collection of basic guards to facilitate chat w llms",
        "template_version": "0.0.1",
        "guards": [
            {
                "id": "chatbot",
                "name": "Chatbot",
                "validators": [
                    {
                        "id": "guardrails/detect_pii",
                        "on": "msg_history",
                        "onFail": "filter",
                        "kwargs": {"pii_entities": ["PHONE_NUMBER"]},
                    },
                    {
                        "id": "guardrails/detect_pii",
                        "on": "$",
                        "onFail": "filter",
                        "kwargs": {"pii_entities": ["LOCATION"]},
                    },
                ],
                "output_schema": {"type": "string"},
            }
        ],
    }
}


def get_template(template_name: str) -> tuple[dict, str]:
    # if template ends in .json load file from disk relative to the execution directory

    if template_name.endswith(".json"):
        template_file_name = template_name
        file_path = os.path.join(os.getcwd(), template_name)
        with open(file_path, "r") as fin:
            return json.load(fin), template_file_name

    # todo - load this from the hub and create an appropriate file
    return TEMPLATES[template_name], template_file_name
