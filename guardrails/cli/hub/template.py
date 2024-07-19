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

VALID_TEMPLATES = [
    "chatbot",
]

is_valid_template = lambda x: x in VALID_TEMPLATES

# todo - load this from the hub
get_template = lambda x: TEMPLATES[x]
