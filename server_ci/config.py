import json
import os
from guardrails import AsyncGuard

try:
    file_path = os.path.join(os.getcwd(), "guard-template.json")
    with open(file_path, "r") as fin:
        guards = json.load(fin)["guards"] or []
except json.JSONDecodeError:
    print("Error parsing guards from JSON")
    SystemExit(1)

# instantiate guards
guard0 = AsyncGuard.from_dict(guards[0])
