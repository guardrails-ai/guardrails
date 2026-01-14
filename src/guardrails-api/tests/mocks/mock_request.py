from typing import Dict, Optional


class MockRequest:
    method: str
    json: Optional[Dict]
    args: Optional[Dict]
    headers: Optional[Dict]

    def __init__(
        self,
        method: str,
        json: Optional[Dict] = {},
        args: Optional[Dict] = {},
        headers: Optional[Dict] = {},
    ):
        self.method = method
        self.json = json
        self.args = args
        self.headers = headers
