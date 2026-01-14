class HttpError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        cause: str = None,
        fields: dict = None,
        context: str = None,
    ):
        self.status = status
        self.message = message
        self.cause = cause
        self.fields = fields
        self.context = context

    def to_dict(self):
        response = {"status": self.status, "message": self.message}

        if self.cause is not None:
            response["cause"] = self.cause
        if self.fields is not None:
            response["fields"] = self.fields
        if self.context is not None:
            response["context"] = self.context

        return response
