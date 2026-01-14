class HealthCheck:
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message

    def to_dict(self):
        return {"status": self.status, "message": self.message}
