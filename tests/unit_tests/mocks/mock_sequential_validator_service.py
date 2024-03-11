class MockSequentialValidatorService:
    initialized: bool

    def __init__(self, *args, **kwargs):
        self.initialized = True

    def validate(self, *args):
        return "MockSequentialValidatorService.validate", {"sync": True}
