class MockSequentialValidatorService:
    def validate(self, *args):
        return "MockSequentialValidatorService.validate", {"sync": True}
