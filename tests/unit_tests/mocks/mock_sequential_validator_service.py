import asyncio

class MockSequentialValidatorService:
    def validate(self, *args):
        return 'MockSequentialValidatorService.validate', { 'sync': True }