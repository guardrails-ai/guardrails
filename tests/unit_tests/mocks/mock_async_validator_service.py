import asyncio

class MockAsyncValidatorService:
    async def async_validate(self, *args):
        await asyncio.sleep(0.1)
        return True, { 'async': True }