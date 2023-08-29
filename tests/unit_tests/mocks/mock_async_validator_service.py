import asyncio


class MockAsyncValidatorService:
    async def async_validate(self, *args):
        await asyncio.sleep(0.1)
        # The return value doesn't really matter here.
        # We just need something to identify which class and method was called.
        return "MockAsyncValidatorService.async_validate", {"async": True}

    def validate(self, *args):
        return "MockAsyncValidatorService.validate", {"sync": True}
