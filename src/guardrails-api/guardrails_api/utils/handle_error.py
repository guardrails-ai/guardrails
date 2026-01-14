from functools import wraps
import traceback
from guardrails_api.classes.http_error import HttpError
from guardrails_api.utils.logger import logger
from guardrails.errors import ValidationError


from fastapi import HTTPException


def handle_error(func=None):
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except ValidationError as validation_error:
                logger.error(validation_error)
                traceback.print_exception(
                    type(validation_error),
                    validation_error,
                    validation_error.__traceback__,
                )
                raise HTTPException(status_code=400, detail=str(validation_error))
            except HttpError as http_error:
                logger.error(http_error)
                traceback.print_exception(type(http_error), http_error, http_error.__traceback__)
                raise HTTPException(status_code=http_error.status_code, detail=http_error.detail)
            except HTTPException as http_exception:
                logger.error(http_exception)
                traceback.print_exception(
                    type(http_exception), http_exception, http_exception.__traceback__
                )
                raise
            except Exception as e:
                logger.error(e)
                traceback.print_exception(type(e), e, e.__traceback__)
                raise HTTPException(status_code=500, detail="Internal Server Error")

        return wrapper

    if func:
        return decorator(func)
    return decorator
