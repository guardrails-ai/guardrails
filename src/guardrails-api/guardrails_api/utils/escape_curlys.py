from typing import Optional

open_curly_token = "#gr<<#gr"
close_curly_token = "#gr>>#gr"


def escape_curlys(input: Optional[str] = None) -> str:
    if input is not None:
        input = input.replace("{", open_curly_token)
        input = input.replace("}", close_curly_token)
    return input


def descape_curlys(input: Optional[str] = None) -> str:
    if input is not None:
        input = input.replace(open_curly_token, "{")
        input = input.replace(close_curly_token, "}")
    return input
