import json
from typing import Any, Callable, Dict, Optional, Union, List
from opentelemetry.baggage import get_baggage
from opentelemetry import context
from opentelemetry.context import Context
from opentelemetry.trace import Tracer, Span

from guardrails.logger import logger
from guardrails.stores.context import (
    get_tracer as get_context_tracer,
    get_tracer_context,
)


def get_tracer(tracer: Optional[Tracer] = None) -> Optional[Tracer]:
    # TODO: Do we ever need to consider supporting non-otel tracers?
    _tracer = tracer if tracer is not None else get_context_tracer()
    return _tracer


def get_current_context() -> Union[Context, None]:
    otel_current_context = (
        context.get_current()
        if context is not None and hasattr(context, "get_current")
        else None
    )
    tracer_context = get_tracer_context()
    return otel_current_context or tracer_context


def get_span(span: Optional[Span] = None) -> Optional[Span]:
    if span is not None and hasattr(span, "add_event"):
        return span
    try:
        from opentelemetry import trace

        current_context = get_current_context()
        current_span = trace.get_current_span(current_context)
        return current_span
    except Exception as e:
        logger.error(e)
        return None


def serialize(val: Any) -> Optional[str]:
    try:
        if val is None:
            return None
        if hasattr(val, "to_dict"):
            return json.dumps(val.to_dict())
        elif hasattr(val, "__dict__"):
            return json.dumps(val.__dict__)
        elif isinstance(val, dict) or isinstance(val, list):
            return json.dumps(val)
        return str(val)
    except Exception:
        return None


def to_dict(val: Any) -> Dict:
    try:
        if val is None:
            return {}
        elif isinstance(val, dict):
            return val
        elif hasattr(val, "to_dict"):
            return val.to_dict()
        elif hasattr(val, "__dict__"):
            return val.__dict__
        else:
            return dict(val)
    except Exception:
        return {}


def wrap_with_otel_context(
    outer_scope_otel_context: Context, func: Callable[..., Any]
) -> Callable[..., Any]:
    """This function is designed to ensure that a given OpenTelemetry context
    is applied when executing a specified function. It is particularly useful
    for preserving the trace context when a guardrails is executed in a
    different execution flow or when integrating with other frameworks.

    Args:
        outer_scope_otel_context (Context): The OpenTelemetry context to apply
            when executing the function.
        func (Callable[..., Any]): The function to be executed within
            the given OpenTelemetry context.

    Returns:
        Callable[..., Any]: A wrapped version of 'func' that, when called,
            executes with 'outer_scope_otel_context' applied.
    """

    def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        # Attach the specified OpenTelemetry context before executing 'func'
        token = context.attach(outer_scope_otel_context)
        try:
            # Execute 'func' within the attached context
            return func(*args, **kwargs)
        finally:
            # Ensure the context is detached after execution
            #   to maintain correct context management
            context.detach(token)

    return wrapped_func


def add_user_attributes(span: Span):
    try:
        client_ip = get_baggage("client.ip") or "unknown"
        user_agent = get_baggage("http.user_agent") or "unknown"
        referrer = get_baggage("http.referrer") or "unknown"
        user_id = get_baggage("user.id") or "unknown"
        organization = get_baggage("organization") or "unknown"
        app = get_baggage("app") or "unknown"

        span.set_attribute("client.ip", str(client_ip))
        span.set_attribute("http.user_agent", str(user_agent))
        span.set_attribute("http.referrer", str(referrer))
        span.set_attribute("user.id", str(user_id))
        span.set_attribute("organization", str(organization))
        span.set_attribute("app", str(app))
    except Exception as e:
        logger.warning("Error loading baggage user information", e)
        pass


def redact(value: str) -> str:
    """Redacts all but the last four characters of the given string.

    Args:
        value (str): The string to be redacted.

    Returns:
        str: The redacted string with all but the last four characters
              replaced by asterisks.
    """
    redaction_length = len(value) - 4
    stars = "*" * redaction_length
    return f"{stars}{value[-4:]}"


def ismatchingkey(
    target_key: str,
    keys_to_match: tuple[str, ...] = ("key", "token", "password"),
) -> bool:
    """Check if the target key contains any of the specified keys to match.

    Args:
        target_key (str): The key to be checked.
        keys_to_match (tuple[str, ...], optional): A tuple of keys to match
                against the target key. Defaults to ("key", "token").

    Returns:
        bool: True if any of the keys to match are found in the target key,
              False otherwise.
    """
    for k in keys_to_match:
        if k in target_key:
            return True
    return False


def can_convert_to_dict(s: str) -> bool:
    """Check if a string can be converted to a dictionary.

    This function attempts to load the input string as JSON. If successful,
    it returns True, indicating that the string can be converted to a dictionary.
    Otherwise, it catches ValueError and TypeError exceptions and returns False.

    Args:
        s (str): The input string to be checked.

    Returns:
        bool: True if the string can be converted to a dictionary, False otherwise.
    """
    try:
        json.loads(s)
        return True
    except (ValueError, TypeError):
        return False


def recursive_key_operation(
    data: Optional[Union[Dict[str, Any], List[Any], str]],
    operation: Callable[[str], str],
    keys_to_match: List[str] = ["key", "token", "password"],
) -> Optional[Union[Dict[str, Any], List[Any], str]]:
    """Recursively traverses a dictionary, list, or JSON string and applies a
    specified operation to the values of keys that match any in the
    `keys_to_match` list. This function is useful for masking sensitive data
    (e.g., keys, tokens, passwords) in nested structures.

    Args:
        data (Optional[Union[Dict[str, Any], List[Any], str]]): The input data
            to traverse. This can bea dictionary, list, or JSON string. If a
            JSON string is provided, it will be parsed into a dictionary before
            processing.

        operation (Callable[[str], str]): A function that takes a string value
            and returns a modified string. This operation is applied to the values
            of keys that match any in `keys_to_match`.
        keys_to_match (List[str]): A list of keys to search for in the data. If
            a key matche any in this list, the corresponding value will be processed
            by the `operation`. Defaults to ["key", "token", "password"].

    Returns:
        Optional[Union[Dict[str, Any], List[Any], str]]: The modified data structure
        with the operation applied to the values of matched keys. The return type
        matches the input type (dict, list, or str).
    """
    if isinstance(data, str) and can_convert_to_dict(data):
        data_dict = json.loads(data)
        data = str(recursive_key_operation(data_dict, operation, keys_to_match))
    elif isinstance(data, dict):
        for key, value in data.items():
            if ismatchingkey(key, tuple(keys_to_match)) and isinstance(value, str):
                # Apply the operation to the value of the matched key
                data[key] = operation(value)
            else:
                # Recursively process nested dictionaries or lists
                data[key] = recursive_key_operation(value, operation, keys_to_match)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = recursive_key_operation(data[i], operation, keys_to_match)

    return data
