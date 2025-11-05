# Formatters

## BaseFormatter

```python
class BaseFormatter(ABC)
```

A Formatter takes an LLM Callable and wraps the method into an abstract
callable.

Used to perform manipulations of the input or the output, like JSON
constrained- decoding.

## JsonFormatter

```python
class JsonFormatter(BaseFormatter)
```

A formatter that uses Jsonformer to ensure the shape of structured data
for Hugging Face models.

