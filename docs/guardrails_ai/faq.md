# Frequently Asked Questions

## I'm encountering an XMLSyntaxError when creating a `Guard` object from a `RAIL` specification. What should I do?

Make sure that you are escaping the `&` character in your `RAIL` specification. The `&` character has a special meaning in XML, and so you need to escape it with `&amp;`. For example, if you have a prompt like this:

```xml
<prompt>
    This is a prompt with an & character.
</prompt>
```

You need to escape the `&` character like this:

```xml
<prompt>
    This is a prompt with an &amp; character.
</prompt>
```

If you're still encountering issues, please [open an issue](https://github.com/ShreyaR/guardrails/issues/new) and we'll help you out!
