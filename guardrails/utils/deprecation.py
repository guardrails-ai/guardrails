from string import Template

deprecation_message = Template(
    """'${name}' is deprecated and will be removed in \
versions ${removal_version} and beyond. Use ${replacement} instead."""
)
