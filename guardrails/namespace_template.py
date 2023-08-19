import string


class NamespaceTemplate(string.Template):
    delimiter = "$"
    idpattern = r"[a-z][_a-z0-9.]*"
