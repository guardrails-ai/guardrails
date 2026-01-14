import os
from lxml import etree as ET


class ConstantsContainer:
    def __init__(self):
        self._constants = {}
        self.fill_constants()

    def fill_constants(self) -> None:
        self_file_path = os.path.dirname(__file__)
        self_dirname = os.path.dirname(self_file_path)
        constants_file = os.path.abspath(
            os.path.join(self_dirname, "..", "constants.xml")
        )

        with open(constants_file, "r") as f:
            xml = f.read()

        parser = ET.XMLParser(encoding="utf-8", resolve_entities=False)
        parsed_constants = ET.fromstring(xml, parser=parser)

        for child in parsed_constants:
            if isinstance(child, ET._Comment):
                continue
            if isinstance(child, str):
                continue

            constant_name = child.tag
            constant_value = child.text
            self._constants[constant_name] = constant_value

    def __getitem__(self, key):
        return self._constants[key]

    def __setitem__(self, key, value):
        self._constants[key] = value

    def __delitem__(self, key):
        del self._constants[key]

    def __iter__(self):
        return iter(self._constants)

    def __len__(self):
        return len(self._constants)

    def __contains__(self, key):
        return key in self._constants

    def __repr__(self):
        return repr(self._constants)

    def __str__(self):
        return str(self._constants)

    def items(self):
        return self._constants.items()

    def keys(self):
        return self._constants.keys()

    def values(self):
        return self._constants.values()
