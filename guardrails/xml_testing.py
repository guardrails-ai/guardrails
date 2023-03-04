from xml.etree import ElementTree as ET


def get_xml():
    """Get an XML file from the user."""
    xml_file = 'guardrails/prompt.xml'

    # Parse XML file.
    with open(xml_file, "r") as f:
        xml = f.read()

    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.fromstring(xml, parser=parser)

    # tree = ElementTree.fromstring(xml)
    print(tree)

    breakpoint()


if __name__ == "__main__":
    get_xml()