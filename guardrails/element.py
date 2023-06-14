from typing import Any, List, Optional, Union
import lxml.etree as ET

from guardrails.validators import Validator

# class Element:
#     def __init__(self, tag, attrib=None, **kwargs):
#         self.tag = tag
#         self.attrib = attrib or {}
#         self.children = []
#         self.text = kwargs.get("text", None)

#     def set(self, key, value):
#         self.attrib[key] = value

#     def append(self, child):
#         self.children.append(child)

#     def tostring(self):
#         attrs = " ".join([f'{k}="{v}"' for k, v in self.attrib.items()])
#         start_tag = f"<{self.tag} {attrs}>" if attrs else f"<{self.tag}>"
#         end_tag = f"</{self.tag}>"
#         children = "".join([child.tostring() for child in self.children])
#         return f"{start_tag}{self.text}{children}{end_tag}"


class Element:
    """
    Element class
    """

    def __init__(
        self,
        element_type: str,
        name: str,
        description: Optional[str],
        value: Optional[Any] = None,
        path: Optional[List["Element"]] = None,
        children: List["Element"] = [],
        validators: List[Validator] = [],
    ):
        self.type = element_type
        self.name = name
        self.description = description
        self.value = value
        self.path = path
        self.children = children
        self.validators = validators

    def __repr__(self):
        return f"{self.name}({self.value})"

    def __str__(self):
        return f"{self.name}({self.value})"

    def __eq__(self, other):
        if isinstance(other, Element):
            return self.name == other.name and self.value == other.value
        return False

    def __hash__(self):
        return hash((self.name, self.value))

    def __iter__(self):
        return iter(self.children)

    @classmethod
    def from_xml(cls, element: ET._Element) -> "Element":
        """Create an Element from an XML element."""
        element_type = element.tag

        if "name" in element.attrib:
            name = element.attrib["name"]
        else:
            raise KeyError(f"Element {element_type} does not have a name attribute")

        if "description" in element.attrib:
            description = element.attrib["description"]
        else:
            description = None

        if "format" in element.attrib:
            # TODO(shreya): Get list of validators here. Can probably reuse some
            # functions from other places.
            validators = []
        else:
            validators = []

        children = list(element)
        if children:
            children = [Element.from_xml(child) for child in children]

        return cls(
            element_type=element_type,
            name=name,
            description=description,
            children=children,
            validators=validators,
        )

    def add_children(self, children: Union["Element", List["Element"]]) -> None:
        if isinstance(children, Element):
            self.children.append(children)
        elif isinstance(children, list):
            self.children.extend(children)

    def tostring(self) -> str:
        # TODO(shreya): Figure out the best way to represent element as string (do we want to convert to XML?)
        # """Convert the Element to an XML string."""
        # element = ET.Element(self.type, name=self.name, description=self.description)
        # for child in self.children:
        #     element.append(child.tostring())
        # return ET.tostring(element, pretty_print=True).decode("utf-8")
        pass
