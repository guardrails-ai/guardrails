"""XML utilities."""
from typing import Dict, List, Union
import warnings

from lxml import etree as ET

from guardrails.datatypes import registry as types_registry
from guardrails.validators import types_to_validators, validators_registry, Validator



