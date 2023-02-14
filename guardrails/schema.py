from guardrails.prompt_repo import PromptRepo
from guardrails.types import BaseType, String, URL, Email, Date, Time, Percentage, CodeSnippet, Float
from dataclasses import dataclass

from jinja2 import Environment


class Registry:
    def __init__(self):
        self.methods = []

    def register(self, method):
        self.methods.append(method)


def register_method(method):
    def wrapper(self, *args, **kwargs):
        setattr(self.__class__, method.__name__, wrapper)
        if not hasattr(self, "_registry"):
            self._registry = Registry()
        self._registry.register(method)
        return method(self, *args, **kwargs)
    return wrapper


@dataclass
class Schema:

    prompt_repo: PromptRepo = None

    # Write post init method for the dataclass.
    def __post_init__(self):
        self.attributes = [attr for attr in dir(self) if isinstance(getattr(self, attr), BaseType)]

        # Call all methods of the subclass that start with `validate_`.
        self.validator_methods = [method for method in dir(self) if callable(getattr(self, method)) and method.startswith("validate_")]

    # def (self, prompt_repo: PromptRepo=None):
        # List all of the attributes of the subclass that are of type `guardrails.types`.
        # self.attributes = [attr for attr in dir(self) if isinstance(getattr(self, attr), BaseType)]
        # for attr in self.attributes:
            # Get the type of the attribute.
            # attr_type = getattr(self, attr)
            # Get the name of the attribute.
            # attr_name = attr_type.name
            # Get the validator of the attribute.
            # attr_validator = attr_type.validator
            # Add the prompt to the prompt repo.
            # prompt_repo.add_prompt(
                # prompt_name=attr_name,
                # prompt_template=attr_type,
                # validator=attr_validator
            # )

    def to_grammar(self):
        pass

    def add_schema_to_prompt(self):
        pass

    def as_json(self):
        pass

    @register_method
    def my_method(self):
        print("my_method called")

    def create_prompt_response_schema(self):
        template_str = "Questions:\n"
        for i, attr in enumerate(self.attributes):
            attr_name = getattr(self, attr).name
            attr_prompt = getattr(self, attr).prompt_template
            template_str += f"{i}. {attr_name}: {attr_prompt}"
        
        template_str += "\nTo answer these questions, respond in this format:\n"

        for i, attr in enumerate(self.attributes):
            




@dataclass
class ToS(Schema):
    fees: BaseType = String(name='Fees')
    interest_rates: BaseType = Float(name='Interest Rates')
    limitations: BaseType = String(name='Limitations')
    liability: BaseType = String(name='Liability')
    privacy: BaseType = String(name='Privacy')
    disputes: BaseType = String(name='Disputes')
    account_termination: BaseType = String(name='Account Termination')
    regulatory_oversight: BaseType = String(name='Regulatory Oversight')

    # Example of correctness
    # @validator
    def validate_fees(self, interest_rates: str, **kwargs):
        # Make sure that interest rates contain some fractional percentage somewhere.
        pass

    def to_grammar(self):
        pass

    def add_schema_to_prompt(self):
        pass

    def as_json(self):
        pass
