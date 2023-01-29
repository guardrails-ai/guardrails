
import re


class Prompt:

    def __init__(self, source: str):
        self.source = source

        # Get all the variable names in the source string.
        # Variable names are surounded by curly braces, and may optionally contain a colon and a type.
        self.variable_names = re.findall(r'\{([\w\d_]+)\}', self.source)

    def add_output_schema_to_prompt(self, output_schema: str):
        self.output_schema = output_schema

    def append_to_prompt(self, text: str):
        self.source += text

    def __str__(self) -> str:
        return self.source

    def get_prompt_variables(self):
        return self.variable_names

    def format(self, **kwargs):
        """Format the prompt using the given keyword arguments."""
        return self.source.format(**kwargs)

    def make_vars_optional(self):
        """Make all variables in the prompt optional."""
        for var in self.variable_names:
            self.source = self.source.replace(f"{{{var}}}", f"{{{var}:}}")


class PromptRepo:
    def __init__(self):
        self.prompts = []

    def add_prompt(self, prompt_name: str, prompt_template: Prompt):
        # Make sure that the prompt name is not already in the repo.
        for prompt in self.prompts:
            if prompt["name"] == prompt_name:
                raise ValueError(f"Prompt with name {prompt_name} already exists.")
        self.prompts.append({"name": prompt_name, "template": prompt_template})

    def get_prompts(self):
        """Return a list of all prompts in the repo."""
        return self.prompts

    def get_prompt(self, prompt_name: str):
        """Return the prompt template for the given prompt name. If the prompt name is not found, return None."""
        for prompt in self.prompts:
            if prompt["name"] == prompt_name:
                return prompt["template"]
        return None

    def update_prompt(self, prompt_name: str, prompt_template: Prompt):
        """Update the prompt template for the given prompt name. If the prompt name is not found, raise an error."""
        found_prompt = False
        for prompt in self.prompts:
            if prompt["name"] == prompt_name:
                prompt["template"] = prompt_template
                return
        if not found_prompt:
            raise ValueError(f"Prompt with name {prompt_name} not found.")
