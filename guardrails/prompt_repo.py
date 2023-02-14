from jinja2 import Template


class Prompt(Template):

    def add_response_schema_to_prompt(self, response_schema: str):
        self.response_schema = response_schema


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
        return self.prompts

    def update_prompt(self, prompt_name: str, prompt_template: Prompt):
        for prompt in self.prompts:
            if prompt["name"] == prompt_name:
                prompt["template"] = prompt_template
