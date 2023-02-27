from guardrails.prompt_repo import PromptRepo, Prompt
# from guardrails.types import DataType, String, URL, Email, Date, Time, Percentage, CodeSnippet, Float
from dataclasses import dataclass
from guardrails.validators import Validator, FormValidator
from guardrails.exceptions import SchemaMismatchException

import re
from copy import deepcopy

from pyparsing import CaselessKeyword, Regex

from typing import Dict, List, Optional, Union, Any, Type, TypeVar, Generic, Callable, Tuple, cast


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


class Field:
    def __init__(self, prompt: Optional[Prompt] = None, validator: Union[Validator, List[Validator]] = None):
        self._prompt = prompt

        # Check if validator is a class or an instance. If it is a class, instantiate it.
        if isinstance(validator, type):
            validator = validator()
        elif isinstance(validator, list):
            for i, v in enumerate(validator):
                if isinstance(v, type):
                    validator[i] = v()

        if isinstance(validator, list):
            self.form_validator = None
            self.content_validators = []
            for v in validator:
                if isinstance(v, Validator):
                    if isinstance(v, FormValidator):
                        self.form_validator = v
                    else:
                        self.content_validators.append(v)
                else:
                    raise ValueError("{v} must be of type Validator.")
            if self.form_validator is None:
                raise ValueError("Must have a form validator in the list of validators.")
        elif isinstance(validator, Validator):
            if not isinstance(validator, FormValidator):
                raise ValueError("Must have a form validator in the list of validators.")
            self.form_validator = validator
            self.content_validators = []
        else:
            raise ValueError("{validator} must be of type Validator or List[Validator].")

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: Prompt):
        self._prompt = prompt

    @classmethod
    def from_validator(cls, validator: Union[Validator, List[Validator]]):
        return cls(validator=validator)


class Schema:

    def __init__(
        self,
        _schema: List[Dict[str, Field]],
        name2idx: Dict[str, int],
        prompt_repo: Optional[PromptRepo] = None,
        base_prompt: Optional[Prompt] = None,
    ):
        self.name2idx = name2idx
        self._schema = _schema  
        self._registry = Registry()
        self._prompt_repo = prompt_repo
        self._base_prompt = base_prompt
        with open('openai_api_key.txt', 'r') as f:
            self.openai_api_key = f.read()

    def llm_ask(self, prompt):
        from openai import Completion
        llm_output = Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=2048,
            api_key=self.openai_api_key
        )
        return llm_output['choices'][0]['text']

    def to_prompt_repo(self):
        prompt_repo = PromptRepo()
        for field in self._schema:
            prompt_repo.add_prompt(field['name'], field['prompt'])
        return prompt_repo

    @property
    def prompt_repo(self):
        return self._prompt_repo

    @prompt_repo.setter
    def prompt_repo(self, prompt_repo: PromptRepo):
        self._prompt_repo = prompt_repo
        for prompt in self.prompt_repo.get_prompts():
            name = prompt['name']
            prompt_template = prompt['template']
            idx = self.name2idx[name]
            self._schema[idx]['field'].prompt = prompt_template

    @property
    def base_prompt(self):
        return self._base_prompt

    @base_prompt.setter
    def base_prompt(self, base_prompt: Prompt):
        self._base_prompt = base_prompt

    def validate_form_for_field(self, idx: int, name: str, field: Field, text: str) -> Tuple[str, str]:
        # combined_grammar = CaselessKeyword(f"{idx}. {name}:") + field.form_validator.grammar
        combined_grammar = CaselessKeyword(f"{name}:") + field.form_validator.grammar
        matched_form = list(combined_grammar.scan_string(text, maxMatches=1))
        if len(matched_form):
            text = text[matched_form[0][2]:]
            return matched_form[0][0][1]
        else:
            raise SchemaMismatchException

    def get_merged_prompt(self, format_prompts: str, content_prompts: str):
        """Merge the format and content prompts into a single prompt."""
        prompt = deepcopy(self.base_prompt)
        prompt.append_to_prompt(content_prompts)
        prompt.append_to_prompt(format_prompts)
        prompt.append_to_prompt("""
Try to be as correct and concise as possible. Find all relevant information in the document and answer the questions, even if the answer is 'None'.
If you are unsure of the answer, enter 'None'. If you answer incorrectly, you will be asked again until you get it right which is expensive.""")
        return prompt

    def merge_form_content_debugging(self, list_1, list_2):
        # List 1 and list 2 are both ordered lists
        # They may have different lengths and elements
        # We want to merge them in such a way that the elements are in the same order as they were in the original lists

        # Write code below
        # TODO(shreya): This doesn't work right now -- fix it.
        merged_list = []
        i = 0
        j = 0
        while i < len(list_1) and j < len(list_2):
            if self.name2idx[list_1[i]['name']] < self.name2idx[list_2[j]['name']]:
                merged_list.append(list_1[i])
                i += 1
            elif self.name2idx[list_1[i]['name']] > self.name2idx[list_2[j]['name']]:
                copied_item = deepcopy(list_2[j])
                copied_item['debugging'] = False
                merged_list.append(copied_item)
                j += 1
            else:
                merged_list.append(list_1[i])
                i += 1
                j += 1

        while i < len(list_1):
            merged_list.append(list_1[i])
            i += 1

        while j < len(list_2):
            copied_item = deepcopy(list_2[j])
            copied_item['debugging'] = False
            merged_list.append(copied_item)
            j += 1

        return merged_list

    def get_format_prompts(self, attributes: List[Dict], llm_output: Optional[str]):
        source = "\nTo answer these questions, respond in this format:\n"
        for attr in attributes:
            attr_name = attr['name']
            attr_field = attr['field']
            debugging = attr.get('debugging', False)
            attr_form = attr_field.form_validator.grammar_as_text
            source += f"{attr_name}: << {attr_form} >>"
            if debugging:
                attr_idx = self.name2idx[attr_name]
                attr_debug_prompt = attr_field.form_validator.debug(
                    # llm_output, placeholder=CaselessKeyword(f"{attr_idx}. {attr_name}:"))
                    llm_output, placeholder=CaselessKeyword(f"{attr_name}:"))
                source += f". {attr_debug_prompt}"
            source += "\n"
        return source

    def get_content_prompts(self, attributes: List[Dict], extracted_object: Dict[str, Any]):
        source = "\nQuestions:\n"
        for attr in attributes:
            attr_name = attr['name']
            attr_field = attr['field']
            attr_prompt = attr_field.prompt
            source += f"{attr_name}: {attr_prompt}"

            debugging = attr.get('debugging', False)
            if debugging:
                field = extracted_object[attr_name]
                # attr_idx = self.name2idx[attr_name]
                attr_debug_prompt = attr_field.form_validator.debug(field)
                source += f". {attr_debug_prompt}"
            source += "\n"
        return source

    def extract_schemified_response(self, text: str):
        """Validate the structure of the text."""

        extracted_object = {}
        prev_llm_output = None
        format_attributes = deepcopy(self._schema)
        content_attributes = deepcopy(self._schema)
        attributes_to_extract = deepcopy(self._schema)

        iteration = 0

        while(True):
            print('\n\n\n\nIteration =', iteration)
            print('Prev llm output =', prev_llm_output)

            iteration += 1
            format_prompts = self.get_format_prompts(
                format_attributes,
                llm_output=prev_llm_output
            )
            content_prompts = self.get_content_prompts(
                content_attributes,
                extracted_object=extracted_object,
            )

            final_prompt = self.get_merged_prompt(format_prompts, content_prompts)
            llm_output = self.llm_ask(final_prompt.format(document=text))
            print(f'HERE LLM Output: {llm_output}')
            print('Finished printing LLM Output')

            format_attributes = []
            content_attributes = []

            for i, attr in enumerate(attributes_to_extract):
                field = attr['field']
                name = attr['name']
                try:
                    extracted_object[name] = self.validate_form_for_field(i, name, field, llm_output)

                    for content_validator in field.content_validators:
                        if not content_validator.validate(extracted_object[name]):
                            attr['debugging'] = True
                            content_attributes.append(attr)
                except SchemaMismatchException:
                    attr['debugging'] = True
                    format_attributes.append(attr)
                    continue

            prev_llm_output = llm_output

            print(f'LLM Output: {llm_output}')
            print(f'Prev LLM Output: {prev_llm_output}')
            print('Finished printing LLM Output')

            old_content_attributes = deepcopy(content_attributes)
            content_attributes = self.merge_form_content_debugging(
                content_attributes, format_attributes)

            format_attributes = self.merge_form_content_debugging(
                format_attributes, old_content_attributes)
            attributes_to_extract = content_attributes

            if len(attributes_to_extract) == 0:
                break

        print(f'Schema: {self._schema}')

        return extracted_object

    def add_to_prompt(self, prompt: Prompt) -> Prompt:
        """Add the schema to the prompt."""

        # assert self.prompts_available(), "All fields must have prompts set."
        prompt_copy = deepcopy(prompt)

        prompt_str = "\n\nQuestions:\n"
        form_str = "\nTo answer these questions, respond in this format:\n"

        for i, item in enumerate(self._schema):
            attr_name = item['name']
            attr_prompt = item['field'].prompt
            attr_form = item['field'].form_validator.grammar_as_text
            prompt_str += f"{i}. {attr_name}: {attr_prompt}\n"
            form_str += f"{i}. {attr_name}: << {attr_form} >>\n"

        prompt_copy.append_to_prompt(prompt_str)
        prompt_copy.append_to_prompt(form_str)
        prompt_copy.append_to_prompt("""
Try to be as correct and concise as possible. Find all relevant information in the document and answer the questions, even if the answer is 'None'.
If you are unsure of the answer, enter 'None'. If you answer incorrectly, you will be asked again until you get it right which is expensive.""")
        return prompt_copy

    def get_validated_llm_output(self, document: str) -> Dict[str, Any]:
        """Get the validated output of the LLM."""

        llm_output = self.llm(self.base_prompt.format(document=document))
        return self.validate_and_debug_text(llm_output)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Union[Field, Validator]],
        base_prompt: Prompt,
        prompt_repo: Optional[PromptRepo] = None,
    ):
        # Use a name2idx dictionary to maintain the order of the fields in the schema.
        # TODO(shreya): This is a temporary workaround, reevaluate this later.
        _schema = []
        name2idx = {}

        for idx, (name, field) in enumerate(data.items()):
            name2idx[name] = idx

            if isinstance(field, Validator):
                field = Field.from_validator(field)

            _schema.append({'name': name, 'field': field})

        return cls(_schema=_schema, name2idx=name2idx, prompt_repo=prompt_repo,
            base_prompt=base_prompt
        )
