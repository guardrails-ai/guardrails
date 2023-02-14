import guardrails as gd


class String(gd.Types):
    def __init__(name: Optional[str]):
        pass


class ToS(gd.Schema):

    fees: gd.String(name='Fees')
    interest_rates: gd.String(name='Interest Rates')
    code_snippet: gd.CodeSnippet(validator=None, name="Code")

    # Example of correctness
    @validator
    def validate_fees(self, interest_rates: str, **kwargs):
        # Make sure that interest rates contain some fractional percentage somewhere.
        pass


    def to_grammar(self):
        pass

    def add_schema_to_prompt(self):
        pass


    def as_json(self):
        pass

class Schema: ...

class PromptRepo:

    def __init__(self, schema: Schema):
        self.schema = schema

    


# ----------

import guardrails as gd

# Then we get PDFs.
pdfs = get_pdfs()

# User has to define a Terms of Service schema containing the following:
# 	fees: str
# 	interest_rates: str
# 	limitations: str
# 	liability: str
# 	privacy: str
# 	disputes: str
# 	account_termination: str
# 	regulatory_oversight: str

schema = gd.Schema(
    fees=gd.String(name='Fees', ),
    interest_rates=gd.Float(name='Interest Rates'),
    limitations=gd.String(name='Limitations'),
    liability=gd.String(name='Liability'),
    privacy=gd.String(name='Privacy'),
    disputes=gd.String(name='Disputes'),
    account_termination=gd.String(name='Account Terminat ion'),
    regulatory_oversight=gd.String(name='Regulatory Oversight'),
)

"""
    - Stateful execution of the LLM (i.e. keep some context around dialog style)
    Schema  ---|--> PromptRepo --|--> Prompt ---|--> Grammar ---|--> Schema
                     |      ^          ^                X
                     |------|          -----------------|
"""



# Then we call LLM on PDFs, and give them back info in schema they requested, after validation.
def llm(schema, pdfs, prompt_something):

    prompt = ...

    # Run the LLM
    response = manifest.run(prompt)

    # Run the response through Guardrails
    result: dict = gd.finalize(response)

    # result is a dictionary, containing keys
    # 	- fees (str)
    # 	- interest_rates (str)
    # 	- limitations (str)
    # 	- liability (str)
    # 	- privacy (str)
    # 	- disputes (str)
    # 	- account_termination (str)
    # 	- regulatory_oversight (str)

    # We can then return the result to the user.
    return result








    