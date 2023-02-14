import guardrails as gd


class String(gd.Types):
	def __init__(name: Optional[str]):
		pass


class ToS(gd.Schema):


	fees: gd.String(name='Fees', description="What fees and charges are associated with my account, including account maintenance fees, transaction fees, and overdraft fees?")
	interest_rates: gd.String(name='Interest Rates')
	code_snippet: gd.CodeSnippet(validator: , name="Code")

	# Example of correctness
	@validator
	def validate_fees(self, interest_rates: str, **kwargs):
		# Make sure that interest rates contain some fractional percentage somewhere.



	def to_grammar()


	def add_schema_to_prompt():


	def as_json():



----------

import guardrails as gd

# Then we get PDFs.

# User has to define a Terms of Service schema containing the following:
# 	fees: str
# 	interest_rates: str
# 	limitations: str
# 	liability: str
# 	privacy: str
# 	disputes: str
# 	account_termination: str
# 	regulatory_oversight: str


# Then we call LLM on PDFs, and give them back info in schema they requested, after validation.
def llm():
	# Run the LLM
	response = manifest.run(prompt)

	# Run the response through Guardrails
	result: dict = gd.finalize(response)

	# result is a dictionary, containing









	