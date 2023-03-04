import re

from pyparsing import CaselessKeyword, Regex


fees = CaselessKeyword("1. Fees:") + Regex(".+", flags=re.IGNORECASE)
interest_rates = CaselessKeyword("2. Interest Rates:") + Regex(".+", flags=re.IGNORECASE)
limitations = CaselessKeyword("3. Limitations:") + Regex(".+", flags=re.IGNORECASE)
liability = CaselessKeyword("4. Liability:") + Regex(".+", flags=re.IGNORECASE)
privacy = CaselessKeyword("5. Privacy:") + Regex(".+", flags=re.IGNORECASE)
disputes = CaselessKeyword("6. Disputes:") + Regex(".+", flags=re.IGNORECASE)
account_termination = CaselessKeyword("7. Account Termination:") + Regex(".+", flags=re.IGNORECASE)
regulatory_oversight = CaselessKeyword("8. Regulatory Oversight:") + Regex(".+", flags=re.IGNORECASE)


terms_and_conditions = fees + interest_rates + limitations + liability + privacy + disputes + account_termination + regulatory_oversight
