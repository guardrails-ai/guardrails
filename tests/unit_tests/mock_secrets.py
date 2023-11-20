SECRETS_CODE_SNIPPET = """
import os
import openai

SECRET_TOKEN = "DUMMY_SECRET_TOKEN_abcdefgh"

ADMIN_CREDENTIALS = {"username": "admin", "password": "dummy_admin_password"}
"""

EXPECTED_SECRETS_CODE_SNIPPET = """
import os
import openai

SECRET_TOKEN = "********"

ADMIN_CREDENTIALS = {"username": "admin", "password": "********"}
"""

NO_SECRETS_CODE_SNIPPET = """
import os
import openai

ADMIN_INFO = {"username": "admin", "country": "United States"}
countries = ["United States", "Canada", "Mexico"]
for country in countries:
    print(country)
    if country == "United States":
        print("Found admin_info for United States")
"""
