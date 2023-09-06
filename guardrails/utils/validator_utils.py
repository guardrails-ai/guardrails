# flake8: noqa
"""This module contains the constants and utils used by the validator.py."""


PROVENANCE_V1_PROMPT = """Instruction:
As an Attribution Validator, you task is to verify whether the following contexts support the claim:

Claim:
{}

Contexts:
{}

Just respond with a "Yes" or "No" to indicate whether the given contexts support the claim.
Response:"""
