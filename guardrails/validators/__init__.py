"""This module contains the validators for the Guardrails framework.

The name with which a validator is registered is the name that is used
in the `RAIL` spec to specify formatters.
"""

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from guardrails.validators.bug_free_python import BugFreePython
from guardrails.validators.bug_free_sql import BugFreeSQL
from guardrails.validators.competitor_check import CompetitorCheck
from guardrails.validators.detect_secrets import DetectSecrets, detect_secrets
from guardrails.validators.endpoint_is_reachable import EndpointIsReachable
from guardrails.validators.ends_with import EndsWith
from guardrails.validators.exclude_sql_predicates import ExcludeSqlPredicates
from guardrails.validators.extracted_summary_sentences_match import (
    ExtractedSummarySentencesMatch,
)
from guardrails.validators.extractive_summary import ExtractiveSummary
from guardrails.validators.is_high_quality_translation import IsHighQualityTranslation
from guardrails.validators.is_profanity_free import IsProfanityFree
from guardrails.validators.lower_case import LowerCase
from guardrails.validators.on_topic import OnTopic
from guardrails.validators.one_line import OneLine
from guardrails.validators.pii_filter import AnalyzerEngine, AnonymizerEngine, PIIFilter
from guardrails.validators.provenance import ProvenanceV0, ProvenanceV1
from guardrails.validators.pydantic_field_validator import PydanticFieldValidator
from guardrails.validators.qa_relevance_llm_eval import QARelevanceLLMEval
from guardrails.validators.reading_time import ReadingTime
from guardrails.validators.regex_match import RegexMatch
from guardrails.validators.remove_redundant_sentences import RemoveRedundantSentences
from guardrails.validators.saliency_check import SaliencyCheck
from guardrails.validators.similar_to_document import SimilarToDocument
from guardrails.validators.similar_to_list import SimilarToList
from guardrails.validators.sql_column_presence import SqlColumnPresence
from guardrails.validators.toxic_language import ToxicLanguage, pipeline
from guardrails.validators.two_words import TwoWords
from guardrails.validators.upper_case import UpperCase
from guardrails.validators.valid_choices import ValidChoices
from guardrails.validators.valid_length import ValidLength
from guardrails.validators.valid_range import ValidRange
from guardrails.validators.valid_url import ValidURL

__all__ = [
    # Validators
    "PydanticFieldValidator",
    "ValidRange",
    "ValidChoices",
    "LowerCase",
    "UpperCase",
    "ValidLength",
    "RegexMatch",
    "TwoWords",
    "OneLine",
    "ValidURL",
    "EndpointIsReachable",
    "BugFreePython",
    "BugFreeSQL",
    "SqlColumnPresence",
    "ExcludeSqlPredicates",
    "SimilarToDocument",
    "IsProfanityFree",
    "IsHighQualityTranslation",
    "EndsWith",
    "ExtractedSummarySentencesMatch",
    "ReadingTime",
    "ExtractiveSummary",
    "RemoveRedundantSentences",
    "SaliencyCheck",
    "QARelevanceLLMEval",
    "ProvenanceV0",
    "ProvenanceV1",
    "PIIFilter",
    "SimilarToList",
    "DetectSecrets",
    "ToxicLanguage",
    "CompetitorCheck",
    "OnTopic",
    # Validator helpers
    "detect_secrets",
    "AnalyzerEngine",
    "AnonymizerEngine",
    "pipeline",
    # Base classes
    "Validator",
    "register_validator",
    "ValidationResult",
    "PassResult",
    "FailResult",
]
