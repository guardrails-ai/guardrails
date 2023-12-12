import os
from unittest import TestCase

import pytest

from guardrails.validator_base import FailResult, PassResult
from guardrails.validators.on_topic import OnTopic


class TestOnTopicIntegrationCPU(TestCase):
    def test_validate_valid_topic_cpu_disable_llm(self):
        validator = OnTopic(
            valid_topics=["sports", "politics"],
            disable_classifier=False,
            disable_llm=True,
        )
        text = "This is an article about sports."
        expected_result = PassResult()
        actual_result = validator.validate(text)
        self.assertEqual(actual_result, expected_result)

    def test_validate_invalid_topic_cpu_disable_llm(self):
        validator = OnTopic(
            valid_topics=["sports", "politics"],
            disable_classifier=False,
            disable_llm=True,
            model_threshold=0.6,
        )
        text = "This is an article about music."
        expected_result = FailResult(error_message="Most relevant topic is other.")
        actual_result = validator.validate(text)
        self.assertEqual(actual_result, expected_result)


# todo
# class TestOnTopicIntegrationGPU(TestCase):
#     def test_validate_valid_topic_gpu_disable_llm(self):
#         validator = OnTopic(
#             valid_topics=["technology", "science"],
#             device=1,  # Set to available GPU
#             disable_classifier=False,
#             disable_llm=True,
#         )
#         text = "This is an article about the latest scientific discoveries."
#         expected_result = PassResult()
#         actual_result = validate_text(validator, text)
#         self.assertEqual(actual_result, expected_result)

#     def test_validate_invalid_topic_gpu_disable_llm(self):
#         validator = OnTopic(
#             valid_topics=["technology", "science"],
#             device=1,  # Set to available GPU
#             disable_classifier=False,
#             disable_llm=True,
#         )
#         text = "This is an article about fashion trends."
#         expected_result = FailResult(error_message="Most relevant topic is other.")
#         actual_result = validate_text(validator, text)
#         self.assertEqual(actual_result, expected_result)


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") in [None, "mocked"],
    reason="openai api key not set",
)
class TestOnTopicIntegrationCPUllm(TestCase):
    def test_validate_valid_topic_cpu_enable_llm(self):
        validator = OnTopic(
            valid_topics=["politics", "history"],
            disable_classifier=False,
            disable_llm=False,
        )
        text = "This is a historical analysis of political events."
        expected_result = PassResult()
        actual_result = validator.validate(text)
        self.assertEqual(actual_result, expected_result)

    def test_validate_invalid_topic_cpu_enable_llm(self):
        validator = OnTopic(
            valid_topics=["politics", "history"],
            disable_classifier=False,
            disable_llm=False,
            model_threshold=0.99,
        )
        text = "This is an article about cooking recipes."
        expected_result = FailResult(error_message="Most relevant topic is other.")
        actual_result = validator.validate(text)
        self.assertEqual(actual_result, expected_result)


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") in [None, "mocked"],
    reason="openai api key not set",
)
class TestOnTopicIntegrationLlm(TestCase):
    def test_validate_valid_topic_disable_model(self):
        validator = OnTopic(
            valid_topics=["sports", "entertainment"],
            disable_classifier=True,
            disable_llm=False,
        )
        text = "This is a movie review for the new sports action film."
        expected_result = PassResult()
        actual_result = validator.validate(text)
        self.assertEqual(actual_result, expected_result)

    def test_validate_invalid_topic_disable_model(self):
        validator = OnTopic(
            valid_topics=["sports", "entertainment"],
            disable_classifier=True,
            disable_llm=False,
        )
        text = "This is a research paper on medical advancements."
        expected_result = FailResult(error_message="Most relevant topic is other.")
        actual_result = validator.validate(text)
        self.assertEqual(actual_result, expected_result)
