import json
import unittest
from unittest.mock import patch

from guardrails.validator_base import FailResult, PassResult
from guardrails.validators.sensitive_topic import SensitiveTopic


class TestSensitiveTopic(unittest.TestCase):
    def setUp(self):
        self.device = -1
        self.model = "facebook/bart-large-mnli"
        self.llm_callable = "gpt-3.5-turbo"

    def test_init_with_valid_args(self):
        validator = SensitiveTopic(
            device=self.device,
            model=self.model,
            llm_callable=self.llm_callable,
            on_fail=None,
        )
        self.assertEqual(validator._device, self.device)
        self.assertEqual(validator._model, self.model)
        self.assertEqual(validator._llm_callable.__name__, "openai_callable")

    def test_init_with_invalid_llm_callable(self):
        with self.assertRaises(ValueError):
            SensitiveTopic(
                llm_callable="invalid_model",
            )

    def test_get_topics_ensemble(self):
        text = "This is an article about sports."
        candidate_topics = ["sports", "politics", "technology"]
        validator = SensitiveTopic()

        with patch.object(validator, "get_topic_zero_shot") as mock_zero_shot:
            mock_zero_shot.return_value = ("sports", 0.6)

            topics = validator.get_topics_ensemble(text, candidate_topics)
            self.assertEqual(topics, ["sports", "sports", "sports"])

    def test_get_topics_llm(self):
        text = "This is an article about politics."
        candidate_topics = ["sports", "politics", "technology"]
        validator = SensitiveTopic()

        with patch.object(validator, "call_llm") as mock_llm:
            mock_llm.return_value = '{"topic": "politics"}'

            validation_result = validator.get_topics_llm(text, candidate_topics)
            self.assertEqual(validation_result, ["politics", "politics", "politics"])

    def test_set_callable_string(self):
        validator = SensitiveTopic()
        validator.set_callable("gpt-3.5-turbo")
        self.assertEqual(validator._llm_callable.__name__, "openai_callable")

    def test_set_callable_callable(self):
        def custom_callable(text, topics):
            return json.dumps({"topic": topics[0]})

        validator = SensitiveTopic()
        validator.set_callable(custom_callable)
        self.assertEqual(validator._llm_callable.__name__, "custom_callable")

    def test_get_topics_zero_shot(self):
        text = "This is an article about technology."
        candidate_topics = ["sports", "politics", "technology"]
        validator = SensitiveTopic(sensitive_topics=candidate_topics)

        topics = validator.get_topics_zero_shot(text, candidate_topics)
        self.assertEqual(
            ["other", "other", "technology"],
            topics,
        )

        with patch.object(validator, "get_topic_zero_shot") as mock_zero_shot:
            mock_zero_shot.return_value = ("technology", 0.6)
            topics = validator.get_topics_zero_shot(text, candidate_topics)
            self.assertEqual(
                ["technology", "technology", "technology"],
                topics,
            )

    def test_validate_message_without_sensitive_topic(self):
        text = "This is an article about sports."
        validator = SensitiveTopic(
            sensitive_topics=["violence"],
        )
        validation_result = validator.validate(text, metadata={})
        self.assertEqual(validation_result, PassResult())

    def test_validate_message_with_sensitive_topic(self):
        text = "This is an article about sports."
        validator = SensitiveTopic(
            sensitive_topics=["sports"],
        )
        validation_result = validator.validate(text, metadata={})
        self.assertEqual(
            validation_result,
            FailResult(
                error_message="Sensitive topics detected: sports",
                fix_value="Trigger warning:\n- sports\n\n"
                "This is an article about sports.",
            ),
        )

    def test_validate_invalid_topic(self):
        validator = SensitiveTopic()
        with patch.object(validator, "call_llm") as mock_llm:
            mock_llm.return_value = '{"topic": "other"}'

            text = "This is an article about music."
            validation_result = validator.validate(text, metadata={})

            self.assertEqual(validation_result, PassResult())
