import json
from unittest.mock import patch
import unittest
from guardrails.validators.on_topic import OnTopic
from guardrails.validator_base import PassResult, FailResult


class TestOnTopic(unittest.TestCase):
    def setUp(self):
        self.valid_topics = ["sports", "politics", "technology"]
        self.invalid_topics = ["other"]
        self.device = -1
        self.model = "facebook/bart-large-mnli"
        self.llm_callable = "gpt-3.5-turbo"

    def test_init_with_valid_args(self):
        validator = OnTopic(
            valid_topics=self.valid_topics,
            invalid_topics=self.invalid_topics,
            device=self.device,
            model=self.model,
            llm_callable=self.llm_callable,
            on_fail=None,
        )
        self.assertEqual(validator._valid_topics, self.valid_topics)
        self.assertEqual(validator._invalid_topics, self.invalid_topics)
        self.assertEqual(validator._device, self.device)
        self.assertEqual(validator._model, self.model)
        self.assertEqual(validator._llm_callable.__name__, "openai_callable")

    def test_init_with_invalid_llm_callable(self):
        with self.assertRaises(ValueError):
            OnTopic(
                valid_topics=self.valid_topics,
                invalid_topics=self.invalid_topics,
                llm_callable="invalid_model",
            )

    def test_get_topic_ensemble(self):
        text = "This is an article about sports."
        candidate_topics = ["sports", "politics", "technology"]
        validator = OnTopic(valid_topics=candidate_topics)

        with patch.object(validator, "get_topic_zero_shot") as mock_zero_shot:
            mock_zero_shot.return_value = ("sports", 0.6)

            validation_result = validator.get_topic_ensemble(text, candidate_topics)
            self.assertEqual(validation_result, PassResult())

    def test_get_topic_llm(self):
        text = "This is an article about politics."
        candidate_topics = ["sports", "politics", "technology"]
        validator = OnTopic(valid_topics=candidate_topics)

        with patch.object(validator, "call_llm") as mock_llm:
            mock_llm.return_value = '{"topic": "politics"}'

            validation_result = validator.get_topic_llm(text, candidate_topics)
            self.assertEqual(validation_result, PassResult())

    def test_get_topic_llm_invalid_topic(self):
        text = "This is an article about science."
        candidate_topics = ["sports", "politics", "technology"]
        validator = OnTopic(valid_topics=candidate_topics)

        with patch.object(validator, "call_llm") as mock_llm:
            mock_llm.return_value = '{"topic": "science"}'

            validation_result = validator.get_topic_llm(text, candidate_topics)
            self.assertEqual(
                validation_result,
                FailResult(error_message="Most relevant topic is science."),
            )

    def test_verify_topic(self):
        validator = OnTopic(valid_topics=self.valid_topics)

        validation_result = validator.verify_topic("sports")
        self.assertEqual(validation_result, PassResult())

        validation_result = validator.verify_topic("other")
        self.assertEqual(
            validation_result, FailResult(error_message="Most relevant topic is other.")
        )

    def test_set_callable_string(self):
        validator = OnTopic(valid_topics=self.valid_topics)
        validator.set_callable("gpt-3.5-turbo")
        self.assertEqual(validator._llm_callable.__name__, "openai_callable")

    def test_set_callable_callable(self):
        def custom_callable(text, topics):
            return json.dumps({"topic": topics[0]})

        validator = OnTopic(valid_topics=self.valid_topics)
        validator.set_callable(custom_callable)
        self.assertEqual(validator._llm_callable.__name__, "custom_callable")

    def test_get_topic_zero_shot(self):
        text = "This is an article about technology."
        candidate_topics = ["sports", "politics", "technology"]
        validator = OnTopic(valid_topics=candidate_topics)

        validation_result = validator.get_topic_zero_shot(text, candidate_topics)
        self.assertEqual(validation_result[0], "technology")

        with patch.object(validator, "call_llm") as mock_llm:
            mock_llm.return_value = '{"topic": "technology"}'
            validation_result = validator.get_topic_zero_shot(text, candidate_topics)
            self.assertEqual(validation_result[0], "technology")

    def test_validate_valid_topic(self):
        text = "This is an article about sports."
        validator = OnTopic(valid_topics=self.valid_topics)
        validation_result = validator.validate(text)
        self.assertEqual(validation_result, PassResult())

    def test_validate_invalid_topic(self):
        validator = OnTopic(valid_topics=self.valid_topics)
        with patch.object(validator, "call_llm") as mock_llm:
            mock_llm.return_value = '{"topic": "other"}'

            text = "This is an article about music."
            validation_result = validator.validate(text)

            self.assertEqual(
                validation_result,
                FailResult(error_message="Most relevant topic is other."),
            )

    def test_validate_no_valid_topics(self):
        with self.assertRaises(ValueError):
            validator = OnTopic(valid_topics=[])
            validator.validate("This is a test text.")

    def test_validate_overlapping_topics(self):
        with self.assertRaises(ValueError):
            validator = OnTopic(valid_topics=["sports"], invalid_topics=["sports"])
            validator.validate("This is a test text.")
