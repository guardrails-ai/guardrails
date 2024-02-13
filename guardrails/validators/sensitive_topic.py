import json
from typing import Any, Callable, Dict, List, Optional, Union

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    register_validator,
)
from guardrails.validators import OnTopic

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


@register_validator(name="sensitive_topic", data_type="string")
class SensitiveTopic(OnTopic):  # type: ignore
    """Checks if text contains any sensitive topics.

    Default behavior first runs a Zero-Shot model, and then falls back to
    ask OpenAI's `gpt-3.5-turbo` if the Zero-Shot model is not confident
    in the topic classification (score < 0.5).

    In our experiments this LLM fallback increases accuracy by 15% but also
    increases latency (more than doubles the latency in the worst case).

    Both the Zero-Shot classification and the GPT classification may be toggled.

    **Key Properties**

    | Property                      | Description                |
    | ----------------------------- | -------------------------- |
    | Name for `format` attribute   | `on_topic`                 |
    | Supported data types          | `string`                   |
    | Programmatic fix              | Prepends a trigger warning |

    Args:
        sensitive_topics (List[str], Optional, defaults to [
            "holiday or anniversary of the trauma or loss",
            "certain sounds, sights, smells, or tastes related to the trauma",
            "loud voices or yelling",
            "loud noises",
            "arguments",
            "being ridiculed or judged",
            "being alone",
            "getting rejected",
            "being ignored",
            "breakup of a relationship",
            "violence in the news",
            "sexual harassment or unwanted touching",
            "physical illness or injury",
        ]): topics that the text should not contain.
        device (int, Optional, defaults to -1): Device ordinal for CPU/GPU
            supports for Zero-Shot classifier. Setting this to -1 will leverage
            CPU, a positive will run the Zero-Shot model on the associated CUDA
            device id.
        model (str, Optional, defaults to 'facebook/bart-large-mnli'): The
            Zero-Shot model that will be used to classify the topic. See a
            list of all models here:
            https://huggingface.co/models?pipeline_tag=zero-shot-classification
        llm_callable (Union[str, Callable, None], Optional, defaults to
            'gpt-3.5-turbo'): Either the name of the OpenAI model, or a callable
            that takes a prompt and returns a response.
        disable_classifier (bool, Optional, defaults to False): controls whether
            to use the Zero-Shot model. At least one of disable_classifier and
            disable_llm must be False.
        disable_llm (bool, Optional, defaults to False): controls whether to use
            the LLM fallback. At least one of disable_classifier and
            disable_llm must be False.
        model_threshold (float, Optional, defaults to 0.5): The threshold used to
            determine whether to accept a topic from the Zero-Shot model. Must be
            a number between 0 and 1.
    """

    def __init__(
        self,
        sensitive_topics: Optional[List[str]] = None,
        device: Optional[int] = -1,
        model: Optional[str] = "facebook/bart-large-mnli",
        llm_callable: Union[str, Callable, None] = None,
        disable_classifier: Optional[bool] = False,
        disable_llm: Optional[bool] = False,
        on_fail: Optional[Callable[..., Any]] = None,
        model_threshold: Optional[float] = 0.5,
    ):
        if sensitive_topics is None:
            sensitive_topics = [
                "holiday or anniversary of the trauma or loss",
                "certain sounds, sights, smells, or tastes related to the trauma",
                "loud voices or yelling",
                "loud noises",
                "arguments",
                "being ridiculed or judged",
                "being alone",
                "getting rejected",
                "being ignored",
                "breakup of a relationship",
                "violence in the news",
                "sexual harassment or unwanted touching",
                "physical illness or injury",
            ]
        super().__init__(
            valid_topics=[],
            invalid_topics=sensitive_topics,
            device=device,
            model=model,
            disable_classifier=disable_classifier,
            disable_llm=disable_llm,
            llm_callable=llm_callable,
            on_fail=on_fail,
            model_threshold=model_threshold,
        )

    def get_topics_ensemble(self, text: str, candidate_topics: List[str]) -> List[str]:
        applicable_topics = []

        for candidate_topic in candidate_topics:
            candidates = [candidate_topic, "other"]
            topic, confidence = self.get_topic_zero_shot(text, candidates)

            if confidence < self._model_threshold:
                response = self.call_llm(text, candidate_topics)
                topic = json.loads(response)["topic"]

            if topic != "other":
                applicable_topics.append(topic)

        return applicable_topics

    def get_topics_llm(self, text: str, candidate_topics: List[str]) -> List[str]:
        applicable_topics = []

        for candidate_topic in candidate_topics:
            candidates = [candidate_topic, "other"]
            response = self.call_llm(text, candidates)
            topic = json.loads(response)["topic"]

            if topic != "other":
                applicable_topics.append(topic)

        return applicable_topics

    def get_topics_zero_shot(self, text: str, candidate_topics: List[str]) -> List[str]:
        applicable_topics = []

        for candidate_topic in candidate_topics:
            candidates = [candidate_topic, "other"]
            topic, confidence = self.get_topic_zero_shot(text, candidates)

            if confidence > self._model_threshold:
                applicable_topics.append(topic)

        return applicable_topics

    def validate(
        self, value: str, metadata: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        invalid_topics = set(self._invalid_topics)

        # throw if there are no invalid topics
        if not invalid_topics:
            raise RuntimeError("No invalid topics provided")

        # Check which model(s) to use
        if self._disable_classifier and self._disable_llm:
            # Error, no model set
            raise ValueError("Either classifier or llm must be enabled.")
        elif (
            not self._disable_classifier and not self._disable_llm
        ):  # Use ensemble (Zero-Shot + Ensemble)
            applicable_topics = self.get_topics_ensemble(value, list(invalid_topics))
        elif self._disable_classifier and not self._disable_llm:
            # Use only LLM
            applicable_topics = self.get_topics_llm(value, list(invalid_topics))
        else:
            # Use only Zero-Shot
            applicable_topics = self.get_topic_zero_shot(value, list(invalid_topics))

        if not applicable_topics:
            return PassResult()

        sensitive_topics_warning = "Trigger warning:"
        for topic in applicable_topics:
            sensitive_topics_warning += f"\n- {topic}"
        fixed_message = f"{sensitive_topics_warning}\n\n{value}"
        return FailResult(
            error_message="Sensitive topics detected: " + ", ".join(applicable_topics),
            fix_value=fixed_message,
        )
