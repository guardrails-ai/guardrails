from typing import Any, Callable, Dict, List, Union, cast

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    import nltk  # type: ignore
except ImportError:
    nltk = None  # type: ignore

if nltk is not None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


@register_validator(name="toxic-language", data_type="string")
class ToxicLanguage(Validator):
    """Validates that the generated text is toxic.

    **Key Properties**
    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `toxic-language`                  |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        threshold: The confidence threshold (model inference) for toxicity.
            Defaults to 0.5.
        validation_method: Whether to validate at the sentence level or
            over the full text. Must be one of `sentence` or `full`.
            Defaults to `sentence`

    This validator uses the pre-trained multi-label model from HuggingFace -
    `unitary/unbiased-toxic-roberta` to check whether the generated text is toxic.
    If the model predicts any label of: `toxicity`, `severe_toxicity`,
    `obscene`, `threat`, `insult`, `identity_attack`, or `sexual_explicit` with
    confidence higher than the specified threshold, the validator fails and returns
    the generated text with the toxic sentences / entire text removed. Else the
    validator returns the generated text as it is.

    If validation_method is `sentence`, the validator will remove the sentences
    that are predicted to be toxic and return the remaining sentences. If
    validation_method is `full`, the validator will remove the entire text if
    the prediction is deemed toxic and return an empty string.

    In our experiments, a threshold of 0.5 worked best, hence set as default here.
    However, you can try different values of threshold to see what works best for
    your use case.
    Link for experiments: https://wandb.ai/ml-guardrails/toxic-language-experiments
    """

    def __init__(
        self,
        threshold: float = 0.5,
        validation_method: str = "sentence",
        on_fail: Union[Callable[..., Any], None] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail, threshold=threshold, validation_method=validation_method, **kwargs
        )
        self._threshold = float(threshold)
        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
        self._validation_method = validation_method

        # Check if transformers.pipeline is imported
        if pipeline is None:
            raise ValueError(
                "You must install transformers in order to "
                "use the ToxicLanguage validator."
                "Install it using `pip install transformers`."
            )

        # Define the model, pipeline and labels
        self._model_name = "unitary/unbiased-toxic-roberta"
        self._detoxify_pipeline = pipeline(
            "text-classification",
            model=self._model_name,
            function_to_apply="sigmoid",
            top_k=None,
            padding="max_length",
            truncation=True,
        )
        self._labels = [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
            "sexual_explicit",
        ]

    def get_toxicity(self, value: str) -> List[str]:
        """Check whether the generated text is toxic.

        Returns the labels predicted by the model with
        confidence higher than the threshold.

        Args:
            value (str): The generated text.

        Returns:
            pred_labels (bool): Labels predicted by the model
            with confidence higher than the threshold.
        """

        # Get the model predictions and the list of labels
        # with confidence higher than the threshold
        pred_labels = []
        if value:
            results = self._detoxify_pipeline(value)
            if results:
                results = cast(List[List[Dict[str, Any]]], results)
                for label_info in results[0]:
                    label, score = label_info["label"], label_info["score"]
                    if label in self._labels and score > self._threshold:
                        pred_labels.append(label)
        return pred_labels

    def validate_each_sentence(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that each sentence in the generated text is toxic."""

        if nltk is None:
            raise ImportError(
                "`nltk` is required for `ToxicLanguage` validator. "
                "Please install it with `pip install nltk`."
            )
        # Split the value into sentences using nltk sentence tokenizer.
        sentences = nltk.sent_tokenize(value)

        unsupported_sentences, supported_sentences = [], []
        for sentence in sentences:
            if sentence:
                pred_labels = self.get_toxicity(sentence)
                if pred_labels:
                    unsupported_sentences.append(sentence)
                else:
                    supported_sentences.append(sentence)

        if unsupported_sentences:
            unsupported_sentences = "- " + "\n- ".join(unsupported_sentences)
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"The following sentences in your response"
                    "were found to be toxic:\n"
                    f"\n{unsupported_sentences}"
                ),
                fix_value="\n".join(supported_sentences),
            )
        return PassResult(metadata=metadata)

    def validate_full_text(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that the entire generated text is toxic."""

        pred_labels = self.get_toxicity(value)
        if pred_labels:
            return FailResult(
                metadata=metadata,
                error_message=(
                    "The generated text was found to be:\n" + ",".join(pred_labels)
                ),
                fix_value="",
            )
        return PassResult()

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        if not value:
            raise ValueError("Value cannot be empty.")

        if self._validation_method == "sentence":
            return self.validate_each_sentence(value, metadata)
        elif self._validation_method == "full":
            return self.validate_full_text(value, metadata)
        else:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
