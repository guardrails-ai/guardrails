from typing import Any, Dict, cast

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    download_model = None
    load_from_checkpoint = None


@register_validator(name="is-high-quality-translation", data_type="string")
class IsHighQualityTranslation(Validator):
    """Validates that the translation is of high quality.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `is-high-quality-translation`     |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Other parameters: Metadata
        translation_source (str): The source of the translation.

    This validator uses one of the reference-free models from Unbabel/COMET
    to check the quality of the translation. Specifically, it uses the
    `Unbabel/wmt22-cometkiwi-da` model.

    Unbabel/COMET details: https://github.com/Unbabel/COMET
    Model details: https://huggingface.co/Unbabel/wmt22-cometkiwi-da

    Pre-requisites:
        - Install the `unbabel-comet` from source:
            `pip install git+https://github.com/Unbabel/COMET`
        - Please accept the model license from:
            https://huggingface.co/Unbabel/wmt22-cometkiwi-da
        - Login into Huggingface Hub using:
            huggingface-cli login --token $HUGGINGFACE_TOKEN
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if download_model is None or load_from_checkpoint is None:
            raise RuntimeError(
                "is-high-quality-translation validator requires "
                "unbabel-comet to be installed. Please install it using "
                "`pip install git+https://github.com/Unbabel/COMET`."
            )
        self._model_name = "Unbabel/wmt22-cometkiwi-da"
        self._quality_threshold = 0.5

        try:
            # Download the model
            print("\nDownloading the model. This may take a while the 1st time...")
            model_path = download_model(self._model_name)

            # Load the model
            print("\nLoading the model from checkpoint...")
            self.model = load_from_checkpoint(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Error while downloading the model {self._model_name} "
                "from COMET: {e}.\n Please review the validator "
                "documentation for more details on the pre-requisites."
                "Ensure that you are logged into Huggingface Hub."
            ) from e

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if "translation_source" not in metadata:
            raise RuntimeError(
                "is-high-quality-translation validator expects "
                "`translation_source` key in metadata"
            )

        model_output = self.model.predict(
            [{"src": metadata["translation_source"], "mt": value}],
            accelerator="cpu",
        )
        model_output = cast(Any, model_output)
        translation_quality = model_output.scores[0]
        print(f"Translation quality: {translation_quality}")
        if translation_quality < self._quality_threshold:
            return FailResult(
                error_message=f"{value} is a low quality translation. "
                "Hence, not returning.",
                fix_value="",
            )
        return PassResult()
