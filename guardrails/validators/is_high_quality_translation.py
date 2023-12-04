import os
from typing import Any, Dict

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="is-high-quality-translation", data_type="string")
class IsHighQualityTranslation(Validator):
    """Using inpiredco.critique to check if a translation is high quality.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `is-high-quality-translation`     |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Other parameters: Metadata
        translation_source (str): The source of the translation.
    """

    required_metadata_keys = ["translation_source"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from inspiredco.critique import Critique  # type: ignore

            inspiredco_api_key = os.environ.get("INSPIREDCO_API_KEY")
            if not inspiredco_api_key:
                raise ValueError(
                    "The INSPIREDCO_API_KEY environment variable must be set"
                    "in order to use the is-high-quality-translation validator!"
                )

            self._critique = Critique(api_key=inspiredco_api_key)

        except ImportError:
            raise ImportError(
                "`is-high-quality-translation` validator requires the `inspiredco`"
                "package. Please install it with `pip install inspiredco`."
            )

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if "translation_source" not in metadata:
            raise RuntimeError(
                "is-high-quality-translation validator expects "
                "`translation_source` key in metadata"
            )
        src = metadata["translation_source"]
        prediction = self._critique.evaluate(
            metric="comet",
            config={"model": "unbabel_comet/wmt21-comet-qe-da"},
            dataset=[{"source": src, "target": value}],
        )
        quality = prediction["examples"][0]["value"]
        if quality < -0.1:
            return FailResult(
                error_message=f"{value} is a low quality translation."
                "Please return a higher quality output.",
                fix_value="",
            )
        return PassResult()
