import os
import warnings
from typing import Any, Callable, Dict, List, Tuple, Union

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

try:
    import detect_secrets  # type: ignore
except ImportError:
    detect_secrets = None


@register_validator(name="detect-secrets", data_type="string")
class DetectSecrets(Validator):
    """Validates whether the generated code snippet contains any secrets.

    **Key Properties**
    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `detect-secrets`                  |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        None

    This validator uses the detect-secrets library to check whether the generated code
    snippet contains any secrets. If any secrets are detected, the validator fails and
    returns the generated code snippet with the secrets replaced with asterisks.
    Else the validator returns the generated code snippet.

    Following are some caveats:
        - Multiple secrets on the same line may not be caught. e.g.
            - Minified code
            - One-line lists/dictionaries
            - Multi-variable assignments
        - Multi-line secrets may not be caught. e.g.
            - RSA/SSH keys

    Example:
        ```py

        guard = Guard.from_string(validators=[
            DetectSecrets(on_fail="fix")
        ])
        guard.parse(
            llm_output=code_snippet,
        )
        ```
    """

    def __init__(self, on_fail: Union[Callable[..., Any], None] = None, **kwargs):
        super().__init__(on_fail, **kwargs)

        # Check if detect-secrets is installed
        if detect_secrets is None:
            raise ValueError(
                "You must install detect-secrets in order to "
                "use the DetectSecrets validator."
            )
        self.temp_file_name = "temp.txt"
        self.mask = "********"

    def get_unique_secrets(self, value: str) -> Tuple[Dict[str, Any], List[str]]:
        """Get unique secrets from the value.

        Args:
            value (str): The generated code snippet.

        Returns:
            unique_secrets (Dict[str, Any]): A dictionary of unique secrets and their
                line numbers.
            lines (List[str]): The lines of the generated code snippet.
        """
        try:
            # Write each line of value to a new file
            with open(self.temp_file_name, "w") as f:
                f.writelines(value)
        except Exception as e:
            raise OSError(
                "Problems creating or deleting the temporary file. "
                "Please check the permissions of the current directory."
            ) from e

        try:
            # Create a new secrets collection
            from detect_secrets import settings
            from detect_secrets.core.secrets_collection import SecretsCollection

            secrets = SecretsCollection()

            # Scan the file for secrets
            with settings.default_settings():
                secrets.scan_file(self.temp_file_name)
        except ImportError:
            raise ValueError(
                "You must install detect-secrets in order to "
                "use the DetectSecrets validator."
            )
        except Exception as e:
            raise RuntimeError(
                "Problems with creating a SecretsCollection or "
                "scanning the file for secrets."
            ) from e

        # Get unique secrets from these secrets
        unique_secrets = {}
        for secret in secrets:
            _, potential_secret = secret
            actual_secret = potential_secret.secret_value
            line_number = potential_secret.line_number
            if actual_secret not in unique_secrets:
                unique_secrets[actual_secret] = [line_number]
            else:
                # if secret already exists, avoid duplicate line numbers
                if line_number not in unique_secrets[actual_secret]:
                    unique_secrets[actual_secret].append(line_number)

        try:
            # File no longer needed, read the lines from the file
            with open(self.temp_file_name, "r") as f:
                lines = f.readlines()
        except Exception as e:
            raise OSError(
                "Problems reading the temporary file. "
                "Please check the permissions of the current directory."
            ) from e

        try:
            # Delete the file
            os.remove(self.temp_file_name)
        except Exception as e:
            raise OSError(
                "Problems deleting the temporary file. "
                "Please check the permissions of the current directory."
            ) from e
        return unique_secrets, lines

    def get_modified_value(
        self, unique_secrets: Dict[str, Any], lines: List[str]
    ) -> str:
        """Replace the secrets on the lines with asterisks.

        Args:
            unique_secrets (Dict[str, Any]): A dictionary of unique secrets and their
                line numbers.
            lines (List[str]): The lines of the generated code snippet.

        Returns:
            modified_value (str): The generated code snippet with secrets replaced with
                asterisks.
        """
        # Replace the secrets on the lines with asterisks
        for secret, line_numbers in unique_secrets.items():
            for line_number in line_numbers:
                lines[line_number - 1] = lines[line_number - 1].replace(
                    secret, self.mask
                )

        # Convert lines to a multiline string
        modified_value = "".join(lines)
        return modified_value

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        # Check if value is a multiline string
        if "\n" not in value:
            # Raise warning if value is not a multiline string
            warnings.warn(
                "The DetectSecrets validator works best with "
                "multiline code snippets. "
                "Refer validator docs for more details."
            )

            # Add a newline to value
            value += "\n"

        # Get unique secrets from the value
        unique_secrets, lines = self.get_unique_secrets(value)

        if unique_secrets:
            # Replace the secrets on the lines with asterisks
            modified_value = self.get_modified_value(unique_secrets, lines)

            return FailResult(
                error_message=(
                    "The following secrets were detected in your response:\n"
                    + "\n".join(unique_secrets.keys())
                ),
                fix_value=modified_value,
            )
        return PassResult()
