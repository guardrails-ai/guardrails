import json
from unittest.mock import patch, MagicMock

import pytest

import guardrails.hub as hub_module


@pytest.fixture(autouse=True)
def reset_export_map_cache():
    """Reset the module-level cache before each test."""
    hub_module._export_map_cache = None
    yield
    hub_module._export_map_cache = None


@pytest.fixture
def registry_dir(tmp_path):
    guardrails_dir = tmp_path / ".guardrails"
    guardrails_dir.mkdir()
    return guardrails_dir


@pytest.fixture
def registry_file(registry_dir):
    return registry_dir / "hub_registry.json"


def _write_registry(registry_file, validators):
    registry = {"version": 1, "validators": validators}
    registry_file.write_text(json.dumps(registry))


def test_getattr_resolves_registered_validator(tmp_path, registry_file):
    _write_registry(
        registry_file,
        {
            "guardrails/test-validator": {
                "import_path": "guardrails_grhub_test_validator",
                "exports": ["TestValidator"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-test-validator",
            }
        },
    )

    mock_module = MagicMock()
    mock_module.TestValidator = "resolved_class"

    with (
        patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)),
        patch(
            "guardrails.hub.importlib.import_module",
            return_value=mock_module,
        ) as mock_import,
    ):
        result = hub_module.__getattr__("TestValidator")

    assert result == "resolved_class"
    mock_import.assert_called_once_with("guardrails_grhub_test_validator")


def test_getattr_caches_in_globals(tmp_path, registry_file):
    _write_registry(
        registry_file,
        {
            "guardrails/test-validator": {
                "import_path": "guardrails_grhub_test_validator",
                "exports": ["TestValidator"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-test-validator",
            }
        },
    )

    mock_module = MagicMock()
    mock_module.TestValidator = "resolved_class"

    with (
        patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)),
        patch(
            "guardrails.hub.importlib.import_module",
            return_value=mock_module,
        ) as mock_import,
    ):
        result1 = hub_module.__getattr__("TestValidator")
        assert result1 == "resolved_class"
        assert mock_import.call_count == 1

        # After caching, the attr should be in globals
        assert "TestValidator" in hub_module.__dict__

        # Clean up
        del hub_module.__dict__["TestValidator"]


def test_getattr_raises_attribute_error_for_unknown(tmp_path, registry_file):
    _write_registry(registry_file, {})

    with patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)):
        with pytest.raises(
            AttributeError,
            match="has no attribute 'NonExistent'",
        ):
            hub_module.__getattr__("NonExistent")


def test_getattr_raises_import_error_for_missing_module(tmp_path, registry_file):
    _write_registry(
        registry_file,
        {
            "guardrails/missing": {
                "import_path": "guardrails_grhub_missing",
                "exports": ["MissingValidator"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-missing",
            }
        },
    )

    with (
        patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)),
        patch(
            "guardrails.hub.importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'guardrails_grhub_missing'"
            ),
        ),
    ):
        with pytest.raises(ImportError, match="Cannot import 'MissingValidator'"):
            hub_module.__getattr__("MissingValidator")


def test_getattr_raises_import_error_for_missing_attribute(tmp_path, registry_file):
    """Module loads but the registered export name does not exist."""
    _write_registry(
        registry_file,
        {
            "guardrails/test-validator": {
                "import_path": "guardrails_grhub_test_validator",
                "exports": ["NonExistentClass"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-test-validator",
            }
        },
    )

    mock_module = MagicMock(spec=[])

    with (
        patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)),
        patch(
            "guardrails.hub.importlib.import_module",
            return_value=mock_module,
        ),
    ):
        with pytest.raises(ImportError, match="Cannot import 'NonExistentClass'"):
            hub_module.__getattr__("NonExistentClass")


def test_dir_includes_registered_exports(tmp_path, registry_file):
    _write_registry(
        registry_file,
        {
            "guardrails/test-validator": {
                "import_path": "guardrails_grhub_test_validator",
                "exports": ["TestValidator", "TestHelper"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-test-validator",
            }
        },
    )

    with patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)):
        result = hub_module.__dir__()

    assert "TestValidator" in result
    assert "TestHelper" in result


def test_dir_with_no_registry(tmp_path):
    """__dir__ should not crash when no registry file exists."""
    with patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)):
        result = hub_module.__dir__()

    assert isinstance(result, list)
    assert "_load_registry" in result
    assert "_build_export_map" in result


def test_empty_registry(tmp_path):
    with patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)):
        with pytest.raises(
            AttributeError,
            match="has no attribute 'SomeName'",
        ):
            hub_module.__getattr__("SomeName")


def test_corrupt_registry_falls_back_gracefully(tmp_path, registry_dir):
    """Corrupt JSON should fall back to AttributeError, not crash."""
    (registry_dir / "hub_registry.json").write_text("{not valid json")

    with patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)):
        with pytest.raises(
            AttributeError,
            match="has no attribute 'DetectPII'",
        ):
            hub_module.__getattr__("DetectPII")


def test_export_name_collision_last_wins(tmp_path, registry_file):
    """When two validators export the same name, last in dict wins."""
    _write_registry(
        registry_file,
        {
            "org-a/validator": {
                "import_path": "org_a_grhub_validator",
                "exports": ["SharedName"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "org-a-grhub-validator",
            },
            "org-b/validator": {
                "import_path": "org_b_grhub_validator",
                "exports": ["SharedName"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "org-b-grhub-validator",
            },
        },
    )

    with patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)):
        export_map = hub_module._build_export_map()
        assert export_map["SharedName"] == "org_b_grhub_validator"


def test_build_export_map_skips_empty_exports(tmp_path, registry_file):
    """Entry with empty exports list produces no entries."""
    _write_registry(
        registry_file,
        {
            "guardrails/no-exports": {
                "import_path": "guardrails_grhub_no_exports",
                "exports": [],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-no-exports",
            }
        },
    )

    with patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)):
        export_map = hub_module._build_export_map()
        assert len(export_map) == 0


def test_build_export_map_handles_missing_keys(tmp_path, registry_file):
    """Entry with missing import_path or exports should not crash."""
    _write_registry(registry_file, {"guardrails/broken": {}})

    with patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)):
        export_map = hub_module._build_export_map()
        assert len(export_map) == 0


def test_get_export_map_caches_result(tmp_path, registry_file):
    """_get_export_map should only build the map once."""
    _write_registry(
        registry_file,
        {
            "guardrails/test": {
                "import_path": "guardrails_grhub_test",
                "exports": ["Test"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-test",
            }
        },
    )

    with patch("guardrails.hub.os.getcwd", return_value=str(tmp_path)):
        result1 = hub_module._get_export_map()
        result2 = hub_module._get_export_map()
        assert result1 is result2
