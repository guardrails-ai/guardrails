import pytest
from tests.unit_tests.mocks.mock_file import MockFile


class TestRC:
    def test_load(self, mocker):
        expanduser_mock = mocker.patch("guardrails.classes.rc.expanduser")
        expanduser_mock.return_value = "/Home"

        import os

        join_spy = mocker.spy(os.path, "join")

        mock_file = MockFile()
        mock_open = mocker.patch("guardrails.classes.rc.open")
        mock_open.return_value = mock_file

        readlines_spy = mocker.patch.object(mock_file, "readlines")
        readlines_spy.return_value = ["key1=val1", "key2=val2"]
        close_spy = mocker.spy(mock_file, "close")

        from guardrails.classes.rc import RC

        mock_from_dict = mocker.patch.object(RC, "from_dict")

        RC.load()

        assert expanduser_mock.called is True
        join_spy.assert_called_once_with("/Home", ".guardrailsrc")

        assert mock_open.call_count == 1
        assert readlines_spy.call_count == 1
        assert close_spy.call_count == 1
        # This is supposed to look wrong; since this method is on the super,
        # it doesn't care if the key values are actually correct.
        # Something to watch out for.
        mock_from_dict.assert_called_once_with({"key1": "val1", "key2": "val2"})

    @pytest.mark.parametrize("no_metrics", [True, False, None])
    def test_load_backfill_no_metrics_true(self, mocker, no_metrics):
        expanduser_mock = mocker.patch("guardrails.classes.rc.expanduser")
        expanduser_mock.return_value = "/Home"

        import os

        join_spy = mocker.spy(os.path, "join")

        mock_file = MockFile()
        mock_open = mocker.patch("guardrails.classes.rc.open")
        mock_open.return_value = mock_file
        readlines_spy = mocker.patch.object(mock_file, "readlines")
        readlines_spy.return_value = [f"no_metrics={no_metrics}"]
        close_spy = mocker.spy(mock_file, "close")

        from guardrails.classes.rc import RC

        mock_from_dict = mocker.patch.object(RC, "from_dict")

        RC.load()

        assert expanduser_mock.called is True
        join_spy.assert_called_once_with("/Home", ".guardrailsrc")

        assert mock_open.call_count == 1
        assert readlines_spy.call_count == 1
        assert close_spy.call_count == 1
        # This is supposed to look wrong; since this method is on the super,
        # it doesn't care if the key values are actually correct.
        # Something to watch out for.

        expected_dict = {"enable_metrics": not no_metrics}
        if no_metrics is None:
            expected_dict = {}

        mock_from_dict.assert_called_once_with(expected_dict)

    @pytest.mark.parametrize(
        ("rc_line", "expected_value"),
        [
            ('token=""', ""),
            ("token=''", ""),
            ('token="my-secret-key"', "my-secret-key"),
            ("token='my-secret-key'", "my-secret-key"),
            ("token=plain-value", "plain-value"),
            ("token=", ""),
        ],
        ids=[
            "double-quoted-empty",
            "single-quoted-empty",
            "double-quoted-value",
            "single-quoted-value",
            "unquoted-value",
            "bare-empty",
        ],
    )
    def test_load_strips_surrounding_quotes(self, mocker, rc_line, expected_value):
        mocker.patch("guardrails.classes.rc.expanduser", return_value="/Home")

        mock_file = MockFile()
        mocker.patch("guardrails.classes.rc.open", return_value=mock_file)
        mocker.patch.object(mock_file, "readlines", return_value=[rc_line])

        from guardrails.classes.rc import RC

        mock_from_dict = mocker.patch.object(RC, "from_dict")

        RC.load()

        call_args = mock_from_dict.call_args[0][0]
        assert call_args["token"] == expected_value
