from tests.unit_tests.mocks.mock_file import MockFile


def test_from_rc_file(mocker):
    # TODO: Re-enable this once we move nltk.download calls to individual validator repos.
    # Right now, it fires during our import chain, causing this to blow up
    mocker.patch("nltk.data.find")
    mocker.patch("nltk.download")
    
    
    expanduser_mock = mocker.patch("guardrails.cli.hub.credentials.expanduser")
    expanduser_mock.return_value = "/Home"

    import os
    join_spy = mocker.spy(os.path, "join")

    
    mock_file = MockFile()
    mock_open = mocker.patch("guardrails.cli.hub.credentials.open")
    mock_open.return_value = mock_file

    readlines_spy = mocker.patch.object(mock_file, "readlines")
    readlines_spy.return_value = ["key1=val1", "key2=val2"]
    close_spy = mocker.spy(mock_file, "close")

    from guardrails.cli.hub.credentials import Credentials

    mock_from_dict = mocker.patch.object(Credentials, "from_dict")

    Credentials.from_rc_file()

    assert expanduser_mock.called == True
    join_spy.assert_called_once_with("/Home", ".guardrailsrc")

    assert mock_open.call_count == 1
    assert readlines_spy.call_count == 1
    assert close_spy.call_count == 1
    mock_from_dict.assert_called_once_with({
        "key1": "val1",
        "key2": "val2"
    })