import pytest
from unittest.mock import ANY as AnyMatcher, MagicMock
from guardrails_api.classes.http_error import HttpError

from guardrails_api.models.guard_item import GuardItem
from guardrails_api.models.guard_item_audit import GuardItemAudit
from tests.mocks.mock_postgres_client import MockPostgresClient
from tests.mocks.mock_guard_client import MockGuardStruct
from unittest.mock import call


def test_init(mocker):
    mock_pg_client = MockPostgresClient()
    mocker.patch(
        "guardrails_api.clients.pg_guard_client.PostgresClient",
        return_value=mock_pg_client,
    )

    from guardrails_api.clients.pg_guard_client import PGGuardClient

    pg_guard_client = PGGuardClient()

    assert pg_guard_client.initialized is True
    assert isinstance(pg_guard_client.pgClient, MockPostgresClient)
    assert pg_guard_client.pgClient == mock_pg_client


class TestGetGuard:
    def test_get_latest(self, mocker):
        mock_pg_client = MockPostgresClient()
        mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
        mock_session = mock_pg_client.SessionLocal()
        mocker.patch(
            "guardrails_api.clients.pg_guard_client.PostgresClient",
            return_value=mock_pg_client,
        )
        query_spy = mock_session.query
        query_spy.return_value = mock_session

        filter_by_spy = mock_session.filter_by
        filter_by_spy.return_value = mock_session

        mock_first = mock_session.first
        latest_guard = MockGuardStruct()
        mock_first.return_value = latest_guard

        mock_from_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.from_guard_item"
        )
        mock_from_guard_item.return_value = latest_guard

        from guardrails_api.clients.pg_guard_client import PGGuardClient

        guard_client = PGGuardClient()

        result = guard_client.get_guard("guard")

        query_spy.assert_called_once_with(GuardItem)
        filter_by_spy.assert_called_once_with(name="guard")
        assert mock_first.call_count == 1
        mock_from_guard_item.assert_called_once_with(latest_guard)

        assert result == latest_guard

    def test_with_as_of_date(self, mocker):
        mock_pg_client = MockPostgresClient()
        mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
        mock_session = mock_pg_client.SessionLocal()
        mocker.patch(
            "guardrails_api.clients.pg_guard_client.PostgresClient",
            return_value=mock_pg_client,
        )
        query_spy = mock_session.query
        query_spy.return_value = mock_session

        filter_by_spy = mock_session.filter_by
        filter_by_spy.return_value = mock_session

        filter_spy = mock_session.filter
        filter_spy.return_value = mock_session

        order_by_spy = mock_session.order_by
        order_by_spy.return_value = mock_session

        mock_first = mock_session.first
        latest_guard = MockGuardStruct(name="latest")
        previous_guard = MockGuardStruct(name="previous")
        mock_first.side_effect = [latest_guard, previous_guard]

        mock_from_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.from_guard_item"
        )
        mock_from_guard_item.return_value = previous_guard

        from guardrails_api.clients.pg_guard_client import PGGuardClient

        guard_client = PGGuardClient()

        result = guard_client.get_guard("guard", as_of_date="2024-03-06")

        assert query_spy.call_count == 2
        query_calls = [call(GuardItem), call(GuardItemAudit)]
        query_spy.assert_has_calls(query_calls)

        filter_by_calls = [call(name="guard"), call(name="guard")]
        assert filter_by_spy.call_count == 2
        filter_by_spy.assert_has_calls(filter_by_calls)

        replaced_on_exp = GuardItemAudit.replaced_on > "2024-03-06"
        filter_spy_call = filter_spy.call_args[0][0]
        assert replaced_on_exp.compare(filter_spy_call)

        replaced_on_order_exp = GuardItemAudit.replaced_on.asc()
        order_by_spy_call = order_by_spy.call_args[0][0]
        assert replaced_on_order_exp.compare(order_by_spy_call)

        assert mock_first.call_count == 2
        mock_from_guard_item.assert_called_once_with(previous_guard)

        assert result == previous_guard

    def test_raises_not_found(self, mocker):
        mock_pg_client = MockPostgresClient()
        mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
        mock_session = mock_pg_client.SessionLocal()
        mocker.patch(
            "guardrails_api.clients.pg_guard_client.PostgresClient",
            return_value=mock_pg_client,
        )
        # Mock the query method to return a mock query object
        mock_query = mock_session.query.return_value

        # Mock the filter_by method to return a mock filter object
        mock_filter_by = mock_query.filter_by.return_value
        mock_first = mock_filter_by.first
        # Mock the first method on the mock filter object to return None
        mock_filter_by.first.return_value = None
        mock_from_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.from_guard_item"
        )

        from guardrails_api.clients.pg_guard_client import PGGuardClient

        guard_client = PGGuardClient()

        with pytest.raises(HttpError) as exc_info:
            guard_client.get_guard("guard")

        assert mock_first.call_count == 1
        assert mock_from_guard_item.call_count == 0

        assert isinstance(exc_info.value, HttpError)
        assert exc_info.value.status == 404
        assert exc_info.value.message == "NotFound"
        assert exc_info.value.cause == "A Guard with the name guard does not exist!"


def test_get_guard_item(mocker):
    mock_pg_client = MockPostgresClient()
    mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
    mock_session = mock_pg_client.SessionLocal()
    mocker.patch(
        "guardrails_api.clients.pg_guard_client.PostgresClient",
        return_value=mock_pg_client,
    )

    query_spy = mock_session.query
    query_spy.return_value = mock_session

    filter_by_spy = mock_session.filter_by
    filter_by_spy.return_value = mock_session

    mock_first = mock_session.first
    latest_guard = MockGuardStruct(name="latest")
    mock_first.return_value = latest_guard

    from guardrails_api.clients.pg_guard_client import PGGuardClient

    guard_client = PGGuardClient()

    result = guard_client.get_guard_item("guard")

    query_spy.assert_called_once_with(GuardItem)
    filter_by_spy.assert_called_once_with(name="guard")
    assert mock_first.call_count == 1

    assert result == latest_guard


def test_get_guards(mocker):
    mock_pg_client = MockPostgresClient()
    mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
    mock_session = mock_pg_client.SessionLocal()
    mocker.patch(
        "guardrails_api.clients.pg_guard_client.PostgresClient",
        return_value=mock_pg_client,
    )

    # Ensure that query returns the mock session itself
    mock_session.query.return_value = mock_session
    query_spy = mock_session.query

    guard_one = MockGuardStruct(name="guard one")
    guard_two = MockGuardStruct(name="guard two")
    guards = [guard_one, guard_two]
    # Mock the all method on the mock session
    mock_session.all.return_value = guards
    mock_all = mock_session.all

    mock_from_guard_item = mocker.patch("guardrails_api.clients.pg_guard_client.from_guard_item")
    mock_from_guard_item.side_effect = [guard_one, guard_two]

    from guardrails_api.clients.pg_guard_client import PGGuardClient

    guard_client = PGGuardClient()

    result = guard_client.get_guards()

    query_spy.assert_called_once_with(GuardItem)
    assert mock_all.call_count == 1

    assert mock_from_guard_item.call_count == 2
    from_guard_item_calls = [call(guard_one), call(guard_two)]
    mock_from_guard_item.assert_has_calls(from_guard_item_calls)

    assert result == [guard_one, guard_two]


def test_create_guard(mocker):
    mock_guard = MockGuardStruct()
    mock_pg_client = MockPostgresClient()
    mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
    mock_guard_struct_init_spy = mocker.spy(MockGuardStruct, "__init__")
    mock_session = mock_pg_client.SessionLocal()
    mocker.patch(
        "guardrails_api.clients.pg_guard_client.PostgresClient",
        return_value=mock_pg_client,
    )
    mocker.patch("guardrails_api.clients.pg_guard_client.GuardItem", new=MockGuardStruct)

    add_spy = mocker.spy(mock_session, "add")
    commit_spy = mocker.spy(mock_session, "commit")

    mock_from_guard_item = mocker.patch("guardrails_api.clients.pg_guard_client.from_guard_item")
    mock_from_guard_item.return_value = mock_guard

    from guardrails_api.clients.pg_guard_client import PGGuardClient

    guard_client = PGGuardClient()

    result = guard_client.create_guard(mock_guard)

    mock_guard_struct_init_spy.assert_called_once_with(
        AnyMatcher,
        name="mock-guard",
        description="mock guard description",
        railspec=mock_guard.to_dict(),
        num_reasks=None,
    )

    assert add_spy.call_count == 1
    mock_guard_item = add_spy.call_args[0][0]
    assert isinstance(mock_guard_item, MockGuardStruct)
    assert mock_guard_item.name == "mock-guard"
    # assert isinstance(mock_guard_item.railspec, MockRailspec)
    assert mock_guard_item.description == "mock guard description"
    assert commit_spy.call_count == 1

    assert result == mock_guard


class TestUpdateGuard:
    def test_raises_not_found(self, mocker):
        mock_guard = MockGuardStruct()
        mock_pg_client = MockPostgresClient()
        mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
        mock_session = mock_pg_client.SessionLocal()
        mocker.patch(
            "guardrails_api.clients.pg_guard_client.PostgresClient",
            return_value=mock_pg_client,
        )
        mock_get_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.PGGuardClient.get_guard_item"
        )
        mock_get_guard_item.return_value = None

        commit_spy = mocker.spy(mock_session, "commit")
        mock_from_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.from_guard_item"
        )

        from guardrails_api.clients.pg_guard_client import PGGuardClient

        guard_client = PGGuardClient()

        with pytest.raises(HttpError) as exc_info:
            guard_client.update_guard("mock-guard", mock_guard)

        assert isinstance(exc_info.value, HttpError)
        assert exc_info.value.status == 404
        assert exc_info.value.message == "NotFound"
        assert exc_info.value.cause == "A Guard with the name mock-guard does not exist!"

        assert commit_spy.call_count == 0
        assert mock_from_guard_item.call_count == 0

    def test_updates_guard_item(self, mocker):
        old_guard = MockGuardStruct()
        old_guard_item = GuardItem(
            name=old_guard.name,
            railspec=old_guard.to_dict(),
            # FIXME: IGuard doesn't appear to have num_reasks
            num_reasks=None,
            description=old_guard.description,
        )
        updated_guard = MockGuardStruct(description="updated description")

        mock_pg_client = MockPostgresClient()
        mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
        mock_session = mock_pg_client.SessionLocal()
        mocker.patch(
            "guardrails_api.clients.pg_guard_client.PostgresClient",
            return_value=mock_pg_client,
        )
        mock_get_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.PGGuardClient.get_guard_item"
        )
        mock_get_guard_item.return_value = old_guard_item

        commit_spy = mocker.spy(mock_session, "commit")
        mock_from_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.from_guard_item"
        )
        mock_from_guard_item.return_value = updated_guard

        from guardrails_api.clients.pg_guard_client import PGGuardClient

        guard_client = PGGuardClient()

        result = guard_client.update_guard("mock-guard", updated_guard)

        mock_get_guard_item.assert_called_once_with("mock-guard")
        assert commit_spy.call_count == 1
        mock_from_guard_item.assert_called_once_with(old_guard_item)

        # These would have been updated by reference
        # assert old_guard.railspec == updated_guard.railspec.to_dict()
        assert result.description == "updated description"

        assert result == updated_guard


class TestUpsertGuard:
    def test_guard_doesnt_exist_yet(self, mocker):
        input_guard = MockGuardStruct()
        new_guard = MockGuardStruct()
        mock_pg_client = MockPostgresClient()
        mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
        mocker.patch(
            "guardrails_api.clients.pg_guard_client.PostgresClient",
            return_value=mock_pg_client,
        )
        mock_get_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.PGGuardClient.get_guard_item"
        )
        mock_get_guard_item.return_value = None

        commit_spy = mocker.spy(mock_pg_client.db.session, "commit")
        mock_from_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.from_guard_item"
        )
        mock_create_guard = mocker.patch(
            "guardrails_api.clients.pg_guard_client.PGGuardClient.create_guard"
        )
        mock_create_guard.return_value = new_guard

        from guardrails_api.clients.pg_guard_client import PGGuardClient

        guard_client = PGGuardClient()

        result = guard_client.upsert_guard("mock-guard", input_guard)

        mock_get_guard_item.assert_called_once_with("mock-guard")
        assert commit_spy.call_count == 0
        assert mock_from_guard_item.call_count == 0
        mock_create_guard.assert_called_once_with(input_guard)

        assert result == new_guard

    def test_guard_already_exists(self, mocker):
        old_guard = MockGuardStruct()
        old_guard_item = GuardItem(
            name=old_guard.name,
            railspec=old_guard.to_dict(),
            # TODO: IGuard doesn't appear to have num_reasks
            num_reasks=None,
            description=old_guard.description,
        )
        updated_guard = MockGuardStruct(description="updated description")

        mock_pg_client = MockPostgresClient()
        mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
        mocker.patch(
            "guardrails_api.clients.pg_guard_client.PostgresClient",
            return_value=mock_pg_client,
        )

        mock_session = mock_pg_client.SessionLocal()

        mock_get_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.PGGuardClient.get_guard_item"
        )
        mock_get_guard_item.return_value = old_guard_item

        commit_spy = mocker.spy(mock_session, "commit")
        mock_from_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.from_guard_item"
        )
        mock_from_guard_item.return_value = updated_guard

        from guardrails_api.clients.pg_guard_client import PGGuardClient

        guard_client = PGGuardClient()

        result = guard_client.upsert_guard("mock-guard", updated_guard)

        mock_get_guard_item.assert_called_once_with("mock-guard")
        assert commit_spy.call_count == 1
        mock_from_guard_item.assert_called_once_with(old_guard_item)

        # These would have been updated by reference
        # Are they still updated by reference if they're separate objects being mocked?
        # TODO: assuming there's no more railspec on guards?
        # assert old_guard.railspec == updated_guard.railspec.to_dict()
        assert result.description == "updated description"

        assert result == updated_guard


class TestDeleteGuard:
    def test_raises_not_found(self, mocker):
        mock_pg_client = MockPostgresClient()
        mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
        mocker.patch(
            "guardrails_api.clients.pg_guard_client.PostgresClient",
            return_value=mock_pg_client,
        )

        mock_session = mock_pg_client.SessionLocal()

        mock_get_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.PGGuardClient.get_guard_item"
        )
        mock_get_guard_item.return_value = None

        commit_spy = mocker.spy(mock_session, "commit")

        mock_from_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.from_guard_item"
        )

        from guardrails_api.clients.pg_guard_client import PGGuardClient

        guard_client = PGGuardClient()

        with pytest.raises(HttpError) as exc_info:
            guard_client.delete_guard("mock-guard")

        assert isinstance(exc_info.value, HttpError)
        assert exc_info.value.status == 404
        assert exc_info.value.message == "NotFound"
        assert exc_info.value.cause == "A Guard with the name mock-guard does not exist!"

        assert commit_spy.call_count == 0
        assert mock_from_guard_item.call_count == 0

    def test_deletes_guard_item(self, mocker):
        old_guard = MockGuardStruct()
        mock_pg_client = MockPostgresClient()
        mock_pg_client.SessionLocal = MagicMock(return_value=MagicMock())
        mocker.patch(
            "guardrails_api.clients.pg_guard_client.PostgresClient",
            return_value=mock_pg_client,
        )

        mock_session = mock_pg_client.SessionLocal()

        mock_get_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.PGGuardClient.get_guard_item"
        )
        mock_get_guard_item.return_value = old_guard

        # Mock the query and delete operations
        mock_query = mock_session.query.return_value
        mock_filter = mock_query.filter_by.return_value
        mock_filter.first.return_value = old_guard
        mock_from_guard_item = mocker.patch(
            "guardrails_api.clients.pg_guard_client.from_guard_item"
        )
        mock_from_guard_item.return_value = old_guard

        from guardrails_api.clients.pg_guard_client import PGGuardClient

        guard_client = PGGuardClient()

        result = guard_client.delete_guard("mock-guard")

        mock_get_guard_item.assert_called_once_with("mock-guard")
        assert mock_session.delete.call_count == 1
        assert mock_session.commit.call_count == 1
        mock_from_guard_item.assert_called_once_with(old_guard)

        assert result == old_guard
