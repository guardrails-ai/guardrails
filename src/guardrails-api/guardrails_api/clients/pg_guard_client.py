from typing import List
from guardrails_api_client import Guard as GuardStruct
from guardrails_api.classes.http_error import HttpError
from guardrails_api.clients.guard_client import GuardClient
from guardrails_api.models.guard_item import GuardItem
from guardrails_api.clients.postgres_client import PostgresClient
from guardrails_api.models.guard_item_audit import GuardItemAudit


def from_guard_item(guard_item: GuardItem) -> GuardStruct:
    # Temporary fix for the fact that the DB schema is out of date with the API schema
    # For now, we're just storing the serialized guard in the railspec column
    return GuardStruct.from_dict(guard_item.railspec)


class PGGuardClient(GuardClient):
    def __init__(self):
        self.initialized = True
        self.pgClient = PostgresClient()

    def get_db(self):  # generator for local sessions
        db = self.pgClient.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def get_guard(self, guard_name: str, as_of_date: str = None) -> GuardStruct:
        db = next(self.get_db())
        latest_guard_item = db.query(GuardItem).filter_by(name=guard_name).first()
        audit_item = None
        if as_of_date is not None:
            audit_item = (
                db.query(GuardItemAudit)
                .filter_by(name=guard_name)
                .filter(GuardItemAudit.replaced_on > as_of_date)
                .order_by(GuardItemAudit.replaced_on.asc())
                .first()
            )
        guard_item = audit_item if audit_item is not None else latest_guard_item
        if guard_item is None:
            raise HttpError(
                status=404,
                message="NotFound",
                cause="A Guard with the name {guard_name} does not exist!".format(
                    guard_name=guard_name
                ),
            )
        return from_guard_item(guard_item)

    def get_guard_item(self, guard_name: str) -> GuardItem:
        db = next(self.get_db())
        return db.query(GuardItem).filter_by(name=guard_name).first()

    def get_guards(self) -> List[GuardStruct]:
        db = next(self.get_db())
        guard_items = db.query(GuardItem).all()

        return [from_guard_item(gi) for gi in guard_items]

    def create_guard(self, guard: GuardStruct) -> GuardStruct:
        db = next(self.get_db())
        guard_item = GuardItem(
            name=guard.name,
            railspec=guard.to_dict(),
            num_reasks=None,
            description=guard.description,
        )
        db.add(guard_item)
        db.commit()
        return from_guard_item(guard_item)

    def update_guard(self, guard_name: str, guard: GuardStruct) -> GuardStruct:
        db = next(self.get_db())
        guard_item = self.get_guard_item(guard_name)
        if guard_item is None:
            raise HttpError(
                status=404,
                message="NotFound",
                cause="A Guard with the name {guard_name} does not exist!".format(
                    guard_name=guard_name
                ),
            )
        # guard_item.num_reasks = guard.num_reasks
        guard_item.railspec = guard.to_dict()
        guard_item.description = guard.description
        db.commit()
        return from_guard_item(guard_item)

    def upsert_guard(self, guard_name: str, guard: GuardStruct) -> GuardStruct:
        db = next(self.get_db())
        guard_item = self.get_guard_item(guard_name)
        if guard_item is not None:
            guard_item.railspec = guard.to_dict()
            guard_item.description = guard.description
            # guard_item.num_reasks = guard.num_reasks
            db.commit()
            return from_guard_item(guard_item)
        else:
            return self.create_guard(guard)

    def delete_guard(self, guard_name: str) -> GuardStruct:
        db = next(self.get_db())
        guard_item = self.get_guard_item(guard_name)
        if guard_item is None:
            raise HttpError(
                status=404,
                message="NotFound",
                cause="A Guard with the name {guard_name} does not exist!".format(
                    guard_name=guard_name
                ),
            )
        db.delete(guard_item)
        db.commit()
        guard = from_guard_item(guard_item)
        return guard
