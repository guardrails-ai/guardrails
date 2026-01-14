from sqlalchemy import Column, String, Integer
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, CHAR
from guardrails_api.clients.postgres_client import Base


class GuardItemAudit(Base):
    __tablename__ = "guards_audit"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    railspec = Column(JSONB, nullable=False)
    num_reasks = Column(Integer, nullable=True)
    description = Column(String, nullable=True)
    # owner = Column(String, nullable=False)
    replaced_on = Column(TIMESTAMP, nullable=False)
    # replaced_by = Column(String, nullable=False)
    operation = Column(CHAR, nullable=False)

    def __init__(
        self,
        id,
        name,
        railspec,
        num_reasks,
        description,
        # owner = None
        replaced_on,
        # replaced_by
        operation,
    ):
        self.id = id
        self.name = name
        self.railspec = railspec
        self.num_reasks = num_reasks
        self.description = description
        self.replaced_on = replaced_on
        self.operation = operation
        # self.owner = owner
