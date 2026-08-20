from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Render's free-tier filesystem is ephemeral -- set DATABASE_URL (e.g. to a
# free Render Postgres instance) so routing telemetry survives restarts.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./router.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app.models import telemetry  # noqa: F401  (ensures model is registered)

    Base.metadata.create_all(bind=engine)
