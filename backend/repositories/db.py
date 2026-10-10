from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.models.orm import Base
from backend.utils.config import get_settings
from backend.utils.errors import RepositoryError

_engine: Engine | None = None
_engine_url: str | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _engine_url, _SessionFactory
    url = get_settings().dentoroute_database_url
    if _engine is None or _engine_url != url:
        if url.startswith("sqlite:///"):
            Path(url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {})
        Base.metadata.create_all(_engine)
        _SessionFactory = sessionmaker(_engine, expire_on_commit=False)
        _engine_url = url
    return _engine


def reset_engine() -> None:
    global _engine, _engine_url, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine, _engine_url, _SessionFactory = None, None, None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Database failures become a human-readable RepositoryError."""
    get_engine()
    assert _SessionFactory is not None
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        raise RepositoryError("The local database is unavailable or returned an error.", detail=str(exc)) from exc
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
