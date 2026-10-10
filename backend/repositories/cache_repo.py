"""Local cache for Jev responses so identical cases are never re-sent."""
from __future__ import annotations

import json
from typing import Any

from backend.models.orm import JevCacheEntry
from backend.repositories.db import session_scope


def get_cached(key: str) -> dict[str, Any] | None:
    with session_scope() as s:
        row = s.get(JevCacheEntry, key)
        if row is None:
            return None
        return {
            "response": json.loads(row.response_json),
            "latency_ms": row.latency_ms,
            "usage": json.loads(row.usage_json) if row.usage_json else None,
            "request_id": row.request_id,
            "model": row.model,
        }


def put_cached(
    key: str, model: str, response: dict[str, Any], latency_ms: float | None,
    usage: dict[str, int] | None, request_id: str | None,
) -> None:
    with session_scope() as s:
        s.merge(
            JevCacheEntry(
                key=key, model=model, response_json=json.dumps(response), latency_ms=latency_ms,
                usage_json=json.dumps(usage) if usage else None, request_id=request_id,
            )
        )


def clear_cache() -> int:
    with session_scope() as s:
        n = s.query(JevCacheEntry).delete()
        return int(n)
