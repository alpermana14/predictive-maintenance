"""Append-only audit log (CAEE revision: R1-11 auditability, R2-20 provenance, R1-9 operator study).

Each line of audit_log.jsonl is one JSON record: approvals and rejections of work orders and
the decisions of the retrieval-sufficiency gate. Every record carries the SHA-256 hash of the
previous record, so a deleted or edited line breaks the chain and can be detected with
verify_chain().
"""
import hashlib
import json
import os
import threading
from datetime import datetime, timezone

AUDIT_PATH = os.getenv(
    "AUDIT_LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_log.jsonl"),
)
_LOCK = threading.Lock()
_GENESIS = "0" * 64


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _last_hash() -> str:
    if not os.path.exists(AUDIT_PATH):
        return _GENESIS
    last = None
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last)["hash"] if last else _GENESIS


def log_event(event: str, **fields) -> dict:
    """Append one record and return it."""
    with _LOCK:
        record = {
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            **fields,
            "prev_hash": _last_hash(),
        }
        body = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        record["hash"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return record


def verify_chain() -> tuple[bool, int]:
    """Return (chain intact, number of records checked)."""
    if not os.path.exists(AUDIT_PATH):
        return True, 0
    prev, n = _GENESIS, 0
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            stored = rec.pop("hash")
            body = json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str)
            if rec.get("prev_hash") != prev or hashlib.sha256(body.encode("utf-8")).hexdigest() != stored:
                return False, n
            prev, n = stored, n + 1
    return True, n
