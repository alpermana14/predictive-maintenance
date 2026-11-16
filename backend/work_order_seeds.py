"""Seed work orders of the demo (seed_work_orders.json): the two earlier records shown in the FOI 2026 demo video.

They are fixtures WO-E and WO-C of the agent benchmark (sufficiency-gated-maintenance-agent repository): two
routine-maintenance records that do not match the 5 May 2026 vibration and noise change, so the
retrieval-sufficiency gate rejects them for that query until an operator approves a matching work order.

Used by the demo deployment (main.py start-up, when SEED_WORK_ORDERS=1 as set in the Dockerfile), by the local
May 2026 replay (demo/replay_may2026.py) and by demo/seed_work_orders.py. The records have fixed ids, so seeding
a store twice never duplicates them, and a record already present is left as it is.
"""
import json
import os

from langchain_core.documents import Document

SEED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_work_orders.json")


def load_seeds(path: str = SEED_FILE) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def seed_ids(seeds=None) -> list:
    return [s["id"] for s in (seeds or load_seeds())]


def ensure_seeded(store, seeds=None) -> list:
    """Adds the seed records missing from the Chroma store; returns the ids that were added."""
    seeds = seeds or load_seeds()
    present = set(store.get(ids=seed_ids(seeds))["ids"])
    missing = [s for s in seeds if s["id"] not in present]
    if missing:
        docs = [Document(page_content=s["content"],
                         metadata={"id": s["id"], "created_at": s["created_at"], "status": "human_approved",
                                   "approved_by": "seed", "source": s["source"]})
                for s in missing]
        store.add_documents(docs, ids=[s["id"] for s in missing])
    return [s["id"] for s in missing]
