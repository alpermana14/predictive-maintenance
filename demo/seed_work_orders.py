"""Adds the demo's seed work orders (backend/seed_work_orders.json) to a work-order store.

The demo deployment adds them itself at start-up (SEED_WORK_ORDERS=1 in backend/Dockerfile) and the May 2026
replay (demo/replay_may2026.py) seeds its own temporary store, so this script is only needed for another
persistent store, e.g. the local backend's backend/maintenance_history_db. Records are written with their fixed ids, so
running the script twice does not duplicate them. Stop the backend first: Chroma stores are not meant to be
written by two processes at once.

Usage (backend/.env or the environment must hold OPENAI_API_KEY, used for the embeddings):
  python demo/seed_work_orders.py                         # backend/maintenance_history_db
  python demo/seed_work_orders.py --remove                # take the seed records out again
"""
import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DEMO_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
import work_order_seeds  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=os.path.join(REPO_ROOT, "backend", "maintenance_history_db"),
                    help="Chroma directory of the work-order store")
    ap.add_argument("--remove", action="store_true", help="remove the seed records instead of adding them")
    args = ap.parse_args()
    load_dotenv(os.path.join(REPO_ROOT, "backend", ".env"))
    # Same embedding model as backend/chat_engine.py, so the retrieval-sufficiency gate sees the same distances
    store = Chroma(persist_directory=args.store, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    if args.remove:
        store.delete(ids=work_order_seeds.seed_ids())
        print(f"[SEED] removed the seed records from {args.store}")
    else:
        added = work_order_seeds.ensure_seeded(store)
        print(f"[SEED] {args.store}: {'added ' + ', '.join(added) if added else 'seed records already present'}")
    print(f"[SEED] store now holds {store._collection.count()} records")  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
