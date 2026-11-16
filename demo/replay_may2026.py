"""Demo replay: the system as it was on 5 May 2026 at 15:00 (local use only).

This is the setup of the demo video (FOI 2026): the dashboard and the copilot run on the recorded conveyor data
up to the load test of 5 May 2026, so the agent sees the same machine state as in the video
(z_rms 3.54 mm/s, ISO 10816 zone C, distribution-shift episode open since 08:30, alert level Low).
The state is rebuilt exactly as in the sufficiency-gated-maintenance-agent repository
(experiments/e17_case_study_replay.py and scripts/replay_dashboard.py):

  * recorded data (demo/data/conveyor_export.csv) cut at REPLAY_NOW;
  * every 30-min cycle of the previous 7 days replayed through the backend functions of main.py, with the
    s-IDK^2 scores seeded per step and channel (seed 0), so the shift flags are reproducible;
  * shift episodes opened before 08:00 count as reviewed (the operators were present to add the load);
  * the tuned LightGBM pipeline for the last REPLAY_FORECAST_HOURS, so the forecast and its recent error are
    those the live system would have had (cached in demo/cache, the first start takes a while);
  * the clock is set to 15:05, so the copilot treats the data as real time;
  * the copilot runs at temperature 0 with a fixed seed (REPLAY_DETERMINISTIC=0 restores the live 0.1), so a
    rehearsal and the demo itself give the same answers.

The work-order store is a fresh temporary one holding only the records in backend/seed_work_orders.json
(two routine-maintenance records, fixtures WO-E and WO-C of the agent benchmark). Every start therefore begins
from the same state as the video; work orders approved during the demo last until the server stops.
The audit log is temporary as well, and approval needs no token (APPROVAL_MODE=open, "demo operator").
No MySQL connection, scheduler or iTwin push is used.

Usage (from the repository root, with backend/.env holding OPENAI_API_KEY):
  python demo/replay_may2026.py            # serves the API on http://127.0.0.1:8000
  cd frontend && npm run dev:local         # dashboard on http://localhost:5173
Env: REPLAY_NOW (default "2026-05-05 15:00"), REPLAY_FORECAST_HOURS (default 30), REPLAY_PORT (default 8000)
"""
import os
import pickle
import random
import sys
import tempfile
import time
from datetime import datetime as _dt

import numpy as np
import pandas as pd

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DEMO_DIR)
BACKEND = os.path.join(REPO_ROOT, "backend")

NOW = pd.Timestamp(os.environ.get("REPLAY_NOW", "2026-05-05 15:00"))
REVIEW_AT = pd.Timestamp(os.environ.get("REPLAY_REVIEW_AT", "2026-05-05 08:00"))
FORECAST_HOURS = float(os.environ.get("REPLAY_FORECAST_HOURS", "30"))
EVENT_LOG_DAYS = 7
SEED = 0
PORT = int(os.environ.get("REPLAY_PORT", "8000"))
DETERMINISTIC = os.environ.get("REPLAY_DETERMINISTIC", "1").strip() == "1"  # copilot at temperature 0, fixed seed
LLM_SEED = int(os.environ.get("REPLAY_LLM_SEED", "7"))
DATA_CSV = os.environ.get("REPLAY_DATA_CSV", os.path.join(DEMO_DIR, "data", "conveyor_export.csv"))
CACHE_DIR = os.path.join(DEMO_DIR, "cache")

TMP = tempfile.mkdtemp(prefix="pm_replay_")
os.environ["AUDIT_LOG_PATH"] = os.path.join(TMP, "audit_log.jsonl")  # never the live audit log
os.environ["APPROVAL_MODE"] = "open"  # local demo: approvals are recorded under "demo operator"

os.chdir(BACKEND)  # backend/.env and the manual store
sys.path.insert(0, BACKEND)
import ml_engine  # noqa: E402
import machine_status as R  # noqa: E402
from IDK_square_sliding import IDK_square_sliding  # noqa: E402
import chat_engine  # noqa: E402
import main  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
import work_order_seeds  # noqa: E402


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    df = df[~df.index.duplicated(keep="first")]
    return df[df.index <= NOW]


def seeded_scores(df_cut, ts):
    """s-IDK^2 scores of one cycle with the per-step, per-channel seed of E16/E17 (the live system is not seeded)."""
    out = {}
    for ch in ml_engine.TARGETS:
        X = df_cut[ch].to_numpy(dtype=float)[-ml_engine.IDK_POPULATION:].reshape(-1, 1)
        random.seed(f"e16-{SEED}-{pd.Timestamp(ts)}-{ch}")
        out[ch] = np.asarray(IDK_square_sliding(X, t=ml_engine.IDK_T, psi1=ml_engine.IDK_PSI,
                                                width=ml_engine.IDK_WIDTH, psi2=ml_engine.IDK_PSI)).ravel()
    return out


def fake_clock(now_local):
    class FakeDateTime(_dt):
        @classmethod
        def utcnow(cls):
            return (now_local - pd.Timedelta(hours=8)).to_pydatetime()
    return FakeDateTime


def rebuild_state():
    """Replay every 30-min cycle of the last 7 days up to NOW into main.state (as E17)."""
    t0 = time.time()
    df_all = load_data()
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"forecasts_{NOW:%Y%m%d_%H%M}_{FORECAST_HOURS:g}h.pkl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    cycles = df_all.index[df_all.index > NOW - pd.Timedelta(days=EVENT_LOG_DAYS)]
    forecast_from = NOW - pd.Timedelta(hours=FORECAST_HOURS)
    todo = sum(1 for ts in cycles if ts >= forecast_from and ts not in cache)
    if todo:
        print(f"[REPLAY] training the forecast models for {todo} cycles (first start only, cached afterwards)...")
    st = main.state
    st.forecast_log, st.shift_status, st.shift_events, st.shift_episode = [], {}, [], None
    for i, ts in enumerate(cycles, 1):
        df_cut = df_all[df_all.index <= ts]
        st.data = df_cut
        if ts >= forecast_from:
            if ts not in cache:
                _, cache[ts], _, _, _ = ml_engine.run_pipeline(df_cut)
            st.forecast = cache[ts]
            main.record_forecast(ts, st.forecast)
        else:
            st.forecast = None
        st.anomalies = seeded_scores(df_cut, ts)
        main.update_shift_state(df_cut)
        if ts == REVIEW_AT:
            main.mark_shift_reviewed("replay assumption", "open episodes reviewed before the load test",
                                     source="demo_replay")
        if i % 48 == 0:
            print(f"  cycle {i}/{len(cycles)} {ts} ({time.time() - t0:.0f} s)", flush=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[REPLAY] replayed {len(cycles)} cycles up to {NOW} ({time.time() - t0:.0f} s)")


def seed_history(store_dir):
    """Fresh work-order store holding only the seed records (ids and dates as shown in the demo video)."""
    store = Chroma(persist_directory=store_dir, embedding_function=chat_engine.embeddings)
    work_order_seeds.ensure_seeded(store)
    return store


def setup(with_importance=True):
    """Rebuild the state, seed the store and freeze the clock. Returns the backend main module."""
    m = main
    rebuild_state()
    if with_importance:  # feature importance for the diagnostics panel (not needed by the copilot)
        _, _, _, m.state.importance, m.state.models = ml_engine.run_pipeline(m.state.data)
    m.update_machine_state = lambda: None             # no new data, no MySQL
    m.check_for_new_data = lambda: None
    m.itwin_bridge.setup_itwin_sensors = lambda: None
    m.itwin_bridge.push_to_itwin = lambda *_a, **_k: False
    if DETERMINISTIC:
        # Same model and tools as the live copilot, but temperature 0 and a fixed seed, so a rehearsal and the
        # demo give (near-)identical answers. The live system keeps temperature 0.1 (backend/chat_engine.py).
        from langchain_openai import ChatOpenAI
        demo_llm = ChatOpenAI(model=chat_engine.llm.model_name, temperature=0, seed=LLM_SEED, max_retries=5)
        chat_engine.llm_with_tools = demo_llm.bind_tools(chat_engine.tools)
    store = seed_history(os.path.join(TMP, "history"))
    chat_engine.vectorstore_history = store           # used by the copilot tool
    m.vectorstore_history = store                     # used by the work-order endpoints of main.py
    clock = NOW + pd.Timedelta(minutes=5)
    m.datetime = fake_clock(clock)                    # the copilot sees the data as real time
    m.state.last_update = str(clock)
    m.state.replay_label = f"replay of recorded data up to {NOW:%d %b %Y %H:%M}"
    alert = m.current_alert()
    print(f"[REPLAY] {NOW}: zone {alert['zone']}, alert level {alert['level']}; episode {m.state.shift_episode}")
    print(f"[REPLAY] work-order store: {store._collection.count()} seed records")  # type: ignore[attr-defined]
    return m


def serve():
    import uvicorn
    m = setup()
    m.app.router.on_startup.clear()  # the live start-up (MySQL load, scheduler) must not run
    uvicorn.run(m.app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    serve()
