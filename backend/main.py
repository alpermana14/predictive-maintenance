import os
import secrets
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import BaseModel
from chat_engine import agent_executor, DRAFT_STORE, vectorstore_history
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv 
from typing import Optional
from langchain_core.documents import Document

# Import logic from your ML Engine
from ml_engine import run_pipeline, load_conveyor_data, TARGETS, UNITS, FREQ
import audit
import machine_status as ms_rules

# Import Bentley iTwin IoT Bridge
import itwin_bridge

load_dotenv()

app = FastAPI(title="Predictive Maintenance API")

# ===================== 1. CONFIGURATION =====================

# Allow React (Frontend) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify "http://localhost:5173"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container (Holds the latest analysis in memory)
class MachineState:
    data = None       # Raw sensor dataframe
    forecast = None   # Future predictions dataframe
    anomalies = None  # IDK anomaly scores
    importance = None # Feature importance
    models = None     # Trained LightGBM models
    last_update = None # Timestamp of last successful run
    forecast_log = None  # [(origin timestamp, forecast dataframe)] for the last 7 days
    shift_status = None  # per channel: threshold, latest score and flag (rule B1)
    shift_events = None  # flagged windows of the last 7 days
    shift_episode = None  # open distribution-shift episode, kept until an operator reviews it

state = MachineState()
state.forecast_log = []
state.shift_status = {}
state.shift_events = []

FORECAST_LOG_DAYS = 7
FORECAST_ERROR_WINDOW = pd.Timedelta(hours=24)
FORECAST_POINTS = {"+1 h": 2, "+3 h": 6, "+6 h": 12}  # steps of 30 minutes


def record_forecast(origin, forecast_df):
    """Keep one forecast per data origin so later observations can score it."""
    if forecast_df is None:
        return
    if state.forecast_log and state.forecast_log[-1][0] == origin:
        state.forecast_log[-1] = (origin, forecast_df)
    else:
        state.forecast_log.append((origin, forecast_df))
    cutoff = origin - pd.Timedelta(days=FORECAST_LOG_DAYS)
    state.forecast_log = [(o, f) for o, f in state.forecast_log if o >= cutoff]


def recent_forecast_errors(data: pd.DataFrame, target: str) -> dict:
    """Mean absolute error of earlier forecasts whose target times fall in the last 24 h.

    Only observed values count: points filled by interpolation (error flag) are skipped.
    """
    if data is None or not state.forecast_log:
        return {"status": "not yet available"}
    now = data.index[-1]
    flag_col = f"{target}_error_flag"
    errors = {1: [], 12: []}
    for origin, fdf in state.forecast_log:
        for step in errors:
            ts = origin + step * pd.Timedelta(FREQ)
            if ts > now or ts < now - FORECAST_ERROR_WINDOW or ts not in data.index:
                continue
            if flag_col in data.columns and bool(data.at[ts, flag_col]):
                continue
            if ts in fdf.index:
                errors[step].append(abs(float(fdf.at[ts, target]) - float(data.at[ts, target])))
    out = {}
    for step, label in [(1, "30 min ahead"), (12, "6 h ahead")]:
        vals = errors[step]
        out[label] = ({"mae": round(float(np.mean(vals)), 3), "n": len(vals)}
                      if vals else {"mae": None, "n": 0})
    return out


def update_shift_state(df: pd.DataFrame):
    """Apply rule B1 to the latest s-IDK^2 scores: per-channel status, event log and shift episode."""
    statuses = {}
    new_record = getattr(state, "last_scored", None) != df.index[-1]
    streaks = getattr(state, "flag_streaks", {}) or {}
    for ch, scores in (state.anomalies or {}).items():
        scores = np.asarray(scores).ravel()
        if len(scores):
            st = ms_rules.channel_status(scores, df.index[-len(scores):])
            if new_record:  # count consecutive cycles only when a new 30-min record was scored
                streaks[ch] = streaks.get(ch, 0) + 1 if st["latest_flag"] else 0
            st["raw_flag"] = st["latest_flag"]
            st["latest_flag"] = streaks.get(ch, 0) >= ms_rules.PERSISTENCE_CYCLES
            statuses[ch] = st
    state.flag_streaks, state.last_scored = streaks, df.index[-1]
    state.shift_status = statuses
    state.shift_events = ms_rules.update_events(state.shift_events, statuses, df.index[-1])
    previous = state.shift_episode
    state.shift_episode = ms_rules.update_episode(previous, statuses, df.index[-1])
    opened = state.shift_episode is not None and (previous is None or previous.get("reviewed"))
    if opened:
        audit.log_event("shift_detected", since=state.shift_episode["since"], channels=state.shift_episode["channels"],
                        scores={c: statuses[c]["latest_score"] for c in state.shift_episode["channels"]},
                        thresholds={c: statuses[c]["threshold"] for c in state.shift_episode["channels"]})


def current_alert() -> dict:
    """Alert level (rule B2) from the latest z_rms, the shift episode and the 6 h forecast."""
    if state.data is None:
        return {"level": "Unknown", "reasons": ["system initializing"]}
    z_now = float(state.data["z_rms"].iloc[-1])
    z_fc = float(state.forecast["z_rms"].iloc[-1]) if state.forecast is not None else None
    err = recent_forecast_errors(state.data, "z_rms").get("6 h ahead", {})
    return ms_rules.assess(z_now, state.shift_episode, z_fc, err.get("mae") if isinstance(err, dict) else None)


def current_reminder():
    """Reminder for an episode left unreviewed too long; logged once per episode. Never drafts a work order."""
    if state.data is None:
        return None
    rem = ms_rules.reminder(state.shift_episode, state.data.index[-1])
    if rem and getattr(state, "reminder_logged_since", None) != rem["since"]:
        state.reminder_logged_since = rem["since"]
        audit.log_event("shift_reminder", since=rem["since"], open_hours=rem["open_hours"],
                        channels=rem["channels"], currently_flagged=rem["currently_flagged"])
    return rem


def mark_shift_reviewed(reviewer: str, note: str, source: str):
    if state.shift_episode and not state.shift_episode.get("reviewed"):
        state.shift_episode = {**state.shift_episode, "reviewed": True, "reviewed_by": reviewer}
        audit.log_event("shift_reviewed", reviewer=reviewer, source=source, note=note,
                        since=state.shift_episode["since"], channels=state.shift_episode["channels"])


def build_forecast_summary(data: pd.DataFrame, forecast: pd.DataFrame, targets: list) -> dict:
    """Forecast trajectory (+1 h, +3 h, +6 h), change over 6 h and recent error, per channel."""
    latest = data.iloc[-1]
    summary = {}
    for tgt in targets:
        traj = {label: round(float(forecast[tgt].iloc[step - 1]), 3)
                for label, step in FORECAST_POINTS.items()}
        summary[tgt] = {
            "unit": UNITS.get(tgt, ""),
            "latest_observed": round(float(latest[tgt]), 3),
            "forecast": traj,
            "change_over_6h": round(traj["+6 h"] - float(latest[tgt]), 3),
            "recent_error_last_24h": recent_forecast_errors(data, tgt),
        }
    return summary

# ===================== 2. REAL-TIME SCHEDULER =====================

def update_machine_state():
    """
    Worker function: Runs in the background.
    1. Connects to MySQL
    2. Retrains models (or just predicts)
    3. Detects anomalies
    4. Updates the global 'state' object
    """
    print(f"[INFO] Scheduler starting update cycle at {datetime.now().strftime('%H:%M:%S')}...")
    
    try:
        # Load fresh data from SQL
        df = load_conveyor_data()
        
        if df.empty:
            print("[WARN] Scheduler warning: SQL returned no data.")
            return

        # Run the full ML Pipeline (Training + IDK + Forecast)
        # In a real heavy production system, you might only 'predict' here and 'train' nightly.
        # For this prototype, we do everything to keep it simple.
        state.data, state.forecast, state.anomalies, state.importance, state.models = run_pipeline(df)
        record_forecast(df.index[-1], state.forecast)
        update_shift_state(df)

        state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] Scheduler updated successfully. Last data point: {df.index[-1]}")

        # Push latest data to Bentley iTwin IoT (if configured)
        try:
            itwin_bridge.push_to_itwin(state)
        except Exception as itwin_err:
            print(f"[WARN] iTwin push failed (non-critical): {itwin_err}")
        
    except Exception as e:
        print(f"[ERROR] Scheduler update failed: {str(e)}")

@app.on_event("startup")
def start_realtime_system():
    """Initializes the system and starts the background timer."""
    print("[INFO] System startup: Initializing AI Engine...")
    
    # 1. Run immediately so the dashboard isn't empty when you open it
    update_machine_state()
    
    # 2. Setup Bentley iTwin IoT sensors (one-time registration)
    try:
        itwin_bridge.setup_itwin_sensors()
    except Exception as e:
        print(f"[WARN] iTwin sensor setup failed (non-critical): {e}")
    
    # 3. Retrain and refresh every 30 minutes, the resolution at which the analytics
    # run (one new 30-min sample per cycle). The interval also leaves headroom for
    # retraining time as the history grows.
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_machine_state, 'interval', minutes=30, max_instances=1, coalesce=True)
    scheduler.start()

    print("[INFO] Scheduler is active: retraining and refreshing every 30 minutes.")

# ===================== 3. API ENDPOINTS =====================

# --- Bentley iTwin IoT Endpoints ---

@app.get("/api/itwin/debug2")
def itwin_debug():
    """Returns the raw nodes, devices, and sensors from Bentley."""
    return itwin_bridge.debug_bentley_api()

@app.get("/api/itwin/status")
def itwin_status():
    """Returns the current status of the Bentley iTwin IoT integration."""
    return itwin_bridge.get_status()

@app.post("/api/itwin/push")
def itwin_manual_push():
    """Manually trigger a data push to Bentley iTwin IoT."""
    if state.data is None:
        raise HTTPException(status_code=503, detail="System initializing, no data to push.")
    
    success = itwin_bridge.push_to_itwin(state)
    status_info = itwin_bridge.get_status()
    
    if success:
        return {"status": "success", "detail": status_info["last_push"]}
    else:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Push to iTwin IoT failed",
                "info": status_info["last_push"]
            }
        )

@app.get("/api/summary")
def get_summary():
    """Returns the latest sensor values and overall machine status."""
    if state.data is None:
        raise HTTPException(status_code=503, detail="System initializing, please wait...")
    
    latest = state.data.iloc[-1]
    
    # Calculate ISO 10816 Zone (Simple Logic)
    z_rms = latest.get("z_rms", 0)
    
    if z_rms < 0.71:
        iso_zone = "A" 
        status = "Good"
    elif z_rms < 1.8:
        iso_zone = "B"
        status = "Acceptable"
    elif z_rms < 4.5:
        iso_zone = "C"
        status = "Unsatisfactory"
    else:
        iso_zone = "D"
        status = "Unacceptable"
    
    alert = current_alert()
    return {
        "timestamp": state.last_update,  # The time the AI actually ran
        "data_timestamp": str(latest.name), # The time of the sensor reading
        # the six analysed channels only (motor current is not used, Section 4.1)
        "metrics": {k: latest[k] for k in TARGETS if k in latest.index},
        "status": status,
        "iso_zone": iso_zone,
        "machine_status": alert["level"],     # alert level of rule B2
        "alert_reasons": alert.get("reasons", []),
        "shift_episode": state.shift_episode,
        "shift_reminder": current_reminder(),  # notice for an episode left unreviewed too long
        "replay": getattr(state, "replay_label", None),  # set only by scripts/replay_dashboard.py
    }

@app.get("/api/forecast/{target}")
def get_forecast(target: str):
    """Returns Historical Data + Future Forecast for plotting."""
    if state.data is None:
         raise HTTPException(status_code=503, detail="System initializing")
         
    if target not in TARGETS:
        raise HTTPException(status_code=404, detail=f"Sensor '{target}' not found. Available: {TARGETS}")

    # Get Historical Data (Last 48 steps ~ 24 hours)
    history = state.data[target].iloc[-48:]
    flags = state.data[f"{target}_error_flag"].iloc[-48:]
    # Get Forecast Data
    prediction = state.forecast[target]
    
    return {
        "history_x": history.index.astype(str).tolist(),
        "history_y": history.values.tolist(),
        "history_flags": flags.values.tolist(),
        "forecast_x": prediction.index.astype(str).tolist(),
        "forecast_y": prediction.values.tolist(),
        "unit": UNITS.get(target, "")
    }

@app.get("/api/anomalies/{target}")
def get_anomalies(target: str):
    """s-IDK^2 window scores of one channel with the threshold and status of rule B1."""
    if state.data is None or not state.anomalies:
        raise HTTPException(status_code=503, detail="System initializing")
    if target not in state.anomalies:
        raise HTTPException(status_code=404, detail=f"Sensor '{target}' not found. Available: {TARGETS}")
    scores = np.asarray(state.anomalies[target]).ravel()
    st = state.shift_status.get(target) or ms_rules.channel_status(scores, state.data.index[-len(scores):])
    return {
        "scores": scores.tolist(),
        "timestamps": state.data.index[-len(scores):].astype(str).tolist(),
        "raw_values": state.data[target].iloc[-len(scores):].values.tolist(),
        "threshold": st["threshold"],
        "status": "Shift" if st["latest_flag"] else "Normal",
        "latest_score": st["latest_score"],
    }


class ShiftReviewRequest(BaseModel):
    note: str = ""


@app.post("/api/shift/review")
def review_shift(req: ShiftReviewRequest, x_approver_token: Optional[str] = Header(default=None)):
    """A named approver marks the open distribution-shift episode as reviewed (rule B1)."""
    reviewer = _authorise(x_approver_token)
    if not state.shift_episode or state.shift_episode.get("reviewed"):
        raise HTTPException(status_code=400, detail="No open distribution-shift episode.")
    mark_shift_reviewed(reviewer, req.note.strip(), source="shift_review")
    return {"status": "reviewed", "episode": state.shift_episode}

@app.get("/api/importance")
def get_importance():
    """Returns feature importance for all targets."""
    if state.importance is None:
         return {}
    return state.importance


# ===================== 4. WORK ORDER HISTORY ENDPOINTS =====================

@app.get("/api/work_orders")
def list_work_orders(q: str | None = None):
    """
    Returns a lightweight list of saved work orders.
    If 'q' is provided, performs a similarity search over work orders.
    """
    try:
        items = []

        def _add_item(wid: str, created_at: str | None, full_text: str):
            # Only expose canonical work orders to the UI
            if not wid.startswith("work_order_"):
                return

            # Clean formatting for UI (remove '*' markdown artifacts)
            text_clean = (full_text or "").replace("*", "")

            items.append(
                {
                    "id": wid,
                    "created_at": created_at,
                    "preview": text_clean[:260],
                    "content": text_clean,
                }
            )

        if q:
            docs = vectorstore_history.similarity_search(q, k=50)
            for d in docs:
                meta = d.metadata or {}
                wid = meta.get("id", "")
                created_at = meta.get("created_at")
                _add_item(wid, created_at, d.page_content or "")
        else:
            # Use underlying Chroma collection to fetch all documents
            raw = vectorstore_history._collection.get()  # type: ignore[attr-defined]
            ids = raw.get("ids", []) or []
            docs = raw.get("documents", []) or []
            metas = raw.get("metadatas", []) or []

            for i, doc in enumerate(docs):
                meta = metas[i] if metas and i < len(metas) else {}
                wid = meta.get("id", ids[i] if i < len(ids) else "")
                created_at = meta.get("created_at")
                _add_item(wid, created_at, doc or "")

        # Deduplicate by ID (keep the newest per work_order_xxx)
        dedup = {}
        for it in items:
            wid = it["id"]
            prev = dedup.get(wid)
            if not prev or (it.get("created_at") or "") > (prev.get("created_at") or ""):
                dedup[wid] = it

        items_uniq = list(dedup.values())

        # Sort newest first if timestamps exist
        items_uniq.sort(
            key=lambda x: x.get("created_at") or "",
            reverse=True,
        )
        return {"items": items_uniq}
    except Exception as e:
        print(f"[ERROR] Failed to list work orders: {e}")
        return {"items": []}


@app.get("/api/work_orders/{work_id}")
def get_work_order(work_id: str):
    """
    Returns the full content of a specific work order by its metadata 'id'.
    """
    try:
        raw = vectorstore_history._collection.get(  # type: ignore[attr-defined]
            where={"id": work_id}
        )
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []

        if not docs:
            # Fallback: approximate search if direct filter fails
            hits = vectorstore_history.similarity_search(work_id, k=1)
            if not hits:
                raise HTTPException(status_code=404, detail="Work order not found")
            d = hits[0]
            return {
                "id": d.metadata.get("id", work_id),
                "created_at": d.metadata.get("created_at"),
                "content": d.page_content,
                "metadata": d.metadata,
            }

        doc = docs[0]
        meta = metas[0] if metas else {}
        return {
            "id": meta.get("id", work_id),
            "created_at": meta.get("created_at"),
            "content": doc,
            "metadata": meta,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to fetch work order {work_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving work order")

# ===================== CHATBOT ENDPOINT =====================

class ChatRequest(BaseModel):
    message: str
    session_id: str
    image_base64: Optional[str] = None # New field for the image

def build_historical_summary(df: pd.DataFrame, targets: list) -> dict:
    """
    Builds a compact historical summary for the LLM context.
    Covers last 2 days (96 x 30min steps) and last 7 days (336 steps).
    """
    now = df.index[-1]
    two_days_ago = now - pd.Timedelta(days=2)
    seven_days_ago = now - pd.Timedelta(days=7)

    summary = {}

    for period_label, cutoff in [("last_2_days", two_days_ago), ("last_7_days", seven_days_ago)]:
        period_df = df[df.index >= cutoff][targets]
        if period_df.empty:
            summary[period_label] = "No data available"
            continue

        stats = {}
        for tgt in targets:
            col = period_df[tgt]
            stats[tgt] = {
                "min": round(float(col.min()), 3),
                "max": round(float(col.max()), 3),
                "mean": round(float(col.mean()), 3),
                "std": round(float(col.std()), 3),
                "latest": round(float(col.iloc[-1]), 3),
            }
        # Distribution-shift events (rule B1) of this period; the log keeps the last 7 days
        stats["anomaly_events"] = [e for e in state.shift_events if pd.Timestamp(e["timestamp"]) >= cutoff]
        summary[period_label] = stats

    return summary

#@app.post("/api/chat")
@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """
    Handles Chat Interaction.
    Injects the LATEST machine state into the AI context.
    """

    # 1. Prepare the Live Context
    # We grab the latest data from your global 'state' object
    # (Ensure 'state' is the variable name of your MachineState() instance)

    existing_draft = DRAFT_STORE.get(req.session_id, "")

    if state.data is None:
         # Fallback if system is just starting up
        current_context = {
            "status": "Initializing",
            "last_update": "Pending",
            "session_id": req.session_id,
            "current_draft_text": existing_draft,
        }
    else:
        # --- NEW: CALCULATE ISO ZONE HERE ---
        latest = state.data.iloc[-1]
        # Check if ANY of the target sensors are currently flagged as an error
        error_sensors = [tgt for tgt in TARGETS if latest.get(f"{tgt}_error_flag") == True]
        if error_sensors:
            data_quality_msg = f"WARNING: Sensor Reading Error detected for {', '.join(error_sensors)}. Data is currently interpolated."
        else:
            data_quality_msg = "All sensors are reporting normally."
        z_rms = latest.get("z_rms", 0)
        sensor_time = latest.name  # This is a pandas Timestamp
        now_my = datetime.utcnow() + timedelta(hours=8)

        # Calculate time difference in minutes
        diff = now_my - sensor_time
        minutes_ago = int(diff.total_seconds() / 60)
        if minutes_ago > 30:
            realtime_status = f"NO. The data is NOT real-time. It is {minutes_ago} minutes old (Timestamp: {sensor_time})."
        else:
            realtime_status = f"YES. The data is real-time ({minutes_ago} mins delay)."
        
        if z_rms < 0.71: iso_status = "Zone A (Good)"
        elif z_rms < 1.8: iso_status = "Zone B (Acceptable)"
        elif z_rms < 4.5: iso_status = "Zone C (Unsatisfactory)"
        else: iso_status = "Zone D (Unacceptable)"
    
        historical_summary = build_historical_summary(state.data, TARGETS)
        alert = current_alert()

        current_context = {
            "last_update": str(latest.name),
            "data_quality_warning": data_quality_msg,
            "minutes_ago": minutes_ago,   # <--- Pass the gap
            "realtime_status_msg": realtime_status,
            # Alert level of rule B2 (ISO zone relative to the normal zone, shift episode of rule B1,
            # 6 h forecast), with the reasons the agent must report
            "status": alert["level"],
            "alert_reasons": alert["reasons"],
            "work_order_scope": alert["work_order_scope"],  # fixed rule: monitoring / scheduled / prompt inspection
            "shift_status": {ch: {"latest_score": st["latest_score"], "threshold": st["threshold"],
                                  "flagged": st["latest_flag"]} for ch, st in state.shift_status.items()},
            "iso_10816_status": iso_status,
            "current_vibration": f"{z_rms} mm/s",
            # Pass through session id so tools can use it without asking the user
            "session_id": req.session_id,
            # Forecast trajectory, change over 6 h and recent forecast error per channel
            # (CAEE revision T6). Plain Python types only, for serialization.
            "forecast_summary": (
                build_forecast_summary(state.data, state.forecast, TARGETS)
                if state.forecast is not None
                else "Loading..."
            ),
            "current_draft_text": existing_draft,
             "historical_summary": historical_summary,
        }
        
    # 2. Prepare the Message Content (Handling optional image)
    # This structures the payload dynamically so LangChain / OpenAI know if an image is attached
    #message_content = [{"type": "text", "text": req.message}]
    message_content = []
    if req.image_base64:
        print("[DEBUG] Image attached to chat message.")

        message_content = [
            {"type": "text", "text": req.message},
            {
                "type": "image_url",
                "image_url": {
                    "url": req.image_base64,
                    "detail": "auto"
                } 
            }
        ]
    else:
        # CRITICAL FIX: If there is no image, just pass the plain text string
        message_content = req.message

    # 3. Run the LangGraph Agent
    # 'thread_id' is used by LangGraph to remember conversation history
    config = {"configurable": {"thread_id": req.session_id}}

    output = agent_executor.invoke(
        {
            "messages": [HumanMessage(content=message_content)],
            "machine_state": current_context
        },
        config=config
    )

    # 4. Extract Response
    ai_response = output["messages"][-1].content

    # NEW: Remove markdown symbols for a "clean" look
    clean_response = ai_response.replace("**", "").replace("###", "").replace("#", "")

    # 4. Get Current Draft (if any exists for this session)
    current_draft = DRAFT_STORE.get(req.session_id, "")

    return {
        "response": ai_response,
        "draft": current_draft
    }

# ===================== HUMAN APPROVAL (CAEE revision T7) =====================
# Only named approvers can approve or reject. Tokens are configured in backend/.env as
#   APPROVER_TOKENS=name1:token1,name2:token2
# Without this setting the approval endpoints refuse every request (fail closed).
# APPROVAL_MODE=open (set only in backend/Dockerfile for the demo deployment) skips the token
# check; every action is then recorded under DEMO_APPROVER in the work order and the audit log.
DEMO_APPROVER = "demo operator"


def _approval_open() -> bool:
    return os.getenv("APPROVAL_MODE", "token").strip().lower() == "open"


def _approvers() -> dict:
    pairs = [p.split(":", 1) for p in os.getenv("APPROVER_TOKENS", "").split(",") if ":" in p]
    return {token.strip(): name.strip() for name, token in pairs if name.strip() and token.strip()}


def _authorise(token: Optional[str]) -> str:
    if _approval_open():
        return DEMO_APPROVER
    approvers = _approvers()
    if not approvers:
        raise HTTPException(status_code=503, detail="Approval is disabled: no approvers are configured.")
    for known, name in approvers.items():
        if token and secrets.compare_digest(token, known):
            return name
    audit.log_event("approval_denied", reason="invalid or missing approver token")
    raise HTTPException(status_code=401, detail="Invalid or missing approver token.")


class ApprovalRequest(BaseModel):
    session_id: str
    content: Optional[str] = None  # the text as edited by the operator; None = approve the draft unchanged


class RejectionRequest(BaseModel):
    session_id: str
    reason: str


@app.post("/api/work_orders/approve")
def approve_work_order(req: ApprovalRequest, x_approver_token: Optional[str] = Header(default=None)):
    """
    Human-in-the-Loop Endpoint:
    A named approver finalises the draft created by the AI, optionally after editing it.
    Only the approved text enters the work-order knowledge base.
    """
    approver = _authorise(x_approver_token)
    draft = DRAFT_STORE.get(req.session_id, "")
    if not draft:
        raise HTTPException(status_code=400, detail="No draft found for this session.")

    final_text = req.content if req.content is not None else draft
    content_clean = final_text.replace("*", "").strip()
    if not content_clean:
        raise HTTPException(status_code=400, detail="The work order is empty.")
    modified = content_clean != draft.replace("*", "").strip()

    try:
        now = datetime.utcnow()
        work_order_id = f"work_order_{now.strftime('%Y_%m_%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        doc = Document(
            page_content=content_clean,
            metadata={
                "id": work_order_id,
                "created_at": now.isoformat(),
                "session_id": req.session_id,
                "status": "human_approved",
                "approved_by": approver,
                "modified_by_operator": modified,
                "draft_sha256": audit.content_hash(draft),
                "content_sha256": audit.content_hash(content_clean),
            },
        )
        vectorstore_history.add_documents([doc])
        DRAFT_STORE[req.session_id] = ""
        mark_shift_reviewed(approver, f"work order {work_order_id} approved", source="work_order_approved")
        audit.log_event(
            "work_order_approved", work_order_id=work_order_id, approver=approver,
            session_id=req.session_id, modified_by_operator=modified,
            draft_sha256=audit.content_hash(draft), content_sha256=audit.content_hash(content_clean),
            draft_chars=len(draft), content_chars=len(content_clean),
        )
        return {"status": "success", "work_order_id": work_order_id, "modified": modified}
    except Exception as e:
        print(f"[ERROR] Failed to approve work order for session {req.session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save work order: {str(e)}"
        )


@app.post("/api/work_orders/reject")
def reject_work_order(req: RejectionRequest, x_approver_token: Optional[str] = Header(default=None)):
    """A named approver rejects the AI draft; nothing enters the knowledge base."""
    approver = _authorise(x_approver_token)
    draft = DRAFT_STORE.get(req.session_id, "")
    if not draft:
        raise HTTPException(status_code=400, detail="No draft found for this session.")
    if not req.reason.strip():
        raise HTTPException(status_code=400, detail="A reason is required to reject a draft.")
    DRAFT_STORE[req.session_id] = ""
    mark_shift_reviewed(approver, f"work order draft rejected: {req.reason.strip()}", source="work_order_rejected")
    audit.log_event(
        "work_order_rejected", approver=approver, session_id=req.session_id,
        reason=req.reason.strip(), draft_sha256=audit.content_hash(draft), draft_chars=len(draft),
    )
    return {"status": "rejected"}