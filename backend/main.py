import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import BaseModel
from chat_engine import agent_executor, DRAFT_STORE, vectorstore_history
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv 

# Import logic from your ML Engine
from ml_engine import run_pipeline, load_conveyor_data, TARGETS 

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

state = MachineState()

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
        
        state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] Scheduler updated successfully. Last data point: {df.index[-1]}")
        
    except Exception as e:
        print(f"[ERROR] Scheduler update failed: {str(e)}")

@app.on_event("startup")
def start_realtime_system():
    """Initializes the system and starts the background timer."""
    print("[INFO] System startup: Initializing AI Engine...")
    
    # 1. Run immediately so the dashboard isn't empty when you open it
    update_machine_state()
    
    # 2. Configure the Scheduler to run every 1 minute
    # You can change 'minutes=1' to 'seconds=30' or 'hours=1'
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_machine_state, 'interval', minutes=5)
    scheduler.start()
    
    print("[INFO] Scheduler is active: Auto-refreshing every 5 minutes.")

# ===================== 3. API ENDPOINTS =====================

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
    
    return {
        "timestamp": state.last_update,  # The time the AI actually ran
        "data_timestamp": str(latest.name), # The time of the sensor reading
        "metrics": latest.to_dict(),
        "status": status,
        "iso_zone": iso_zone,
        "machine_status": getattr(state, "status", "Unknown"),
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
        "unit": "mm/s" if "rms" in target else "°C" if "temp" in target else "A"
    }

@app.get("/api/anomalies/{target}")
def get_anomalies(target: str):
    # ... existing checks ...
    scores = state.anomalies.get(target)
    timestamps = state.data.index[-len(scores):].astype(str).tolist()
    
    # --- NEW: Get Raw Values ---
    raw_values = state.data[target].iloc[-len(scores):].values.tolist()
    # ---------------------------

    return {
        "scores": scores.tolist(),
        "timestamps": timestamps,
        "raw_values": raw_values,  # <--- Add this!
        "threshold": float(np.percentile(scores, 5)),
        # ... rest of return ...
    }

    # --- NEW: Get the timestamps for these specific scores ---
    # The scores correspond to the LAST N rows of the data
    timestamps = state.data.index[-len(scores):].astype(str).tolist()
    # ---------------------------------------------------------

    threshold = float(np.percentile(scores, 5))
    latest_score = scores[-1]
    is_anomaly = latest_score < threshold

    return {
        "scores": scores.tolist(),
        "timestamps": timestamps, # <--- Sending dates to frontend
        "threshold": threshold,
        "status": "Anomaly" if is_anomaly else "Normal",
        "latest_score": float(latest_score)
    }

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
    
    
        current_context = {
            "last_update": str(latest.name),
            "data_quality_warning": data_quality_msg,
            "minutes_ago": minutes_ago,   # <--- Pass the gap
            "realtime_status_msg": realtime_status,
            "status": "Anomaly" if state.anomalies and list(state.anomalies.values())[0][-1] < 0.05 else "Normal",
            "iso_10816_status": iso_status, 
            "current_vibration": f"{z_rms} mm/s",
            "is_anomaly": True,  # You can make this dynamic based on real threshold logic
            # Pass through session id so tools can use it without asking the user
            "session_id": req.session_id,
            # Send a tiny summary of the forecast
            # Ensure all values are basic Python types (no NumPy types) for serialization
            "forecast_summary": (
                {k: float(v.iloc[-1]) for k, v in state.forecast.items()}
                if state.forecast is not None
                else "Loading..."
            ),
            "current_draft_text": existing_draft,
        }

    # 2. Run the LangGraph Agent
    # 'thread_id' is used by LangGraph to remember conversation history
    config = {"configurable": {"thread_id": req.session_id}}

    output = agent_executor.invoke(
        {
            "messages": [HumanMessage(content=req.message)], 
            "machine_state": current_context
        },
        config=config
    )

    # 3. Extract Response
    ai_response = output["messages"][-1].content

    # NEW: Remove markdown symbols for a "clean" look
    clean_response = ai_response.replace("**", "").replace("###", "").replace("#", "")

    # 4. Get Current Draft (if any exists for this session)
    current_draft = DRAFT_STORE.get(req.session_id, "")

    return {
        "response": ai_response,
        "draft": current_draft
    }

class ApprovalRequest(BaseModel):
    session_id: str

@app.post("/api/work_orders/approve")
def approve_work_order(req: ApprovalRequest):
    """
    Human-in-the-Loop Endpoint:
    The USER triggers this to finalize the draft created by the AI.
    """
    content = DRAFT_STORE.get(req.session_id, "")
    
    if not content:
        raise HTTPException(status_code=400, detail="No draft found for this session.")

    # 1. Clean the content
    content_clean = content.replace("*", "").strip()

    # 2. Create Canonical ID (One per day)
    today_str = datetime.utcnow().strftime("%Y_%m_%d")
    canonical_id = f"work_order_{today_str}"

    # 3. Check for duplicates (Optional: Logic to append instead of skip)
    existing = vectorstore_history._collection.get(where={"id": canonical_id})
    if existing and existing.get("documents"):
         # For this demo, we append to the existing day's log
         # In production, you might want distinct IDs like work_order_DATE_001
         pass 

    # 4. Save to Vector DB
    doc = Document(
        page_content=content_clean,
        metadata={
            "id": canonical_id,
            "created_at": datetime.utcnow().isoformat(),
            "session_id": req.session_id,
            "status": "human_approved" # Tag it!
        },
    )
    vectorstore_history.add_documents([doc])

    # 5. Clear the Draft
    DRAFT_STORE[req.session_id] = ""

    return {"status": "success", "work_order_id": canonical_id}