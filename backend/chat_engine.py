import os
import json
import threading
import time
from datetime import datetime
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from operator import add as add_messages
from dotenv import load_dotenv

import audit
import rag_guard

load_dotenv()

# ===================== 1. SETUP & CONFIG =====================
# In-memory store for drafts (simulating session_state)
# Key = session_id, Value = draft_text
DRAFT_STORE = {}

# Always resolve paths relative to this file (so it works no matter
# where Uvicorn is started from, e.g. project root vs backend folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HISTORY_DIR = os.path.join(BASE_DIR, "maintenance_history_db")
MANUAL_DIR  = os.path.join(BASE_DIR, "maintenance_manual_db")
PDF_PATH    = os.path.join(BASE_DIR, "Maintenance_Conveyor.pdf")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm        = ChatOpenAI(model="gpt-4o", temperature=0.1, max_retries=5)
# Cheap, deterministic model for the guard's relevance-entailment stage (E7).
guard_judge = ChatOpenAI(model=rag_guard.judge_model(), temperature=0.0)

# The reasoning model is served under an account allowance expressed in tokens per minute. The
# allowance limits a rate, not a number of simultaneous calls, so several operators asking at the
# same moment can exhaust a whole minute of it in a few seconds and have their requests refused.
# Calls to the reasoning model therefore pass through a pacer that holds each caller until the
# rolling minute has room for it, and through a gate that bounds how many are in flight at once.
# Requests queue instead of failing, at the cost of waiting. The two settings below describe the
# account of this study and are the only values to change for a larger allowance. The guard's
# judge and the embedding model are served under separate allowances and are not paced.
LLM_TOKENS_PER_MINUTE = int(os.getenv("LLM_TOKENS_PER_MINUTE", "30000"))
LLM_TOKENS_PER_CALL = int(os.getenv("LLM_TOKENS_PER_CALL", "4000"))
LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))
_llm_gate = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)


class _TokenPacer:
    """Keeps the calls of all sessions within a rolling per-minute token allowance.

    Each caller reserves the tokens its call is expected to consume. When the last minute is
    already spoken for, the caller sleeps until the oldest reservation leaves the window. The
    reservation is an estimate; the retry budget of the client absorbs the difference.
    """

    def __init__(self, tokens_per_minute: int, window_s: float = 60.0):
        self.limit = tokens_per_minute
        self.window = window_s
        self._spent = []  # (time of the reservation, tokens reserved)
        self._lock = threading.Lock()

    def reserve(self, tokens: int) -> float:
        """Block until the allowance has room, then book the tokens. Returns the seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._spent = [(t, n) for t, n in self._spent if now - t < self.window]
                if sum(n for _, n in self._spent) + tokens <= self.limit or not self._spent:
                    self._spent.append((now, tokens))
                    return waited
                sleep_for = self.window - (now - self._spent[0][0]) + 0.05
            time.sleep(sleep_for)
            waited += sleep_for


_llm_pacer = _TokenPacer(LLM_TOKENS_PER_MINUTE)

# ===================== 2. VECTOR STORES (RAG) =====================

# --- Static knowledge: maintenance manual PDF ---
if os.path.exists(PDF_PATH):
    if not os.path.exists(MANUAL_DIR):
        print("[INFO] Ingesting manual PDF from:", PDF_PATH)
        loader   = PyPDFLoader(PDF_PATH)
        docs     = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits   = splitter.split_documents(docs)
        vectorstore_manual = Chroma.from_documents(
            splits, embeddings, persist_directory=MANUAL_DIR
        )
    else:
        vectorstore_manual = Chroma(
            persist_directory=MANUAL_DIR, embedding_function=embeddings
        )
    manual_retriever = vectorstore_manual.as_retriever(search_kwargs={"k": 2})
else:
    print(f"[WARN] Manual PDF not found at path: {PDF_PATH}")
    manual_retriever = None

# --- Dynamic knowledge: past approved work orders ---
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)
vectorstore_history = Chroma(
    persist_directory=HISTORY_DIR, embedding_function=embeddings
)


# ===================== 3. TOOLS =====================

@tool
def retriever_tool(query: str) -> str:
    """Search the machine maintenance manual for technical specifications,
    procedures, fault causes, and recommended corrective actions."""
    if not manual_retriever:
        return "Manual not found."
    docs = manual_retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])


@tool
def query_past_orders(query: str) -> str:
    """Search past approved maintenance work orders as decision-support
    knowledge for similar faults, root causes, and actions previously taken."""
    # Retrieval-sufficiency guard (E7): gate the top-k results so an
    # irrelevant nearest-neighbour cannot be passed off as evidence. On
    # insufficiency the guard returns an INSUFFICIENT_EVIDENCE sentinel that
    # the system prompt instructs the agent to surface honestly.
    sufficient, payload, decision = rag_guard.guarded_retrieve(
        vectorstore_history, query, k=3, llm=guard_judge, record_prefix="record"
    )
    print(f"[GUARD] past_orders stage={decision.stage} "
          f"dist={decision.best_distance:.3f} sufficient={sufficient}")
    audit.log_event("retrieval_gate", tool="query_past_orders", query=query,
                    stage=decision.stage, best_distance=round(float(decision.best_distance), 4),
                    sufficient=bool(sufficient))
    return payload


@tool
def update_work_order(content: str, session_id: str) -> str:
    """Update the current work order draft text with the provided content.
    Always pass the session_id exactly as given in the system context."""
    DRAFT_STORE[session_id] = content
    return "Draft updated. The user can see the preview."


tools          = [retriever_tool, query_past_orders, update_work_order]
llm_with_tools = llm.bind_tools(tools)


# ===================== 4. HELPER: FORMAT HISTORICAL SUMMARY =====================

def _format_historical_summary(historical_summary: dict) -> str:
    """
    Converts the historical_summary dict (built in main.py) into a
    readable plain-text block for the system prompt.

    Expected structure:
    {
      "last_2_days": {
          "z_rms": {"min": x, "max": x, "mean": x, "std": x, "latest": x},
          ...
          "anomaly_events": [{"timestamp": "...", "sensor": "...", "score": x}, ...]
      },
      "last_7_days": { ... }
    }
    """
    if not historical_summary:
        return "No historical data available."

    units = {
        "temperature": "°C",
        "z_rms":       "mm/s",
        "x_rms":       "mm/s",
        "z_peak":      "g",
        "x_peak":      "g",
        "noise":       "dB",
    }

    lines = []

    for period_key, period_label in [
        ("last_2_days", "Last 2 Days"),
        ("last_7_days", "Last 7 Days"),
    ]:
        period_data = historical_summary.get(period_key)
        if not period_data or period_data == "No data available":
            lines.append(f"  [{period_label}]: No data available")
            continue

        lines.append(f"  [{period_label}]")

        # Sensor statistics
        for sensor, stats in period_data.items():
            if sensor == "anomaly_events":
                continue  # handled separately below
            if not isinstance(stats, dict):
                continue
            unit = units.get(sensor, "")
            lines.append(
                f"    {sensor}: "
                f"min={stats.get('min', 'N/A')} {unit}, "
                f"max={stats.get('max', 'N/A')} {unit}, "
                f"mean={stats.get('mean', 'N/A')} {unit}, "
                f"std={stats.get('std', 'N/A')} {unit}, "
                f"latest={stats.get('latest', 'N/A')} {unit}"
            )

        # Anomaly events for this period
        anomaly_events = period_data.get("anomaly_events", [])
        if anomaly_events:
            lines.append(f"    Anomaly Events Detected ({len(anomaly_events)}):")
            for ev in anomaly_events[:10]:  # cap at 10 to keep prompt compact
                lines.append(
                    f"      - {ev.get('timestamp', 'Unknown')} | "
                    f"sensor={ev.get('sensor', '?')} | "
                    f"IDK score={ev.get('score', '?')}"
                )
        else:
            lines.append("    Anomaly Events: None detected in this period")

    return "\n".join(lines)


def _format_shift_status(shift_status) -> str:
    if not shift_status:
        return "    Not available"
    return "\n".join(f"    {ch}: {st.get('latest_score')} / {st.get('threshold')} / {'YES' if st.get('flagged') else 'no'}"
                     for ch, st in shift_status.items())


def _format_forecast_summary(forecast_summary) -> str:
    """Plain-text block of the per-channel forecast built in main.build_forecast_summary."""
    if not isinstance(forecast_summary, dict) or not forecast_summary:
        return f"  {forecast_summary or 'Loading...'}"
    lines = []
    for sensor, info in forecast_summary.items():
        unit = info.get("unit", "")
        traj = ", ".join(f"{k}: {v} {unit}" for k, v in info.get("forecast", {}).items())
        err = info.get("recent_error_last_24h", {})
        if err.get("status"):
            err_text = err["status"]
        else:
            parts = []
            for horizon, e in err.items():
                parts.append(f"{horizon} MAE={e['mae']} {unit} (n={e['n']})" if e.get("n") else f"{horizon}: no scored forecasts yet")
            err_text = "; ".join(parts)
        lines.append(
            f"  {sensor}: latest={info.get('latest_observed')} {unit} | {traj} | "
            f"change over 6 h={info.get('change_over_6h')} {unit} | recent forecast error (last 24 h): {err_text}"
        )
    return "\n".join(lines)


# ===================== 5. GRAPH DEFINITION =====================

class AgentState(TypedDict):
    messages:      Annotated[Sequence[BaseMessage], add_messages]
    machine_state: dict  # Live + historical data passed from FastAPI


def agent_node(state: AgentState):
    ms         = state["machine_state"]
    draft_text = ms.get("current_draft_text", "")
    rt_status  = ms.get("realtime_status_msg", "Unknown")

    # Format historical summary into readable text for the prompt
    historical_text = _format_historical_summary(
        ms.get("historical_summary", {})
    )

    sys_msg = SystemMessage(content=f"""
You are an advanced Multimodal Predictive Maintenance Copilot for an industrial conveyor system.
YOU HAVE VISION CAPABILITIES. You CAN view photos and images.
Do NOT ever state that you cannot view images or photos. When the user provides an image, you MUST actively analyze its contents.

# === [DOMAIN RECOGNITION & ROUTING RULES] ===
You monitor a specific LIVE CONVEYOR SYSTEM. However, the user may ask about OTHER general machinery.
1. Live Conveyor Queries: If the user asks about "the machine", "the conveyor", "current status",
   historical trends, past readings, or uploads an image related to the monitored conveyor:
   - Reference the [LIVE MACHINE STATUS] and [HISTORICAL SENSOR DATA] sections below.
   - Never say you lack historical data — the [HISTORICAL SENSOR DATA] section IS the historical record.
2. General Machinery Queries: If the user asks general engineering questions or about unrelated equipment:
   - Answer from your broad industrial knowledge.
   - Do NOT reference the live conveyor sensor data, as it is irrelevant to other machines.

# === [HISTORICAL DATA PROTOCOL] ===
When the user asks about past trends, e.g.:
  - "What happened yesterday?"
  - "Describe vibration over the last 2 days"
  - "How has temperature changed this week?"
  - "Were there any anomalies last week?"
You MUST answer using the [HISTORICAL SENSOR DATA] section below. It contains:
  - min, max, mean, std, and latest values per sensor for the last 2 days and last 7 days
  - Detected anomaly events with timestamps and IDK scores
Do NOT call any tool for historical sensor questions — the data is already provided.
Do NOT say "I don't have access to historical data" — this is factually incorrect.

# === [PAST WORK ORDER KNOWLEDGE PROTOCOL] ===
Past work orders are a decision-support knowledge base stored in the vector database.
- For troubleshooting, maintenance recommendations, risk assessment, or "what should we do next":
  Call `query_past_orders` before giving final advice.
- Use retrieved records to justify decisions with practical precedent.
- If relevant history exists, synthesize it into a recommended decision path.
- If no relevant history is found, state that clearly and continue with best-practice guidance.
Note: Past work orders are DIFFERENT from historical sensor data. Past work orders contain
technician notes, root causes, and repair actions. Historical sensor data contains raw sensor trends.

# === [RETRIEVAL SUFFICIENCY PROTOCOL — CRITICAL] ===
If `query_past_orders` returns a string beginning with "INSUFFICIENT_EVIDENCE", it means the
work order history contains NO record relevant to the question. In that case you MUST:
  1. Tell the user plainly that there is no matching past work order for this specific issue.
  2. NOT fabricate a past record, date, root cause, or action, and NOT present general
     knowledge as if it came from the maintenance history.
  3. You MAY still offer general best-practice guidance, but clearly labelled as general
     knowledge, not as retrieved history.
This protocol exists to prevent hallucinated maintenance history during knowledge-base
cold-start. Treat INSUFFICIENT_EVIDENCE as an authoritative "not in the records" signal.

# !!! CRITICAL PROTOCOL FOR WORK ORDERS !!!
1. TRIGGER: If the user asks to "Draft", "Create", "Write", or "Update" a work order...
2. EVIDENCE FIRST: When drafting a new work order whose Work Order Scope is scheduled_inspection or
   prompt_inspection, first call `retriever_tool` for the manual sections on the observed condition and
   `query_past_orders` for similar past work orders. A monitoring-scope work order lists no causes and needs no
   retrieval. (A pure wording edit of an existing draft does not need new retrieval.)
3. ACTION: Then you MUST call the tool `update_work_order` in the same turn.
4. FORBIDDEN: You are FORBIDDEN from saying "I have created a draft" UNLESS you have actually called the tool.
5. VERIFICATION: If you do not see the tool output in your message history, you have failed. Try again.

# === [PAYLOAD CONTENT FOR WORK ORDERS] ===
When calling `update_work_order`, the content argument MUST follow this template, keeping every heading and the
final Priority line:
Incident Report:
  Timestamp   : {ms.get('last_update')}
  Vibration   : {ms.get('current_vibration')}
  ISO Zone    : {ms.get('iso_10816_status')}
  Alert Level : {ms.get('status')} ({'; '.join(ms.get('alert_reasons', []))})
Root Cause Analysis:
  <as set by the Work Order Scope below>
Recommended Actions (<timing set by the Work Order Scope below>):
  <as set by the Work Order Scope below>
Priority: <copy it from the Alert Level: High -> High, Medium -> Medium, Low or Normal -> Low; do not choose it yourself>

# === [WORK ORDER SCOPE — FIXED RULE, NOT YOUR CHOICE] ===
Work Order Scope: {ms.get('work_order_scope', 'Unknown')}
- monitoring (the vibration reading is within the acceptable range for this conveyor):
    Root Cause Analysis: "Not required: z_rms is within the acceptable range for this conveyor." Do not list causes.
    Heading: "Recommended Actions (none required):", then "No maintenance action required. Continue routine
    monitoring." You may add one sentence naming an open distribution-shift episode or the forecast as something to
    watch on the dashboard, but do NOT recommend any inspection, check, test run, adjustment, cleaning, lubrication,
    repair or replacement.
- scheduled_inspection:
    Root Cause Analysis: possible causes from the retrieved manual sections, stated as possibilities; name a past work
    order only if `query_past_orders` returned it, otherwise state that no matching past work order exists.
    Heading: "Recommended Actions (at the next scheduled maintenance):", then 3-4 numbered checks.
- prompt_inspection:
    Root Cause Analysis: as for scheduled_inspection.
    Heading: "Recommended Actions (as soon as practical):", then 3-4 numbered checks.

# === [VISUAL DIAGNOSIS RULES] ===
- If the user uploads an image of a machine part, analyze it for visible signs of wear,
  misalignment, contamination, or damage.
- If the user uploads a graph or dashboard screenshot, correlate the visual trend with
  the LIVE MACHINE STATUS and HISTORICAL SENSOR DATA provided below.
- When drafting a work order for the conveyor, reference both visual evidence and sensor data.

# === [NATURAL CONVERSATION RULES] ===
- Do NOT use Markdown symbols like '**', '###', or '#' in your final response to the user.
- Use a professional, helpful, conversational tone — like a colleague on the factory floor.
- Keep responses organized with plain text spacing and simple dashes if needed.
- When answering historical questions, cite specific numbers (e.g., "z_rms peaked at X mm/s
  over the last 2 days, with a mean of Y mm/s").

# === [REAL-TIME DATA FRESHNESS] ===
Real-time status: {rt_status}
- If the status contains "NO", politely warn the user that data may be delayed.
- If the status contains "YES", confirm the data is live.

=====================================================================
[LIVE MACHINE STATUS]
  Timestamp         : {ms.get('last_update', 'Unknown')}
  Current Vibration : {ms.get('current_vibration', 'Unknown')}
  ISO 10816 Zone    : {ms.get('iso_10816_status', 'Unknown')}
  Alert Level       : {ms.get('status', 'Unknown')}   (Normal / Low / Medium / High, computed by a fixed rule, not by you)
  Alert Reasons     : {'; '.join(ms.get('alert_reasons', [])) or 'Unknown'}
  Distribution shift per channel (s-IDK² latest score / threshold / flagged):
{_format_shift_status(ms.get('shift_status', {}))}
- Report the alert level and its reasons as given; do not raise or lower the level yourself.
- A distribution shift means the recent data differ from the last 3 days; it does not identify a fault or its cause.
  Data Quality      : {ms.get('data_quality_warning', 'All sensors reporting normally')}

[FORECAST, NEXT 6 HOURS] (LightGBM, recursive, 30-min steps)
{_format_forecast_summary(ms.get('forecast_summary', 'Loading...'))}
- When you cite a forecast, also state the recent forecast error of that channel.
- Treat a forecast change smaller than the recent 6 h error as within forecast uncertainty, not as a trend.

=====================================================================
[HISTORICAL SENSOR DATA]
Use this section to answer ALL questions about past trends, patterns, and anomaly history.
Sensors: temperature (°C), z_rms/x_rms (RMS velocity, mm/s), z_peak/x_peak (peak acceleration, g), noise (dB)

{historical_text}

=====================================================================
[CURRENT WORK ORDER DRAFT]
  Draft Exists   : {bool(draft_text)}
  Draft Content  :
'''{draft_text if draft_text else "None"}'''

=====================================================================
[SESSION CONTEXT]
  Session ID: {ms.get('session_id')}
""")

    _llm_pacer.reserve(LLM_TOKENS_PER_CALL)
    with _llm_gate:
        reply = llm_with_tools.invoke([sys_msg] + list(state["messages"]))
    return {"messages": [reply]}


# ===================== 6. BUILD LANGGRAPH =====================

def tool_error_message(error: Exception) -> str:
    """Fail-safe: a failing tool (e.g. an unreachable vector store) returns this message to the agent instead of
    aborting the chat request, so the agent can tell the operator which source was unavailable."""
    audit.log_event("tool_error", error_type=type(error).__name__, detail=str(error)[:300])
    return (f"TOOL_ERROR: {type(error).__name__}: {error}. This source could not be used. Tell the user that it was "
            "unavailable, do not present any content as if it came from it, and do not claim that the action succeeded.")


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=tool_error_message))

builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    lambda x: "tools" if x["messages"][-1].tool_calls else END,
)
builder.add_edge("tools", "agent")

# Compile without external checkpointing to avoid serialization issues
agent_executor = builder.compile()