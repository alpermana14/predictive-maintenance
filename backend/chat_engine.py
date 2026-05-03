import os
import json
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
llm        = ChatOpenAI(model="gpt-4o", temperature=0.1)

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
    docs = vectorstore_history.similarity_search(query, k=3)
    if not docs:
        return "No relevant past records found."
    return "\n".join([f"record: {d.page_content}" for d in docs])


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
        "current":     "A",
        "temperature": "°C",
        "z_rms":       "mm/s",
        "x_rms":       "mm/s",
        "z_peak":      "mm/s",
        "x_peak":      "mm/s",
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

# !!! CRITICAL PROTOCOL FOR WORK ORDERS !!!
1. TRIGGER: If the user asks to "Draft", "Create", "Write", or "Update" a work order...
2. ACTION: You MUST call the tool `update_work_order` IMMEDIATELY.
3. FORBIDDEN: You are FORBIDDEN from saying "I have created a draft" UNLESS you have actually called the tool.
4. VERIFICATION: If you do not see the tool output in your message history, you have failed. Try again.

# === [PAYLOAD CONTENT FOR WORK ORDERS] ===
When calling `update_work_order`, the content argument MUST include:
  - Incident Report:
      Timestamp  : {ms.get('last_update')}
      Vibration  : {ms.get('current_vibration')}
      ISO Zone   : {ms.get('iso_10816_status')}
      IDK Status : {ms.get('status')}
  - Root Cause Analysis (based on sensor trends and manual retrieval if needed)
  - Recommended Actions: 3-4 numbered technical checks
  - Priority: High / Medium / Low

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
  IDK Anomaly Status: {ms.get('status', 'Unknown')}
  Data Quality      : {ms.get('data_quality_warning', 'All sensors reporting normally')}
  Forecast (next 6h): {json.dumps(ms.get('forecast_summary', 'Loading...'), indent=2)}

=====================================================================
[HISTORICAL SENSOR DATA]
Use this section to answer ALL questions about past trends, patterns, and anomaly history.
Sensors: current (A), temperature (°C), z_rms/x_rms/z_peak/x_peak (mm/s), noise (dB)

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

    return {"messages": [llm_with_tools.invoke([sys_msg] + list(state["messages"]))]}


# ===================== 6. BUILD LANGGRAPH =====================

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    lambda x: "tools" if x["messages"][-1].tool_calls else END,
)
builder.add_edge("tools", "agent")

# Compile without external checkpointing to avoid serialization issues
agent_executor = builder.compile()