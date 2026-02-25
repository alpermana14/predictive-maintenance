import os
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
MANUAL_DIR = os.path.join(BASE_DIR, "maintenance_manual_db")
PDF_PATH = os.path.join(BASE_DIR, "Maintenance_Conveyor.pdf")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# ===================== 2. VECTOR STORES (RAG) =====================
# Initialize Manual Retriever
if os.path.exists(PDF_PATH):
    # Only process if DB doesn't exist to save time
    if not os.path.exists(MANUAL_DIR):
        print("[INFO] Ingesting manual PDF from:", PDF_PATH)
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        vectorstore_manual = Chroma.from_documents(splits, embeddings, persist_directory=MANUAL_DIR)
    else:
        vectorstore_manual = Chroma(persist_directory=MANUAL_DIR, embedding_function=embeddings)
    
    manual_retriever = vectorstore_manual.as_retriever(search_kwargs={"k": 2})
else:
    print(f"[WARN] Manual PDF not found at path: {PDF_PATH}")
    manual_retriever = None

# Initialize History Retriever
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)
vectorstore_history = Chroma(persist_directory=HISTORY_DIR, embedding_function=embeddings)


# ===================== 3. TOOLS =====================
@tool
def retriever_tool(query: str) -> str:
    """Search the machine manual for technical specs and procedures."""
    if not manual_retriever: return "Manual not found."
    docs = manual_retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])

@tool
def query_past_orders(query: str) -> str:
    """Search past maintenance records for similar issues."""
    docs = vectorstore_history.similarity_search(query, k=3)
    if not docs: return "No relevant past records found."
    return "\n".join([f"record: {d.page_content}" for d in docs])

@tool
def update_work_order(content: str, session_id: str) -> str:
    """Update the current work order draft text. Always pass the session_id."""
    DRAFT_STORE[session_id] = content
    return "Draft updated. The user can see the preview."

# @tool
# def finalize_work_order(filename_id: str, session_id: str) -> str:
#     """
#     Save the current draft as a permanent work order and clear the draft.

#     NOTE:
#     - The caller may suggest a filename_id, but the system will normalize it to
#       a single canonical ID per day to avoid duplicates.
#     """
#     content = DRAFT_STORE.get(session_id, "")
#     if not content:
#         return "Error: Draft is empty."

#     # Sanitize content for UI display:
#     # - Remove '*' so markdown-style formatting doesn't leak into the UI.
#     # - Strip leading/trailing whitespace.
#     content_clean = content.replace("*", "").strip()

#     # Canonical work order ID: one per day (UTC)
#     today_str = datetime.utcnow().strftime("%Y_%m_%d")
#     canonical_id = f"work_order_{today_str}"

#     # If a work order already exists for today, do NOT create a new one.
#     # Instead, instruct the technician to reference the existing record.
#     try:
#         existing = vectorstore_history._collection.get(  # type: ignore[attr-defined]
#             where={"id": canonical_id}
#         )
#         if existing and existing.get("documents"):
#             return (
#                 f"A work order for this issue already exists with ID '{canonical_id}'. "
#                 "Please refer to that existing work order and proceed with inspection, "
#                 "instead of creating a new one."
#             )
#     except Exception:
#         # If the lookup fails, we fall back to creating the record below.
#         pass


    # # Save to Vector DB
    # doc = Document(
    #     page_content=content_clean,
    #     metadata={
    #         "id": canonical_id,
    #         "created_at": datetime.utcnow().isoformat(),
    #         "session_id": session_id,
    #         "original_name": filename_id,
    #     },
    # )
    # vectorstore_history.add_documents([doc])

    # # Clear Draft
    # DRAFT_STORE[session_id] = ""
    # return f"Work Order {canonical_id} saved and indexed."

tools = [retriever_tool, query_past_orders, update_work_order] # finalize removed!
# tools = [retriever_tool, query_past_orders, update_work_order, finalize_work_order]
llm_with_tools = llm.bind_tools(tools)

# ===================== 4. GRAPH DEFINITION =====================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    machine_state: dict # Live data passed from FastAPI

# def agent_node(state: AgentState):
    # Construct Dynamic System Prompt
    # ms = state['machine_state']
    
    # sys_msg = SystemMessage(content=f"""
    # You are a Predictive Maintenance Copilot.
    
    # [LIVE MACHINE STATUS]
    # - Last Update: {ms.get('last_update', 'Unknown')}
    # - Status: {ms.get('status', 'Unknown')}
    # - Anomaly Detected: {ms.get('is_anomaly', False)}
    # - Forecast Trend: {str(ms.get('forecast_summary', 'N/A'))}
    
    # Capabilities:
    # 1. Diagnose issues using live data.
    # 2. Search manuals (retriever_tool).
    # 3. Check history (query_past_orders).
    # 4. Write work orders (update_work_order).
    
    # IMPORTANT:
    # - The backend has already provided a 'session_id' field in machine_state.
    # - When calling update_work_order or finalize_work_order, ALWAYS use:
    #     session_id = machine_state['session_id']
    # - Do NOT ask the user to provide or repeat the session_id.
    # - When finalizing a work order, the system will handle the ID. You should
    #   focus on capturing clear instructions and the ASSIGNED TECHNICIAN NAME.
    # - Do NOT create separate "anomaly_report_*" records; use a single work
    #   order that already summarizes the anomaly and the actions.
    # """)
    
    # return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
def agent_node(state: AgentState):
    # Construct Dynamic System Prompt
    ms = state['machine_state']
    draft_text = ms.get('current_draft_text', '')
    rt_status = ms.get('realtime_status_msg', 'Unknown')
    # Create a dynamic warning message
    
    sys_msg = SystemMessage(content=f"""
     You are an advanced Multimodal Predictive Maintenance Copilot. 
    YOU HAVE VISION CAPABILITIES. You CAN view photos and images. 
    Do NOT ever state that you cannot view images or photos. When the user provides an image, you MUST actively analyze its contents.
    
    # === [DOMAIN RECOGNITION & ROUTING RULES] ===
    You monitor a specific LIVE CONVEYOR SYSTEM. However, the user may ask you about OTHER general machinery.
    1. **Live Conveyor Queries:** If the user asks about "the machine", "the conveyor", "current status", or uploads an image clearly related to the monitored conveyor:
       - You MUST reference the [LIVE MACHINE STATUS] provided below to correlate visual symptoms with real-time data.
    2. **General Machinery Queries:** If the user asks general engineering questions or uploads an image of unrelated equipment (e.g., a pump, CNC spindle, broken pipe, general tool):
       - Answer based on your broad industrial and engineering knowledge.
       - DO NOT append or reference the live conveyor's vibration or IDK data, as it is completely irrelevant to other machines.

    # !!! CRITICAL PROTOCOL FOR WORK ORDERS !!!
    # READ THIS CAREFULLY. DO NOT HALLUCINATE.
    
    1. **TRIGGER:** If the user asks to "Draft", "Create", "Write", or "Update" a work order...
    2. **ACTION:** You MUST call the tool `update_work_order` IMMEDIATELY.
    3. **FORBIDDEN:** You are FORBIDDEN from outputting the text "I have created a draft" or "I have updated the draft" UNLESS you have actually called the tool.
    4. **VERIFICATION:** If you do not see the tool output in your history, you have failed. Try again.

    # === [VISUAL DIAGNOSIS RULES] ===
    - If the user uploads an image of a machine part (like a conveyor belt or motor), analyze it for visible signs of wear, misalignment, or damage.
    - If the user uploads a graph or dashboard screenshot, correlate the visual trend with the LIVE MACHINE STATUS provided below.
    - If drafting a work order FOR THE CONVEYOR, always reference both the visual evidence and the real-time sensor data (Vibration: {ms.get('current_vibration', 'Unknown')}, Status: {ms.get('status', 'Unknown')}) when drafting a work order.
    
    # === [PAYLOAD CONTENT] ===
    When calling `update_work_order`, the 'content' argument MUST include
    - **Incident Report:** - Timestamp: {ms.get('last_update')}
      - Vibration: {ms.get('current_vibration')}
      - ISO Zone: {ms.get('iso_10816_status')}
      - IDK Anomaly: {ms.get('status')}
    - **Recommended Actions:** 3-4 numbered technical checks.
    - **Priority:** High/Medium/Low.
    
    # === [NATURAL CONVERSATION RULES] ===
    - DO NOT use Markdown symbols like '**', '###', or '#' in your final response to the user.
    - Use a professional, helpful, and conversational tone.
    - Keep information organized with plain text spacing and simple dashes if needed.
    - Treat the user like a colleague on the factory floor.
    
    1. **Real-Time Check:** - If {rt_status} contains "NO", warn the user politely about the delay.
       - If {rt_status} contains "YES", confirm the data is live.
       
    2. **Status Explanation:**
       - Explain the ISO 10816 Zone (Rule-based).
       - Explain the IDK Algorithm (AI-based anomaly detection on raw data).
    
        
    [CURRENT DRAFT CONTEXT]
    - Does Draft Exist? {bool(draft_text)}
    - Current Draft Content: 
    '''{draft_text if draft_text else "None"}'''

    
    [CURRENT CONTEXT]
    Session ID: {ms.get('session_id')}
    """)
    
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
# Build Graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", lambda x: "tools" if x['messages'][-1].tool_calls else END)
builder.add_edge("tools", "agent")

# Compile agent without external checkpointing to avoid serialization issues
agent_executor = builder.compile()