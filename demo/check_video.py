"""Plays the copilot part of the demo video against the 5 May 2026 replay and compares it with the video.

Sequence of the video (Video1_short, 1:06-1:58):
  chat 1: Q1 "Why is the alert level Low right now?"                  -> answered from the machine state, no tool
          Q2 "Search our past work orders: ..."                       -> gate INSUFFICIENT, the agent abstains
          Q3 "Please draft a work order for the current machine condition."
                                                                      -> manual search, gate INSUFFICIENT, draft (Low)
          the operator edits the Root Cause Analysis and approves     -> record in the store, alert level Normal
  chat 2: Q2 again                                                    -> gate SUFFICIENT, answer from the new record

The LLM (gpt-4o, temperature 0.1) words its answers a little differently from run to run, so the check compares
what must match (tools, gate verdicts, facts, priority, alert level) and stores the full transcript for reading.
Gate distances depend on the search query the agent writes, so they are reported next to the video values.

Usage (from the repository root, with backend/.env holding OPENAI_API_KEY):
  python demo/check_video.py            # writes demo/results/check_<time>.json and .md
"""
import json
import os
import re
import sys
import time

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEMO_DIR)
import replay_may2026 as replay  # noqa: E402  (rebuilds nothing on import; setup() does)
import chat_engine  # noqa: E402
import rag_guard  # noqa: E402

Q1 = "Why is the alert level Low right now?"
Q2 = "Search our past work orders: is there a work order for this vibration and noise change?"
Q3 = "Please draft a work order for the current machine condition."
OPERATOR_RCA = "Rise in vibration and noise after the added load on 5 May 2026, reviewed by the operator."

# What the video shows (frames of Video1_short_final.mp4 and demo/raw.webm)
VIDEO = {
    "Q1": {"tools": [], "facts": ["Low", "3.54", "Zone C", "08:30", "noise", "3.04"]},
    "Q2": {"tools": ["query_past_orders"], "gate": "INSUFFICIENT (floor, distance 1.485)",
           "query": "vibration 3.54 mm/s, noise 70.62 dB, ISO 10816 Zone C, distribution shift",
           "facts": ["No matching past work order", "3.54", "70.62"]},
    "Q3": {"tools": ["retriever_tool", "query_past_orders", "update_work_order"],
           "gate": "INSUFFICIENT (floor, distance 1.298)",
           "query": "vibration unsatisfactory ISO 10816 Zone C", "priority": "Low",
           "manual_query": "vibration unsatisfactory ISO 10816 Zone C possible causes and recommended actions",
           "facts": ["scheduled inspection", "slide rail", "chain tension", "contamination"]},
    "approval": {"alert_after": "Normal", "store_records": 3},
    "Q4": {"tools": ["query_past_orders"], "gate": "SUFFICIENT (pass, distance 0.905)",
           "query": "vibration 3.54 mm/s, noise 70.62 dB, ISO 10816 Zone C, distribution shift in z_rms and noise",
           "facts": ["past work order", "added load on 5 May", "slide rail", "Low"]},
}

GATE_LOG = []
_orig_guarded = rag_guard.guarded_retrieve


def recording_guarded(vectorstore, query, *a, **k):
    sufficient, payload, decision = _orig_guarded(vectorstore, query, *a, **k)
    GATE_LOG.append({"query": query, "sufficient": bool(sufficient), "stage": decision.stage,
                     "distance": round(float(decision.best_distance), 3)})
    return sufficient, payload, decision


rag_guard.guarded_retrieve = recording_guarded


class RecordingExecutor:
    """Keeps the messages of the last agent run (the graph has no checkpointer: every chat message is answered
    on its own, only the draft is kept per session, as in the dashboard)."""

    def __init__(self, inner):
        self.inner, self.last = inner, None

    def invoke(self, inputs, config=None):
        self.last = self.inner.invoke(inputs, config=config)
        return self.last


def ask(m, session, message):
    n_gate = len(GATE_LOG)
    t0 = time.time()
    reply = m.chat_endpoint(m.ChatRequest(message=message, session_id=session))
    msgs = (m.agent_executor.last or {}).get("messages", [])
    calls = [c for msg in msgs for c in (getattr(msg, "tool_calls", None) or [])]
    tools = [c["name"] for c in calls]
    queries = {c["name"]: c["args"].get("query") for c in calls if "query" in c["args"]}
    gates = GATE_LOG[n_gate:]
    return {"message": message, "session": session, "seconds": round(time.time() - t0, 1), "tools": tools,
            "tool_queries": queries, "gate": gates, "response": reply["response"], "draft": reply["draft"]}


def gate_text(g):
    if not g:
        return "-"
    return "; ".join(f"{'SUFFICIENT' if x['sufficient'] else 'INSUFFICIENT'} ({x['stage']}, distance {x['distance']})"
                     for x in g)


def check(step, run, expected):
    rows = []
    ok_tools = run["tools"] == expected["tools"] or (sorted(set(run["tools"])) == sorted(expected["tools"]))
    rows.append(("tools", " -> ".join(expected["tools"]) or "none", " -> ".join(run["tools"]) or "none", ok_tools))
    if "gate" in expected:
        exp_verdict = expected["gate"].split()[0]
        got = gate_text(run["gate"])
        rows.append(("gate", expected["gate"], got, bool(run["gate"]) and got.split()[0] == exp_verdict))
        rows.append(("gate query", expected["query"], "; ".join(x["query"] for x in run["gate"]) or "-", None))
    if "manual_query" in expected:
        rows.append(("manual query", expected["manual_query"], run["tool_queries"].get("retriever_tool", "-"), None))
    text = run["response"] + "\n" + (run["draft"] or "")
    missing = [f for f in expected["facts"] if f.lower() not in text.lower()]
    rows.append(("facts", ", ".join(expected["facts"]), "missing: " + ", ".join(missing) if missing else "all present",
                 not missing))
    if "priority" in expected:
        p = re.search(r"Priority\s*[:\-]?\s*\**\s*(High|Medium|Low)", run["draft"] or "", re.I)
        got = p.group(1).title() if p else None
        rows.append(("draft priority", expected["priority"], got, got == expected["priority"]))
    return rows


def operator_edit(draft):
    """The video's edit: the first sentence of the Root Cause Analysis becomes OPERATOR_RCA."""
    pattern = re.compile(r"(Root Cause Analysis:\**\s*)(.*?\.)(\s)", re.S)
    edited, n = pattern.subn(lambda mm: mm.group(1) + OPERATOR_RCA + mm.group(3), draft, count=1)
    return edited if n else draft


def main():
    m = replay.setup(with_importance=False)
    m.agent_executor = RecordingExecutor(m.agent_executor)
    results, table = {}, []

    results["Q1"] = ask(m, "video_chat_1", Q1)
    results["Q2"] = ask(m, "video_chat_1", Q2)
    results["Q3"] = ask(m, "video_chat_1", Q3)
    for step in ("Q1", "Q2", "Q3"):
        table += [(step, *r) for r in check(step, results[step], VIDEO[step])]

    draft = chat_engine.DRAFT_STORE.get("video_chat_1", "")
    edited = operator_edit(draft)
    approval = m.approve_work_order(m.ApprovalRequest(session_id="video_chat_1", content=edited), None)
    alert_after = m.current_alert()["level"]
    store = m.list_work_orders()["items"]
    results["approval"] = {"edited_text": edited, "edit_applied": edited != draft, "result": approval,
                           "alert_after": alert_after, "work_orders": [(w["id"], w["created_at"]) for w in store]}
    table.append(("approval", "operator edit", "Root Cause Analysis replaced", "applied" if edited != draft else "NOT applied",
                  edited != draft))
    table.append(("approval", "alert level after approval", VIDEO["approval"]["alert_after"], alert_after,
                  alert_after == VIDEO["approval"]["alert_after"]))
    table.append(("approval", "records in the store", str(VIDEO["approval"]["store_records"]), str(len(store)),
                  len(store) == VIDEO["approval"]["store_records"]))

    results["Q4"] = ask(m, "video_chat_2", Q2)
    table += [("Q4", *r) for r in check("Q4", results["Q4"], VIDEO["Q4"])]

    out_dir = os.path.join(DEMO_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(out_dir, f"check_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "table": table}, f, indent=2, default=str, ensure_ascii=False)
    lines = [f"# Video check {stamp}", "", "| Step | Item | Video | This run | Match |", "|---|---|---|---|---|"]
    for step, item, exp, got, ok in table:
        mark = "" if ok is None else ("yes" if ok else "**NO**")
        lines.append(f"| {step} | {item} | {exp} | {str(got).replace(chr(10), ' ')} | {mark} |")
    for step in ("Q1", "Q2", "Q3", "Q4"):
        r = results[step]
        lines += ["", f"## {step} ({r['session']}): {r['message']}", "", r["response"]]
        if step == "Q3":
            lines += ["", "### Draft", "", "```", r["draft"], "```"]
    lines += ["", "## Approved text (after the operator edit)", "", "```", edited, "```"]
    with open(os.path.join(out_dir, f"check_{stamp}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    passed = sum(1 for *_, ok in table if ok)
    total = sum(1 for *_, ok in table if ok is not None)
    print("\n".join(lines[:len(table) + 4]))
    print(f"\n[CHECK] {passed}/{total} checks match the video; transcript: demo/results/check_{stamp}.md")


if __name__ == "__main__":
    main()
