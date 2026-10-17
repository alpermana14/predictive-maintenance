"""Retrieval-sufficiency guard for the dual-layer RAG copilot.

Problem it solves (the manuscript's own documented failure, prompts 12 & 13):
when the experiential store (past work orders) does not contain a record
relevant to the query, naive top-k retrieval STILL returns its k nearest
records. The agent then treats that irrelevant content as evidence and, when
it is unhelpful, silently falls back to parametric LLM knowledge -> a
confident, ungrounded answer (hallucination). This is exactly the cold-start
regime of a self-enriching knowledge base, so it is not an edge case: it is
the dominant regime early in deployment.

The guard interposes a sufficiency decision between retrieval and the agent:

  Stage 1 - distance floor (free, deterministic):
     reject if the CLOSEST retrieved record is farther than TAU in embedding
     space. text-embedding-3-small vectors are unit-norm, so L2 distance and
     cosine similarity are monotonically related (L2^2 = 2 - 2*cos); a single
     distance threshold is therefore a cosine-similarity threshold.

  Stage 2 - relevance entailment (optional, one cheap LLM call):
     even a "near" record can be topically off (the paper's case: a load-change
     record retrieved for a bearing-fault query). A yes/no LLM check asks
     whether the retrieved text actually contains information that helps answer
     THIS query. Catches the false-positives Stage 1 cannot.

On insufficiency the guard returns a sentinel the agent is instructed to
surface honestly ("no relevant history exists") instead of answering from
general knowledge. The guard NEVER invents content; it only gates.

All behaviour is configurable via env vars so experiments can toggle/sweep it:
  RAG_GUARD_ENABLED     (default "1")
  RAG_GUARD_TAU         L2 distance floor (default 1.10; ~cos 0.40)
  RAG_GUARD_ENTAILMENT  (default "1"; disable only to trade safety for latency)
  RAG_GUARD_JUDGE_MODEL (default "gpt-4o-mini")

NOTE (E7 finding): the distance floor ALONE is not enough. The failure that
matters — an off-topic query landing on a near record (e.g. a bearing-fault
query retrieving a load-change record at L2 distance 0.99, under the 1.10
floor) — is only caught by the entailment stage. In evaluation, floor-only
left a weak model (gpt-3.5-turbo) hallucinating 1/3 cold-start prompts;
floor+entailment drove that to 0/3. Entailment therefore defaults ON.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

INSUFFICIENT_PREFIX = "INSUFFICIENT_EVIDENCE"


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def guard_enabled() -> bool:
    return _env_flag("RAG_GUARD_ENABLED", "1")


def entailment_enabled() -> bool:
    return _env_flag("RAG_GUARD_ENTAILMENT", "1")


def tau() -> float:
    return float(os.getenv("RAG_GUARD_TAU", "1.10"))


def judge_model() -> str:
    return os.getenv("RAG_GUARD_JUDGE_MODEL", "gpt-4o-mini")


@dataclass
class GuardDecision:
    sufficient: bool
    reason: str
    best_distance: float
    stage: str  # "floor", "entailment", "pass", "disabled", "empty"


def _entailment_ok(query: str, evidence: str, llm=None) -> bool:
    """One yes/no LLM call: does `evidence` help answer `query`?"""
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=judge_model(), temperature=0.0)
    prompt = (
        "You are a strict retrieval-relevance judge for a maintenance copilot.\n"
        "Decide whether the RECORD below contains information that materially "
        "helps answer the QUERY about a specific machine's maintenance history. "
        "A record about a DIFFERENT fault type or topic does NOT count, even if "
        "it is about the same machine.\n\n"
        f"QUERY: {query}\n\nRECORD: {evidence}\n\n"
        "Answer with exactly one word: YES or NO."
    )
    resp = llm.invoke(prompt).content.strip().upper()
    return resp.startswith("Y")


def assess(
    query: str,
    docs_with_scores: List[Tuple[object, float]],
    llm=None,
) -> GuardDecision:
    """Decide whether retrieved (Document, distance) pairs are sufficient.

    Lower distance = more similar (Chroma similarity_search_with_score).
    """
    if not guard_enabled():
        return GuardDecision(True, "guard disabled", float("nan"), "disabled")
    if not docs_with_scores:
        return GuardDecision(False, "no records in store", float("inf"), "empty")

    best_doc, best_dist = min(docs_with_scores, key=lambda ds: ds[1])

    if best_dist > tau():
        return GuardDecision(
            False,
            f"closest record distance {best_dist:.3f} exceeds floor {tau():.3f}",
            best_dist,
            "floor",
        )

    if entailment_enabled():
        content = getattr(best_doc, "page_content", str(best_doc))
        if not _entailment_ok(query, content, llm=llm):
            return GuardDecision(
                False,
                "closest record is near but topically irrelevant (entailment=NO)",
                best_dist,
                "entailment",
            )

    return GuardDecision(True, "sufficient evidence", best_dist, "pass")


def guarded_retrieve(
    vectorstore,
    query: str,
    k: int = 3,
    llm=None,
    record_prefix: str = "record",
) -> Tuple[bool, str, GuardDecision]:
    """Retrieve with sufficiency gating.

    Returns (sufficient, payload, decision). `payload` is either the joined
    record text (sufficient) or an INSUFFICIENT_EVIDENCE sentinel the agent is
    told to surface honestly.
    """
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    decision = assess(query, docs_with_scores, llm=llm)

    if not decision.sufficient:
        sentinel = (
            f"{INSUFFICIENT_PREFIX}: {decision.reason}. "
            "No past work order sufficiently matches this query. Tell the user "
            "that no relevant maintenance history exists for this specific issue "
            "and do NOT answer it from general knowledge or fabricate a record."
        )
        return False, sentinel, decision

    payload = "\n".join(
        f"{record_prefix}: {getattr(d, 'page_content', str(d))}"
        for d, _ in docs_with_scores
    )
    return True, payload, decision
