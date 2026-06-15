"""
server.py
─────────
Stage 2 FastMCP Server — streamable-http transport on port 8000.

Exposes
  Tool     → reflection_tool           : 2-stage Critique + Correction via MCP Sampling.
                                          No LLM initialised on the server; all completions
                                          are delegated back to the client via Sampling.
  Resource → knowledge://domain/{query}: Hierarchical CRAG resource:
                                          multi-query expansion (Sampling) →
                                          hierarchical retrieval →
                                          3-branch ToT grading (LLM via Sampling) →
                                          optional Tavily fallback.

Run
  uv run --package mcp-server mcp-server
  # or directly:
  python -m mcp_server.server
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime
from typing import Any

from fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from .knowledge_base import search_knowledge_base

# ── Server-side logging ────────────────────────────────────────────────────────
# stdout + forwarded to client via ctx.info() / ctx.debug() (dual-stream pattern)
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [SERVER] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── FastMCP initialisation ─────────────────────────────────────────────────────
mcp = FastMCP(
    name="ThinkingAgentServer",
    instructions=(
        "Production MCP server for Stage 2. "
        "Call reflection_tool to critique + correct a draft answer via MCP Sampling. "
        "Read knowledge://domain/{query} for hierarchical CRAG domain knowledge."
    ),
)


# ── Health endpoint ────────────────────────────────────────────────────────────
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Container/orchestration readiness probe — also logs the connection attempt."""
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Connection attempt | endpoint=/health | client_ip={client_ip}")
    return PlainTextResponse("OK")


# ─────────────────────────────────────────────────────────────────────────────
#  TOOL: reflection_tool
#  MCP Sampling: server delegates BOTH LLM calls back to the client's model.
#  The server holds NO API keys and instantiates NO LLM.
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def reflection_tool(tool_input: str, ctx: Context) -> str:
    """
    Two-stage reflection via MCP Sampling — the server holds NO LLM or API keys.

    Stage 1 — Critique : Identifies hallucinations, gaps, and unsupported claims
                         by delegating to the client's LLM via MCP Sampling.
    Stage 2 — Correction: Rewrites the draft fixing every issue found.
                          Delegated to the client via MCP Sampling.
                          Skipped when is_sufficient=true.

    Action Input (JSON string):
        {"draft_answer": "<current best answer>", "original_query": "<user question>"}

    Returns:
        JSON string — { critique, is_sufficient, corrected_answer }
    """
    ts = _ts()
    logger.info(f"reflection_tool invoked at {ts}")
    await ctx.info(f"[{ts}] reflection_tool invoked — starting 2-stage critique loop")

    # ── Parse single-string input (ReAct text-format passes one string) ───────
    try:
        data           = json.loads(_strip(tool_input))
        draft_answer   = data.get("draft_answer", "")
        original_query = data.get("original_query", "")
    except (json.JSONDecodeError, AttributeError):
        draft_answer   = tool_input
        original_query = ""

    if not draft_answer:
        return json.dumps({"error": "draft_answer is required"})

    # ── Stage 1: Critique via MCP Sampling ────────────────────────────────────
    logger.debug("Stage 1 — sending critique sampling request to client")
    await ctx.debug("Stage 1 — dispatching Critique sampling request to client LLM")

    critique_system = (
        "You are a strict Critic AI auditing a draft answer. "
        "Return ONLY a raw JSON object with exactly these keys: "
        '{"critique": "<detailed findings>", "is_sufficient": <true|false>}. '
        "No markdown fences. No preamble."
    )
    critique_prompt = (
        f"ORIGINAL QUERY:\n{original_query}\n\nDRAFT ANSWER:\n{draft_answer}"
    )

    # MCP spec: content block must be structured as TextContent object
    critique_result = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=critique_prompt),
            )
        ],
        system_prompt=critique_system,
        max_tokens=512,
    )
    critique_raw = (
        critique_result.content.text
        if hasattr(critique_result.content, "text")
        else str(critique_result.content)
    )

    try:
        cp            = json.loads(_strip(critique_raw))
        critique_text = cp.get("critique", critique_raw)
        is_sufficient = bool(cp.get("is_sufficient", False))
    except Exception:
        critique_text = critique_raw
        is_sufficient = False

    logger.info(f"Stage 1 complete | is_sufficient={is_sufficient}")
    await ctx.info(f"Stage 1 complete | is_sufficient={is_sufficient}")

    # ── Stage 2: Correction via MCP Sampling ──────────────────────────────────
    if is_sufficient:
        corrected_answer = draft_answer
        logger.info("Stage 2 skipped — draft already sufficient")
        await ctx.info("Stage 2 skipped — critic marked draft as sufficient")
    else:
        logger.debug("Stage 2 — sending correction sampling request to client")
        await ctx.debug("Stage 2 — dispatching Correction sampling request to client LLM")

        correction_system = (
            "You are a Correction AI. Fix every issue identified in the critique. "
            "Return ONLY a raw JSON object: "
            '{"corrected_answer": "<improved fact-grounded answer>"}. '
            "No markdown fences. No preamble."
        )
        correction_prompt = (
            f"ORIGINAL QUERY:\n{original_query}\n\n"
            f"DRAFT ANSWER:\n{draft_answer}\n\n"
            f"CRITIQUE:\n{critique_text}"
        )

        correction_result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=correction_prompt),
                )
            ],
            system_prompt=correction_system,
            max_tokens=512,
        )
        correction_raw = (
            correction_result.content.text
            if hasattr(correction_result.content, "text")
            else str(correction_result.content)
        )

        try:
            corrected_answer = json.loads(_strip(correction_raw)).get(
                "corrected_answer", correction_raw
            )
        except Exception:
            corrected_answer = correction_raw

        logger.info("Stage 2 correction complete")
        await ctx.info("Stage 2 correction complete — refined answer assembled")

    return json.dumps(
        {
            "critique":         critique_text,
            "is_sufficient":    is_sufficient,
            "corrected_answer": corrected_answer,
        },
        indent=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  RESOURCE: knowledge://domain/{query}
#  CRAG pipeline:
#    (1) Multi-query expansion  → MCP Sampling (delegated to client LLM)
#    (2) Hierarchical retrieval → keyword scoring across L1→L2→L3 chunks
#    (3) ToT evaluation         → LLM 3-branch reasoning via MCP Sampling
#    (4) Tavily fallback        → triggered when avg ToT score < 0.6
# ─────────────────────────────────────────────────────────────────────────────
@mcp.resource("knowledge://domain/{query}")
async def crag_knowledge_resource(query: str, ctx: Context) -> str:
    """
    Hierarchical CRAG resource — enterprise AI/ML knowledge base.

    Pipeline
    ────────
    1. Multi-Query Expansion  : Expands the raw query into 3 semantic variants
                                via MCP Sampling (delegated to client's LLM).
    2. Hierarchical Retrieval : Searches L1 topic summaries → L2/L3 chunks
                                using two-level keyword overlap scoring.
    3. ToT Evaluation         : 3-branch Tree-of-Thought grading performed by
                                the client LLM via MCP Sampling — Specificity,
                                Completeness, Novelty rated per branch then
                                voted (server holds no LLM).
    4. Tavily Fallback        : Triggered when avg ToT score < 0.6 or
                                no internal chunks found.

    URI: knowledge://domain/{query}
    """
    ts = _ts()
    logger.info(f"CRAG resource queried | query='{query}' | {ts}")
    await ctx.info(f"[{ts}] CRAG resource invoked | query='{query}'")

    # ── Step 1: Multi-Query Expansion via MCP Sampling ────────────────────────
    logger.debug("CRAG Step 1: Multi-query expansion")
    await ctx.debug("CRAG Step 1 — dispatching Multi-Query Expansion to client LLM")

    expansion_system = (
        "Generate exactly 3 semantically distinct search query variants for retrieval. "
        "Return ONLY a raw JSON array of 3 strings. No markdown. No preamble."
    )
    expansion_prompt = (
        f"Original query: '{query}'\n"
        "Generate 3 variants approaching the topic from different angles."
    )

    try:
        expansion_result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=expansion_prompt),
                )
            ],
            system_prompt=expansion_system,
            max_tokens=200,
        )
        expansion_raw    = (
            expansion_result.content.text
            if hasattr(expansion_result.content, "text") else "[]"
        )
        expanded_queries = json.loads(_strip(expansion_raw))
        if not isinstance(expanded_queries, list):
            raise ValueError("not a list")
        expanded_queries = [query] + [str(q) for q in expanded_queries[:3]]
    except Exception as exc:
        logger.warning(f"Query expansion failed ({exc}) — using original only")
        expanded_queries = [query]

    await ctx.info(f"Multi-query expansion: {len(expanded_queries)} queries ready")

    # ── Step 2: Hierarchical Retrieval ────────────────────────────────────────
    logger.debug(f"CRAG Step 2: Hierarchical retrieval for {len(expanded_queries)} queries")
    await ctx.debug("CRAG Step 2 — Hierarchical Indexing retrieval started")

    retrieved_chunks: list[dict] = []
    seen_sigs: set[str]          = set()

    for q in expanded_queries:
        for chunk in search_knowledge_base(q, top_k=3):
            sig = chunk["content"][:80]
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                retrieved_chunks.append(chunk)

    logger.info(f"Hierarchical retrieval: {len(retrieved_chunks)} unique chunks found")
    await ctx.info(f"Hierarchical retrieval: {len(retrieved_chunks)} unique chunks found")

    # ── Step 3: Tree-of-Thought Evaluation (LLM reasoning via MCP Sampling) ──
    # ToT is an LLM reasoning paradigm: the client's LLM explores multiple
    # independent reasoning branches over the same evidence, then deliberates
    # and votes to converge on a grade. The server holds no LLM — every branch
    # is sampled from the client's model. Three orthogonal dimensions are
    # graded (specificity / completeness / novelty), each measuring a distinct
    # property so lexical overlap is never scored twice.
    logger.debug("CRAG Step 3: Tree-of-Thought evaluation (LLM via sampling)")
    await ctx.debug("CRAG Step 3 — Initiating ToT Evaluation on retrieved chunks")

    tot_scores = await _run_tot_evaluation(query, retrieved_chunks, ctx)
    avg_score  = sum(tot_scores.values()) / max(len(tot_scores), 1)

    await ctx.info(
        f"ToT scores — Specificity:{tot_scores['specificity']:.2f} "
        f"Completeness:{tot_scores['completeness']:.2f} "
        f"Novelty:{tot_scores['novelty']:.2f} | avg={avg_score:.2f}"
    )
    logger.info(f"ToT evaluation complete | avg={avg_score:.2f} | scores={tot_scores}")

    # ── Step 4: Tavily Fallback ───────────────────────────────────────────────
    use_fallback     = avg_score < 0.6 or len(retrieved_chunks) == 0
    fallback_context = ""

    if use_fallback:
        logger.info("Tavily fallback triggered — internal context below threshold")
        await ctx.info("Tavily fallback triggered — avg ToT score below 0.6")
        fallback_context = await _tavily_fallback(query)
        await ctx.info("Tavily fallback complete — web context appended")

    # ── Assemble resource payload ──────────────────────────────────────────────
    internal_context = "\n\n".join(
        f"[{c['topic'].upper()} — {c['level']}]\n{c['content']}"
        for c in retrieved_chunks[:5]
    )
    combined = internal_context
    if fallback_context:
        combined += "\n\n--- WEB FALLBACK RESULTS ---\n" + fallback_context

    payload: dict[str, Any] = {
        "query":            query,
        "expanded_queries": expanded_queries,
        "tot_scores":       tot_scores,
        "avg_tot_score":    round(avg_score, 3),
        "fallback_used":    use_fallback,
        "internal_context": internal_context,
        "fallback_context": fallback_context,
        "combined_context": combined,
    }

    logger.info(f"CRAG resource assembled | fallback_used={use_fallback}")
    await ctx.info(f"CRAG resource complete | fallback_used={use_fallback}")
    return json.dumps(payload, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _strip(raw: str) -> str:
    """Remove markdown code fences Llama models may emit."""
    return (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _clamp01(x: float) -> float:
    """Clamp a value into the [0.0, 1.0] grading range."""
    return max(0.0, min(1.0, x))


async def _run_tot_evaluation(
    query: str,
    chunks: list[dict],
    ctx: Context,
    n_branches: int = 3,
) -> dict[str, float]:
    """
    Tree-of-Thought relevance grading of the retrieved context.

    ToT is an LLM reasoning paradigm, so the reasoning is performed by the
    client's LLM via MCP Sampling — the server holds no LLM or API keys. The
    model explores ``n_branches`` INDEPENDENT reasoning paths over the same
    evidence, scores each branch, then DELIBERATES and VOTES to converge on a
    final grade (self-consistency over the thought tree).

    The three dimensions are deliberately ORTHOGONAL so each measures a
    distinct property — they do not all reduce to lexical overlap:
      • Specificity  — how *precisely* the context targets the query intent.
      • Completeness — whether *all* sub-aspects of the query are covered.
      • Novelty      — how information-rich / non-generic the context is.

    Falls back to a deterministic lexical heuristic only when sampling is
    unavailable (e.g. the client LLM errors). Returns scores in [0.0, 1.0].
    """
    if not chunks:
        logger.warning("ToT evaluation: no chunks — returning zeros")
        return {"specificity": 0.0, "completeness": 0.0, "novelty": 0.0}

    # Compact the evidence so the sampling prompt stays within the TPM budget.
    evidence = "\n\n".join(
        f"[CHUNK {i + 1} | {c.get('topic', '?')} | {c.get('level', '?')}]\n"
        f"{c['content'][:500]}"
        for i, c in enumerate(chunks[:5])
    )

    tot_system = (
        "You are a retrieval-grading reasoner using the Tree-of-Thought method. "
        f"Explore EXACTLY {n_branches} INDEPENDENT reasoning branches that each "
        "judge how well the RETRIEVED CONTEXT answers the QUERY. For every branch "
        "rate three ORTHOGONAL dimensions in [0,1] — judge MEANING, not word overlap:\n"
        "  - specificity: how precisely and directly the context targets the exact "
        "intent of the query (penalise generic or tangential text);\n"
        "  - completeness: whether ALL sub-aspects needed to fully answer the query "
        "are covered by the context (penalise partial coverage);\n"
        "  - novelty: how information-rich and non-obvious the context is (penalise "
        "content that merely restates the query or is boilerplate).\n"
        "Then DELIBERATE across the branches and VOTE to converge on a final score "
        "per dimension. Return ONLY raw JSON with this exact shape: "
        '{"branches":[{"thought":"<one line>","specificity":<num>,'
        '"completeness":<num>,"novelty":<num>}],'
        '"final":{"specificity":<num>,"completeness":<num>,"novelty":<num>}}. '
        "No markdown fences. No preamble."
    )
    tot_prompt = f"QUERY:\n{query}\n\nRETRIEVED CONTEXT:\n{evidence}"

    try:
        result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=tot_prompt),
                )
            ],
            system_prompt=tot_system,
            max_tokens=600,
        )
        raw = (
            result.content.text
            if hasattr(result.content, "text")
            else str(result.content)
        )
        parsed   = json.loads(_strip(raw))
        branches = parsed.get("branches", []) or []
        final    = parsed.get("final") or {}

        def _vote(dim: str) -> float:
            """Average a dimension across the explored branches (the ToT vote)."""
            vals = [
                float(b[dim])
                for b in branches
                if isinstance(b, dict) and isinstance(b.get(dim), (int, float))
            ]
            return sum(vals) / len(vals) if vals else 0.0

        scores: dict[str, float] = {}
        for dim in ("specificity", "completeness", "novelty"):
            # Prefer the model's own deliberated `final`; otherwise vote ourselves.
            v = final.get(dim)
            scores[dim] = round(
                _clamp01(float(v)) if isinstance(v, (int, float)) else _vote(dim), 3
            )

        await ctx.debug(
            f"ToT explored {len(branches)} reasoning branches → voted {scores}"
        )
        logger.info(f"ToT (LLM/sampling) | branches={len(branches)} | scores={scores}")
        return scores

    except Exception as exc:
        logger.warning(f"ToT sampling failed ({exc}) — deterministic fallback")
        await ctx.debug(f"ToT sampling unavailable ({exc}); using deterministic fallback")
        return _heuristic_relevance_fallback(query, chunks)


def _heuristic_relevance_fallback(query: str, chunks: list[dict]) -> dict[str, float]:
    """
    Deterministic relevance grade used ONLY when LLM sampling is unavailable.

    This is a lexical safety net, NOT Tree-of-Thought. Each dimension is
    computed from a STRUCTURALLY DISTINCT statistic so the same overlap is
    never scored twice:
      • Specificity  — mean per-chunk precision: how focused each chunk is on
                       the query (|q ∩ chunk| / |chunk|).
      • Completeness — set-level coverage of the query across the UNION of all
                       chunks (|q ∩ ⋃chunks| / |q|) — recall, not per-chunk.
      • Novelty      — IDF-weighted overlap rewarding rare, informative terms.
    """
    query_terms = _tokenize(query)
    all_docs    = [_tokenize(c["content"]) for c in chunks]
    if not query_terms or not all_docs:
        return {"specificity": 0.0, "completeness": 0.0, "novelty": 0.0}
    N = len(all_docs)

    # Specificity — mean per-chunk precision (focus of each chunk on the query).
    precisions  = [len(query_terms & ct) / len(ct) if ct else 0.0 for ct in all_docs]
    specificity = _clamp01(sum(precisions) / len(precisions) * 5)  # scale sparse hits

    # Completeness — collective query coverage across the UNION of chunks (recall).
    union_terms  = set().union(*all_docs)
    completeness = len(query_terms & union_terms) / len(query_terms)

    # Novelty — IDF-weighted overlap rewarding rare, informative shared terms.
    novelty_scores = []
    for ct in all_docs:
        overlap = query_terms & ct
        if not overlap:
            novelty_scores.append(0.0)
            continue
        idf_sum = sum(
            math.log((N + 1) / (sum(1 for d in all_docs if t in d) + 1)) for t in overlap
        )
        novelty_scores.append(min(idf_sum / (len(overlap) * 3), 1.0))
    novelty = sum(novelty_scores) / len(novelty_scores)

    return {
        "specificity":  round(specificity, 3),
        "completeness": round(completeness, 3),
        "novelty":      round(novelty, 3),
    }


async def _tavily_fallback(query: str) -> str:
    """Web search via Tavily — triggered when internal knowledge scores below threshold."""
    try:
        from tavily import TavilyClient  # type: ignore

        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            logger.warning("TAVILY_API_KEY not set — fallback skipped")
            return "[Tavily unavailable: TAVILY_API_KEY not set on server]"

        client   = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=3, search_depth="advanced")
        results  = response.get("results", [])

        if not results:
            return "[Tavily: no results returned]"

        return "\n\n".join(
            f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')[:400]}"
            for r in results
        )
    except Exception as exc:
        logger.error(f"Tavily fallback error: {exc}")
        return f"[Tavily fallback error: {exc}]"


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """uv script entry point — starts streamable-http server on port 8000."""
    logger.info("=" * 60)
    logger.info("ThinkingAgentServer starting up")
    logger.info("Transport  : streamable-http")
    logger.info("Endpoint   : http://0.0.0.0:8000/mcp")
    logger.info("Tools      : reflection_tool")
    logger.info("Resources  : knowledge://domain/{query}")
    logger.info("Awaiting client connections...")
    logger.info("=" * 60)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
