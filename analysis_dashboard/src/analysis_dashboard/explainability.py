"""
explainability.py
─────────────────
Stage 4 — Local Explainability Audit Engine (proxy SHAP + proxy LIME).

Given a targeted runtime failure mode (an error trace in the vector store), this
engine builds a local, post-hoc explanation of *why* the system trended toward
that anomaly, combining three signals:

  1. Graph-relational context  — the adjacent Neo4j/log subgraph (graph_context).
  2. Proxy SHAP (structured)   — exact Shapley values over a surrogate model on
     numeric execution features (payload length, latency, token volume, call
     frequency): marginal feature contributions across *all* feature subsets,
     identifying which metric most skewed the system toward the anomaly.
  3. Proxy LIME (unstructured) — perturbs the log text by systematically masking
     token subsets and regresses the change in the surrogate's error-confidence
     on token presence, yielding localized token-importance coefficients.

Surrogate models are fit with scikit-learn when both classes are present;
otherwise a deterministic heuristic scorer is used so audits always run. The
proxy algorithms themselves are implemented from scratch on numpy (the brief
asks for *proxy* SHAP/LIME), with shap/lime/scikit-learn/numpy all installed.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from .graph_context import entries_for_session, fetch_graph_context, list_sessions  # noqa: E402
from .logger import analysis_log  # noqa: E402
from .store_reader import LOG_ROOT, get_reader  # noqa: E402

sns.set_theme(style="whitegrid")

FEATURE_NAMES = ["payload_length", "latency_ms", "token_estimate", "call_frequency"]
AUDIT_REPORT_PATH = os.environ.get("AUDIT_REPORT_PATH", "explainability_audit_report.json")
CHART_DIR = os.environ.get("CHART_DIR", "charts")

_ERROR_TERMS = {
    "error", "errors", "fail", "failed", "failure", "exception", "injected",
    "malformed", "invalid", "timeout", "timed", "refused", "traceback", "429",
    "fallback", "catastrophic", "degraded", "unrecoverable", "fault", "drop",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _both_classes(y: np.ndarray) -> bool:
    return y.size > 0 and float(y.min()) != float(y.max())


# ─────────────────────────────────────────────────────────────────────────────
#  Surrogate "black-box" models (sklearn when trainable, else heuristic)
# ─────────────────────────────────────────────────────────────────────────────
class StructuredSurrogate:
    """Predicts P(error) from a numeric feature vector."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.baseline = X.mean(axis=0) if X.size else np.zeros(len(FEATURE_NAMES))
        self.mu = X.mean(axis=0) if X.size else np.zeros(len(FEATURE_NAMES))
        self.sigma = (X.std(axis=0) + 1e-9) if X.size else np.ones(len(FEATURE_NAMES))
        self.mode = "heuristic"
        if _both_classes(y) and len(y) >= 6:
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler

                self.scaler = StandardScaler().fit(X)
                self.clf = LogisticRegression(max_iter=1000).fit(self.scaler.transform(X), y)
                self.mode = "sklearn_logreg"
            except Exception as exc:  # pragma: no cover
                analysis_log.warning(f"structured surrogate training failed ({exc}) — heuristic")

    def proba(self, x: np.ndarray) -> float:
        if self.mode == "sklearn_logreg":
            try:
                return float(self.clf.predict_proba(self.scaler.transform(x.reshape(1, -1)))[0, 1])
            except Exception:
                pass
        z = (x - self.mu) / self.sigma
        return float(1.0 / (1.0 + math.exp(-float(z.sum()))))


class TextSurrogate:
    """Predicts P(error) from log text content."""

    def __init__(self, texts: list[str], y: np.ndarray) -> None:
        self.mode = "heuristic"
        if _both_classes(y) and len(y) >= 6:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression

                self.vec = TfidfVectorizer(max_features=600).fit(texts)
                self.clf = LogisticRegression(max_iter=1000).fit(self.vec.transform(texts), y)
                self.mode = "sklearn_tfidf_logreg"
            except Exception as exc:  # pragma: no cover
                analysis_log.warning(f"text surrogate training failed ({exc}) — heuristic")

    def proba(self, text: str) -> float:
        if self.mode == "sklearn_tfidf_logreg":
            try:
                return float(self.clf.predict_proba(self.vec.transform([text]))[0, 1])
            except Exception:
                pass
        toks = set(_tokenize(text))
        hits = len(toks & _ERROR_TERMS)
        return min(1.0, hits / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def _records_to_features(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the numeric matrix, labels, texts, and per-record metadata."""
    # call_frequency = number of interactions sharing a (session, component).
    freq: dict[tuple[str, str], int] = {}
    for r in records:
        e = r["entry"]
        freq[(e.get("session_id", ""), e.get("component", ""))] = (
            freq.get((e.get("session_id", ""), e.get("component", "")), 0) + 1
        )

    rows, texts, labels, meta = [], [], [], []
    for r in records:
        e = r["entry"]
        content = e.get("content", "") or ""
        rows.append(
            [
                float(len(content)),
                float(e.get("latency_ms", 0.0) or 0.0),
                float(e.get("token_estimate", 0) or 0),
                float(freq[(e.get("session_id", ""), e.get("component", ""))]),
            ]
        )
        texts.append(content)
        labels.append(1 if e.get("status") == "error" else 0)
        meta.append(
            {
                "key": r["key"],
                "namespace": ".".join(r["namespace"]),
                "session_id": e.get("session_id"),
                "component": e.get("component"),
                "tool_name": e.get("tool_name"),
                "status": e.get("status"),
                "error": e.get("error"),
                "content": content,
                "content_preview": content[:200],
            }
        )
    return {
        "X": np.array(rows, dtype=float) if rows else np.zeros((0, len(FEATURE_NAMES))),
        "y": np.array(labels, dtype=int),
        "texts": texts,
        "meta": meta,
    }


def _all_records(limit: int = 2000) -> list[dict[str, Any]]:
    return get_reader().list_entries(namespace_prefix=(LOG_ROOT,), limit=limit)


# ─────────────────────────────────────────────────────────────────────────────
#  Proxy SHAP — exact Shapley values over the structured surrogate
# ─────────────────────────────────────────────────────────────────────────────
def proxy_shap(model: StructuredSurrogate, x: np.ndarray) -> dict[str, Any]:
    """
    Exact Shapley values: for each feature, average the marginal contribution
    v(S∪{i}) - v(S) over ALL coalitions S of the other features (features not
    in S are held at the dataset baseline). Identifies which metric most
    skewed the surrogate toward predicting the anomaly.
    """
    n = len(FEATURE_NAMES)
    baseline = model.baseline.copy()

    def v(subset: tuple[int, ...]) -> float:
        xv = baseline.copy()
        for i in subset:
            xv[i] = x[i]
        return model.proba(xv)

    shap = np.zeros(n)
    idx = list(range(n))
    for i in idx:
        others = [j for j in idx if j != i]
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                w = (math.factorial(len(S)) * math.factorial(n - len(S) - 1)) / math.factorial(n)
                shap[i] += w * (v(tuple(S) + (i,)) - v(tuple(S)))

    values = {FEATURE_NAMES[i]: round(float(shap[i]), 5) for i in idx}
    most = max(values, key=lambda k: abs(values[k])) if values else None
    return {
        "model": model.mode,
        "baseline_value": round(model.proba(baseline), 5),
        "predicted_error_proba": round(model.proba(x), 5),
        "values": values,
        "most_influential_feature": most,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Proxy LIME — token-masking perturbation + weighted local linear fit
# ─────────────────────────────────────────────────────────────────────────────
def proxy_lime(
    model: TextSurrogate,
    text: str,
    n_samples: int = 150,
    max_tokens: int = 60,
    seed: int = 42,
    top_k: int = 12,
) -> dict[str, Any]:
    """
    Perturb the log sequence by systematically masking token subsets, measure
    the variation in the surrogate's error-confidence, and fit a locality-
    weighted linear model (Ridge) of token-presence → confidence. The
    coefficients are the localized token-importance estimates.
    """
    tokens = _tokenize(text)[:max_tokens]
    if not tokens:
        return {"model": model.mode, "base_confidence": 0.0, "n_samples": 0,
                "token_importances": [], "most_influential_token": None}

    m = len(tokens)
    rng = np.random.default_rng(seed)
    base_conf = model.proba(text)

    masks = rng.integers(0, 2, size=(n_samples, m))
    masks[0, :] = 1  # the unperturbed original
    scores = np.empty(n_samples)
    for r in range(n_samples):
        kept = [tokens[j] for j in range(m) if masks[r, j] == 1]
        scores[r] = model.proba(" ".join(kept))

    # Locality weighting: samples closer to the original (fewer masked) weigh more.
    frac_masked = 1.0 - masks.mean(axis=1)
    weights = np.exp(-(frac_masked ** 2) / (0.25 ** 2))

    try:
        from sklearn.linear_model import Ridge

        coefs = Ridge(alpha=1.0).fit(masks, scores, sample_weight=weights).coef_
    except Exception:  # pragma: no cover
        # Fallback: importance = base - mean(score when token masked).
        coefs = np.array(
            [base_conf - scores[masks[:, j] == 0].mean() if (masks[:, j] == 0).any() else 0.0
             for j in range(m)]
        )

    # Aggregate repeated tokens by mean coefficient.
    agg: dict[str, list[float]] = {}
    for tok, c in zip(tokens, coefs):
        agg.setdefault(tok, []).append(float(c))
    importances = [
        {"token": tok, "coefficient": round(float(np.mean(cs)), 5)}
        for tok, cs in agg.items()
    ]
    importances.sort(key=lambda d: abs(d["coefficient"]), reverse=True)
    top = importances[:top_k]
    return {
        "model": model.mode,
        "base_confidence": round(float(base_conf), 5),
        "n_samples": n_samples,
        "n_tokens_analyzed": m,
        "token_importances": top,
        "most_influential_token": top[0]["token"] if top else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Audit orchestration
# ─────────────────────────────────────────────────────────────────────────────
def _pick_session(session_id: Optional[str]) -> Optional[str]:
    if session_id:
        return session_id
    sessions = list_sessions()
    if not sessions:
        return None
    # Prefer the session with the most errors (most to explain).
    return sessions[0]["session_id"]


def _pick_target(feats: dict[str, Any], session_id: str, target_key: Optional[str],
                 model: StructuredSurrogate) -> Optional[int]:
    meta = feats["meta"]
    if target_key:
        for i, m in enumerate(meta):
            if m["key"] == target_key:
                return i
    # Otherwise: the highest-anomaly error trace in the session, else highest overall.
    in_session = [i for i, m in enumerate(meta) if m["session_id"] == session_id]
    errs = [i for i in in_session if meta[i]["status"] == "error"]
    pool = errs or in_session or list(range(len(meta)))
    if not pool:
        return None
    return max(pool, key=lambda i: model.proba(feats["X"][i]))


def run_explainability_audit(
    session_id: Optional[str] = None,
    target_key: Optional[str] = None,
    n_lime_samples: int = 150,
) -> dict[str, Any]:
    """
    Run a full local explainability audit for one targeted failure mode and
    return the structured report (also writable to JSON via write_audit_report).
    """
    analysis_log.info(f"Explainability audit | session={session_id} target={target_key}")
    records = _all_records()
    if not records:
        return {"status": "no_data", "message": "log store is empty"}

    feats = _records_to_features(records)
    structured_model = StructuredSurrogate(feats["X"], feats["y"])
    text_model = TextSurrogate(feats["texts"], feats["y"])

    sid = _pick_session(session_id)
    if sid is None:
        return {"status": "no_data", "message": "no sessions found"}

    ti = _pick_target(feats, sid, target_key, structured_model)
    if ti is None:
        return {"status": "no_target", "message": "no target trace found", "session_id": sid}

    target = feats["meta"][ti]
    x = feats["X"][ti]

    # 1. Graph-relational context hydration (Neo4j → log-derived fallback).
    graph = fetch_graph_context(sid)

    # 2. Proxy SHAP over structured execution signatures.
    shap_result = proxy_shap(structured_model, x)

    # 3. Proxy LIME over the unstructured log text.
    lime_result = proxy_lime(text_model, target["content"], n_samples=n_lime_samples)

    failure_mode = (
        f"{target.get('component')} / {target.get('tool_name')} "
        f"[{target.get('status')}]: {target.get('error') or 'anomalous trace'}"
    )
    summary = (
        f"Targeted failure mode → {failure_mode}. "
        f"Proxy SHAP attributes the structured skew primarily to "
        f"'{shap_result['most_influential_feature']}' "
        f"(predicted error confidence {shap_result['predicted_error_proba']}). "
        f"Proxy LIME flags the token '{lime_result['most_influential_token']}' as the "
        f"strongest local text driver. Graph context: {graph['counts']['agent_actions']} "
        f"adjacent AgentAction(s) and {graph['counts']['server_calls']} MCPServerCall(s) "
        f"hydrated via {graph['source']}."
    )

    report = {
        "schema": "explainability_audit_report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_id": sid,
        "target": {
            "key": target["key"],
            "namespace": target["namespace"],
            "component": target["component"],
            "tool_name": target["tool_name"],
            "status": target["status"],
            "error": target["error"],
            "content_preview": target["content_preview"],
        },
        "failure_mode": failure_mode,
        "graph_context": {
            "source": graph["source"],
            "session_present": graph["session_present"],
            "counts": graph["counts"],
            "matched_agent_actions": graph.get("agent_actions", [])[:8],
            "matched_server_calls": graph.get("server_calls", [])[:8],
            "context_text": graph["context_text"],
        },
        "structured_features": {
            name: round(float(val), 4) for name, val in zip(FEATURE_NAMES, x)
        },
        "proxy_shap": shap_result,
        "proxy_lime": lime_result,
        "summary": summary,
    }
    analysis_log.info(f"Audit complete | {summary}")
    return report


def write_audit_report(report: dict[str, Any], path: str | None = None) -> str:
    """Persist the audit report to the deliverable JSON file."""
    out = path or AUDIT_REPORT_PATH
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    analysis_log.info(f"Explainability audit report written → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  SHAP horizontal bar chart renderer (for the dashboard / evidence)
# ─────────────────────────────────────────────────────────────────────────────
def render_shap_chart(report: dict[str, Any], save_path: str | None = None) -> str:
    """Render proxy-SHAP feature weights as a horizontal bar chart PNG."""
    values = report.get("proxy_shap", {}).get("values", {})
    os.makedirs(CHART_DIR, exist_ok=True)
    out = save_path or os.path.join(CHART_DIR, "shap_feature_weights.png")
    if not values:
        return out
    names = list(values.keys())
    weights = [values[n] for n in names]
    order = np.argsort(np.abs(weights))
    names = [names[i] for i in order]
    weights = [weights[i] for i in order]
    colors = ["#d1495b" if w >= 0 else "#2e7d8c" for w in weights]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, weights, color=colors)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_title("Proxy SHAP — structured feature influence on the anomaly")
    ax.set_xlabel("Shapley value (→ pushes toward error)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    analysis_log.info(f"SHAP chart saved → {out}")
    return out
