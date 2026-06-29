# REFLECTION_STAGE4.md — Resilience, Edgeless Orchestration & Black-Box XAI

## Deterministic vs. Edgeless Graph Orchestration

Discarding `add_edge` in favour of nodes that return `Command(goto=...)` trades a
static, inspectable topology for runtime flexibility. The structural benefit is
that control flow becomes *data*: `initial_ingest_node` computes a plan and each
node pops its next hop, so adding a capability never means rewiring the graph —
routing adapts per request. The operational hazard is that the graph is no longer
self-documenting. With a single `START` edge, a static visualization shows
isolated nodes; the real path only exists at execution time. Testing shifts from
asserting edges to asserting *emitted commands* — you unit-test each node's
`goto` decision given a state, and integration-test that the plan queue drains to
`END`. Debugging is harder because a wrong destination is a value bug, not a
missing wire, so I log every hop (`[edgeless] node → goto`) to reconstruct traces.

## The Reality of Black-Box XAI in Language Models

LLMs emit token-conditional probabilities over an enormous vocabulary, not
decisions across a smooth feature boundary, so SHAP/LIME assumptions are
strained. Proxy LIME masks tokens and regresses a *surrogate's* confidence on
presence — but language is non-additive: masking one token shifts the meaning of
its neighbours, violating LIME's local linearity, and my Ridge coefficients are
estimates of a TF-IDF surrogate, not the generative model. Proxy SHAP's exact
Shapley values are faithful *to the surrogate*, yet the surrogate is a logistic
approximation of "error vs. success," not the agent's true reasoning. These
post-hoc scores are therefore directional and comparative, useful for ranking
which latency/token feature or which log token co-varies with failure — but they
are correlational attributions, not causal explanations of the LLM itself.

## Static Fallbacks vs. Dynamic Context Self-Healing

`RunnableWithFallbacks` is cheap, deterministic, and fast: a known alternative
fires with zero extra tokens, ideal for predictable faults (bad schema, a dead
endpoint, the hardcoded absolute payload). Dynamic LLM self-healing — reinjecting
the `error_trace` and asking the model to correct itself — is flexible but costs
latency, tokens, and adds non-determinism and its own failure surface. In
production I reserve LLM self-healing for genuinely novel, semantically rich
failures, and keep deterministic retries/fallbacks (and a non-raising final
layer) as the cheap, reliable backbone — which is why my self-heal step treats
the LLM as best-effort and the deterministic re-drive as the actual recovery.
