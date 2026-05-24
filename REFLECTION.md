# REFLECTION.md — Stage 2 Architectural Reflections

## The Architecture Shift

Moving from Stage 1's monolithic LangChain agent to a decoupled MCP client-server system produces a fundamental operational change: **tools become network-addressable microservices**. The operational advantages are significant — the server can be deployed, scaled, updated, and versioned independently of the client. Multiple agents from different teams can consume the same Reflection tool and CRAG resource simultaneously. Hot-swapping the knowledge base requires no agent restart. However, the bottleneck is equally clear: every tool invocation now crosses a network boundary. What was a local Python function call with microsecond latency becomes an HTTP round-trip, introducing ~20–100ms overhead per step. In a 12-iteration ReAct loop invoking three tools, this compounds to over 3 seconds of pure network overhead — significant for latency-sensitive applications.

## The Sampling Paradox

MCP Sampling is the protocol's most architecturally distinctive feature and its most counter-intuitive one. The server — which *owns* the Reflection logic — has no LLM. It constructs Critique and Correction prompts, then routes them to the *client's* model via `ctx.session.create_message()`. The server is the brains; the client is the brawn.

The security implication is profound: the server never stores API keys, reducing the blast radius of a server compromise. The structural challenge is equally real: the server cannot predict the client's model capabilities or response format. A misbehaving or low-quality client model silently degrades the server's logic. Rigorous JSON-only system prompts and fence-stripping in both server and client mitigate this, but the coupling is inherent to the pattern.

## State & Context Management

Hierarchical chunking altered data flow fundamentally. Instead of passing a flat list of retrieved chunks to the agent, the CRAG resource returns a structured JSON payload: `expanded_queries`, `tot_scores`, `fallback_used`, and `combined_context`. The agent only reads `combined_context`, but the metadata travels with every resource response. This enables downstream observability — the agent's intermediate steps log shows not just *what* context was retrieved, but *how confidently* the ToT judges rated it, providing a grading paper trail that flat RAG completely lacks.
