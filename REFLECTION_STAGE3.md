# Stage 3 Reflection — Hierarchical Log Persistence & Graph Knowledge Mapping

## The Structural Evolution of Observability

Moving from a flat `.log` stream to a vector-embedded, hierarchically-namespaced
`SqliteStore` changes observability from *reading* to *querying*. A flat file is
append-only prose: grep-able by exact token, but blind to meaning and structure.
The trade-off is real — the store adds an embedding model on the write path
(~tens of ms per entry), a binary DB instead of a human-tailable file, and the
discipline of a validated schema. What we buy is decisive: every interaction is
typed (`tool_invocation`, `resource_read`, `sampling_request`), scoped to a
dot-separated namespace (`logs.mcp.client.sampling`), and semantically indexed on
its `content`. Automated anomaly detection shifts from brittle regex rules
("match `ERROR`") to similarity search ("find interactions *like* a slow,
token-heavy sampling stall") plus structured aggregation (latency moving
averages, error-rate-by-component). We keep the flat file (`mcp_agent_system.log`)
for human tailing and add the store (`mcp_agent_log.db`) for machine reasoning —
the two are complementary, not either/or.

## Graph-Relational Knowledge Mapping

A property graph fits multi-agent MCP traces because the data *is* a graph: a
session triggers agent actions, which route to server calls, which depend back on
client sampling. In SQL this is a thicket of join tables and recursive CTEs to
walk a single causal chain; in a document store the relationships live only
inside opaque blobs. Neo4j makes the edges first-class — `(:Session)-[:TRIGGERED]
->(:AgentAction)-[:ROUTED_TO]->(:MCPServerCall)-[:DEPENDS_ON]->(:AgentAction)` —
so variable-depth traversals ("which sampling calls did this session's CRAG read
ultimately depend on?") are native pattern matches, not query gymnastics. For the
inverted client/server loops that MCP Sampling creates, that traversal-native
model is the natural representation.

## Data Type Handling in AI Pipelines

The sharpest lesson was serialization across the decoupled boundary. JSON has no
integer-key concept, so a `config_map` written with `{0: ...}` silently returns
`{"0": ...}` after a round-trip — a latent type-mismatch fault the moment the
reader does integer arithmetic on keys. The fix is explicit, validated
round-tripping: `LogEntry.to_store_value()` stringifies on write and
`from_store_value()` / `restore_int_keys()` casts back to `int` on read, in *both*
the writer and the independent analysis process. Strict Pydantic schemas also
forced nested `requestedSchema` keys and structural arrays to be declared rather
than assumed, turning a class of "works until it doesn't" bugs into validation
errors caught at the boundary.
