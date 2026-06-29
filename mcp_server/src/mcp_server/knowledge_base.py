"""
knowledge_base.py
─────────────────
Hierarchical enterprise knowledge base for the CRAG resource.

Structure
  L1 — Topic summaries  : one paragraph per domain area.
  L2 — Section chunks   : detailed explanations with examples.
  L3 — Code blocks      : runnable snippets and structured data.

Search uses keyword overlap with L1 summaries to identify relevant
topics, then scores L2/L3 chunks within those topics by keyword +
content overlap — no vector DB required, zero external dependencies.
"""

from __future__ import annotations

import re
from typing import Any


# ── Level 1: Topic Summaries (used for topic routing) ─────────────────────────
TOPIC_SUMMARIES: dict[str, str] = {
    "agentic_ai": (
        "Agentic AI systems are autonomous agents that reason, plan, and act using LLMs "
        "as their cognitive core. Key frameworks: LangChain, LangGraph, AutoGen, CrewAI. "
        "Agents use external tools (search, code execution) and follow paradigms like "
        "ReAct (Reason + Act) for structured decision-making loops."
    ),
    "mcp_protocol": (
        "Model Context Protocol (MCP) is an open standard for connecting AI agents to "
        "external tools, data sources, and services. FastMCP provides a Pythonic "
        "implementation. Primitives: tools, resources, prompts, sampling, elicitation. "
        "Transports: stdio and streamable-http. Endpoint: http://host:port/mcp."
    ),
    "rag_crag": (
        "Retrieval-Augmented Generation (RAG) enriches LLM responses with retrieved "
        "context. Corrective RAG (CRAG) adds a grading step — verifies chunk relevance "
        "before generation. Hierarchical CRAG combines multi-query expansion, "
        "hierarchical indexing, Tree-of-Thought evaluation, and Tavily web fallback."
    ),
    "langchain": (
        "LangChain is an LLM application framework with chains, agents, and tools. "
        "Core primitives: LLMs, PromptTemplates, OutputParsers, Tools, Agents. "
        "LCEL (LangChain Expression Language) enables composable chain construction "
        "using the | pipe operator. create_react_agent + AgentExecutor = ReAct loop."
    ),
    "react_tot": (
        "ReAct (Reasoning + Acting) interleaves Thought, Action, Observation steps. "
        "Tree-of-Thought (ToT) generates multiple parallel reasoning paths then uses a "
        "Judge to select the strongest. ToT improves accuracy on complex, multi-step "
        "reasoning by exploring the solution space before committing."
    ),
    "groq_models": (
        "Groq is an AI inference provider using LPU hardware for ultra-low latency. "
        "Supports Llama 3.x, Mixtral, Gemma. Free tier: llama-3.1-8b-instant 6 000 TPM, "
        "llama-3.3-70b-versatile 12 000 TPM. LangChain integration: langchain-groq. "
        "Class: ChatGroq(model='llama-3.3-70b-versatile', temperature=0.3)."
    ),
    "fastmcp_server": (
        "FastMCP is the Pythonic MCP framework. Key decorators: @mcp.tool(), "
        "@mcp.resource(), @mcp.prompt(). Context methods: ctx.sample(), ctx.info(), "
        "ctx.debug(), ctx.elicit(), ctx.report_progress(). "
        "Run: mcp.run(transport='streamable-http', host='0.0.0.0', port=8000)."
    ),
    "observability": (
        "Enterprise observability for AI systems: structured logging with timestamps, "
        "dual-stream log capture (CLIENT + SERVER), distributed tracing, and "
        "intermediate-step inspection. MCP forwards server logs to clients via "
        "ctx.info() / ctx.debug() notification messages. Python logging module is "
        "preferred over print() for production systems."
    ),
}


# ── Level 2 & 3: Detailed chunks ──────────────────────────────────────────────
KNOWLEDGE_BASE: dict[str, list[dict[str, Any]]] = {

    "agentic_ai": [
        {
            "level": "L2-concept",
            "topic": "agentic_ai",
            "keywords": ["agent", "autonomous", "react", "reason", "act", "tool", "loop"],
            "content": (
                "An AI Agent is an LLM-powered system with four capabilities:\n"
                "(1) Perception — reads user inputs and tool observations.\n"
                "(2) Reasoning — plans next action via LLM Thought step.\n"
                "(3) Action — executes a selected tool with appropriate arguments.\n"
                "(4) Memory — short-term context window; optional long-term vector store.\n"
                "The ReAct paradigm (Yao et al., 2022) formalises this as a "
                "Thought → Action → Observation loop repeated until a Final Answer."
            ),
        },
        {
            "level": "L2-frameworks",
            "topic": "agentic_ai",
            "keywords": ["langchain", "langgraph", "autogen", "crewai", "framework", "production", "enterprise"],
            "content": (
                "Production agent framework comparison (2025):\n"
                "• LangChain / LangGraph  — best for single + multi-agent with state graphs, "
                "human-in-the-loop, and streaming. Mature ecosystem.\n"
                "• AutoGen (Microsoft)    — multi-agent conversation framework; strong for "
                "code-generation workflows and agent debates.\n"
                "• CrewAI                 — role-based task delegation; readable YAML configs.\n"
                "Enterprise recommendation: LangGraph for stateful, long-running pipelines; "
                "LangChain for single-agent prototypes and tool-use heavy tasks."
            ),
        },
        {
            "level": "L3-code",
            "topic": "agentic_ai",
            "keywords": ["create_react_agent", "agentexecutor", "code", "implementation", "groq"],
            "content": (
                "LangChain ReAct agent — minimal working setup with Groq:\n"
                "```python\n"
                "from langchain.agents import create_react_agent, AgentExecutor\n"
                "from langchain_groq import ChatGroq\n"
                "from langchain import hub\n"
                "\n"
                "llm   = ChatGroq(model='llama-3.1-8b-instant', temperature=0, max_tokens=1024)\n"
                "prompt = hub.pull('hwchase17/react')\n"
                "agent  = create_react_agent(llm=llm, tools=tools, prompt=prompt)\n"
                "executor = AgentExecutor(\n"
                "    agent=agent, tools=tools,\n"
                "    verbose=True, max_iterations=12, max_execution_time=180,\n"
                "    return_intermediate_steps=True,\n"
                ")\n"
                "result = executor.invoke({'input': 'your query here'})\n"
                "```"
            ),
        },
    ],

    "mcp_protocol": [
        {
            "level": "L2-primitives",
            "topic": "mcp_protocol",
            "keywords": ["mcp", "tool", "resource", "prompt", "sampling", "elicitation", "primitive"],
            "content": (
                "MCP defines five primitives:\n"
                "• Tools      — callable functions the agent can invoke (POST /mcp).\n"
                "• Resources  — readable data sources identified by URI (e.g. knowledge://...).\n"
                "• Prompts    — reusable parameterised prompt templates.\n"
                "• Sampling   — server requests an LLM completion from the *client's* model.\n"
                "• Elicitation— server requests structured user input from the client.\n"
                "Sampling is the key inversion: the server has no LLM; the client's model "
                "handles all completions routed through ctx.session.create_message()."
            ),
        },
        {
            "level": "L2-transport",
            "topic": "mcp_protocol",
            "keywords": ["streamable-http", "stdio", "transport", "http", "endpoint", "port"],
            "content": (
                "MCP transport options:\n"
                "• stdio           : Local subprocess. Simple, no networking. Suitable for "
                "local desktop AI assistants and MCP-compatible IDE clients.\n"
                "• streamable-http : HTTP + Server-Sent Events. Stateless per-request; "
                "supports concurrent clients, NAT traversal, cloud deployment.\n"
                "FastMCP setup: mcp.run(transport='streamable-http', host='0.0.0.0', port=8000)\n"
                "Default endpoint: http://localhost:8000/mcp\n"
                "Client connection: Client('http://localhost:8000/mcp')"
            ),
        },
        {
            "level": "L3-sampling-code",
            "topic": "mcp_protocol",
            "keywords": ["sampling", "ctx", "create_message", "SamplingMessage", "TextContent", "code"],
            "content": (
                "MCP Sampling from server — canonical content-block pattern:\n"
                "```python\n"
                "from mcp.types import SamplingMessage, TextContent\n"
                "\n"
                "result = await ctx.session.create_message(\n"
                "    messages=[\n"
                "        SamplingMessage(\n"
                "            role='user',\n"
                "            content=TextContent(type='text', text=your_prompt),\n"
                "        )\n"
                "    ],\n"
                "    system_prompt='You are a critic AI...',\n"
                "    max_tokens=512,\n"
                ")\n"
                "# result.content is a TextContent object\n"
                "generated_text = result.content.text\n"
                "```\n"
                "The client's sampling_handler executes the actual LLM call "
                "and returns the text. The server receives the generated text."
            ),
        },
    ],

    "rag_crag": [
        {
            "level": "L2-rag-concept",
            "topic": "rag_crag",
            "keywords": ["rag", "retrieval", "augmented", "generation", "vector", "embedding", "chunk"],
            "content": (
                "Standard RAG pipeline:\n"
                "(1) Index   — chunk documents, embed with encoder, store in FAISS/Chroma.\n"
                "(2) Retrieve— embed query, cosine similarity search returns top-k chunks.\n"
                "(3) Augment — prepend chunks to LLM prompt as grounding context.\n"
                "(4) Generate— LLM produces a grounded response.\n"
                "Limitation: retrieved chunks may be irrelevant or hallucinated. "
                "CRAG adds a correctiveness step to filter noise before generation."
            ),
        },
        {
            "level": "L2-crag",
            "topic": "rag_crag",
            "keywords": ["crag", "corrective", "grading", "relevance", "fallback", "tavily", "web search"],
            "content": (
                "Corrective RAG (CRAG) pipeline:\n"
                "1. Retrieve : Standard similarity search.\n"
                "2. Grade    : LLM evaluates each chunk — 'relevant' or 'irrelevant'.\n"
                "3. Correct  :\n"
                "   • All irrelevant → Tavily web search fallback.\n"
                "   • Some relevant  → use only relevant chunks.\n"
                "   • All relevant   → proceed normally.\n"
                "4. Generate : LLM generates from corrected context.\n"
                "Hierarchical CRAG extends this with multi-query expansion (3 semantic "
                "variants), hierarchical L1→L2→L3 indexing, and ToT-based grading "
                "across 3 judge perspectives (Specificity, Completeness, Accuracy)."
            ),
        },
        {
            "level": "L2-hierarchical-indexing",
            "topic": "rag_crag",
            "keywords": ["hierarchical", "chunking", "parent", "child", "summary", "level", "index"],
            "content": (
                "Hierarchical indexing strategy:\n"
                "• L1 (Summaries)  : Topic-level abstractions; matched first against query.\n"
                "• L2 (Sections)   : Detailed explanations with examples (300-800 tokens).\n"
                "• L3 (Code/Data)  : Granular code blocks and structured reference data.\n"
                "Retrieval flow:\n"
                "  query → keyword match against L1 summaries → score topic relevance\n"
                "  → drill into matching topic's L2/L3 chunks → rank by overlap score\n"
                "This reduces noise vs flat chunk retrieval and improves precision "
                "for technical queries by using topic routing before chunk scoring."
            ),
        },
    ],

    "langchain": [
        {
            "level": "L2-lcel",
            "topic": "langchain",
            "keywords": ["lcel", "chain", "pipe", "compose", "expression", "invoke", "async"],
            "content": (
                "LCEL (LangChain Expression Language) — composable chains via | operator:\n"
                "```python\n"
                "from langchain_core.prompts import ChatPromptTemplate\n"
                "from langchain_core.output_parsers import StrOutputParser\n"
                "\n"
                "chain = ChatPromptTemplate.from_messages([...]) | llm | StrOutputParser()\n"
                "# Sync\n"
                "result = chain.invoke({'key': 'value'})\n"
                "# Async (inside async def)\n"
                "result = await chain.ainvoke({'key': 'value'})\n"
                "# Stream\n"
                "for chunk in chain.stream({'key': 'value'}): print(chunk, end='')\n"
                "```\n"
                "LCEL chains auto-parallelise independent branches and support "
                ".with_fallbacks(), .with_retry(), and .with_config() modifiers."
            ),
        },
        {
            "level": "L2-tools",
            "topic": "langchain",
            "keywords": ["tool", "decorator", "@tool", "single string", "react", "wrapper"],
            "content": (
                "LangChain @tool for ReAct agents — single-string input pattern:\n"
                "```python\n"
                "import json\n"
                "from langchain_core.tools import tool\n"
                "\n"
                "@tool\n"
                "def my_tool(tool_input: str) -> str:\n"
                "    '''\n"
                "    Description the agent reads to decide when to use this tool.\n"
                "    Action Input must be a JSON string: {\"key\": \"value\"}\n"
                "    '''\n"
                "    data = json.loads(tool_input)\n"
                "    return process(data['key'])\n"
                "```\n"
                "Important: Text-based ReAct (hwchase17/react prompt) passes Action Input "
                "as ONE string. Multi-parameter tools cause ValidationError. Always use "
                "a single tool_input: str parameter and parse JSON inside the function."
            ),
        },
    ],

    "fastmcp_server": [
        {
            "level": "L2-server-setup",
            "topic": "fastmcp_server",
            "keywords": ["fastmcp", "server", "tool", "resource", "context", "mcp", "decorator"],
            "content": (
                "FastMCP server skeleton:\n"
                "```python\n"
                "from fastmcp import FastMCP, Context\n"
                "from mcp.types import SamplingMessage, TextContent\n"
                "\n"
                "mcp = FastMCP('MyServer')\n"
                "\n"
                "@mcp.tool()\n"
                "async def my_tool(param: str, ctx: Context) -> str:\n"
                "    await ctx.info('Tool invoked')            # forwarded to client log\n"
                "    result = await ctx.session.create_message(\n"
                "        messages=[SamplingMessage(\n"
                "            role='user',\n"
                "            content=TextContent(type='text', text=param)\n"
                "        )],\n"
                "        max_tokens=256,\n"
                "    )\n"
                "    return result.content.text\n"
                "\n"
                "@mcp.resource('knowledge://{query}')\n"
                "async def my_resource(query: str, ctx: Context) -> str:\n"
                "    await ctx.debug('Resource queried')\n"
                "    return 'context data'\n"
                "\n"
                "def main():\n"
                "    mcp.run(transport='streamable-http', host='0.0.0.0', port=8000)\n"
                "```"
            ),
        },
    ],

    "react_tot": [
        {
            "level": "L2-react",
            "topic": "react_tot",
            "keywords": ["react", "thought", "action", "observation", "final answer", "loop"],
            "content": (
                "ReAct loop structure (per hwchase17/react prompt):\n"
                "Thought: <agent reasons about what to do next>\n"
                "Action: <tool_name>\n"
                "Action Input: <tool argument — one JSON string for text-format ReAct>\n"
                "Observation: <tool output appended by executor>\n"
                "... (repeat Thought/Action/Observation)\n"
                "Thought: I now know the final answer\n"
                "Final Answer: <complete answer to the user's question>\n\n"
                "Key failure modes:\n"
                "• 413 TPM exceeded: directive too long, reduce with compact system prompt.\n"
                "• Iteration limit: agent loops without concluding; "
                "add 'After reflection, write Final Answer immediately.'"
            ),
        },
        {
            "level": "L2-tot",
            "topic": "react_tot",
            "keywords": ["tree of thought", "tot", "persona", "judge", "paths", "parallel", "reasoning"],
            "content": (
                "Tree-of-Thought (ToT) implementation pattern:\n"
                "1. Define 3 personas: Analytical Expert, Systems Thinker, Devil's Advocate.\n"
                "2. Invoke each persona in sequence (or parallel) on the same question.\n"
                "3. Judge LLM evaluates all 3 paths on: accuracy, completeness, risk, actionability.\n"
                "4. Return winning path + judge verdict.\n"
                "Groq TPM note: Sequential calls are safer on free tier. "
                "3 persona calls + 1 judge = 4 LLM invocations per ToT cycle — "
                "budget ~3 000 tokens for the full cycle on llama-3.3-70b-versatile."
            ),
        },
    ],

    "groq_models": [
        {
            "level": "L2-setup",
            "topic": "groq_models",
            "keywords": ["groq", "llama", "api", "setup", "langchain-groq", "model", "rate limit", "tpm"],
            "content": (
                "Groq with LangChain setup:\n"
                "```python\n"
                "import os\n"
                "from langchain_groq import ChatGroq\n"
                "\n"
                "# Fast outer-loop model\n"
                "react_llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0, max_tokens=1024)\n"
                "# Capable inner model for critique/judge\n"
                "meta_llm  = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.3, max_tokens=1024)\n"
                "```\n"
                "Free tier hard limits:\n"
                "  llama-3.1-8b-instant   : 6 000 TPM, 30 RPM\n"
                "  llama-3.3-70b-versatile: 12 000 TPM, 30 RPM\n"
                "Mitigation: compact directives, max_tokens caps, sequential (not parallel) calls."
            ),
        },
    ],

    "observability": [
        {
            "level": "L2-logging",
            "topic": "observability",
            "keywords": ["logging", "log", "timestamp", "client", "server", "agent_system", "dual stream"],
            "content": (
                "Dual-stream logging pattern for MCP systems:\n"
                "```python\n"
                "import logging\n"
                "\n"
                "# CLIENT logger — own operations\n"
                "client_logger = logging.getLogger('CLIENT')\n"
                "# SERVER logger — entries forwarded via MCP log_handler callback\n"
                "server_logger = logging.getLogger('SERVER')\n"
                "\n"
                "# File handler shared by both — agent_system.log\n"
                "fh = logging.FileHandler('agent_system.log', mode='a')\n"
                "fh.setFormatter(logging.Formatter(\n"
                "    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',\n"
                "    datefmt='%Y-%m-%d %H:%M:%S'\n"
                "))\n"
                "client_logger.addHandler(fh)\n"
                "server_logger.addHandler(fh)\n"
                "```\n"
                "MCP server forwards logs via: await ctx.info('message') → client log_handler."
            ),
        },
    ],
}


def search_knowledge_base(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """
    Two-level hierarchical keyword search.

    Algorithm
    ─────────
    L1 pass — score each topic by keyword overlap with its summary.
    L2/L3 pass — within relevant topics, score each chunk by:
        • keyword-field overlap  (weight 2.0 — explicit signals)
        • content-field overlap  (weight 0.5 — implicit signals)
        • L1 topic boost         (weight 1.0 — coherence reward)

    Args:
        query: Raw search query string.
        top_k: Maximum chunks to return (ranked by score, descending).

    Returns:
        List of chunk dicts, best matches first.
    """
    query_terms = set(re.findall(r"\b\w+\b", query.lower()))
    if not query_terms:
        return []

    # ── L1: topic routing ─────────────────────────────────────────────────────
    topic_scores: dict[str, float] = {}
    for topic, summary in TOPIC_SUMMARIES.items():
        summary_terms = set(re.findall(r"\b\w+\b", summary.lower()))
        overlap = len(query_terms & summary_terms)
        if overlap > 0:
            topic_scores[topic] = overlap / max(len(query_terms), 1)

    if topic_scores:
        relevant_topics = [t for t, s in topic_scores.items() if s > 0.04]
    else:
        # No L1 match — search every topic (broad fallback)
        relevant_topics = list(KNOWLEDGE_BASE.keys())

    # ── L2/L3: chunk scoring ───────────────────────────────────────────────────
    scored: list[tuple[float, dict]] = []
    for topic in relevant_topics:
        for chunk in KNOWLEDGE_BASE.get(topic, []):
            kw_terms      = set(chunk.get("keywords", []))
            content_terms = set(re.findall(r"\b\w+\b", chunk["content"].lower()))
            kw_score      = len(query_terms & kw_terms) * 2.0
            content_score = len(query_terms & content_terms) * 0.5
            topic_boost   = topic_scores.get(topic, 0.0) * 1.0
            total         = kw_score + content_score + topic_boost
            if total > 0:
                scored.append((total, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]
