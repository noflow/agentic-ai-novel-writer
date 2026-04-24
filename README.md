# Agentic AI Framework

A learn-by-building Python framework for agentic AI — from a single agent
to multi-agent collaboration.

## Roadmap

- **Phase 1** ✅ Single Agent with Tool Use
- **Phase 2** ✅ Specialized Agents + Orchestrator
- **Phase 3** 🔜 Dynamic Multi-Agent Communication

## Project Structure

```
agentic-ai/
├── agent.py           # Core Agent class with the agentic loop
├── config.py          # Configuration and model settings
├── specialists.py     # Phase 2: Researcher, Writer, Critic agents
├── orchestrator.py    # Phase 2: Pipeline orchestrator
├── tools/
│   ├── __init__.py    # Tool registry
│   ├── base.py        # Base tool class
│   ├── calculator.py  # Math evaluation tool
│   ├── web_search.py  # Web search (simulated — swap for real API)
│   ├── file_ops.py    # Read/write files
│   └── clock.py       # Current date/time
├── main.py            # Interactive CLI (single + pipeline modes)
└── README.md
```

## Quick Start

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"
python main.py
```

## Phase 1: Single Agent

Chat directly with an AI agent that can reason and use tools.
The core "agentic loop": Think → Pick Tool → Execute → Observe → Repeat.

```
🧑 You → Assistant: What's 15% of 847.50?
  [Assistant] Turn 1... → Tool call
    → Calling calculator({"expression": "847.50 * 0.15"})
    ← Result: 127.125
  [Assistant] Turn 2... → Final answer
🤖 Assistant: 15% of 847.50 is 127.125, or $127.13.
```

## Phase 2: Specialist Agents & Pipelines

Three specialized agents, each with their own personality and tools:

| Agent       | Role                          | Tools                        |
|-------------|-------------------------------|------------------------------|
| Researcher  | Gathers & analyzes info       | web search, calculator, clock, file read |
| Writer      | Creates polished content      | file read, file write        |
| Critic      | Reviews & suggests fixes      | file read, calculator        |

### Using Pipelines

Switch to pipeline mode to chain agents together:

```
🧑 You: mode pipeline
📍 Pipeline mode — multi-agent collaboration

🧑 Task (full_pipeline): Write a blog post about why sleep is important

  → Handing off to: Researcher
    [researches the topic, uses tools]
  ← Researcher finished

  → Handing off to: Writer
    [writes first draft using research]
  ← Writer finished

  → Handing off to: Critic
    [reviews draft, provides feedback]
  ← Critic finished

  → Handing off to: Writer
    [revises draft based on feedback]
  ← Writer finished

PIPELINE COMPLETE
```

Available pipelines:
- `research_only` — just gather info
- `write_only` — just write content
- `research_and_write` — research then write
- `full_pipeline` — research → write → critique → revise

### Switching Agents

In single-agent mode, you can talk to any specialist directly:

```
🧑 You: switch researcher
✓ Switched to Researcher

🧑 You → Researcher: What do we know about dark matter?
```

## Adding Custom Tools

```python
from tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something cool"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to process"}
        },
        "required": ["query"]
    }

    def run(self, query: str) -> str:
        return f"Processed: {query}"
```

Register it in `tools/__init__.py`, then give it to whichever agents need it.

## Key Concepts

### The Agentic Loop (agent.py)
```
User Goal → LLM thinks → Tool call? → Execute tool → Feed result back → Loop
                        → Text response? → Done!
```

### Tool System (tools/)
Tools are Python classes with a name, description, JSON schema, and `run()` method.
The Agent converts these to Anthropic's API format automatically.

### Specialist Agents (specialists.py)
Each agent gets a unique system prompt (personality) and a curated set of tools.
The same Agent class powers all of them — specialization comes from configuration.

### Orchestrator (orchestrator.py)
Defines pipelines — sequences of agent handoffs. Each agent's output becomes
the next agent's input. No agent knows about the others; the orchestrator
manages all the routing.

## What's Next? (Phase 3)

Phase 3 will add dynamic multi-agent communication:
- Agents that can **request help** from other agents mid-task
- A **message bus** for agent-to-agent communication
- **Shared memory** so agents can build on each other's work
- **Dynamic routing** — the orchestrator decides which agent to call based on context
