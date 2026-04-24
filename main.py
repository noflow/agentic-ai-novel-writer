"""
Interactive CLI for the Agentic AI framework.

Supports two modes:
  - Single agent: Chat with one agent directly
  - Pipeline mode: Run multi-agent pipelines via the Orchestrator

Run with:  python main.py

Commands:
  quit / exit       -- stop the program
  reset             -- clear conversation history
  tools             -- list current agent's tools
  mode single       -- switch to single-agent mode
  mode pipeline     -- switch to pipeline mode
  agents            -- list available specialist agents
  pipelines         -- list available pipelines
  switch <name>     -- switch to a different agent (researcher/writer/critic/assistant)
"""

from __future__ import annotations
import sys
from config import check_api_key
from agent import Agent
from tools import create_default_registry
from specialists import create_researcher, create_writer, create_critic
from orchestrator import Orchestrator


def print_banner():
    print()
    print("=" * 60)
    print("  [AI]  AGENTIC AI -- Phases 1 & 2")
    print("=" * 60)
    print()
    print("  Modes:")
    print("    mode single     -- chat with one agent")
    print("    mode pipeline   -- run multi-agent pipelines")
    print()
    print("  Commands:")
    print("    quit / exit     -- stop the program")
    print("    reset           -- clear conversation history")
    print("    tools           -- list available tools")
    print("    agents          -- list specialist agents")
    print("    pipelines       -- list multi-agent pipelines")
    print("    switch <name>   -- switch agent (researcher/writer/critic)")
    print()
    print("-" * 60)


def create_agents() -> dict[str, Agent]:
    """Create all available agents."""
    return {
        "assistant": Agent(name="Assistant", tool_registry=create_default_registry()),
        "researcher": create_researcher(),
        "writer": create_writer(),
        "critic": create_critic(),
    }


def run_single_mode(agents: dict[str, Agent], current_name: str) -> str:
    """Run the single-agent interactive loop. Returns new agent name if switched."""
    agent = agents[current_name]
    print(f"\n  >>> Single-agent mode | Active: {agent.name}")
    print(f"  {agent}\n")

    while True:
        try:
            user_input = input(f"\n[You] You -> {agent.name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! ")
            sys.exit(0)

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "exit"):
            print("\nGoodbye! ")
            sys.exit(0)

        if cmd == "reset":
            agent.reset()
            print("  OK Conversation history cleared.")
            continue

        if cmd == "tools":
            print(f"\n  {agent.name}'s tools:")
            for tool in agent.tool_registry.list_tools():
                print(f"    - {tool.name}: {tool.description[:60]}...")
            continue

        if cmd == "agents":
            print("\n  Available agents:")
            for name, a in agents.items():
                marker = " <- active" if name == current_name else ""
                print(f"    - {name}: {a.name}{marker}")
            print("\n  Use 'switch <name>' to change agents.")
            continue

        if cmd == "pipelines":
            print("\n  Use 'mode pipeline' to run multi-agent pipelines.")
            continue

        if cmd.startswith("switch "):
            new_name = cmd.split(" ", 1)[1].strip()
            if new_name in agents:
                return new_name  # Signal to switch
            else:
                print(f"  Unknown agent '{new_name}'. Available: {', '.join(agents.keys())}")
                continue

        if cmd.startswith("mode "):
            mode = cmd.split(" ", 1)[1].strip()
            if mode == "pipeline":
                return "__pipeline__"  # Signal to switch mode
            elif mode == "single":
                print("  Already in single-agent mode.")
                continue
            else:
                print(f"  Unknown mode '{mode}'. Use 'single' or 'pipeline'.")
                continue

        # --- Run the agent --------------------------------
        print()
        try:
            response = agent.run(user_input)
            print(f"\n[AI] {agent.name}: {response}")
        except Exception as e:
            print(f"\nERROR: Error: {e}")
            print("   (Make sure ANTHROPIC_API_KEY is set)")

    return current_name


def run_pipeline_mode(orchestrator: Orchestrator):
    """Run the pipeline interactive loop."""
    print("\n  >>> Pipeline mode -- multi-agent collaboration")
    print(f"  Available pipelines: {', '.join(orchestrator.list_pipelines())}")
    print()
    print("  Usage: Type your task and it will run through the selected pipeline.")
    print("  Change pipeline with: use <pipeline_name>")
    print()

    current_pipeline = "full_pipeline"
    print(f"  Active pipeline: {current_pipeline}\n")

    while True:
        try:
            user_input = input(f"\n[You] Task ({current_pipeline}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! ")
            sys.exit(0)

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "exit"):
            print("\nGoodbye! ")
            sys.exit(0)

        if cmd == "pipelines":
            print("\n  Available pipelines:")
            for name in orchestrator.list_pipelines():
                marker = " <- active" if name == current_pipeline else ""
                print(f"    - {name}{marker}")
            continue

        if cmd.startswith("use "):
            name = cmd.split(" ", 1)[1].strip()
            if name in orchestrator.list_pipelines():
                current_pipeline = name
                print(f"  OK Switched to pipeline: {current_pipeline}")
            else:
                available = ", ".join(orchestrator.list_pipelines())
                print(f"  Unknown pipeline '{name}'. Available: {available}")
            continue

        if cmd.startswith("mode "):
            mode = cmd.split(" ", 1)[1].strip()
            if mode == "single":
                return  # Signal to switch back
            elif mode == "pipeline":
                print("  Already in pipeline mode.")
                continue
            else:
                print(f"  Unknown mode '{mode}'. Use 'single' or 'pipeline'.")
                continue

        # --- Run the pipeline -----------------------------
        try:
            results = orchestrator.run(current_pipeline, user_input)

            print(f"\n{'='*60}")
            print("  PIPELINE COMPLETE -- Results Summary")
            print(f"{'='*60}")

            for stage, content in results.items():
                print(f"\n{'-'*40}")
                print(f"  * {stage.upper().replace('_', ' ')}")
                print(f"{'-'*40}")
                # Show first 500 chars of each stage
                preview = content[:500]
                if len(content) > 500:
                    preview += f"\n  ... [{len(content) - 500} more chars]"
                print(f"\n{preview}")

            print(f"\n{'='*60}")

        except Exception as e:
            print(f"\nERROR: Error: {e}")
            print("   (Make sure ANTHROPIC_API_KEY is set)")


def main():
    agents = create_agents()
    orchestrator = Orchestrator(verbose=True)
    current_agent = "assistant"

    print_banner()

    while True:
        result = run_single_mode(agents, current_agent)

        if result == "__pipeline__":
            run_pipeline_mode(orchestrator)
            # When pipeline mode returns, go back to single mode
            print("\n  Returning to single-agent mode...")
            continue
        else:
            current_agent = result
            print(f"\n  OK Switched to {agents[current_agent].name}")


if __name__ == "__main__":
    check_api_key()  # Exits with a helpful message if key is missing
    main()
