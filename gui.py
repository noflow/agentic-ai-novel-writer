"""
Agentic AI -- GUI Interface

Features:
  - Chat with AI agents (single or pipeline mode)
  - Live console log showing agent activity
  - Animated spinner + progress bar while thinking
  - File attach button + drag-and-drop zone
  - Elapsed time counter

Run with:  python gui.py
"""

from __future__ import annotations
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import queue
import json
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import check_api_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_ascii(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


class GUIStream:
    def __init__(self, log_queue: queue.Queue, original_stream):
        self.log_queue = log_queue
        self.original = original_stream

    def write(self, text: str):
        if text.strip():
            self.log_queue.put(text)
        try:
            self.original.write(text)
        except (UnicodeEncodeError, UnicodeDecodeError):
            self.original.write(_safe_ascii(text))

    def flush(self):
        self.original.flush()


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
COLORS = {
    "bg_dark":       "#1a1b26",
    "bg_panel":      "#24283b",
    "bg_input":      "#1f2335",
    "bg_console":    "#1a1b26",
    "bg_drop":       "#1f2335",
    "bg_drop_hover": "#292e42",
    "text_primary":  "#c0caf5",
    "text_secondary":"#565f89",
    "text_user":     "#ffffff",
    "text_agent":    "#c0caf5",
    "text_console":  "#9ece6a",
    "text_tool":     "#e0af68",
    "text_error":    "#f7768e",
    "text_info":     "#7aa2f7",
    "accent":        "#7aa2f7",
    "accent_hover":  "#89b4fa",
    "border":        "#3b4261",
    "progress_bg":   "#1f2335",
    "progress_fill": "#7aa2f7",
    "file_tag_bg":   "#292e42",
    "file_tag_fg":   "#9ece6a",
}

SPINNER_FRAMES = ["|  ", "/  ", "-  ", "\\  "]

# File types the agent can read
READABLE_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".py", ".js", ".html", ".css",
    ".xml", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".log",
    ".bat", ".sh", ".sql", ".r", ".java", ".c", ".cpp", ".h",
}


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------
class AgenticGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Agentic AI")
        self.root.geometry("1200x750")
        self.root.minsize(900, 500)
        self.root.configure(bg=COLORS["bg_dark"])

        self.log_queue: queue.Queue = queue.Queue()
        self.response_queue: queue.Queue = queue.Queue()

        self.current_agent_name = "assistant"
        self.agents = {}
        self.orchestrator = None
        self.is_running = False
        self.current_mode = "single"
        self.current_pipeline = "full_pipeline"

        # Attached files (list of file paths)
        self.attached_files: list[str] = []

        # Spinner state
        self.spinner_idx = 0
        self.thinking_start = 0.0
        self.thinking_step = ""
        self.progress_pulse_pos = 0

        sys.stdout = GUIStream(self.log_queue, sys.__stdout__)

        self._build_styles()
        self._build_layout()
        self._init_agents()
        self._poll_queues()

    # -------------------------------------------------------------------
    # Styles
    # -------------------------------------------------------------------
    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=COLORS["bg_dark"])
        style.configure("Panel.TFrame", background=COLORS["bg_panel"])
        style.configure("Title.TLabel", background=COLORS["bg_panel"],
                        foreground=COLORS["text_primary"], font=("Consolas", 11, "bold"))
        style.configure("Status.TLabel", background=COLORS["bg_dark"],
                        foreground=COLORS["text_secondary"], font=("Consolas", 9))
        style.configure("Spinner.TLabel", background=COLORS["bg_panel"],
                        foreground=COLORS["accent"], font=("Consolas", 10, "bold"))
        style.configure("Timer.TLabel", background=COLORS["bg_panel"],
                        foreground=COLORS["text_tool"], font=("Consolas", 9))
        style.configure("Step.TLabel", background=COLORS["bg_panel"],
                        foreground=COLORS["text_info"], font=("Consolas", 9))

    # -------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------
    def _build_layout(self):
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ---- Top bar ----
        topbar = ttk.Frame(main, style="Dark.TFrame")
        topbar.pack(fill=tk.X, pady=(0, 8))

        mode_frame = ttk.Frame(topbar, style="Dark.TFrame")
        mode_frame.pack(side=tk.LEFT)
        ttk.Label(mode_frame, text="Mode:", style="Status.TLabel").pack(side=tk.LEFT, padx=(0, 4))

        self.mode_var = tk.StringVar(value="single")
        for val, label in [("single", "Single Agent"), ("pipeline", "Pipeline")]:
            rb = tk.Radiobutton(
                mode_frame, text=label, variable=self.mode_var, value=val,
                command=self._on_mode_change,
                bg=COLORS["bg_dark"], fg=COLORS["text_primary"],
                selectcolor=COLORS["bg_panel"], activebackground=COLORS["bg_dark"],
                activeforeground=COLORS["text_primary"], font=("Consolas", 9),
                highlightthickness=0, bd=0,
            )
            rb.pack(side=tk.LEFT, padx=4)

        self.selector_frame = ttk.Frame(topbar, style="Dark.TFrame")
        self.selector_frame.pack(side=tk.LEFT, padx=(20, 0))
        self.selector_label = ttk.Label(self.selector_frame, text="Agent:", style="Status.TLabel")
        self.selector_label.pack(side=tk.LEFT, padx=(0, 4))
        self.selector_var = tk.StringVar(value="assistant")
        self.selector_combo = ttk.Combobox(
            self.selector_frame, textvariable=self.selector_var,
            values=["assistant", "researcher", "writer", "critic", "story_director", "novel_writer", "story_critic", "humanizer", "summarizer", "formatter", "continuity_checker"],
            state="readonly", width=16, font=("Consolas", 9),
        )
        self.selector_combo.pack(side=tk.LEFT)
        self.selector_combo.bind("<<ComboboxSelected>>", self._on_selector_change)

        # Provider selector
        provider_frame = ttk.Frame(topbar, style="Dark.TFrame")
        provider_frame.pack(side=tk.LEFT, padx=(20, 0))
        ttk.Label(provider_frame, text="LLM:", style="Status.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        self.provider_var = tk.StringVar(value="anthropic")
        self.provider_combo = ttk.Combobox(
            provider_frame, textvariable=self.provider_var,
            values=["anthropic"],  # Updated in _init_agents with available providers
            state="readonly", width=12, font=("Consolas", 9),
        )
        self.provider_combo.pack(side=tk.LEFT)
        self.provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(topbar, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT)

        # ---- Split pane ----
        paned = tk.PanedWindow(main, orient=tk.HORIZONTAL, bg=COLORS["bg_dark"],
                               sashwidth=6, sashrelief=tk.FLAT, bd=0, opaqueresize=True)
        paned.pack(fill=tk.BOTH, expand=True)

        # -- LEFT: Chat --
        chat_frame = ttk.Frame(paned, style="Panel.TFrame")
        paned.add(chat_frame, minsize=400, width=650)

        chat_header = ttk.Frame(chat_frame, style="Panel.TFrame")
        chat_header.pack(fill=tk.X, padx=12, pady=(12, 4))
        ttk.Label(chat_header, text="CHAT", style="Title.TLabel").pack(side=tk.LEFT)
        tk.Button(chat_header, text="Clear", command=self._clear_chat,
                  bg=COLORS["bg_panel"], fg=COLORS["text_secondary"],
                  activebackground=COLORS["bg_dark"], activeforeground=COLORS["text_primary"],
                  font=("Consolas", 9), bd=0, cursor="hand2", highlightthickness=0).pack(side=tk.RIGHT)

        # Send conversation to pipeline button
        self.pipeline_btn = tk.Button(
            chat_header, text=">> Pipeline", command=self._send_to_pipeline,
            bg=COLORS["bg_panel"], fg=COLORS["text_tool"],
            activebackground=COLORS["bg_dark"], activeforeground=COLORS["text_tool"],
            font=("Consolas", 9, "bold"), bd=0, cursor="hand2", highlightthickness=0,
        )
        self.pipeline_btn.pack(side=tk.RIGHT, padx=(0, 12))

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, state=tk.DISABLED,
            bg=COLORS["bg_panel"], fg=COLORS["text_primary"],
            font=("Consolas", 10), bd=0, padx=12, pady=8,
            insertbackground=COLORS["text_primary"],
            selectbackground=COLORS["accent"], highlightthickness=0,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.chat_display.tag_configure("user_name", foreground=COLORS["accent"], font=("Consolas", 10, "bold"))
        self.chat_display.tag_configure("user_msg", foreground=COLORS["text_user"])
        self.chat_display.tag_configure("agent_name", foreground=COLORS["text_tool"], font=("Consolas", 10, "bold"))
        self.chat_display.tag_configure("agent_msg", foreground=COLORS["text_agent"])
        self.chat_display.tag_configure("system_msg", foreground=COLORS["text_secondary"], font=("Consolas", 9, "italic"))
        self.chat_display.tag_configure("error_msg", foreground=COLORS["text_error"])

        # ---- Thinking indicator ----
        self.thinking_frame = ttk.Frame(chat_frame, style="Panel.TFrame")

        self.spinner_label = ttk.Label(self.thinking_frame, text="", style="Spinner.TLabel")
        self.spinner_label.pack(side=tk.LEFT, padx=(12, 4))
        self.step_label = ttk.Label(self.thinking_frame, text="Thinking...", style="Step.TLabel")
        self.step_label.pack(side=tk.LEFT, padx=(0, 12))
        self.timer_label = ttk.Label(self.thinking_frame, text="0s", style="Timer.TLabel")
        self.timer_label.pack(side=tk.RIGHT, padx=(0, 12))
        self.progress_canvas = tk.Canvas(self.thinking_frame, height=4, bg=COLORS["progress_bg"],
                                         highlightthickness=0, bd=0)
        self.progress_canvas.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 8))

        # ---- Attached files display ----
        self.files_frame = ttk.Frame(chat_frame, style="Panel.TFrame")
        # Only packed when files are attached

        # ---- Input area ----
        input_frame = ttk.Frame(chat_frame, style="Panel.TFrame")
        input_frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        # Attach file button
        self.attach_btn = tk.Button(
            input_frame, text="[+]", command=self._attach_file,
            bg=COLORS["bg_input"], fg=COLORS["text_secondary"],
            activebackground=COLORS["bg_panel"], activeforeground=COLORS["text_primary"],
            font=("Consolas", 10, "bold"), bd=0, padx=8, pady=8,
            cursor="hand2", highlightthickness=0,
        )
        self.attach_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.input_field = tk.Text(
            input_frame, height=3, wrap=tk.WORD,
            bg=COLORS["bg_input"], fg=COLORS["text_primary"],
            font=("Consolas", 10), bd=0, padx=8, pady=8,
            insertbackground=COLORS["text_primary"],
            selectbackground=COLORS["accent"],
            highlightthickness=1, highlightcolor=COLORS["border"],
            highlightbackground=COLORS["border"],
        )
        self.input_field.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 8))
        self.input_field.bind("<Return>", self._on_enter)
        self.input_field.bind("<Shift-Return>", lambda e: None)

        self.send_btn = tk.Button(
            input_frame, text="Send", command=self._send_message,
            bg=COLORS["accent"], fg="#ffffff",
            activebackground=COLORS["accent_hover"], activeforeground="#ffffff",
            font=("Consolas", 10, "bold"), bd=0, padx=16, pady=8,
            cursor="hand2", highlightthickness=0,
        )
        self.send_btn.pack(side=tk.RIGHT)

        # -- RIGHT: Console --
        console_frame = ttk.Frame(paned, style="Panel.TFrame")
        paned.add(console_frame, minsize=300, width=550)

        console_header = ttk.Frame(console_frame, style="Panel.TFrame")
        console_header.pack(fill=tk.X, padx=12, pady=(12, 4))
        ttk.Label(console_header, text="CONSOLE LOG", style="Title.TLabel").pack(side=tk.LEFT)
        tk.Button(console_header, text="Clear", command=self._clear_console,
                  bg=COLORS["bg_panel"], fg=COLORS["text_secondary"],
                  activebackground=COLORS["bg_dark"], activeforeground=COLORS["text_primary"],
                  font=("Consolas", 9), bd=0, cursor="hand2", highlightthickness=0).pack(side=tk.RIGHT)

        self.console_display = scrolledtext.ScrolledText(
            console_frame, wrap=tk.WORD, state=tk.DISABLED,
            bg=COLORS["bg_console"], fg=COLORS["text_console"],
            font=("Consolas", 9), bd=0, padx=12, pady=8,
            insertbackground=COLORS["text_console"],
            selectbackground=COLORS["accent"], highlightthickness=0,
        )
        self.console_display.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.console_display.tag_configure("tool", foreground=COLORS["text_tool"])
        self.console_display.tag_configure("info", foreground=COLORS["text_info"])
        self.console_display.tag_configure("error", foreground=COLORS["text_error"])
        self.console_display.tag_configure("success", foreground=COLORS["text_console"])
        self.console_display.tag_configure("dim", foreground=COLORS["text_secondary"])

        # ---- Setup drag-and-drop (Windows native via tkdnd or fallback) ----
        self._setup_drag_drop()

    # -------------------------------------------------------------------
    # Drag-and-drop
    # -------------------------------------------------------------------
    def _setup_drag_drop(self):
        """Try to enable native drag-and-drop. Falls back gracefully."""
        try:
            # Try tkdnd (if installed: pip install tkinterdnd2)
            from tkinterdnd2 import DND_FILES, TkinterDnD
            # Won't work unless root is TkinterDnD.Tk -- skip for now
            raise ImportError("Using fallback")
        except ImportError:
            pass

        # Fallback: paste file paths with Ctrl+Shift+V
        self.input_field.bind("<Control-Shift-KeyPress-V>", self._paste_file_path)

        # Also accept files dropped as text (some terminals do this)
        self.root.bind("<Control-o>", lambda e: self._attach_file())

    def _paste_file_path(self, event):
        """Handle Ctrl+Shift+V to paste and attach a file path from clipboard."""
        try:
            path = self.root.clipboard_get().strip().strip("\"'")
            if os.path.isfile(path):
                self._add_attached_file(path)
                return "break"
        except Exception:
            pass

    # -------------------------------------------------------------------
    # File attachment
    # -------------------------------------------------------------------
    def _attach_file(self):
        """Open file picker to attach files."""
        filepaths = filedialog.askopenfilenames(
            title="Attach files",
            filetypes=[
                ("Text files", "*.txt *.md *.csv *.json *.log"),
                ("Code files", "*.py *.js *.html *.css *.java *.c *.cpp"),
                ("Config files", "*.yaml *.yml *.toml *.ini *.cfg *.xml"),
                ("All files", "*.*"),
            ],
        )
        for fp in filepaths:
            self._add_attached_file(fp)

    def _add_attached_file(self, filepath: str):
        """Add a file to the attached list and update the UI."""
        if filepath in self.attached_files:
            return  # Already attached

        # Check if readable
        ext = Path(filepath).suffix.lower()
        if ext not in READABLE_EXTENSIONS:
            self._chat_system(
                f"Warning: {Path(filepath).name} may not be a text file. "
                "The agent will try to read it anyway."
            )

        self.attached_files.append(filepath)
        self._update_files_display()
        self._log_console(f"Attached: {filepath}", "info")

    def _remove_attached_file(self, filepath: str):
        """Remove a file from the attached list."""
        if filepath in self.attached_files:
            self.attached_files.remove(filepath)
            self._update_files_display()

    def _update_files_display(self):
        """Show/hide the attached files bar."""
        # Clear existing widgets
        for widget in self.files_frame.winfo_children():
            widget.destroy()

        if not self.attached_files:
            self.files_frame.pack_forget()
            return

        self.files_frame.pack(fill=tk.X, padx=12, pady=(0, 4))

        tk.Label(
            self.files_frame, text="Files:",
            bg=COLORS["bg_panel"], fg=COLORS["text_secondary"],
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))

        for fp in self.attached_files:
            name = Path(fp).name
            tag_frame = tk.Frame(self.files_frame, bg=COLORS["file_tag_bg"],
                                 padx=6, pady=2)
            tag_frame.pack(side=tk.LEFT, padx=2)

            tk.Label(
                tag_frame, text=name,
                bg=COLORS["file_tag_bg"], fg=COLORS["file_tag_fg"],
                font=("Consolas", 9),
            ).pack(side=tk.LEFT)

            # X button to remove
            tk.Button(
                tag_frame, text=" x", command=lambda f=fp: self._remove_attached_file(f),
                bg=COLORS["file_tag_bg"], fg=COLORS["text_error"],
                activebackground=COLORS["file_tag_bg"], activeforeground=COLORS["text_error"],
                font=("Consolas", 8, "bold"), bd=0, cursor="hand2",
                highlightthickness=0, padx=0, pady=0,
            ).pack(side=tk.LEFT)

        # Clear all button
        tk.Button(
            self.files_frame, text="Clear all",
            command=self._clear_attached_files,
            bg=COLORS["bg_panel"], fg=COLORS["text_secondary"],
            activebackground=COLORS["bg_dark"], activeforeground=COLORS["text_primary"],
            font=("Consolas", 8), bd=0, cursor="hand2", highlightthickness=0,
        ).pack(side=tk.RIGHT, padx=(8, 0))

    def _clear_attached_files(self):
        self.attached_files.clear()
        self._update_files_display()

    def _build_file_context(self) -> str:
        """Read attached files and build context string for the agent."""
        if not self.attached_files:
            return ""

        parts = []
        for fp in self.attached_files:
            name = Path(fp).name
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if len(content) > 15_000:
                    content = content[:15_000] + "\n... [truncated]"
                parts.append(f"=== FILE: {name} ===\n{content}\n=== END FILE ===")
                self._log_console(f"  Read {name} ({len(content)} chars)", "dim")
            except Exception as e:
                parts.append(f"=== FILE: {name} ===\nError reading: {e}\n=== END FILE ===")
                self._log_console(f"  Error reading {name}: {e}", "error")

        return "\n\n".join(parts)

    # -------------------------------------------------------------------
    # Thinking animation
    # -------------------------------------------------------------------
    def _show_thinking(self):
        self.thinking_start = time.time()
        self.thinking_step = "Thinking..."
        self.spinner_idx = 0
        self.progress_pulse_pos = 0
        self.thinking_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        self._animate_thinking()

    def _hide_thinking(self):
        self.thinking_frame.pack_forget()

    def _animate_thinking(self):
        if not self.is_running:
            return
        self.spinner_idx = (self.spinner_idx + 1) % len(SPINNER_FRAMES)
        self.spinner_label.config(text=SPINNER_FRAMES[self.spinner_idx])
        self.step_label.config(text=self.thinking_step)

        elapsed = time.time() - self.thinking_start
        if elapsed < 60:
            self.timer_label.config(text=f"{elapsed:.0f}s")
        else:
            self.timer_label.config(text=f"{int(elapsed//60)}m {int(elapsed%60)}s")

        canvas = self.progress_canvas
        canvas.delete("all")
        w = canvas.winfo_width()
        if w > 1:
            pulse_w = max(60, w // 5)
            self.progress_pulse_pos = (self.progress_pulse_pos + 3) % (w + pulse_w)
            x1 = self.progress_pulse_pos - pulse_w
            x2 = self.progress_pulse_pos
            if x1 > w:
                self.progress_pulse_pos = 0
            canvas.create_rectangle(x1, 0, x2, 4, fill=COLORS["progress_fill"], width=0)

        self.root.after(100, self._animate_thinking)

    def _update_thinking_step(self, step: str):
        self.thinking_step = step

    # -------------------------------------------------------------------
    # Agent init
    # -------------------------------------------------------------------
    def _init_agents(self):
        self._log_console("Initializing agents...", "info")

        def init():
            try:
                from agent import Agent
                from tools import create_default_registry
                from specialists import create_researcher, create_writer, create_critic
                from specialists import create_story_director, create_novel_writer, create_story_critic, create_humanizer, create_summarizer, create_formatter, create_continuity_checker, create_story_tracker
                from orchestrator import Orchestrator
                from config import list_available_providers, PROVIDER

                self.agents = {
                    "assistant": Agent(name="Assistant", tool_registry=create_default_registry()),
                    "researcher": create_researcher(),
                    "writer": create_writer(),
                    "critic": create_critic(),
                    "story_director": create_story_director(),
                    "novel_writer": create_novel_writer(),
                    "story_critic": create_story_critic(),
                    "humanizer": create_humanizer(),
                    "summarizer": create_summarizer(),
                    "formatter": create_formatter(),
                    "continuity_checker": create_continuity_checker(),
                    "story_tracker": create_story_tracker(),
                }
                self.orchestrator = Orchestrator(verbose=True)

                # Detect available providers
                available = list_available_providers()
                if available:
                    self.provider_combo.config(values=available)
                    current = PROVIDER if PROVIDER in available else available[0]
                    self.provider_var.set(current)
                    self._log_console(f"Available LLM providers: {', '.join(available)}", "info")
                    self._log_console(f"Active provider: {current}", "info")

                self._log_console("All agents ready!", "success")
                self._log_console(f"Active: {self.agents[self.current_agent_name]}", "dim")
                self._chat_system(
                    "Ready! Type a message or click [+] to attach files. "
                    "Use the LLM dropdown to switch providers."
                )
            except Exception as e:
                self._log_console(f"ERROR: {e}", "error")
                self._chat_system(f"Error initializing: {e}")

        threading.Thread(target=init, daemon=True).start()

    def _rebuild_agents_with_provider(self, provider_name: str):
        """Recreate all agents using a different LLM provider."""
        try:
            from agent import Agent
            from tools import create_default_registry
            from specialists import (create_researcher, create_writer, create_critic,
                                     create_story_director, create_novel_writer, create_story_critic, create_humanizer, create_summarizer, create_formatter, create_continuity_checker, create_story_tracker)
            from orchestrator import Orchestrator
            from config import create_provider

            provider = create_provider(provider_name)
            model = provider.default_model

            self._log_console(f"Switching to {provider_name} ({model})...", "info")

            self.agents = {
                "assistant": Agent(name="Assistant", tool_registry=create_default_registry(),
                                   provider=provider, model=model),
                "researcher": create_researcher(provider=provider, model=model),
                "writer": create_writer(provider=provider, model=model),
                "critic": create_critic(provider=provider, model=model),
                "story_director": create_story_director(provider=provider, model=model),
                "novel_writer": create_novel_writer(provider=provider, model=model),
                "story_critic": create_story_critic(provider=provider, model=model),
                "humanizer": create_humanizer(provider=provider, model=model),
                "summarizer": create_summarizer(provider=provider, model=model),
                "formatter": create_formatter(provider=provider, model=model),
                "continuity_checker": create_continuity_checker(provider=provider, model=model),
                "story_tracker": create_story_tracker(provider=provider, model=model),
            }
            self.orchestrator = Orchestrator(verbose=True, provider=provider, model=model)
            self._log_console(f"All agents now using {provider_name} ({model})", "success")
            self._chat_system(f"Switched to {provider_name} ({model}). Conversations reset.")

        except Exception as e:
            self._log_console(f"Error switching provider: {e}", "error")
            self._chat_system(f"Error: {e}")

    # -------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------
    def _on_enter(self, event):
        if not event.state & 0x1:
            self._send_message()
            return "break"

    def _on_provider_change(self, event=None):
        """Handle provider dropdown change."""
        provider = self.provider_var.get()
        if self.is_running:
            self._chat_system("Wait for the current task to finish first.")
            return
        threading.Thread(
            target=self._rebuild_agents_with_provider,
            args=(provider,), daemon=True
        ).start()

    def _on_mode_change(self):
        mode = self.mode_var.get()
        self.current_mode = mode
        if mode == "single":
            self.selector_label.config(text="Agent:")
            self.selector_combo.config(values=["assistant", "researcher", "writer", "critic", "story_director", "novel_writer", "story_critic", "humanizer", "summarizer", "formatter", "continuity_checker", "story_tracker"])
            self.selector_var.set(self.current_agent_name)
            self._chat_system(f"Single-agent mode. Talking to {self.current_agent_name}.")
        else:
            self.selector_label.config(text="Pipeline:")
            self.selector_combo.config(values=["research_only", "write_only", "research_and_write", "full_pipeline", "novel_plan", "novel_chapter", "novel_start", "write_novel"])
            self.selector_var.set(self.current_pipeline)
            self._chat_system(f"Pipeline mode. Using {self.current_pipeline}.")
        self._log_console(f"Mode: {mode}", "info")

    def _on_selector_change(self, event=None):
        value = self.selector_var.get()
        if self.current_mode == "single":
            self.current_agent_name = value
            if value in self.agents:
                self.agents[value].reset()
            self._chat_system(f"Switched to {value}. Conversation reset.")
            self._log_console(f"Agent: {value}", "info")
        else:
            self.current_pipeline = value
            self._chat_system(f"Pipeline: {value}.")
            self._log_console(f"Pipeline: {value}", "info")

    def _send_to_pipeline(self):
        """Take the current single-agent conversation and send it to a pipeline."""
        if self.is_running:
            self._chat_system("Wait for the current task to finish first.")
            return

        # Get the current agent's conversation history
        agent = self.agents.get(self.current_agent_name)
        if not agent or not agent.conversation_history:
            self._chat_system("Nothing to send -- start a conversation first.")
            return

        # Build a summary of the conversation to pass to the pipeline
        context_parts = []
        for msg in agent.conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Extract text from content blocks
                text_bits = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_bits.append(block.get("text", ""))
                    elif isinstance(block, dict) and block.get("type") == "tool_result":
                        text_bits.append(f"[tool result: {block.get('content', '')[:200]}]")
                text = "\n".join(text_bits)
            else:
                text = str(content)

            if text.strip():
                label = "User" if role == "user" else "Agent"
                context_parts.append(f"{label}: {text[:500]}")

        conversation_summary = "\n\n".join(context_parts[-6:])  # Last 6 exchanges

        # Show pipeline picker dialog
        self._show_pipeline_picker(conversation_summary)

    def _show_pipeline_picker(self, context: str):
        """Show a small dialog to pick which pipeline to run."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Send to Pipeline")
        dialog.geometry("400x300")
        dialog.configure(bg=COLORS["bg_panel"])
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog, text="Send conversation to pipeline",
            bg=COLORS["bg_panel"], fg=COLORS["text_primary"],
            font=("Consolas", 11, "bold"),
        ).pack(pady=(16, 8))

        tk.Label(
            dialog, text="Add instructions (optional):",
            bg=COLORS["bg_panel"], fg=COLORS["text_secondary"],
            font=("Consolas", 9),
        ).pack(anchor=tk.W, padx=16)

        instructions = tk.Text(
            dialog, height=4, wrap=tk.WORD,
            bg=COLORS["bg_input"], fg=COLORS["text_primary"],
            font=("Consolas", 10), bd=0, padx=8, pady=8,
            insertbackground=COLORS["text_primary"],
            highlightthickness=1, highlightcolor=COLORS["border"],
            highlightbackground=COLORS["border"],
        )
        instructions.pack(fill=tk.X, padx=16, pady=8)
        instructions.insert("1.0", "Take this conversation and expand it into a polished document.")

        tk.Label(
            dialog, text="Pipeline:",
            bg=COLORS["bg_panel"], fg=COLORS["text_secondary"],
            font=("Consolas", 9),
        ).pack(anchor=tk.W, padx=16)

        pipeline_var = tk.StringVar(value="full_pipeline")
        pipeline_combo = ttk.Combobox(
            dialog, textvariable=pipeline_var,
            values=["research_and_write", "full_pipeline", "write_only"],
            state="readonly", width=25, font=("Consolas", 10),
        )
        pipeline_combo.pack(padx=16, pady=4, anchor=tk.W)

        def run_it():
            pipeline = pipeline_var.get()
            extra = instructions.get("1.0", tk.END).strip()
            dialog.destroy()

            # Build the full task
            task = f"{extra}\n\n=== CONVERSATION CONTEXT ===\n{context}"

            # Switch to pipeline mode in the UI
            self.current_mode = "pipeline"
            self.current_pipeline = pipeline
            self.mode_var.set("pipeline")
            self.selector_label.config(text="Pipeline:")
            self.selector_combo.config(values=["research_only", "write_only", "research_and_write", "full_pipeline", "novel_plan", "novel_chapter", "novel_start", "write_novel"])
            self.selector_var.set(pipeline)

            self._chat_system(f"Sending conversation to {pipeline} pipeline...")
            self._log_console(f"\nSending to pipeline: {pipeline}", "info")

            # Run it
            self.is_running = True
            self.status_var.set("Working...")
            self.send_btn.config(state=tk.DISABLED, bg=COLORS["border"])
            self._show_thinking()

            threading.Thread(target=self._run_agent, args=(task,), daemon=True).start()

        btn_frame = tk.Frame(dialog, bg=COLORS["bg_panel"])
        btn_frame.pack(fill=tk.X, padx=16, pady=(8, 16))

        tk.Button(
            btn_frame, text="Cancel", command=dialog.destroy,
            bg=COLORS["bg_panel"], fg=COLORS["text_secondary"],
            activebackground=COLORS["bg_dark"], activeforeground=COLORS["text_primary"],
            font=("Consolas", 10), bd=0, cursor="hand2", highlightthickness=0,
            padx=16, pady=6,
        ).pack(side=tk.LEFT)

        tk.Button(
            btn_frame, text="Run Pipeline", command=run_it,
            bg=COLORS["accent"], fg="#ffffff",
            activebackground=COLORS["accent_hover"], activeforeground="#ffffff",
            font=("Consolas", 10, "bold"), bd=0, cursor="hand2", highlightthickness=0,
            padx=16, pady=6,
        ).pack(side=tk.RIGHT)

    def _send_message(self):
        if self.is_running:
            return

        text = self.input_field.get("1.0", tk.END).strip()
        if not text and not self.attached_files:
            return

        self.input_field.delete("1.0", tk.END)

        # Build the full message with file context
        file_context = self._build_file_context()
        if file_context:
            file_names = [Path(f).name for f in self.attached_files]
            display_text = text or f"[Attached: {', '.join(file_names)}]"
            self._chat_user(display_text)
            full_message = f"{file_context}\n\nUser request: {text}" if text else file_context
            self._clear_attached_files()
        else:
            self._chat_user(text)
            full_message = text

        self.is_running = True
        self.status_var.set("Working...")
        self.send_btn.config(state=tk.DISABLED, bg=COLORS["border"])
        self._show_thinking()

        self._log_console(f"\n--- New request ---", "dim")
        self._log_console(f"User: {text[:100]}{'...' if len(text)>100 else ''}", "info")

        threading.Thread(target=self._run_agent, args=(full_message,), daemon=True).start()

    def _run_agent(self, user_message: str):
        try:
            if self.current_mode == "single":
                agent = self.agents.get(self.current_agent_name)
                if not agent:
                    self.response_queue.put(("error", "Agent not initialized"))
                    return
                response = agent.run(user_message)
                self.response_queue.put(("response", response))
            else:
                if not self.orchestrator:
                    self.response_queue.put(("error", "Orchestrator not initialized"))
                    return
                results = self.orchestrator.run(self.current_pipeline, user_message)
                output_parts = []
                for stage, content in results.items():
                    header = stage.upper().replace("_", " ")
                    output_parts.append(f"=== {header} ===\n{content}")
                self.response_queue.put(("response", "\n\n".join(output_parts)))
        except Exception as e:
            self.response_queue.put(("error", _safe_ascii(str(e))))

    # -------------------------------------------------------------------
    # Queue polling
    # -------------------------------------------------------------------
    def _poll_queues(self):
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                tag = "dim"
                ml = msg.lower() if isinstance(msg, str) else ""
                if "-> calling" in ml or "tool call" in ml:
                    tag = "tool"
                    self._update_thinking_step(msg.strip()[:60])
                elif "<- result" in ml:
                    tag = "success"
                    self._update_thinking_step("Processing result...")
                elif "error" in ml:
                    tag = "error"
                elif "handing off" in ml:
                    tag = "info"
                    self._update_thinking_step(msg.strip()[:60])
                elif "turn " in ml:
                    tag = "info"
                    self._update_thinking_step(msg.strip()[:60])
                elif "final answer" in ml:
                    self._update_thinking_step("Composing answer...")
                elif "orchestrator" in ml or "pipeline" in ml:
                    tag = "info"

                self.console_display.config(state=tk.NORMAL)
                self.console_display.insert(tk.END, msg + "\n", tag)
                self.console_display.see(tk.END)
                self.console_display.config(state=tk.DISABLED)
            except queue.Empty:
                break

        while not self.response_queue.empty():
            try:
                msg_type, content = self.response_queue.get_nowait()
                if msg_type == "response":
                    name = self.current_agent_name if self.current_mode == "single" else "Pipeline"
                    self._chat_agent(name, content)
                    elapsed = time.time() - self.thinking_start
                    self._log_console(f"--- Complete ({elapsed:.1f}s) ---\n", "dim")
                elif msg_type == "error":
                    self._chat_error(content)
                    self._log_console(f"ERROR: {content}", "error")

                self.is_running = False
                self._hide_thinking()
                self.status_var.set("Ready")
                self.send_btn.config(state=tk.NORMAL, bg=COLORS["accent"])
            except queue.Empty:
                break

        self.root.after(50, self._poll_queues)

    # -------------------------------------------------------------------
    # Chat helpers
    # -------------------------------------------------------------------
    def _chat_user(self, text: str):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\nYou:\n", "user_name")
        self.chat_display.insert(tk.END, text + "\n", "user_msg")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _chat_agent(self, name: str, text: str):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n{name}:\n", "agent_name")
        self.chat_display.insert(tk.END, text + "\n", "agent_msg")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _chat_system(self, text: str):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n[{text}]\n", "system_msg")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _chat_error(self, text: str):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\nError: {text}\n", "error_msg")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _log_console(self, text: str, tag: str = "dim"):
        self.console_display.config(state=tk.NORMAL)
        self.console_display.insert(tk.END, text + "\n", tag)
        self.console_display.see(tk.END)
        self.console_display.config(state=tk.DISABLED)

    def _clear_chat(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        if self.current_mode == "single" and self.current_agent_name in self.agents:
            self.agents[self.current_agent_name].reset()
        self._chat_system("Chat cleared.")

    def _clear_console(self):
        self.console_display.config(state=tk.NORMAL)
        self.console_display.delete("1.0", tk.END)
        self.console_display.config(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    check_api_key()
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = AgenticGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
