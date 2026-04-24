"""
Orchestrator with bulletproof chapter sequencing.

- Extracts actual chapter numbers from filenames (not just counting files)
- Detects gaps (ch1 exists, ch3 exists, ch2 missing -> writes ch2)
- Continuity Checker agent verifies chapter completeness after writing
- Supports writing a specific chapter number
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from specialists import (
    create_researcher, create_writer, create_critic,
    create_story_director, create_novel_writer, create_story_critic,
    create_humanizer, create_summarizer, create_formatter,
    create_continuity_checker, create_story_tracker,
)
from agent import Agent
from providers import BaseLLMProvider


def _get_output_dir() -> Path:
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    return out


def _extract_chapter_num(filename: str) -> int | None:
    """Extract chapter number from a filename like 'novel_ch3_title.txt' or 'chapter_3.txt'."""
    patterns = [
        r'ch(\d+)',
        r'chapter[_\s]*(\d+)',
        r'chapter[_\s]*(\d+)',
    ]
    lower = filename.lower()
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            return int(match.group(1))
    return None


def _scan_novel_files():
    """Scan output folder with actual chapter number detection."""
    output_dir = _get_output_dir()
    all_files = sorted([f.name for f in output_dir.iterdir() if f.is_file()])

    outline_files = [f for f in all_files if "outline" in f.lower() and f.endswith(".txt")]
    story_so_far = [f for f in all_files if "story_so_far" in f.lower()]

    # Chapter files: .txt, contains ch/chapter, excludes meta files
    chapter_files = [f for f in all_files
                     if f.endswith(".txt")
                     and ("ch" in f.lower() or "chapter" in f.lower())
                     and not any(x in f.lower() for x in
                                ["summary", "review", "formatted", "story_so_far", "outline"])]

    # Extract actual chapter numbers
    chapter_map = {}  # {chapter_num: filename}
    for f in chapter_files:
        num = _extract_chapter_num(f)
        if num is not None:
            chapter_map[num] = f

    existing_nums = sorted(chapter_map.keys())

    # Find gaps
    gaps = []
    if existing_nums:
        for i in range(1, max(existing_nums) + 1):
            if i not in existing_nums:
                gaps.append(i)

    # Determine next chapter to write
    if gaps:
        next_num = gaps[0]  # Fill the first gap
    elif existing_nums:
        next_num = max(existing_nums) + 1
    else:
        next_num = 1

    return {
        "all_files": all_files,
        "outline": outline_files[0] if outline_files else None,
        "story_so_far": story_so_far[0] if story_so_far else None,
        "chapters": chapter_files,
        "chapter_map": chapter_map,
        "existing_nums": existing_nums,
        "gaps": gaps,
        "next_num": next_num,
    }


class Orchestrator:
    def __init__(self, verbose=True, provider=None, model=None):
        self.verbose = verbose
        self._provider = provider
        self._model = model

        self.researcher = create_researcher(provider=provider, model=model)
        self.writer = create_writer(provider=provider, model=model)
        self.critic = create_critic(provider=provider, model=model)

        self._story_director = None
        self._novel_writer = None
        self._story_critic = None
        self._humanizer = None
        self._summarizer = None
        self._formatter = None
        self._continuity_checker = None
        self._story_tracker = None

        self.pipelines = {
            "research_only": self._research_only,
            "write_only": self._write_only,
            "research_and_write": self._research_and_write,
            "full_pipeline": self._full_pipeline,
            "novel_plan": self._novel_plan,
            "novel_chapter": self._novel_chapter,
            "novel_start": self._novel_start,
        }

    # Lazy properties
    @property
    def story_director(self):
        if self._story_director is None:
            self._story_director = create_story_director(provider=self._provider, model=self._model)
        return self._story_director

    @property
    def novel_writer(self):
        if self._novel_writer is None:
            self._novel_writer = create_novel_writer(provider=self._provider, model=self._model)
        return self._novel_writer

    @property
    def story_critic(self):
        if self._story_critic is None:
            self._story_critic = create_story_critic(provider=self._provider, model=self._model)
        return self._story_critic

    @property
    def humanizer(self):
        if self._humanizer is None:
            self._humanizer = create_humanizer(provider=self._provider, model=self._model)
        return self._humanizer

    @property
    def summarizer(self):
        if self._summarizer is None:
            self._summarizer = create_summarizer(provider=self._provider, model=self._model)
        return self._summarizer

    @property
    def formatter(self):
        if self._formatter is None:
            self._formatter = create_formatter(provider=self._provider, model=self._model)
        return self._formatter

    @property
    def continuity_checker(self):
        if self._continuity_checker is None:
            self._continuity_checker = create_continuity_checker(provider=self._provider, model=self._model)
        return self._continuity_checker

    @property
    def story_tracker(self):
        if self._story_tracker is None:
            self._story_tracker = create_story_tracker(provider=self._provider, model=self._model)
        return self._story_tracker

    def run(self, pipeline, task):
        if pipeline not in self.pipelines:
            available = ", ".join(self.pipelines.keys())
            raise ValueError(f"Unknown pipeline '{pipeline}'. Available: {available}")

        self._log(f"\n{'='*60}")
        self._log(f"  ORCHESTRATOR: Running '{pipeline}' pipeline")
        self._log(f"  Task: {task}")
        self._log(f"{'='*60}\n")

        return self.pipelines[pipeline](task)

    # ---------------------------------------------------------------
    # General pipelines
    # ---------------------------------------------------------------
    def _research_only(self, task):
        return {"research": self._run_agent(self.researcher, task)}

    def _write_only(self, task):
        return {"draft": self._run_agent(self.writer, task)}

    def _research_and_write(self, task):
        research = self._run_agent(self.researcher, f"Research thoroughly: {task}")
        draft = self._run_agent(self.writer,
            f"Using this research, create content for: {task}\n\n=== RESEARCH ===\n{research}")
        return {"research": research, "draft": draft}

    def _full_pipeline(self, task):
        research = self._run_agent(self.researcher, f"Research thoroughly: {task}")
        draft_v1 = self._run_agent(self.writer,
            f"Using this research, create content for: {task}\n\n=== RESEARCH ===\n{research}")
        critique = self._run_agent(self.critic,
            f"Review this content. Task: {task}\n\n=== CONTENT ===\n{draft_v1}")
        draft_v2 = self._run_agent(self.writer,
            f"Revise based on feedback. Task: {task}\n\n=== DRAFT ===\n{draft_v1}\n\n=== FEEDBACK ===\n{critique}")
        return {"research": research, "draft_v1": draft_v1, "critique": critique, "draft_v2": draft_v2}

    # ---------------------------------------------------------------
    # Novel pipelines
    # ---------------------------------------------------------------
    def _novel_plan(self, task):
        plan = self._run_agent(self.story_director,
            f"Create a complete novel outline for:\n\n{task}\n\n"
            f"Choose a compelling title. Give each chapter a creative title. "
            f"Define file naming. Save the outline.\n"
            f"IMPORTANT: Calculate the correct number of chapters to hit 70,000-100,000 word target.")
        return {"novel_outline": plan}

    def _novel_chapter(self, task):
        """Write the next missing chapter with full pipeline and continuity checking."""

        # --- Scan and determine chapter number ---
        info = _scan_novel_files()
        outline = info["outline"]
        chapters = info["chapters"]
        chapter_map = info["chapter_map"]
        existing_nums = info["existing_nums"]
        gaps = info["gaps"]
        story_so_far = info["story_so_far"]

        # Check if user specified a chapter number
        user_ch_match = re.search(r'chapter\s*(\d+)', task.lower())
        if user_ch_match:
            ch_num = int(user_ch_match.group(1))
            if ch_num in chapter_map:
                self._log(f"\n  WARNING: Chapter {ch_num} already exists: {chapter_map[ch_num]}")
                self._log(f"  It will be rewritten as requested.")
        else:
            ch_num = info["next_num"]

        self._log(f"\n  === CHAPTER SEQUENCING ===")
        self._log(f"  Chapters found: {existing_nums if existing_nums else 'none'}")
        if gaps:
            self._log(f"  GAPS DETECTED: chapters {gaps} are missing!")
            self._log(f"  Filling gap: writing Chapter {ch_num}")
        else:
            self._log(f"  Writing next: Chapter {ch_num}")

        # Build protected file list
        protected = "\n".join(f"  - {chapter_map[n]} (Chapter {n} -- DO NOT MODIFY)"
                              for n in existing_nums if n != ch_num)

        context = (
            f"=== CHAPTER STATUS ===\n"
            f"Outline: {outline or 'NOT FOUND'}\n"
            f"Chapters completed: {existing_nums if existing_nums else 'none'}\n"
            f"{'MISSING chapters: ' + str(gaps) if gaps else 'No gaps -- sequence is complete so far'}\n"
            f"YOUR ASSIGNMENT: Write Chapter {ch_num}\n"
            f"Running summary: {story_so_far or 'none'}\n"
            f"\nPROTECTED FILES (DO NOT MODIFY):\n{protected}\n"
            f"\nRULES:\n"
            f"- Create a NEW file for Chapter {ch_num}\n"
            f"- The filename MUST contain 'ch{ch_num}' so it can be detected\n"
            f"- NEVER touch any protected files listed above\n"
        )

        # --- Step 1: Write ---
        self._log(f"\n  Step 1: Writing Chapter {ch_num}")
        files_before = set(f.name for f in _get_output_dir().iterdir())

        draft = self._run_agent(self.novel_writer,
            f"{context}\n"
            f"Read the novel outline ('{outline}') and find the outline for Chapter {ch_num}.\n"
            f"{'Read ' + story_so_far + ' for previous chapter context.' if story_so_far else ''}\n"
            f"Write Chapter {ch_num}. Include 'ch{ch_num}' in the filename.\n"
            f"Additional: {task}")

        # Detect the new file
        chapter_file = self._detect_new_chapter(files_before, chapters, ch_num)
        self._log(f"  Chapter file: {chapter_file}")

        # --- Step 2: Summarize ---
        self._log(f"\n  Step 2: Summarizing Chapter {ch_num}")
        summary = self._run_agent(self.summarizer,
            f"Read ONLY '{chapter_file}' (Chapter {ch_num}). Create a summary.\n"
            f"Save as 'summary_{chapter_file}'\n"
            f"{'Append to ' + story_so_far if story_so_far else 'Create a story_so_far file'}.")

        # --- Step 3: Critic ---
        self._log(f"\n  Step 3: Reviewing Chapter {ch_num}")
        review = self._run_agent(self.story_critic,
            f"Review ONLY '{chapter_file}' (Chapter {ch_num}).\n"
            f"Summary:\n{summary}\n\n"
            f"Read outline '{outline}' for reference. Save review as 'review_{chapter_file}'")

        # --- Step 4: Revise ---
        self._log(f"\n  Step 4: Revising Chapter {ch_num}")
        revision = self._run_agent(self.novel_writer,
            f"Revise ONLY '{chapter_file}' (Chapter {ch_num}).\n\n"
            f"=== FEEDBACK ===\n{review}\n\n"
            f"Read '{chapter_file}', apply feedback, overwrite '{chapter_file}'.\n"
            f"Do NOT create a new file. Do NOT touch other chapters.")

        # --- Step 5: Humanize ---
        self._log(f"\n  Step 5: Humanizing Chapter {ch_num}")
        humanized = self._run_agent(self.humanizer,
            f"Humanize ONLY '{chapter_file}' (Chapter {ch_num}).\n"
            f"Summary:\n{summary}\n\n"
            f"Read '{chapter_file}', fix AI patterns, overwrite '{chapter_file}'.")

        # --- Step 6: Story Track (Arc Compliance) ---
        self._log(f"\n  Step 6: Story Track - Verifying Arc Compliance for Chapter {ch_num}")

        arc_check = self._run_agent(self.story_tracker,
            f"Verify Chapter {ch_num} follows the planned novel arc.\n"
            f"Outline: '{outline}'\n"
            f"Story so far: {story_so_far or 'none'}\n"
            f"Current chapter: {chapter_file}\n\n"
            f"Check:\n"
            f"1. Does this chapter belong to the correct act (Beginning/Middle/End)?\n"
            f"2. Is the story progressing through the arc as planned?\n"
            f"3. Any drift from the planned story structure?\n"
            f"4. Are subplots being developed appropriately?\n"
            f"5. Pacing: too fast (skipping beats) or too slow (filler)?\n\n"
            f"Read the outline and story_so_far FIRST, then read the chapter.\n"
            f"Save tracker report as 'tracker_{chapter_file}'.\n"
            f"If drift is detected, provide specific recommendations to get back on track.")

        # --- Step 7: Continuity Check ---
        self._log(f"\n  Step 7: Continuity Check on Chapter {ch_num}")

        # Get the previous chapter file for transition checking
        prev_ch_file = chapter_map.get(ch_num - 1, None) if ch_num > 1 else None

        continuity = self._run_agent(self.continuity_checker,
            f"Check Chapter {ch_num} (file: '{chapter_file}') for completeness.\n"
            f"Outline: '{outline}'\n"
            f"{'Previous chapter: ' + prev_ch_file if prev_ch_file else 'This is the first chapter.'}\n"
            f"{'Story so far: ' + story_so_far if story_so_far else ''}\n\n"
            f"Verify:\n"
            f"1. Chapter ends with a COMPLETE sentence (no cut-off mid-sentence)\n"
            f"2. Chapter has a proper ending that hooks into the next chapter\n"
            f"3. Chapter matches the outline for Chapter {ch_num}\n"
            f"4. {'Transition from Chapter ' + str(ch_num-1) + ' is smooth' if prev_ch_file else 'Opening hooks the reader'}\n"
            f"5. No continuity errors with previous chapters\n"
            f"6. All scenes are complete (no unfinished scenes)\n\n"
            f"If ANY issues are found, fix them by reading '{chapter_file}', "
            f"making corrections, and overwriting '{chapter_file}'.\n"
            f"Save a continuity report as 'continuity_{chapter_file}'.")

        # --- Step 7: Format to .docx ---
        self._log(f"\n  Step 8: Formatting Chapter {ch_num} to Word")
        docx_name = chapter_file.replace(".txt", ".docx")
        formatted = self._run_agent(self.formatter,
            f"Convert ONLY '{chapter_file}' to Word.\n"
            f"Save as '{docx_name}'. Heading: '## Chapter {ch_num}'.\n"
            f"Do NOT convert other files.")

        return {
            "chapter_number": str(ch_num),
            "chapter_file": chapter_file,
            "first_draft": draft,
            "summary": summary,
            "review": review,
            "revised": revision,
            "humanized": humanized,
            "arc_check": arc_check,
            "continuity": continuity,
            "formatted": formatted,
        }

    def _novel_start(self, task):
        """Full novel kickoff."""

        self._log("\n  Phase 1: Research")
        research = self._run_agent(self.researcher,
            f"Research background for a novel about: {task}\n"
            f"Focus on setting, history, culture, technical details.")

        self._log("\n  Phase 2: Story Planning")
        plan = self._run_agent(self.story_director,
            f"Create a novel outline for:\n\n{task}\n\nResearch:\n{research}\n\n"
            f"Choose a TITLE. Give each chapter a creative title. "
            f"Calculate chapter count for 70,000-100,000 word target. "
            f"Define file naming. List all filenames.")

        self._log("\n  Phase 3: Writing Chapter 1")
        files_before = set(f.name for f in _get_output_dir().iterdir())
        chapter_1 = self._run_agent(self.novel_writer,
            f"Read the novel outline, then write Chapter 1. "
            f"Include 'ch1' in the filename. Hook the reader immediately.")

        ch1_file = self._detect_new_chapter(files_before, [], 1)
        self._log(f"  Chapter 1 file: {ch1_file}")

        self._log("\n  Phase 4: Summarizing")
        summary = self._run_agent(self.summarizer,
            f"Read ONLY '{ch1_file}'. Create summary and story_so_far file.")

        self._log("\n  Phase 5: Reviewing")
        review = self._run_agent(self.story_critic,
            f"Review ONLY '{ch1_file}'.\nSummary:\n{summary}\n\n"
            f"Read outline for reference. Save review.")

        self._log("\n  Phase 6: Revising")
        revision = self._run_agent(self.novel_writer,
            f"Revise ONLY '{ch1_file}':\n\n=== REVIEW ===\n{review}\n\n"
            f"Read '{ch1_file}', apply feedback, overwrite '{ch1_file}'.")

        self._log("\n  Phase 7: Humanizing")
        humanized = self._run_agent(self.humanizer,
            f"Humanize ONLY '{ch1_file}'.\nSummary:\n{summary}\n\n"
            f"Read '{ch1_file}', fix AI patterns, overwrite.")
        self._log("\n  Phase 8: Story Track - Arc Compliance")
        arc_check = self._run_agent(self.story_tracker,
            f"Verify Chapter 1 follows the planned novel arc.\n"
            f"Outline: (created in Phase 2)\n"
            f"Current chapter: {ch1_file}\n\n"
            f"Check:\n"
            f"1. Does this chapter properly set up the Beginning (Act 1)?\n"
            f"2. Is the inciting incident present?\n"
            f"3. Are main characters introduced?\n"
            f"4. Is the world established?\n"
            f"5. Does it hook the reader and set up the story goal?\n\n"
            f"Read the outline (created in Phase 2) and the chapter.\n"
            f"Save tracker report as 'tracker_{ch1_file}'.")
        self._log("\n  Phase 9: Continuity Check")
        continuity = self._run_agent(self.continuity_checker,
            f"Check Chapter 1 ('{ch1_file}') for completeness.\n"
            f"Read the outline for reference.\n"
            f"Verify: complete sentences, proper ending with hook, "
            f"no unfinished scenes, matches outline.\n"
            f"Fix any issues by overwriting '{ch1_file}'.\n"
            f"Save report as 'continuity_{ch1_file}'.")

        self._log("\n  Phase 9: Formatting to Word")
        ch1_docx = ch1_file.replace(".txt", ".docx")
        formatted = self._run_agent(self.formatter,
            f"Convert '{ch1_file}' to Word as '{ch1_docx}'. "
            f"Add title page with novel title, then Chapter 1.")

        return {
            "research": research, "novel_outline": plan,
            "chapter_1_file": ch1_file,
            "chapter_1_draft": chapter_1, "chapter_1_summary": summary,
            "chapter_1_review": review, "chapter_1_revised": revision,
            "chapter_1_humanized": humanized, "chapter_1_arc_check": arc_check,
            "chapter_1_continuity": continuity,
            "chapter_1_formatted": formatted,
        }

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    def _detect_new_chapter(self, files_before, existing_chapters, ch_num):
        """Find the chapter file that was just created."""
        files_after = set(f.name for f in _get_output_dir().iterdir())
        new_files = sorted(files_after - files_before)
        new_txt = [f for f in new_files if f.endswith(".txt")
                   and not any(x in f.lower() for x in ["summary", "review"])]

        if new_txt:
            return new_txt[0]

        # Fallback: most recently modified txt not in existing list
        output_dir = _get_output_dir()
        candidates = [(f.name, f.stat().st_mtime) for f in output_dir.iterdir()
                      if f.is_file() and f.suffix == ".txt"
                      and f.name not in existing_chapters
                      and not any(x in f.name.lower() for x in
                                 ["summary", "review", "outline", "story_so_far", "continuity", "tracker"])]
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return f"chapter_{ch_num}.txt"

    def _run_agent(self, agent, message):
        import time as _time
        self._log(f"\n{'-'*50}")
        self._log(f"  -> Handing off to: {agent.name}")
        self._log(f"{'-'*50}")
        agent.reset()
        response = agent.run(message)
        self._log(f"\n  <- {agent.name} finished ({len(response)} chars)")
        _time.sleep(2)
        return response

    def _log(self, message):
        if self.verbose:
            print(message)

    def list_pipelines(self):
        return list(self.pipelines.keys())
