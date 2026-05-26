# ooda_loop.py
"""
OODA Deep Research Loop for QWEN Autonomous Agent Mode.

Implements the Observe → Orient → Decide → Act cognitive cycle as an agentic
research engine that wraps QWEN's existing memory command infrastructure.

QWEN drives this loop autonomously — issuing [SEARCH:], [DISCUSS_WITH_CLAUDE:],
[WEB_SEARCH:], [STORE:] and other commands naturally during OBSERVE turns.
The loop continues until QWEN declares COMPLETE or the safety ceiling is reached.

External Research Command Hierarchy (QWEN decides which to use):
  1. [DISCUSS_WITH_CLAUDE: query]     — PREFERRED external research path.
                                        Claude performs high-quality web search
                                        and synthesizes findings. Use for most
                                        external research needs.
  2. [WEB_SEARCH: topic | turns=N]   — FALLBACK external research path.
                                        QWEN's own DuckDuckGo search pipeline.
                                        Use when: Claude is unavailable, API
                                        limits are reached, or content may be
                                        outside Claude's response boundaries.

Key design principles:
  - Zero changes to deepseek.py — all commands route through process_response()
  - QWEN terminates the loop (COMPLETE) or hits safety ceiling (OODA_MAX_CYCLES)
  - Intermediate thinking displayed via st.status() expandable container
  - Human-initiated only — QWEN cannot self-invoke this loop
  - Ken profile context held for ACT phase only (output framing)
  - Empty search result blocks are compacted before entering accumulated_context
    to prevent prompt bloat across cycles (QWEN Suggestion 1 fix)
  - Sliding window context compression: recent cycles kept in full, older cycles
    compressed to QWEN-generated summaries with fallback truncation. Hard character
    ceiling prevents unbounded prompt growth regardless of cycle count.

Integration points:
  - chatbot.llm.invoke(prompt)              — direct LLM call
  - chatbot.deepseek_enhancer.process_response(text) — command parsing/execution
  - st.status()                              — Streamlit live thinking display
"""

import re
import time
import logging
import datetime
import concurrent.futures
from typing import Tuple, Optional, List, Dict

# --- Module-level logger ---
logger = logging.getLogger(__name__)

# --- Context compression constants (OODA-internal, not user-facing) ---
# Number of most recent cycles to include with full observation text.
# Older cycles are represented by their compressed summaries only.
CONTEXT_FULL_WINDOW = 2

# Hard ceiling on total assembled context string (chars) injected into
# OBSERVE/DECIDE/ACT prompts. Prevents unbounded prompt growth regardless
# of cycle count. 20K chars leaves ample room in the 65K context window
# for prompt structure, task description, and QWEN's response generation.
CONTEXT_BUDGET_CHARS = 30000

# Generous fallback truncation length (chars) when QWEN does not produce
# a parseable CYCLE_SUMMARY block. 600 chars preserves roughly a full
# paragraph — enough to retain specific findings, numbers, and details
# rather than just vague topic references. Still yields ~75-85% reduction
# from a typical 3-5K OBSERVE output.
FALLBACK_SUMMARY_CHARS = 600

# Maximum consecutive degraded cycles (error/parse-failure) before forcing
# synthesis. Prevents a broken loop from burning all remaining cycles on junk
# when the model is producing malformed output or infrastructure is down.
# A successful cycle resets the counter to zero — only unbroken failure
# streaks trigger early termination.
CONSECUTIVE_FAILURE_CEILING = 3

# --- Stagnation detection constants ---
# Jaccard similarity threshold for research query overlap between cycles.
# If a cycle's search queries overlap this much with any single previous
# cycle's queries, it counts as a stagnant cycle. 0.7 = 70% keyword overlap.
STAGNATION_SIMILARITY_THRESHOLD = 0.7

# Consecutive stagnant cycles before injecting a warning into the OBSERVE
# prompt. Two in a row means QWEN searched for essentially the same things
# three times total (the original + 2 repeats) — clear signal to stop.
STAGNATION_CONSECUTIVE_CEILING = 2

# --- Per-call LLM timeout constants (seconds) ---
# OBSERVE is longest because commands like DISCUSS_WITH_CLAUDE involve
# external API calls that execute during process_response().
# Note: timeout applies to llm.invoke() only — process_response() runs
# after invoke returns and has its own execution time on top of this.
LLM_TIMEOUT_OBSERVE = 180
LLM_TIMEOUT_DECIDE = 90
LLM_TIMEOUT_ACT = 240


class OODALoop:
    """
    Autonomous OODA research loop engine.

    Instantiated once in Chatbot.__init__ and reused across turns.
    The run() method is the sole public entry point, called from
    chatbot.process_command() when OODA mode is active.
    """

    def __init__(self, chatbot):
        """
        Initialize OODALoop with a reference to the parent Chatbot instance.

        Args:
            chatbot: The Chatbot instance (provides llm, deepseek_enhancer, etc.)
        """
        try:
            # --- Core reference ---
            self.chatbot = chatbot

            # --- Safety ceiling (loaded from config, fallback to 20) ---
            try:
                from config import OODA_MAX_CYCLES
                self.max_cycles = OODA_MAX_CYCLES
            except ImportError:
                self.max_cycles = 20
                logger.warning("OODA: config.OODA_MAX_CYCLES not found, defaulting to 20")

            # --- Per-run state (reset by run()) ---
            self._task = ""                  # Original user task
            self._accumulated_context = []  # List of dicts: {cycle, full, summary}
            self._command_log = []          # List of dicts: {cycle, commands: [str]}
            self._cycle = 0                 # Current cycle counter
            self._start_time = None         # Datetime of run() start
            self._status_container = None   # st.status() handle
            self._cycle_queries = {}        # Maps cycle → list of normalized research query strings
            self._stagnation_count = 0      # Consecutive stagnant cycles
            self._stagnation_info = ""      # Human-readable stagnation detail for DECIDE injection

            logger.info(f"✅ OODALoop initialized (max_cycles={self.max_cycles})")

        except Exception as e:
            logger.error(f"OODALoop.__init__ error: {e}", exc_info=True)
            raise

    # =========================================================================
    # PUBLIC ENTRY POINT
    # =========================================================================

    def run(self, task: str, status_container) -> str:
        """
        Execute a full OODA research run for the given task.

        Called by chatbot.process_command() when OODA mode is active.
        Drives the Observe → Decide loop until QWEN signals COMPLETE
        or the safety ceiling is reached, then calls _act() for final synthesis.

        Args:
            task (str): The user's research task or question.
            status_container: A Streamlit st.status() context object for live UI updates.

        Returns:
            str: The final synthesized response for display in the chat UI.
        """
        # --- Reset all per-run state ---
        self._task = task.strip()
        self._accumulated_context = []
        self._command_log = []
        self._cycle = 0
        self._consecutive_failures = 0  # Consecutive degraded cycle counter
        self._stagnation_count = 0      # Consecutive stagnant cycle counter
        self._stagnation_info = ""      # Stagnation detail for DECIDE injection
        self._cycle_queries = {}        # Research query tracking per cycle
        self._start_time = datetime.datetime.now()
        self._status_container = status_container
        self._convo_context = ""         # Active session conversation context
        self._zero_results_streak = 0    # Consecutive all-zero-result cycles
        self._convo_context = self.chatbot.build_conversation_context(
            self._task, max_token_pct=0.4
        )
        if self._convo_context:
            logger.info(
                f"OODA: Loaded conversation context "
                f"({len(self._convo_context):,} chars) for task disambiguation"
            )
        else:
            logger.warning(
                "OODA: No conversation context available — "
                "loop will run without session history"
            )

        logger.info(f"🔄 OODA RUN STARTED | Task: {self._task[:80]}...")

        try:
											
            self._update_status("🔍 Starting OODA Deep Research Loop...")

            # ---- Main Observe → Decide loop ----
            while True:
                self._cycle += 1
                logger.info(f"🔄 OODA | Cycle {self._cycle} starting")

                # --- OBSERVE phase ---
                observe_output = self._observe()

                # --- DECIDE phase ---
                decision, decision_detail = self._decide()

                logger.info(
                    f"🔄 OODA | Cycle {self._cycle} DECIDE result: "
                    f"{decision} — {decision_detail[:60]}"
                )

                if decision == "COMPLETE":
					# QWEN is satisfied — break to ACT									
                    self._update_status(
                        f"✅ Cycle {self._cycle} — Research complete: {decision_detail}"
                    )
                    logger.info(f"🔄 OODA | COMPLETE declared at cycle {self._cycle}")
                    break

                # --- Consecutive failure tracking ---
                # Detect degraded cycle: _observe() errors start with
                # "(Observation error" and all _decide()/_parse error/default
                # paths wrap detail strings in parentheses.
                # Clean QWEN responses never start with "(".
                cycle_degraded = (
                    observe_output.startswith("(Observation error") or
                    decision_detail.startswith("(")
                )

                if cycle_degraded:
                    self._consecutive_failures += 1
                    logger.warning(
                        f"🔄 OODA | Degraded cycle {self._cycle} — "
                        f"consecutive failures: {self._consecutive_failures}"
                        f"/{CONSECUTIVE_FAILURE_CEILING}"
                    )
                else:
                    # Clean cycle — reset the streak
                    if self._consecutive_failures > 0:
                        logger.info(
                            f"🔄 OODA | Clean cycle {self._cycle} — "
                            f"resetting consecutive failure counter "
                            f"(was {self._consecutive_failures})"
                        )
                    self._consecutive_failures = 0

                # --- Consecutive failure ceiling ---
                # A broken loop (bloated context, model incoherence, infra down)
                # is caught here instead of burning all remaining cycles on junk.
                if self._consecutive_failures >= CONSECUTIVE_FAILURE_CEILING:
                    logger.warning(
                        f"🔄 OODA | {CONSECUTIVE_FAILURE_CEILING} consecutive "
                        f"degraded cycles — forcing synthesis with "
                        f"accumulated research"
                    )
                    self._update_status(
                        f"⚠️ {CONSECUTIVE_FAILURE_CEILING} consecutive errors "
                        f"detected — synthesizing with research gathered so far."
                    )
                    break

                # CONTINUE — check safety ceiling before next cycle
                if self._cycle >= self.max_cycles:
															  
                    logger.warning(
                        f"🔄 OODA | Safety ceiling reached ({self.max_cycles} cycles) "
                        f"— forcing COMPLETE"
                    )
                    self._update_status(
                        f"⚠️ Maximum research depth ({self.max_cycles} cycles) reached "
                        f"— synthesizing final answer."
                    )
                    break

                # Log gap and continue
                self._update_status(
                    f"🔍 Cycle {self._cycle} — Gap identified: "
                    f"{decision_detail[:80]} — continuing..."
                )

            # ---- ACT phase — final synthesis ----
            self._update_status("✍️ Synthesizing final answer...")
            final_response = self._act()

							 
            elapsed = (datetime.datetime.now() - self._start_time).total_seconds()
            logger.info(
                f"🔄 OODA RUN COMPLETE | Cycles: {self._cycle} | "
                f"Commands: {self._get_total_commands_count()} | Elapsed: {elapsed:.1f}s"
            )

            return final_response

        except Exception as e:
            logger.error(f"OODA run() error: {e}", exc_info=True)
																			 
            return (
                f"⚠️ The OODA research loop encountered an error during cycle {self._cycle}.\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Partial research gathered:**\n"
                + self._format_accumulated_context()
            )

    # =========================================================================
    # OBSERVE PHASE
    # =========================================================================

    def _observe(self) -> str:
        """
        Execute one OBSERVE cycle.

        Presents the task and accumulated context to QWEN via a structured prompt.
        QWEN emits memory commands naturally; these are executed via process_response().
        The processed output is cleaned of verbose empty-result blocks, then stored
        as a structured dict with both full text and a compressed summary for the
        sliding window context system.

        Returns:
            str: Processed LLM output with commands executed and empty blocks compacted.
        """
        try:
            self._update_status(f"🔍 Cycle {self._cycle} — Observing...")

            # --- Build the OBSERVE prompt with accumulated context ---
            prompt = self._build_observe_prompt()

            logger.info(
                f"🔍 OODA OBSERVE | Cycle {self._cycle} | "
                f"Prompt length: {len(prompt)} chars"
            )

            # --- Invoke LLM with timeout protection ---
            raw_output = self._invoke_with_timeout(
                prompt, LLM_TIMEOUT_OBSERVE, "OBSERVE"
            )

            if not raw_output:
                logger.warning(
                    f"🔍 OODA OBSERVE | Cycle {self._cycle} — Empty LLM response"
                )
                raw_output = "(No output from LLM this cycle)"

            logger.info(
                f"🔍 OODA OBSERVE | Cycle {self._cycle} | "
                f"Raw output: {len(raw_output)} chars"
            )

            # --- Extract command signatures BEFORE process_response ---
            # Captures what QWEN asked for so we can build a rich command log
            # and detect stagnation. Does not execute commands — just reads them.
            signatures = self._extract_command_signatures(raw_output)

            # --- Execute memory commands through existing deepseek.py pipeline ---
            # Zero new command logic here — all routing handled by process_response()
            processed_output, commands_executed_count = (
                self.chatbot.deepseek_enhancer.process_response(raw_output)
            )

            # --- Compact verbose empty search blocks BEFORE accumulating ---
            # Fixes QWEN Suggestion 1: "NO RESULTS FOUND" blocks are ~85 chars of
            # formatting markup with zero info value. They bloat every subsequent
            # OBSERVE/DECIDE prompt. Replace with compact single-line notation.
            # Actual search results with content are left completely untouched.
            processed_output = self._clean_empty_search_results(processed_output)

            # --- Zero-results streak tracking ---
            # When all search commands in a cycle return nothing, the memory
            # store does not contain what QWEN is looking for. After 2+
            # consecutive zero-result cycles the DECIDE prompt will direct
            # QWEN to either use DISCUSS_WITH_CLAUDE or declare COMPLETE.
            search_signatures = [
                s for s in signatures
                if any(cmd in s for cmd in ('SEARCH:', 'COMPREHENSIVE_SEARCH:',
                                             'PRECISE_SEARCH:', 'EXACT_SEARCH:'))
            ]
            zero_result_hits = processed_output.count('[SEARCH \u2192 0 results]')

            if search_signatures and zero_result_hits >= len(search_signatures):
                # Every search command this cycle returned nothing
                self._zero_results_streak += 1
                logger.warning(
                    f"OODA ZERO_RESULTS | Cycle {self._cycle} | "
                    f"All {zero_result_hits} search(es) returned 0 results | "
                    f"Streak: {self._zero_results_streak}"
                )
            else:
                # At least one search found results, or no searches were run
                if self._zero_results_streak > 0:
                    logger.info(
                        f"OODA ZERO_RESULTS | Cycle {self._cycle} | "
                        f"Results found — resetting streak "
                        f"(was {self._zero_results_streak})"
                    )
                self._zero_results_streak = 0

            logger.info(
                f"🔍 OODA OBSERVE | Cycle {self._cycle} | "
                f"Commands executed: {commands_executed_count} | "
                f"Signatures extracted: {len(signatures)}"
            )

            # --- Update structured command log ---
            # Stores actual command signatures instead of just counts,
            # so QWEN can see what it already searched in future prompts.
            if signatures:
                self._command_log.append({
                    "cycle": self._cycle,
                    "commands": signatures
                })
                self._update_status(
                    f"🔍 Cycle {self._cycle} — "
                    f"Executed {commands_executed_count} command(s): "
                    f"{', '.join(sig[:50] for sig in signatures[:3])}"
                    f"{'...' if len(signatures) > 3 else ''}"
                )
            else:
                self._update_status(
                    f"🔍 Cycle {self._cycle} — "
                    f"Observation complete (no commands issued)"
                )

            # --- Stagnation detection ---
            # Compares this cycle's research queries against previous cycles.
            # Only runs from cycle 2 onward (need at least one prior cycle).
            if self._cycle >= 2:
                is_stagnant, stagnation_detail = self._check_stagnation()
                if is_stagnant:
                    self._stagnation_count += 1
                    self._stagnation_info = stagnation_detail
                    logger.warning(
                        f"🔄 OODA STAGNATION | Cycle {self._cycle} | "
                        f"Consecutive stagnant: {self._stagnation_count}"
                        f"/{STAGNATION_CONSECUTIVE_CEILING} | {stagnation_detail}"
                    )
                    self._update_status(
                        f"⚠️ Cycle {self._cycle} — Stagnation detected: "
                        f"{stagnation_detail[:80]}"
                    )
                else:
                    # Clean cycle — reset stagnation streak
                    if self._stagnation_count > 0:
                        logger.info(
                            f"🔄 OODA STAGNATION | Cycle {self._cycle} | "
                            f"Clean — resetting stagnation counter "
                            f"(was {self._stagnation_count})"
                        )
                    self._stagnation_count = 0
                    self._stagnation_info = ""

            # --- Extract compressed summary from QWEN's output ---
            # QWEN is instructed to end observations with a CYCLE_SUMMARY: block.
            # If present, we extract it. If not, we fall back to generous truncation.
            # This costs zero extra LLM calls — summary is part of the OBSERVE output.
            summary = self._extract_cycle_summary(processed_output)

            # --- Append structured entry to accumulated context ---
            # Stores both full text (for recent-cycle window) and compressed
            # summary (for older cycles) enabling bounded prompt growth.
            self._accumulated_context.append({
                "cycle": self._cycle,
                "full": processed_output,
                "summary": summary
            })

            # --- Log context compression metrics ---
            total_context_chars = sum(
                len(entry["full"]) for entry in self._accumulated_context
            )
            logger.info(
                f"🔍 OODA OBSERVE | Cycle {self._cycle} | "
                f"Full output: {len(processed_output)} chars | "
                f"Summary: {len(summary)} chars | "
                f"Total accumulated (uncompressed): {total_context_chars} chars"
            )

            return processed_output

        except Exception as e:
            logger.error(
                f"OODA _observe() error at cycle {self._cycle}: {e}",
                exc_info=True
            )
            # --- Error entries use the same dict structure for consistency ---
            # Error messages are already short so full == summary
            error_msg = f"(Observation error at cycle {self._cycle}: {str(e)})"
            self._accumulated_context.append({
                "cycle": self._cycle,
                "full": error_msg,
                "summary": error_msg
            })
            return error_msg

    def _build_observe_prompt(self) -> str:
        """
        Build the structured OBSERVE phase prompt for the current cycle.

        Includes the original task, active conversation context (cycle 1 only),
        accumulated research context, structured command history, stagnation
        warnings (if detected), and guidance on the external research command
        hierarchy.

        Conversation context is injected in cycle 1 only — subsequent cycles
        have accumulated_context which provides richer research continuity.
        This prevents unbounded prompt growth while ensuring QWEN understands
        referential task phrases ("these findings", "this document") from the start.

        Cycle wrap-up warnings are excluded here — they belong in DECIDE where
        the continue/stop decision is made. OBSERVE should focus on gathering.

        Returns:
            str: The complete prompt string to pass to the LLM.
        """
        if self._cycle == 1:
            prior_context_section = (
                "This is your first observation cycle. "
                "No prior research has been gathered yet."
            )
            command_log_section = "None yet."
        else:
            prior_context_section = self._format_accumulated_context()
            command_log_section = self._format_command_log()

        # --- Active conversation context (cycle 1 only) ---
        # Injected so QWEN can resolve referential phrases like "these findings"
        # or "this document" back to what Ken actually provided in the conversation.
        # Only shown in cycle 1 — later cycles carry forward understanding via
        # accumulated_context, and re-injecting it every cycle would bloat prompts.
        convo_context_section = ""
        if self._cycle == 1 and self._convo_context:
            convo_context_section = (
                "\nACTIVE CONVERSATION CONTEXT\n"
                "(Recent messages from this session. Use this to understand what "
                "referential phrases in the task — such as 'these findings', "
                "'this document', or 'what we just discussed' — actually refer to.)\n"
                f"{self._convo_context}\n"
                "END OF CONVERSATION CONTEXT\n"
            )

        # --- Stagnation warning ---
        # Fires when repeated identical queries are detected across cycles.
        stagnation_warning = ""
        if self._stagnation_count >= STAGNATION_CONSECUTIVE_CEILING:
            stagnation_warning = (
                f"\u26a0\ufe0f STAGNATION WARNING: Your last {self._stagnation_count} "
                f"cycle(s) searched for substantially similar terms as earlier "
                f"cycles. {self._stagnation_info} Either explore a genuinely "
                f"different angle or prepare to declare COMPLETE \u2014 repeating "
                f"the same queries will not yield new information."
            )

        # Inject current date so QWEN's web searches are temporally grounded.
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        prompt = f"""You are operating in OODA Deep Research Mode \u2014 Cycle {self._cycle} of {self.max_cycles}.
    Current date: {current_date}

    Your task or research project:
    {self._task}
    {convo_context_section}
    What you have gathered in previous cycles:
    {prior_context_section}

    Research actions already taken this session:
    {command_log_section}

    {stagnation_warning}

    Begin your observation cycle now. Think through what you need to know to address
    this task well, then issue whichever commands you judge necessary.

    AVAILABLE COMMANDS \u2014 use standard syntax:

    Memory Search:
        [SEARCH: query | type=TYPE | limit=N]

    External Research \u2014 choose based on your situation:
        [DISCUSS_WITH_CLAUDE: query]        \u2190 PREFERRED for most external research.
                                            Claude performs high-quality web search
                                            and synthesizes findings directly.
                                            Use this first for external information.

        [WEB_SEARCH: topic | turns=N]      \u2190 FALLBACK \u2014 use autonomously when:
                                            \u2022 Claude is unavailable or rate-limited
                                            \u2022 Content may be outside Claude's
                                                response boundaries
                                            \u2022 You need raw unfiltered web results
                                            \u2022 [DISCUSS_WITH_CLAUDE] fails or times out

    Memory Storage:
        [STORE: content | type=TYPE | confidence=X]

    Issue commands as you reason \u2014 they execute immediately and results are visible
    to you in subsequent cycles. After gathering, summarize clearly what this cycle
    revealed and what critical gaps, if any, remain.

    IMPORTANT \u2014 End your observation with a structured summary using this exact format:

    CYCLE_SUMMARY: [2-3 sentences capturing the key findings, specific data points,
    and any critical gaps identified this cycle. Be specific \u2014 include names, numbers,
    and conclusions. This summary will represent this cycle in future context when
    full observation text is compressed to save context space.]

    You MUST write this CYCLE_SUMMARY even if a [WEB_SEARCH] or [DISCUSS_WITH_CLAUDE]
    command generated its own findings block above. The summary is your own synthesis
    of what was learned — write it as the very last line of your response.
    """
        return prompt
    # =========================================================================
    # DECIDE PHASE
    # =========================================================================

    def _decide(self) -> Tuple[str, str]:
        """
        Execute the DECIDE phase.

        Presents a structured evaluation prompt to QWEN asking whether research
        is sufficient. Parses the response for a CONTINUE or COMPLETE declaration.

        Returns:
            Tuple[str, str]: ('CONTINUE' | 'COMPLETE', detail_string)
																		   
															   
        """
        try:
            self._update_status(
                f"🤔 Cycle {self._cycle} — Evaluating research sufficiency..."
            )

										 
            prompt = self._build_decide_prompt()

            logger.info(
                f"🤔 OODA DECIDE | Cycle {self._cycle} | "
                f"Prompt length: {len(prompt)} chars"
            )

								
            # --- Invoke LLM with timeout protection ---
            decide_output = self._invoke_with_timeout(
                prompt, LLM_TIMEOUT_DECIDE, "DECIDE"
            )

            if not decide_output:
                logger.warning(
                    f"🤔 OODA DECIDE | Cycle {self._cycle} — "
                    f"Empty response, defaulting to CONTINUE"
                )
                return "CONTINUE", "(No DECIDE response — continuing by default)"

            logger.info(
                f"🤔 OODA DECIDE | Cycle {self._cycle} | "
                f"Response: {decide_output[:200]}"
            )

																 
																		 

            return self._parse_decide_response(decide_output)

        except Exception as e:
            logger.error(
                f"OODA _decide() error at cycle {self._cycle}: {e}",
                exc_info=True
            )
            return "CONTINUE", f"(DECIDE error: {str(e)} — continuing)"

    def _build_decide_prompt(self) -> str:
        """
        Build the structured DECIDE phase evaluation prompt.

        Includes cycle wrap-up warnings (only here, not in OBSERVE),
        diminishing returns nudge from cycle 3+, stagnation evidence from
        the query overlap detector, and zero-results warnings when memory
        searches repeatedly return nothing.

        Returns:
            str: The DECIDE prompt string.
        """
        # --- Cycle wrap-up warning (lives in DECIDE only) ---
        wrap_up_warning = self._build_cycle_warning()

        # --- Diminishing returns nudge (fires from cycle 3 onward) ---
        diminishing_returns = ""
        if self._cycle >= 3:
            diminishing_returns = (
                f"You have completed {self._cycle} research cycles. Each additional "
                f"cycle has diminishing returns. Declare COMPLETE unless the remaining "
                f"gap would materially change the quality of your final answer. "
                f'"Good enough to be genuinely useful" is the bar \u2014 not '
                f'"exhaustively complete."'
            )

        # --- Stagnation evidence ---
        # Gives QWEN an external mechanical signal anchored in observed query overlap.
        stagnation_evidence = ""
        if self._stagnation_count > 0 and self._stagnation_info:
            stagnation_evidence = (
                f"EXTERNAL CHECK: {self._stagnation_info} "
                f"If you are not finding new information, this is evidence "
                f"that research is sufficient."
            )

        # --- Zero-results warning ---
        # Fires when all search commands across multiple cycles returned nothing.
        # Steers QWEN toward the correct resolution: use external research or
        # recognize that the needed content is already in the conversation context.
        zero_results_warning = ""
        if self._zero_results_streak >= 2:
            zero_results_warning = (
                f"\u26a0\ufe0f ZERO RESULTS WARNING: Your last "
                f"{self._zero_results_streak} consecutive cycle(s) returned zero "
                f"results from ALL memory searches. The memory store does not "
                f"contain what you are searching for under these query terms.\n"
                f"You have two paths forward:\n"
                f"  1. If this task requires external information not yet in memory, "
                f"use [DISCUSS_WITH_CLAUDE: topic] next cycle \u2014 Claude will "
                f"perform a live web search and return synthesized findings.\n"
                f"  2. If this task asks you to consolidate, interpret, or analyze "
                f"content that is already visible in the ACTIVE CONVERSATION CONTEXT "
                f"above, declare COMPLETE \u2014 the data you need is already in "
                f"front of you, not in memory storage."
            )

        prompt = f"""OODA DECIDE PHASE \u2014 Cycle {self._cycle} of {self.max_cycles}

    My current task or research project:
    {self._task}

    What I have gathered so far across all observation cycles:
    {self._format_accumulated_context()}

    Research actions completed this session:
    {self._format_command_log()}

    {diminishing_returns}

    {stagnation_evidence}

    {zero_results_warning}

    {wrap_up_warning}

    Evaluate honestly:
    1. Do I have a good understanding of this task or research project?
    2. Is the information I have gathered truthful, well-sourced, and sufficiently complete?
    3. Is the information current and actionable for this task?
    4. Would one more focused research cycle meaningfully change or improve my answer?

    You MUST respond with exactly one of the following two formats:

    CONTINUE: [specific gap that remains] \u2192 [exact command or action to address it next cycle]

    COMPLETE: [one clear sentence explaining why the research is now sufficient]

    Your response:"""
        return prompt

    def _parse_decide_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the LLM's DECIDE response into a structured decision.

        Checks for COMPLETE first (more conservative), then CONTINUE.
        Falls back to CONTINUE if neither is found clearly.

        Args:
            response (str): Raw LLM output from the DECIDE prompt.

        Returns:
            Tuple[str, str]: ('CONTINUE'|'COMPLETE', detail_string)
        """
        try:
								  
            text = response.strip()

            # Check COMPLETE first — more conservative default
            complete_match = re.search(
                r'COMPLETE\s*:\s*(.+)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if complete_match:
														
																 
                detail = re.split(r'[\n\r]', complete_match.group(1).strip())[0].strip()
                logger.info(f"🤔 OODA DECIDE | COMPLETE parsed: {detail[:80]}")
                return "COMPLETE", detail

            # Check CONTINUE
            continue_match = re.search(
                r'CONTINUE\s*:\s*(.+)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if continue_match:
														
									  
                detail = re.split(r'[\n\r]', continue_match.group(1).strip())[0].strip()
                logger.info(f"🤔 OODA DECIDE | CONTINUE parsed: {detail[:80]}")
                return "CONTINUE", detail

            # Neither found — log and default to CONTINUE
            logger.warning(
                f"🤔 OODA DECIDE | Could not parse CONTINUE/COMPLETE. "
                f"Defaulting to CONTINUE. Preview: {text[:200]}"
            )
            return "CONTINUE", "(Could not parse clear decision — continuing research)"

        except Exception as e:
            logger.error(f"OODA _parse_decide_response() error: {e}", exc_info=True)
            return "CONTINUE", f"(Parse error: {str(e)} — continuing)"

    # =========================================================================
    # ACT PHASE
    # =========================================================================

    def _act(self) -> str:
        """
        Execute the ACT phase — final synthesis of all accumulated research.

        Builds a synthesis prompt that includes all accumulated context plus
        Ken's profile framing pulled from Qdrant, instructing QWEN to produce
        a complete, well-structured final response for display in the chat UI.

        Returns:
            str: The final synthesized response string with OODA header.
        """
        try:
            self._update_status("✍️ Generating final synthesized response...")

											
            prompt = self._build_act_prompt()

            logger.info(
                f"✍️ OODA ACT | Prompt length: {len(prompt)} chars | "
                f"Cycles: {self._cycle} | Commands: {self._get_total_commands_count()}"
            )

													
            # --- Invoke LLM with timeout protection ---
            final_output = self._invoke_with_timeout(
                prompt, LLM_TIMEOUT_ACT, "ACT"
            )

            if not final_output:
                logger.warning("✍️ OODA ACT | Empty response from LLM")
                final_output = (
                    "⚠️ The synthesis step returned an empty response. "
                    "The accumulated research is available below.\n\n"
                    + self._format_accumulated_context()
                )

            # Process any commands QWEN emits during synthesis (e.g. final [STORE:])
																	  
            processed_final, final_cmd_count = (
                self.chatbot.deepseek_enhancer.process_response(final_output)
            )

            if final_cmd_count > 0:
                logger.info(
                    f"✍️ OODA ACT | {final_cmd_count} command(s) processed during synthesis"
                )

            # Prepend OODA summary header
            elapsed = (datetime.datetime.now() - self._start_time).total_seconds()
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            # --- Determine exit reason for header display ---
            # Lets Ken see at a glance whether the run completed cleanly,
            # hit the max ceiling, or bailed on consecutive failures.
            if self._consecutive_failures >= CONSECUTIVE_FAILURE_CEILING:
                exit_flag = (
                    " ⚠️ (degraded — consecutive errors forced early synthesis)"
                )
            elif self._cycle >= self.max_cycles:
                exit_flag = " ⚠️ (safety ceiling reached)"
            else:
                exit_flag = ""

            header = (
                f"**🔄 OODA Research Complete** — "
                f"{self._cycle} cycle(s) · "
                f"{self._get_total_commands_count()} command(s) · "
                f"{minutes}m {seconds}s{exit_flag}\n\n"
                f"---\n\n"
            )

            return header + processed_final

        except Exception as e:
            logger.error(f"OODA _act() error: {e}", exc_info=True)
            return (
                f"⚠️ Error during final synthesis: {str(e)}\n\n"
                f"**Accumulated research from {self._cycle} cycle(s):**\n\n"
                + self._format_accumulated_context()
            )

    def _build_act_prompt(self) -> str:
        """
        Build the ACT phase synthesis prompt.

        Ken's profile context is introduced here — not in OBSERVE/DECIDE —
        so research is conducted objectively but the final answer is framed
        appropriately for the person receiving it.

        STORE guidance uses a lighter-touch approach because synthesized content
        is already preserved via:
          1. Active conversation context window
          2. ai_communication Qdrant entries from OBSERVE phase Claude calls
          3. Conversation summary at 85% token threshold
        The STORE instruction targets precision-retrievable facts only —
        specific protocols, measurements, and schedules worth surfacing
        independently of conversation context months from now.

        Returns:
            str: The ACT synthesis prompt string.
        """
					  
        ken_context = self._retrieve_ken_context_for_framing()

			 
        elapsed = (datetime.datetime.now() - self._start_time).total_seconds()

        prompt = f"""OODA ACT PHASE — Final Synthesis

You have completed {self._cycle} research cycle(s) on the following task:

TASK:
{self._task}

COMPLETE RESEARCH GATHERED ({len(self._accumulated_context)} cycles):
{self._format_accumulated_context()}

COMMANDS EXECUTED THIS SESSION:
{self._format_command_log()}

CONTEXT ABOUT THE PERSON RECEIVING THIS ANSWER:
{ken_context}

Research time: {elapsed:.0f} seconds across {self._cycle} cycle(s).

Now synthesize everything you have gathered into a final, complete response.
Your response should:
- Directly address the original task with specific, well-supported information
- Be truthful — clearly distinguish confirmed findings from inference
- Be actionable and framed appropriately for the person receiving it
- Cite which cycle or source each key finding came from where relevant
- Be honest about any gaps that remain despite the research conducted
																	   

OPTIONAL — PRECISION STORAGE:
If this task produced specific, actionable findings worth retrieving
independently of conversation context (protocols, measurements, schedules,
or recommendations with specific numbers), store 2-3 as discrete searchable
memories AFTER your written response:

[STORE: [actual finding in full sentences with specific details and numbers
— not a title or label] | type=ooda_synthesis | confidence=0.X]

Useful precision storage — specific enough to be valuable standalone:
✅ [STORE: Ken BJJ recovery Mon/Fri: replace 30min passive hot tub with
4 cycles contrast therapy — 4min at 40°C then 1min cold at 15-18°C,
finish cold. Evidence: superior inflammation reduction vs heat alone for
masters athletes age 50+. | type=ooda_synthesis | confidence=0.9]

Not useful — already captured in conversation summary:
❌ [STORE: Recovery Protocol Optimization - 53yr BJJ]
❌ [STORE: Protocol summary]
❌ [STORE: Research findings]

Each stored item must contain the actual recommendation with specific
details — minimum 30 words. Store only findings precise enough that
they would be useful months from now without any surrounding context.

Please provide your complete final answer now, followed by any STORE commands:"""
        return prompt

    def _retrieve_ken_context_for_framing(self) -> str:
        """
        Lightweight Qdrant search for Ken's profile to frame the ACT output.

        Called only during the ACT phase. Returns a neutral fallback string
        if the search fails — this must never crash the ACT phase.

        Returns:
            str: Brief profile context string, or neutral fallback.
        """
        try:
																				
																
            #  matches VectorDB.search() signature
            results = self.chatbot.vector_db.search(
                query="Ken profile personal context farm family",
                k=3
            )

            if results:
														   
                context_parts = []
                for r in results[:3]:
                    content = r.get('content', r.get('text', ''))
                    if content and len(content) > 20:
                        context_parts.append(f"- {content[:200]}")

                if context_parts:
                    logger.info(
                        f"✍️ OODA ACT | Retrieved {len(context_parts)} Ken context snippets"
                    )
                    return "\n".join(context_parts)

            logger.info("✍️ OODA ACT | No Ken context retrieved — using neutral framing")
            return "No specific personal context retrieved — provide a generally applicable response."

        except Exception as e:
            logger.warning(f"✍️ OODA ACT | Ken context retrieval failed: {e}")
            return "Personal context unavailable — provide a generally applicable response."

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _invoke_with_timeout(self, prompt: str, timeout_seconds: int,
                             phase_label: str) -> str:
        """
        Invoke the LLM with a timeout ceiling to prevent indefinite hangs.

        Wraps self.chatbot.llm.invoke() in a ThreadPoolExecutor with a deadline.
        On timeout, returns an error string that the existing degraded-cycle
        detector will catch (starts with "(Observation error" or similar).

        Note: ThreadPoolExecutor cannot kill the underlying Ollama inference
        thread — it only stops waiting. The GPU will still be busy until
        inference completes or Ollama's own timeout fires. But from Streamlit's
        perspective, the OODA loop is no longer blocked indefinitely.

        Args:
            prompt (str): The complete prompt string to send to the LLM.
            timeout_seconds (int): Maximum seconds to wait for a response.
            phase_label (str): Human-readable label for logging (e.g. "OBSERVE").

        Returns:
            str: LLM response text, or error string on timeout/failure.
        """
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.chatbot.llm.invoke, prompt)
                try:
                    result = future.result(timeout=timeout_seconds)
                    return result
                except concurrent.futures.TimeoutError:
                    logger.error(
                        f"🕐 OODA TIMEOUT | {phase_label} phase exceeded "
                        f"{timeout_seconds}s at cycle {self._cycle}"
                    )
                    self._update_status(
                        f"⚠️ Cycle {self._cycle} — {phase_label} timed out "
                        f"after {timeout_seconds}s"
                    )
                    return (
                        f"(Observation error at cycle {self._cycle}: "
                        f"LLM timeout after {timeout_seconds}s in {phase_label} phase)"
                    )
        except Exception as e:
            logger.error(
                f"OODA _invoke_with_timeout error ({phase_label}): {e}",
                exc_info=True
            )
            return (
                f"(Observation error at cycle {self._cycle}: "
                f"{phase_label} invocation failed: {str(e)})"
            )

    def _extract_command_signatures(self, raw_output: str) -> List[str]:
        """
        Extract command signatures from raw LLM output before process_response.

        Parses the bracketed command syntax to capture what QWEN asked for —
        the actual query text, not just the command type. This enables:
          1. Rich command log (QWEN can see what it already searched)
          2. Stagnation detection (compare queries across cycles)

        Does NOT execute any commands — just reads the brackets.
        STORE commands are logged with truncated content to keep the log readable.

        Args:
            raw_output (str): Raw LLM output text before command execution.

        Returns:
            List[str]: List of human-readable command signature strings.
                       e.g. ["SEARCH: apricot pruning timing | type=knowledge",
                             "DISCUSS_WITH_CLAUDE: frost dates Methow Valley",
                             "STORE: (1 item)"]
        """
        signatures = []
        research_queries = []  # Subset used for stagnation detection

        try:
            # --- Pattern for research commands (SEARCH, DISCUSS_WITH_CLAUDE, WEB_SEARCH) ---
            # These are the commands where query text matters for stagnation detection.
            research_patterns = [
                (r'\[\s*SEARCH\s*:\s*((?:[^\[\]]|\[[^\[\]]*\])*?)\s*\]', 'SEARCH'),
                (r'\[\s*DISCUSS_WITH_CLAUDE\s*:\s*(.*?)\s*\]', 'DISCUSS_WITH_CLAUDE'),
                (r'\[\s*WEB_SEARCH\s*:\s*(.*?)\s*(?:\|\s*turns=\d+)?\s*\]', 'WEB_SEARCH'),
                (r'\[\s*COMPREHENSIVE_SEARCH\s*:\s*(.*?)\s*\]', 'COMPREHENSIVE_SEARCH'),
                (r'\[\s*PRECISE_SEARCH\s*:\s*(.*?)\s*\]', 'PRECISE_SEARCH'),
                (r'\[\s*EXACT_SEARCH\s*:\s*(.*?)\s*\]', 'EXACT_SEARCH'),
            ]

            for pattern, cmd_name in research_patterns:
                for match in re.finditer(pattern, raw_output, re.IGNORECASE):
                    query_text = match.group(1).strip()
                    if query_text and len(query_text) > 2:
                        # Truncate very long queries for readability
                        display_query = query_text[:120] + "..." if len(query_text) > 120 else query_text
                        signatures.append(f"{cmd_name}: {display_query}")
                        research_queries.append(query_text)

            # --- Pattern for STORE commands (log presence, not full content) ---
            store_matches = re.findall(
                r'\[\s*STORE\s*:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]',
                raw_output, re.IGNORECASE
            )
            if store_matches:
                signatures.append(f"STORE: ({len(store_matches)} item(s))")

            # --- Pattern for other notable commands ---
            if re.search(r'\[\s*REFLECT\s*\]', raw_output, re.IGNORECASE):
                signatures.append("REFLECT")
            if re.search(r'\[\s*COGNITIVE_STATE\s*:', raw_output, re.IGNORECASE):
                signatures.append("COGNITIVE_STATE")

            # --- Store research queries for stagnation detection ---
            self._cycle_queries[self._cycle] = research_queries

            logger.info(
                f"🔍 OODA SIGNATURES | Cycle {self._cycle} | "
                f"Extracted {len(signatures)} signatures, "
                f"{len(research_queries)} research queries"
            )

        except Exception as e:
            # Signature extraction must never crash the research loop
            logger.warning(
                f"OODA _extract_command_signatures error (non-critical): {e}"
            )

        return signatures

    def _check_stagnation(self) -> Tuple[bool, str]:
        """
        Check if the current cycle's research queries overlap significantly
        with any previous cycle's queries using Jaccard similarity.

        Compares normalized keyword sets — converts queries to lowercase,
        splits on whitespace and common delimiters, then measures overlap.
        This catches the obvious case of QWEN re-searching the same topic
        with minor phrasing variations.

        Only considers research queries (SEARCH, DISCUSS_WITH_CLAUDE,
        WEB_SEARCH) — not STORE or REFLECT commands.

        Returns:
            Tuple[bool, str]: (is_stagnant, detail_string)
                is_stagnant: True if overlap exceeds STAGNATION_SIMILARITY_THRESHOLD
                detail_string: Human-readable description of the overlap found
        """
        try:
            current_queries = self._cycle_queries.get(self._cycle, [])

            # --- No queries this cycle means no research was attempted ---
            # This could be a pure reasoning/storage cycle — not stagnation.
            if not current_queries:
                return False, ""

            # --- Normalize current cycle's queries into a keyword set ---
            current_keywords = self._normalize_queries_to_keywords(current_queries)

            if not current_keywords:
                return False, ""

            # --- Compare against each previous cycle's keywords ---
            highest_similarity = 0.0
            most_similar_cycle = 0

            for prev_cycle, prev_queries in self._cycle_queries.items():
                if prev_cycle >= self._cycle:
                    continue  # Only compare against earlier cycles
                if not prev_queries:
                    continue

                prev_keywords = self._normalize_queries_to_keywords(prev_queries)
                if not prev_keywords:
                    continue

                # --- Jaccard similarity: |intersection| / |union| ---
                intersection = current_keywords & prev_keywords
                union = current_keywords | prev_keywords
                similarity = len(intersection) / len(union) if union else 0.0

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_cycle = prev_cycle

            # --- Evaluate against threshold ---
            if highest_similarity >= STAGNATION_SIMILARITY_THRESHOLD:
                overlap_keywords = (
                    current_keywords &
                    self._normalize_queries_to_keywords(
                        self._cycle_queries.get(most_similar_cycle, [])
                    )
                )
                detail = (
                    f"Cycle {self._cycle} queries overlap "
                    f"{highest_similarity:.0%} with cycle {most_similar_cycle}. "
                    f"Shared terms: {', '.join(sorted(overlap_keywords)[:8])}"
                )
                logger.info(
                    f"🔍 OODA STAGNATION CHECK | {detail}"
                )
                return True, detail
            else:
                logger.debug(
                    f"🔍 OODA STAGNATION CHECK | Cycle {self._cycle} | "
                    f"Highest similarity: {highest_similarity:.0%} "
                    f"(threshold: {STAGNATION_SIMILARITY_THRESHOLD:.0%}) — no stagnation"
                )
                return False, ""

        except Exception as e:
            # Stagnation detection must never crash the research loop
            logger.warning(
                f"OODA _check_stagnation error (non-critical): {e}"
            )
            return False, ""

    def _normalize_queries_to_keywords(self, queries: List[str]) -> set:
        """
        Convert a list of query strings into a normalized keyword set.

        Strips common stop words, lowercases, and splits on whitespace
        and delimiters. Returns a set suitable for Jaccard comparison.

        Args:
            queries (List[str]): Raw query strings from command signatures.

        Returns:
            set: Normalized keyword set (lowercase, no stop words).
        """
        # Common stop words that don't carry research topic signal
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'not', 'no', 'it', 'its', 'this', 'that', 'these', 'those',
            'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why',
            'about', 'type', 'limit', 'turns', 'query', 'search', 'find',
        }

        keywords = set()
        for query in queries:
            # Split on whitespace, pipes, equals signs, common delimiters
            tokens = re.split(r'[\s|=,;:]+', query.lower())
            for token in tokens:
                # Strip non-alphanumeric edges, keep meaningful tokens
                clean = token.strip('()[]{}"\'.!?')
                if clean and len(clean) >= 3 and clean not in stop_words:
                    keywords.add(clean)

        return keywords

    def _format_command_log(self) -> str:
        """
        Format the structured command log for injection into prompts.

        Converts the list-of-dicts command log into a human-readable string
        that shows QWEN exactly what it searched for in each cycle. This
        replaces the old flat counter format ("Cycle 4: 3 command(s) executed")
        with actionable detail ("Cycle 4: SEARCH: apricot pruning timing").

        Returns:
            str: Formatted command log string, or "None recorded." if empty.
        """
        if not self._command_log:
            return "None recorded."

        try:
            sections = []
            for entry in self._command_log:
                cycle_num = entry.get("cycle", "?")
                commands = entry.get("commands", [])
                if commands:
                    # Format each command on its own line, indented under the cycle
                    cmd_lines = "\n  ".join(commands)
                    sections.append(f"Cycle {cycle_num}:\n  {cmd_lines}")
                else:
                    sections.append(f"Cycle {cycle_num}: (no commands)")

            return "\n".join(sections)

        except Exception as e:
            logger.warning(
                f"OODA _format_command_log error (non-critical): {e}"
            )
            # Fallback: dump raw structure
            return str(self._command_log)

    def _get_total_commands_count(self) -> int:
        """
        Count total commands across all cycles from the structured command log.

        Used for summary headers and logging where a flat count is needed.

        Returns:
            int: Total number of command signatures recorded across all cycles.
        """
        try:
            return sum(
                len(entry.get("commands", []))
                for entry in self._command_log
            )
        except Exception:
            return 0

    def _clean_empty_search_results(self, text: str) -> str:
        """
        Compact verbose empty search result blocks before accumulating context.

        When deepseek.py finds no results it formats the response as a
        multi-line block:
            **===== SEARCH RESULTS =====**
            **NO RESULTS FOUND**
            **===== END OF SEARCH =====**

        Inside OODA's accumulated_context, these blocks add ~85 chars of
        formatting markup per empty search with zero information value.
        On a deep research run with many cycles this becomes significant
        prompt bloat that pushes useful context out of the window.

        This method replaces each empty block with a compact single-line token
        that preserves the information (a search ran and found nothing) while
        eliminating the noise. Search results that DID return content are
        left completely untouched.

        Addresses QWEN Suggestion 1 from orchard guide test (2026-03-28):
        "clearer separation between structural guidance and operational
        notifications."

        Args:
            text (str): Processed LLM output after command execution.

        Returns:
            str: Text with empty result blocks compacted to single-line tokens.
                 Actual search results are returned unchanged.
        """
        try:
            # Match the full NO RESULTS block with flexible whitespace handling.
            # {5,} allows for slight formatting variations in the === delimiters.
            # DOTALL allows . to match newlines within the block.
            no_results_pattern = re.compile(
                r'\*\*={5,}\s*SEARCH RESULTS\s*={5,}\*\*\s*'
                r'\*\*NO RESULTS FOUND\*\*\s*'
                r'\*\*={5,}\s*END OF SEARCH\s*={5,}\*\*',
                re.IGNORECASE | re.DOTALL
            )

            original_len = len(text)

            # Replace with compact token — preserves the fact that a search
            # ran and returned nothing, without consuming prompt space
            cleaned = no_results_pattern.sub('[SEARCH → 0 results]', text)

            if len(cleaned) != original_len:
                savings = original_len - len(cleaned)
                logger.info(
                    f"🔍 OODA CLEAN | Compacted empty search block(s) | "
                    f"Saved {savings} chars from accumulated context"
                )

            return cleaned

        except Exception as e:
            # Never let a cleanup step block the research loop
            logger.warning(
                f"OODA _clean_empty_search_results() error (non-critical): {e}"
            )
            return text  # Return original unchanged on any error

    def _build_cycle_warning(self) -> str:
        """
        Build a contextually appropriate wrap-up warning based on current cycle.

        Returns an empty string in early cycles, escalating text near ceiling.

        Returns:
            str: Warning text for injection into OBSERVE/DECIDE prompts,
                 or empty string if no warning is warranted.
        """
        remaining = self.max_cycles - self._cycle

        if remaining <= 0:
																	 
            return (
                "⚠️ CRITICAL: You have reached the maximum research depth. "
                "You MUST declare COMPLETE in your DECIDE response. "
                "Synthesize what you have — do not initiate new research."
            )
        elif remaining <= 2:
            return (
                f"⚠️ IMPORTANT: Only {remaining} research cycle(s) remaining. "
                "Begin wrapping up and preparing for synthesis unless a truly "
                "critical gap remains."
            )
        elif remaining <= 4:
            return (
                f"ℹ️ NOTE: {remaining} research cycles remaining before maximum depth. "
                "Consider whether you have enough to answer the task well."
            )
        else:
			# Early cycles — no warning needed									
            return ""

    def _format_accumulated_context(self) -> str:
        """
        Format accumulated context using a sliding window compression strategy.

        Recent cycles (last CONTEXT_FULL_WINDOW entries) are included with their
        complete observation text — QWEN needs this detail for coherent reasoning
        about what it just found. Older cycles are represented only by their
        compressed summaries, preserving awareness of earlier findings without
        consuming the context window.

        A hard character ceiling (CONTEXT_BUDGET_CHARS) is enforced as a final
        safety net, trimming from the oldest entries first.

        This method is called by _build_observe_prompt(), _build_decide_prompt(),
        and _build_act_prompt() — all three benefit from bounded context.

        Returns:
            str: Formatted context string with windowed compression, or
                 placeholder if no research has been gathered yet.
        """
        if not self._accumulated_context:
            return "(No research gathered yet.)"

        try:
            total_entries = len(self._accumulated_context)

            # --- Split into older (summary-only) and recent (full text) ---
            # If we have fewer entries than the window, everything is "recent"
            if total_entries <= CONTEXT_FULL_WINDOW:
                older_entries = []
                recent_entries = self._accumulated_context
            else:
                older_entries = self._accumulated_context[:-CONTEXT_FULL_WINDOW]
                recent_entries = self._accumulated_context[-CONTEXT_FULL_WINDOW:]

            # --- Build formatted sections ---
            sections = []

            # Older cycles: compressed summary only
            if older_entries:
                sections.append("=== Earlier Research (compressed summaries) ===")
                for entry in older_entries:
                    cycle_num = entry.get("cycle", "?")
                    summary = entry.get("summary", "(no summary available)")
                    sections.append(
                        f"--- Cycle {cycle_num} Summary ---\n{summary}"
                    )
                sections.append("")  # Blank line separator

            # Recent cycles: full observation text
            if recent_entries:
                if older_entries:
                    # Only add this header when there's a contrast with older entries
                    sections.append("=== Recent Research (full detail) ===")
                for entry in recent_entries:
                    cycle_num = entry.get("cycle", "?")
                    full_text = entry.get("full", "(no observation text)")
                    sections.append(
                        f"--- Cycle {cycle_num} Observation ---\n{full_text}"
                    )

            # --- Assemble and enforce hard character ceiling ---
            assembled = "\n".join(sections)
            bounded = self._enforce_context_budget(assembled, older_entries)

            # --- Log compression metrics for debugging ---
            uncompressed_total = sum(
                len(entry.get("full", "")) for entry in self._accumulated_context
            )
            logger.info(
                f"📊 OODA CONTEXT | Entries: {total_entries} | "
                f"Older (summary): {len(older_entries)} | "
                f"Recent (full): {len(recent_entries)} | "
                f"Uncompressed: {uncompressed_total} chars | "
                f"Assembled: {len(bounded)} chars | "
                f"Compression: {(1 - len(bounded) / max(uncompressed_total, 1)) * 100:.0f}%"
            )

            return bounded

        except Exception as e:
            # Context formatting must never crash the research loop.
            # Fall back to a simple dump of whatever we have.
            logger.error(
                f"OODA _format_accumulated_context() error: {e}",
                exc_info=True
            )
            fallback_parts = []
            for entry in self._accumulated_context:
                if isinstance(entry, dict):
                    fallback_parts.append(
                        entry.get("summary", entry.get("full", str(entry)))
                    )
                else:
                    # Safety net for any unexpected non-dict entries
                    fallback_parts.append(str(entry))
            return "\n".join(fallback_parts)

    def _extract_cycle_summary(self, processed_output: str) -> str:
        """
        Extract the CYCLE_SUMMARY block from QWEN's OBSERVE output.

        QWEN is instructed to end its observation with:
            CYCLE_SUMMARY: [2-3 sentences with key findings and gaps]

        If QWEN complies, we extract and return that summary. If not (the model
        didn't follow the instruction), we fall back to a generous truncation
        of the full output — first FALLBACK_SUMMARY_CHARS characters plus a
        [truncated] marker.

        This costs zero extra LLM calls. The summary is part of the OBSERVE
        output that QWEN was already generating.

        Args:
            processed_output (str): The full processed OBSERVE output after
                                    command execution and empty-block cleaning.

        Returns:
            str: Extracted summary (typically 100-400 chars) or fallback
                 truncation (FALLBACK_SUMMARY_CHARS chars).
        """
        try:
            # --- Attempt to parse CYCLE_SUMMARY block ---
            # Look for the marker, then capture everything after it.
            # The summary may span multiple lines if QWEN wrote a detailed one.
            summary_match = re.search(
                r'CYCLE_SUMMARY\s*:\s*(.+)',
                processed_output,
                re.IGNORECASE | re.DOTALL
            )

            if summary_match:
                # Extract the summary text after the marker
                raw_summary = summary_match.group(1).strip()

                # Trim to first meaningful block — stop at double newline or
                # if QWEN started issuing commands after the summary
                # (e.g. a stray [STORE:] after the summary line)
                clean_summary = re.split(
                    r'\n\s*\n|\[(?:SEARCH|STORE|DISCUSS_WITH_CLAUDE|WEB_SEARCH)',
                    raw_summary
                )[0].strip()

                if len(clean_summary) >= 20:
                    # Successful extraction — log and return
                    logger.info(
                        f"🔍 OODA SUMMARY | Cycle {self._cycle} | "
                        f"Extracted CYCLE_SUMMARY: {len(clean_summary)} chars"
                    )
                    return clean_summary

                # Summary was present but too short to be useful — fall through
                logger.warning(
                    f"🔍 OODA SUMMARY | Cycle {self._cycle} | "
                    f"CYCLE_SUMMARY found but too short ({len(clean_summary)} chars) "
                    f"— using fallback truncation"
                )

            # --- Fallback: generous truncation of full output ---
            # No parseable CYCLE_SUMMARY found — take the first
            # FALLBACK_SUMMARY_CHARS characters. This is still a major
            # reduction from typical 3-5K OBSERVE outputs.
            if len(processed_output) <= FALLBACK_SUMMARY_CHARS:
                # Output is short enough to use in full as its own summary
                logger.info(
                    f"🔍 OODA SUMMARY | Cycle {self._cycle} | "
                    f"No CYCLE_SUMMARY found, output short enough to use in full "
                    f"({len(processed_output)} chars)"
                )
                return processed_output
            else:
                truncated = processed_output[:FALLBACK_SUMMARY_CHARS].rstrip()
                # Try to break at the last sentence boundary for cleaner reading
                last_period = truncated.rfind('.')
                last_newline = truncated.rfind('\n')
                # Use whichever boundary is later (closer to the end)
                break_point = max(last_period, last_newline)
                if break_point > FALLBACK_SUMMARY_CHARS * 0.5:
                    # Only use the boundary if it's in the back half,
                    # otherwise we'd lose too much content
                    truncated = truncated[:break_point + 1]

                logger.info(
                    f"🔍 OODA SUMMARY | Cycle {self._cycle} | "
                    f"No CYCLE_SUMMARY found — fallback truncation: "
                    f"{len(truncated)} chars from {len(processed_output)} chars"
                )
                return truncated + " [truncated]"

        except Exception as e:
            # Summary extraction must never crash the research loop
            logger.warning(
                f"OODA _extract_cycle_summary() error (non-critical): {e}"
            )
            # Ultra-safe fallback — just take the first chunk
            return processed_output[:FALLBACK_SUMMARY_CHARS] + " [truncated]"

    def _enforce_context_budget(self, assembled: str, older_entries: list) -> str:
        """
        Enforce the hard character ceiling on assembled context.

        If the assembled context string exceeds CONTEXT_BUDGET_CHARS, trims
        from the oldest summaries first (top of the string) to bring it under
        budget. Recent full-text entries are preserved as long as possible since
        they contain the most relevant detail for current reasoning.

        This is the final safety net — it guarantees bounded prompt size
        regardless of how many cycles ran or how verbose QWEN's outputs were.

        Args:
            assembled (str): The fully formatted context string from
                             _format_accumulated_context().
            older_entries (list): The older context entries (summary-only) that
                                  can be trimmed first if budget is exceeded.

        Returns:
            str: Context string guaranteed to be within CONTEXT_BUDGET_CHARS,
                 or the original if already under budget.
        """
        try:
            if len(assembled) <= CONTEXT_BUDGET_CHARS:
                # Under budget — no trimming needed
                return assembled

            # --- Over budget: rebuild with progressively fewer older entries ---
            overage = len(assembled) - CONTEXT_BUDGET_CHARS
            logger.warning(
                f"📊 OODA BUDGET | Context exceeds budget by {overage} chars "
                f"({len(assembled)}/{CONTEXT_BUDGET_CHARS}) — trimming oldest entries"
            )

            # Strategy: drop oldest summary entries one at a time until
            # the re-assembled context fits. This preserves the most recent
            # older summaries which are more likely to be relevant.
            # We rebuild by re-calling format logic with fewer entries rather
            # than doing string surgery, to keep the output clean.

            # Simple approach: truncate from the front of the assembled string
            # at section boundaries (--- Cycle N ---) to preserve structure
            trimmed = assembled
            entries_dropped = 0

            while len(trimmed) > CONTEXT_BUDGET_CHARS:
                # Find the first cycle section boundary after the header
                section_break = trimmed.find("\n--- Cycle", 10)
                if section_break == -1:
                    # No more sections to drop — hard truncate from front
                    trimmed = "...[earlier research trimmed for context budget]...\n" + \
                              trimmed[-(CONTEXT_BUDGET_CHARS - 60):]
                    entries_dropped += 1
                    break

                # Drop everything up to (but not including) the next section
                trimmed = trimmed[section_break:].lstrip('\n')
                entries_dropped += 1

            # Re-add a header if we dropped entries
            if entries_dropped > 0:
                header = (
                    f"[{entries_dropped} earlier cycle(s) trimmed for context budget]\n\n"
                )
                trimmed = header + trimmed

                logger.info(
                    f"📊 OODA BUDGET | Trimmed {entries_dropped} section(s) | "
                    f"Final size: {len(trimmed)} chars"
                )

            return trimmed

        except Exception as e:
            # Budget enforcement must never crash the loop
            logger.warning(
                f"OODA _enforce_context_budget() error (non-critical): {e}"
            )
            # Hard ceiling fallback — just truncate with a notice
            if len(assembled) > CONTEXT_BUDGET_CHARS:
                return (
                    "[context truncated for budget]\n" +
                    assembled[-(CONTEXT_BUDGET_CHARS - 40):]
                )
            return assembled

    def _update_status(self, message: str) -> None:
        """
        Write a status update to the Streamlit st.status() container.

        Gracefully handles None container or unavailable Streamlit context
        (e.g. unit tests). Never raises — a failed status update must not
        interrupt the research loop.

        Args:
            message (str): Status message to display.
        """
        try:
			# Always log — even if Streamlit isn't available												  
            logger.info(f"OODA STATUS | {message}")

            if self._status_container is not None:
				# st.status() containers support .update() and .write()
                # Use write() to append messages progressively													   
                self._status_container.write(message)

        except Exception as e:
			# Never let a status update crash the loop										  
            logger.warning(f"OODA _update_status() failed (non-critical): {e}")