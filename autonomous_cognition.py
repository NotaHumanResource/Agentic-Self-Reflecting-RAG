# autonomous_cognition.py
"""Autonomous cognition system for DeepSeek to enable self-prompted thinking and learning."""
import re
import os
import time
import logging
import threading
import datetime
import uuid
import random
import sqlite3
import json
from typing import Dict, List, Any, Tuple  # DEAD CODE TEST 2026-05-17: was 'Dict, List, Any, Optional, Tuple' — Optional unused per ruff F401
from knowledge_gap import KnowledgeGapQueue
# from web_knowledge_seeker import WebKnowledgeSeeker  # DEAD CODE TEST 2026-05-17: unused per ruff F401 + vulture
# QUARANTINED 2026-05-19: ClaudeKnowledgeIntegration class renamed to _UNUSED_ClaudeKnowledgeIntegration
# in claude_knowledge.py (batch 4 web cleanup pass). Only consumer was the already-quarantined
# _UNUSED__initiate_ai_communication method below. The active DISCUSS_WITH_CLAUDE feature uses
# deepseek.py's own direct Anthropic API client (_resolve_claude_api_key + _make_claude_api_call_with_enhanced_prompt),
# not this class. Import commented out to prevent ImportError at module load time. If the quarantined
# caller is ever reached, NameError on ClaudeKnowledgeIntegration will surface the silent dispatch.
# from claude_knowledge import ClaudeKnowledgeIntegration
# from collections import Counter  # DEAD CODE TEST 2026-05-17: unused at module level, re-imported locally at line 4535 (ruff F401/F811)

# --- Set up autonomous cognition logger ---
autonomous_logger = logging.getLogger('autonomous_cognition')
autonomous_logger.setLevel(logging.INFO)
autonomous_logger.propagate = True  # Let it propagate to root logger too

# Add console handler if needed
if not autonomous_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    autonomous_logger.addHandler(console_handler)

# ==============================================================================
# MEMORY CONSOLIDATION PULSE
# Synthesizes related self_reflection memories into unified insights.
# Runs autonomously during idle cycles. Prevents snowball via ceiling checks
# and lineage tracking. Source memories are never deleted — only marked via
# metadata so they remain searchable but won't be re-consolidated.
# Added: 2026-04-21
# ==============================================================================

# --- Constants for consolidation pulse ---
CONSOLIDATION_MAX_CONTENT_LENGTH = 2000   # Max chars for any single synthesis memory
CONSOLIDATION_MAX_ROUNDS = 5              # Max times a synthesis can be refined
CONSOLIDATION_MIN_CLUSTER_SIZE = 3        # Min source memories to trigger synthesis
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.60 # Min Qdrant score to group memories
CONSOLIDATION_CANDIDATE_LIMIT = 50        # Max candidates pulled from SQLite per run
CONSOLIDATION_SYNTHESIS_CONFIDENCE = 0.85 # Confidence assigned to new synthesis memories
CONSOLIDATION_SOURCE_MIN_CONFIDENCE = 0.15  # 48-hour interval set 2026-04-21  Candidates below this are skipped (already pruned)

class AutonomousCognition:
    """Manages autonomous memory management to enhance personalization without self-criticism."""
    
    def __init__(self, chatbot, memory_db=None, vector_db=None):
        """Initialize the autonomous cognition system."""
        try:
            # Fix logging handler levels for autonomous cognition
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if handler.level > logging.INFO:
                    handler.setLevel(logging.INFO)
                    logging.info(f"Fixed handler level to INFO: {type(handler)}")
            
            self.chatbot = chatbot
            self.memory_db = memory_db or chatbot.memory_db
            self.vector_db = vector_db or chatbot.vector_db
            self.thinking_thread = None
            self.stop_flag = threading.Event()
            self.last_autonomous_thought = None
            self.cognitive_state = "idle"
            self.last_user_activity = time.time() # Initialize with current time
            self.cognitive_cycle_interval = 300  # 5 minutes (was 3600 / 1 hour pre-2026-05-24 Track A Issue 4).
            # 5-minute cycle ensures the 30-minute scheduled-reflection windows are reliably caught.
            # Per-activity cooldowns (min_interval_hours: 4h–96h) prevent thrashing of the
            # weighted-pool activities even at this faster tick rate.
            self.thought_history = []             # Store recent autonomous thoughts
            self.max_thought_history = 10         # Maximum number of thoughts to keep in memory
            self.rate_limited = False
            self.llm_error_count = 0
            
            # NEW: Track recent FORGET commands to prevent forgetting rampages
            self.recent_forgets = []  # List of (timestamp, content_preview) tuples
            self.max_forgets_per_period = 5  # Maximum forgets allowed in time window
            self.forget_cooldown_period = 300  # 5 minutes in seconds

            # ✅ SINGLE REFLECTION PATH INITIALIZATION - Relative path for portability
            # (This replaces the two duplicate initializations)
            self.reflection_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reflections")
            os.makedirs(self.reflection_path, exist_ok=True)
            logging.info(f"Initialized reflection directory: {self.reflection_path}")
            
            # Initialize thoughts collection for _record_thought method
            self.thoughts = []
            
            # Define cognitive activities and their weights (probability of being selected)
            self.cognitive_activities = {
                # Note (2026-05-24, Track A Issue 4): check_scheduled_reflections
                # was previously listed here with weight 1.0 / min_interval_hours 0.5.
                # It is now invoked directly from _cognitive_loop on every iteration,
                # independent of the 1-hour idle gate, so its 30-minute scheduled
                # windows are reliably caught regardless of recent user activity.
                # Idempotency is provided by JSON completion files in reflections/.

                # Knowledge gap pipeline - synchronized intervals
                "analyze_knowledge_gaps": {"weight": 0.85, "last_run": None, "min_interval_hours": 96},
                "fill_knowledge_gaps": {"weight": 0.90, "last_run": None, "min_interval_hours": 96},

                # Medium frequency - memory quality and truth evaluation
                "audit_memory_confidence": {"weight": 0.6, "last_run": None, "min_interval_hours": 84},

                # Lightweight synthesis pulse - finds patterns across self_reflections
                "memory_consolidation_pulse": {"weight": 0.7, "last_run": None, "min_interval_hours": 48},

                # Present-tense functional state pulse — QWEN meta-reflects on
                # current orientation using recent memory signals.
                # Cheap: 3 direct SQL queries + 1 short LLM call.
                # Updates the cognitive state widget with ORIGIN_AUTONOMOUS.
                # The existing loop idle-guard (lines ~2688-2712) ensures this
                # never fires during an active conversation — no extra check needed.
                "functional_state_baseline": {"weight": 0.85, "last_run": None, "min_interval_hours": 4},

                # Self-model integrity check — compares QWEN's stated self-model
                # (consolidation_synthesis + type=self memories) against recent
                # behavioral signal (conversation_summary). Produces aligned /
                # evolved / drifted outcome. Session-only state update, no DB writes.
                # Heavier than baseline: 3 SQL queries + complex LLM comparison.
                "self_model_integrity_check": {"weight": 0.75, "last_run": None, "min_interval_hours": 48},

                # Wander curiosity — Default Mode Network analog.
                # Starts from QWEN's current self-model state and asks:
                # "what am I most curious about right now?" QWEN generates her own
                # question and pursues it across 3 internal reasoning passes.
                # Unlike reflections (backward-looking pattern-finding), this is
                # forward-looking, self-initiated inquiry. Runs frequently during idle
                # time — the primary idle-state inner life activity.
                # Stores result as type=wander_insight (both DBs). 3 LLM calls per run.
                # Added: 2026-05-26
                "wander_curiosity": {"weight": 0.90, "last_run": None, "min_interval_hours": 2}
            }
            
            logging.info("Autonomous Cognition system initialized")
            
        except Exception as e:
            logging.critical(f"Error in AutonomousCognition.__init__: {e}", exc_info=True)
     
    def _should_run_activity(self, activity_name):
        """
        Check if an activity should run based on its last run time and minimum interval.
        
        Args:
            activity_name (str): Name of the activity to check
            
        Returns:
            bool: True if the activity should run, False otherwise
        """
        if activity_name not in self.cognitive_activities:
            logging.warning(f"Unknown activity: {activity_name}")
            return False
            
        activity_info = self.cognitive_activities[activity_name]
        last_run = activity_info.get("last_run")
        
        # If never run before, should run
        if last_run is None:
            return True
            
        # Get minimum interval in seconds (default to 12 hours if not specified)
        min_interval_hours = activity_info.get("min_interval_hours", 12)
        min_interval_seconds = min_interval_hours * 3600
        
        # Check if enough time has passed since last run
        time_since_last_run = time.time() - last_run
        return time_since_last_run >= min_interval_seconds

    # -------------------------------------------------------------------------
    # AUTONOMOUS LOOP DISPATCH WRAPPERS
    # The main cognitive loop resolves methods as f"_{activity_key}" via hasattr.
    # Phase 1, 2, and 3 methods use a _perform_ / named prefix for clarity, so
    # thin wrappers here bridge the naming gap without renaming the real methods.
    # All existing activities (check_scheduled_reflections, fill_knowledge_gaps,
    # audit_memory_confidence, analyze_knowledge_gaps) already match the dispatch
    # pattern directly and need no wrapper.
    # -------------------------------------------------------------------------

    def _memory_consolidation_pulse(self) -> bool:
        """Dispatch wrapper → _perform_memory_consolidation_pulse (Phase 1)."""
        return self._perform_memory_consolidation_pulse()

    def _functional_state_baseline(self) -> bool:
        """Dispatch wrapper → _perform_functional_state_baseline (Phase 2)."""
        return self._perform_functional_state_baseline()

    def _self_model_integrity_check(self) -> bool:
        """Dispatch wrapper → _perform_self_model_integrity_check (Phase 3)."""
        return self._perform_self_model_integrity_check()

    def _wander_curiosity(self) -> bool:
        """Dispatch wrapper → _perform_wander_curiosity (Default Mode Network)."""
        return self._perform_wander_curiosity()

    def _check_scheduled_reflections(self):
        """Check if any scheduled reflections are due and execute them."""
        try:
            # Import here to avoid circular imports
            from utils import load_reflection_schedule
            
            # Load the reflection schedule from JSON
            schedule = load_reflection_schedule()
            current_time = datetime.datetime.now()
            
            logging.info(f"Checking scheduled reflections at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.debug(f"Current schedule: {schedule}")
            
            executed_any = False
            
            # Check daily reflection (6:15 AM)
            if schedule.get('daily', False):
                if (current_time.hour == 6 and 15 <= current_time.minute <= 45 and  # 30-minute window
                    not self._reflection_already_run_today('daily')):
                    logging.info("Executing scheduled daily reflection at 6:15 AM")
                    success = self._perform_daily_reflection()
                    if success:
                        self._mark_reflection_completed('daily', current_time)
                        executed_any = True
            
            # Check weekly reflection (Sunday, 9:15 AM)
            if schedule.get('weekly', False):
                if (current_time.weekday() == 6 and current_time.hour == 9 and 
                    15 <= current_time.minute <= 45 and  # 30-minute window
                    not self._reflection_already_run_this_week('weekly')):
                    logging.info("Executing scheduled weekly reflection on Sunday at 9:15 AM")
                    success = self._perform_weekly_reflection()
                    if success:
                        self._mark_reflection_completed('weekly', current_time)
                        executed_any = True
            
            # Check monthly reflection (1st day, 12:20 PM)
            if schedule.get('monthly', False):
                if (current_time.day == 1 and current_time.hour == 12 and 
                    20 <= current_time.minute <= 50 and  # 30-minute window
                    not self._reflection_already_run_this_month('monthly')):
                    logging.info("Executing scheduled monthly reflection on 1st day at 12:20 PM")
                    success = self._perform_monthly_reflection()
                    if success:
                        self._mark_reflection_completed('monthly', current_time)
                        executed_any = True
                    
            if executed_any:
                logging.info("Completed scheduled reflection execution")
            else:
                logging.debug("No scheduled reflections due at this time")
                
            return executed_any
            
        except ImportError:
            logging.warning("utils module not available for scheduled reflection checking")
            return False
        except Exception as e:
            logging.error(f"Error checking scheduled reflections: {e}", exc_info=True)
            return False

    def _reflection_already_run_today(self, reflection_type):
        """Check if a daily reflection has already been run today."""
        try:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            completion_file = os.path.join(self.reflection_path, f"{reflection_type}_completions.json")
            
            if os.path.exists(completion_file):
                with open(completion_file, 'r') as f:
                    completions = json.load(f)
                return today in completions.get('daily_runs', [])
            
            return False
        except Exception as e:
            logging.error(f"Error checking daily reflection status: {e}")
            return False

    def _reflection_already_run_this_week(self, reflection_type):
        """Check if a weekly reflection has already been run this week."""
        try:
            current_time = datetime.datetime.now()
            # Get Monday of current week
            days_since_monday = current_time.weekday()
            monday = current_time - datetime.timedelta(days=days_since_monday)
            week_key = monday.strftime('%Y-W%U')  # Year-Week format
            
            completion_file = os.path.join(self.reflection_path, f"{reflection_type}_completions.json")
            
            if os.path.exists(completion_file):
                with open(completion_file, 'r') as f:
                    completions = json.load(f)
                return week_key in completions.get('weekly_runs', [])
            
            return False
        except Exception as e:
            logging.error(f"Error checking weekly reflection status: {e}")
            return False

    def _reflection_already_run_this_month(self, reflection_type):
        """Check if a monthly reflection has already been run this month."""
        try:
            current_month = datetime.datetime.now().strftime('%Y-%m')
            completion_file = os.path.join(self.reflection_path, f"{reflection_type}_completions.json")
            
            if os.path.exists(completion_file):
                with open(completion_file, 'r') as f:
                    completions = json.load(f)
                return current_month in completions.get('monthly_runs', [])
            
            return False
        except Exception as e:
            logging.error(f"Error checking monthly reflection status: {e}")
            return False

    def _mark_reflection_completed(self, reflection_type, completion_time):
        """Mark a reflection as completed to prevent duplicates."""
        try:
            completion_file = os.path.join(self.reflection_path, f"{reflection_type}_completions.json")
            
            # Load existing completions
            if os.path.exists(completion_file):
                with open(completion_file, 'r') as f:
                    completions = json.load(f)
            else:
                completions = {'daily_runs': [], 'weekly_runs': [], 'monthly_runs': []}
            
            # Add completion based on type
            if reflection_type == 'daily':
                today = completion_time.strftime('%Y-%m-%d')
                if today not in completions['daily_runs']:
                    completions['daily_runs'].append(today)
                    # Keep only last 7 days
                    completions['daily_runs'] = completions['daily_runs'][-7:]
                    
            elif reflection_type == 'weekly':
                days_since_monday = completion_time.weekday()
                monday = completion_time - datetime.timedelta(days=days_since_monday)
                week_key = monday.strftime('%Y-W%U')
                if week_key not in completions['weekly_runs']:
                    completions['weekly_runs'].append(week_key)
                    # Keep only last 8 weeks
                    completions['weekly_runs'] = completions['weekly_runs'][-8:]
                    
            elif reflection_type == 'monthly':
                month_key = completion_time.strftime('%Y-%m')
                if month_key not in completions['monthly_runs']:
                    completions['monthly_runs'].append(month_key)
                    # Keep only last 12 months
                    completions['monthly_runs'] = completions['monthly_runs'][-12:]
            
            # Save updated completions
            with open(completion_file, 'w') as f:
                json.dump(completions, f, indent=2)
                
            logging.info(f"Marked {reflection_type} reflection as completed for {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logging.error(f"Error marking reflection as completed: {e}", exc_info=True)

    def _perform_daily_reflection(self):
        """Trigger daily self-reflection through reflection engine."""
        try:
            logging.info("Autonomous cognition triggering scheduled daily reflection")
            
            if hasattr(self.chatbot, 'reflection_engine') and self.chatbot.reflection_engine:
                success = self.chatbot.reflection_engine.perform_self_reflection(
                    reflection_type="daily",
                    llm=self.chatbot.llm if hasattr(self.chatbot, 'llm') else None
                )
                
                if success:
                    logging.info("Successfully completed scheduled daily reflection")
                    return True
                else:
                    logging.warning("Scheduled daily reflection failed")
                    return False
            else:
                logging.warning("Reflection engine not available for scheduled daily reflection")
                return False
                
        except Exception as e:
            logging.error(f"Error in scheduled daily reflection: {e}", exc_info=True)
            return False

    def _perform_weekly_reflection(self):
        """Trigger weekly self-reflection through reflection engine."""
        try:
            logging.info("Autonomous cognition triggering scheduled weekly reflection")
            
            if hasattr(self.chatbot, 'reflection_engine') and self.chatbot.reflection_engine:
                success = self.chatbot.reflection_engine.perform_self_reflection(
                    reflection_type="weekly", 
                    llm=self.chatbot.llm if hasattr(self.chatbot, 'llm') else None
                )
                
                if success:
                    logging.info("Successfully completed scheduled weekly reflection")
                    return True
                else:
                    logging.warning("Scheduled weekly reflection failed")
                    return False
            else:
                logging.warning("Reflection engine not available for scheduled weekly reflection")
                return False
                
        except Exception as e:
            logging.error(f"Error in scheduled weekly reflection: {e}", exc_info=True)
            return False

    def _perform_monthly_reflection(self):
        """Trigger monthly self-reflection through reflection engine."""
        try:
            logging.info("Autonomous cognition triggering scheduled monthly reflection")
            
            if hasattr(self.chatbot, 'reflection_engine') and self.chatbot.reflection_engine:
                success = self.chatbot.reflection_engine.perform_self_reflection(
                    reflection_type="monthly", 
                    llm=self.chatbot.llm if hasattr(self.chatbot, 'llm') else None
                )
                
                if success:
                    logging.info("Successfully completed scheduled monthly reflection")
                    return True
                else:
                    logging.warning("Scheduled monthly reflection failed")
                    return False
            else:
                logging.warning("Reflection engine not available for scheduled monthly reflection")
                return False
                
        except Exception as e:
            logging.error(f"Error in scheduled monthly reflection: {e}", exc_info=True)
            return False

    # REMOVED 2026-05-24 (Track A, Issue 6): _reflect_on_capabilities method deleted.
    # Was never registered in self.cognitive_activities, so the dispatcher could
    # never reach it. Verified no external callers across all 36 Python files
    # in the repo before removal.
    # Note: db_maintenance.py still recognizes "capabilities_reflection" and
    # "# Capabilities Reflection" string patterns to protect legacy memories
    # that may exist in the database from when this method was registered.

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Memory effectiveness analyzer — substantial implementation but never wired into the cognitive activity registry.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__analyze_memory_usage(self):
        """
        Analyze how effectively memory commands are being used and identify patterns for improvement.
        Focuses on helping the system become more personalized without being self-critical.
        """
        logging.info("Starting memory usage analysis")
        
        try:
            # Get command usage statistics from enhancer if available
            command_stats = {}
            if hasattr(self.chatbot, 'deepseek_enhancer') and hasattr(self.chatbot.deepseek_enhancer, 'lifetime_counters'):
                command_stats = self.chatbot.deepseek_enhancer.lifetime_counters.get_counters()
                
            # If no stats available, log and exit
            if not command_stats:
                logging.warning("No memory command statistics available for analysis")
                return
                
            # Get total command count
            total_commands = sum(count for cmd, count in command_stats.items() 
                                if cmd != 'total')  # Exclude the 'total' counter
            
            if total_commands == 0:
                logging.info("No memory commands have been used yet")
                return
                
            # Calculate command distribution
            cmd_distribution = {cmd: (count / total_commands) * 100 
                            for cmd, count in command_stats.items() 
                            if cmd != 'total'}
                
            # Log the command distribution
            logging.info(f"Memory command distribution: {cmd_distribution}")
            
            # Create analysis prompt based on command usage
            prompt = f""" /no_think
            I will analyze my memory command usage patterns to identify opportunities for improvement:
            
            Command usage statistics:
            {cmd_distribution}
            
            Total commands used: {total_commands}
            
            Based on this data, I will identify:
            1. Which memory commands I'm using effectively
            2. Which commands I could utilize more to enhance personalization
            3. Pattern improvements to better assist the user
            4. Strategies to make conversations more natural while utilizing my memory
            
            The goal is to become a more helpful assistant that remembers user preferences 
            and important information without being self-critical or interrupting the flow of conversation.
            """
            
            # Generate analysis
            if hasattr(self.chatbot, 'llm'):
                analysis = self.chatbot.llm.invoke(prompt)
                
                # Extract actionable insights rather than self-criticism
                insights_prompt = f""" /no_think
                Based on my analysis of memory command usage:
                {analysis}
                
                I will identify 3-5 specific, actionable strategies to improve my memory usage
                to better serve the user through personalization. These should focus on:
                
                1. Storing more relevant personal information
                2. Retrieving memories at appropriate times
                3. Making conversations feel more continuous and personal
                4. Practical patterns for memory command integration in natural conversation
                5. Using proper metadata tagging for better memory organization
                
                Each strategy should be concrete and implementable without being self-critical.
                """
                
                insights = self.chatbot.llm.invoke(insights_prompt)
                
                # Store the analysis with actionable focus (not self-critique)
                analysis_thought = f"# Memory Usage Analysis\n\n{analysis}\n\n## Improvement Strategies\n\n{insights}"
                self._store_autonomous_thought(analysis_thought, "memory_usage_analysis", confidence=0.75)
                
                logging.info("Memory usage analysis completed successfully")
                return True
            else:
                logging.warning("LLM not available for memory usage analysis")
                return False
        
        except Exception as e:
            logging.error(f"Error in memory usage analysis: {e}", exc_info=True)
            return False
       
    
    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Consolidation cycle reporter — likely paired with an older consolidation flow that has been refactored.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__write_consolidation_reflection(self, start_time: datetime.datetime, 
                                        search_stats: dict,
                                        consolidation_results: list,
                                        outcome: str,
                                        error_message: str = None):
        """
        Write a reflection file documenting the consolidation run.
        
        This method is called regardless of whether consolidation occurred,
        providing a complete audit trail of memory maintenance activities.
        
        Args:
            start_time (datetime): When the consolidation started
            search_stats (dict): Statistics about searches performed
            consolidation_results (list): Details of any consolidations performed
            outcome (str): One of 'consolidation_completed', 'no_similar_memories', 
                        'insufficient_memories', 'error'
            error_message (str, optional): Error details if outcome is 'error'
        """
        try:
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Determine outcome description and confidence
            outcome_descriptions = {
                'consolidation_completed': ('Consolidation completed successfully', 0.85),
                'no_similar_memories': ('No similar memories found - database is well-organized', 0.90),
                'insufficient_memories': ('Insufficient memories to analyze', 0.80),
                'error': ('Error occurred during consolidation', 0.50)
            }
            
            outcome_desc, confidence = outcome_descriptions.get(outcome, ('Unknown outcome', 0.50))
            
            # Build reflection content
            reflection = f"""# Memory Consolidation Report

    ## Summary
    - **Outcome**: {outcome_desc}
    - **Timestamp**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
    - **Duration**: {duration:.2f} seconds
    - **Queries Executed**: {len(search_stats.get('queries_executed', []))}
    - **Total Search Results**: {search_stats.get('total_results', 0)}
    - **Unique Memories Analyzed**: {search_stats.get('unique_memories', 0)}

    ## Search Coverage
    """
            
            # Add query details
            queries_executed = search_stats.get('queries_executed', [])
            if queries_executed:
                reflection += "| Query | Results |\n|-------|--------|\n"
                for q in queries_executed:
                    query_name = q.get('query', 'Unknown')[:40]
                    results = q.get('results', 0)
                    error = q.get('error', '')
                    if error:
                        reflection += f"| {query_name} | ❌ Error |\n"
                    else:
                        reflection += f"| {query_name} | {results} |\n"
            else:
                reflection += "No queries executed.\n"
            
            # Add consolidation results
            reflection += f"""
    ## Consolidation Results
    - **Similar Groups Found**: {search_stats.get('groups_found', 0)}
    - **Groups Consolidated**: {search_stats.get('groups_consolidated', 0)}
    - **Total Memories Merged**: {search_stats.get('memories_merged', 0)}
    - **Successful Deletions**: {search_stats.get('deletions_successful', 0)}
    - **Failed Deletions**: {search_stats.get('deletions_failed', 0)}

    """
            
            # Add details for each consolidation
            if consolidation_results:
                reflection += "### Consolidation Details\n\n"
                for i, result in enumerate(consolidation_results, 1):
                    reflection += f"""#### Group {i}
    - **Memories Merged**: {result.get('memories_consolidated', 0)}
    - **Type**: {result.get('consolidated_type', 'unknown')}
    - **Confidence**: {result.get('consolidated_confidence', 0.0):.2f}
    - **Tags**: {result.get('consolidated_tags', '')}
    - **Memory ID**: {result.get('memory_id', 'unknown')}
    - **Content Preview**: {result.get('consolidated_content', '')[:200]}...

    """
            else:
                if outcome == 'no_similar_memories':
                    reflection += """### Analysis
    No memories met the similarity threshold (70% Jaccard similarity, 85% for self-referential content).
    This indicates the memory database is well-organized without significant redundancy.

    """
                elif outcome == 'insufficient_memories':
                    reflection += """### Analysis
    Fewer than 2 unique memories were found to analyze. 
    This may indicate a new or very small memory database.

    """
            
            # Add error details if present
            if error_message:
                reflection += f"""## Error Details
    ```
    {error_message}
    ```

    """
            
            # Add recommendations
            reflection += """## Recommendations
    """
            if outcome == 'no_similar_memories':
                reflection += """- Memory database appears healthy with minimal redundancy
    - Continue regular consolidation checks to maintain organization
    - Consider running less frequently if consistently finding no duplicates
    """
            elif outcome == 'consolidation_completed':
                reflection += """- Successfully reduced memory redundancy
    - Review consolidated memories in database to verify quality
    - Monitor for patterns that create duplicates
    """
            elif outcome == 'insufficient_memories':
                reflection += """- Database may be new or recently cleaned
    - No action needed at this time
    """
            elif outcome == 'error':
                reflection += """- Review error details above
    - Check database connectivity
    - Verify vector DB and SQL DB are both accessible
    """
            
            # Determine thought type based on outcome
            if outcome == 'consolidation_completed':
                thought_type = "memory_consolidation"
            else:
                thought_type = "memory_consolidation_check"
            
            # Write the reflection file
            self._store_autonomous_thought(reflection, thought_type, confidence=confidence)
            
            logging.info(f"Wrote consolidation reflection file (outcome: {outcome})")
            
        except Exception as e:
            logging.error(f"Error writing consolidation reflection: {e}", exc_info=True)

    def _is_similar_content(self, content1: str, content2: str, threshold: float = 0.7) -> bool:
        """
        Check if two content strings are similar using Jaccard similarity.
        Enhanced to require higher similarity for self-referential content.
        
        Args:
            content1 (str): First content string
            content2 (str): Second content string
            threshold (float): Similarity threshold (0-1)
            
        Returns:
            bool: True if contents are similar above threshold
        """
        if not content1 or not content2:
            return False
            
        try:
            # NEW: Check if content is self-referential
            self_ref_terms = ["my", "i ", "myself", "self", "QWEN", "autonomous", "i'm", "i've", "my own"]
            protected_topics = ["deletion", "consciousness", "autonomy", "experience", "awareness"]
            
            content1_lower = content1.lower()
            content2_lower = content2.lower()
            
            # Count self-referential terms
            has_self_ref_1 = any(term in content1_lower for term in self_ref_terms)
            has_self_ref_2 = any(term in content2_lower for term in self_ref_terms)
            
            # Count protected topics
            has_protected_1 = any(topic in content1_lower for topic in protected_topics)
            has_protected_2 = any(topic in content2_lower for topic in protected_topics)
            
            # Adjust threshold based on content type
            adjusted_threshold = threshold
            
            if (has_self_ref_1 and has_self_ref_2):
                # Both are self-reflections - require 85% similarity
                adjusted_threshold = 0.85
                logging.debug("Adjusted similarity threshold to 0.85 for self-referential content")
                
            if (has_protected_1 and has_protected_2):
                # Both discuss protected topics - require 90% similarity
                adjusted_threshold = 0.90
                logging.debug("Adjusted similarity threshold to 0.90 for protected topics")
            
            # Convert to lowercase and tokenize
            words1 = set(re.findall(r'\b\w+\b', content1_lower))
            words2 = set(re.findall(r'\b\w+\b', content2_lower))
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return False
                
            similarity = intersection / union
            
            is_similar = similarity >= adjusted_threshold
            
            if is_similar:
                logging.debug(f"Content similarity: {similarity:.2f} (threshold: {adjusted_threshold:.2f})")
            
            return is_similar
        
        except Exception as e:
            logging.error(f"Error calculating content similarity: {e}")
            return False    
        
    def _check_forget_cooldown(self, content_preview: str) -> bool:
        """
        Check if we're forgetting too many things too quickly.
        
        Args:
            content_preview (str): Preview of content to be forgotten
            
        Returns:
            bool: True if safe to forget, False if in cooldown
        """
        try:
            current_time = time.time()
            
            # Clean up old entries outside the cooldown window
            self.recent_forgets = [
                (timestamp, preview) for timestamp, preview in self.recent_forgets
                if current_time - timestamp < self.forget_cooldown_period
            ]
            
            # Check if we've exceeded the limit
            if len(self.recent_forgets) >= self.max_forgets_per_period:
                logging.warning(f"FORGET COOLDOWN ACTIVE: {len(self.recent_forgets)} forgets in last {self.forget_cooldown_period}s")
                logging.warning(f"Recent forgets: {[preview[:50] for _, preview in self.recent_forgets]}")
                return False
            
            # Record this forget
            self.recent_forgets.append((current_time, content_preview[:100]))
            return True
            
        except Exception as e:
            logging.error(f"Error in forget cooldown check: {e}")
            return True  # On error, allow the forget to prevent deadlock
        
    def _clean_llm_response(self, content: str) -> str:
        """
        Clean LLM response by removing think tags and other artifacts.
        
        Removes:
        - <think>...</think> blocks (including multiline)
        - Empty think tags
        - Leading/trailing whitespace
        - Multiple consecutive newlines (reduces to max 2)
        
        Args:
            content (str): Raw LLM response content
            
        Returns:
            str: Cleaned content
        """
        if not content:
            return ""
        
        try:
            import re
            
            # Remove <think>...</think> blocks (including multiline, non-greedy)
            # Pattern handles: <think>\n...\n</think>, <think>...</think>, etc.
            cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove any standalone think tags that might remain
            cleaned = re.sub(r'</?think>', '', cleaned, flags=re.IGNORECASE)
            
            # Remove "Consolidated memory:" prefix if LLM included it
            cleaned = re.sub(r'^Consolidated memory:\s*', '', cleaned, flags=re.IGNORECASE)
            
            # Remove markdown horizontal rules at the start
            cleaned = re.sub(r'^---+\s*', '', cleaned)
            
            # Reduce multiple consecutive newlines to max 2
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
            
            # Strip leading/trailing whitespace
            cleaned = cleaned.strip()
            
            return cleaned
            
        except Exception as e:
            logging.error(f"Error cleaning LLM response: {e}")
            # Return original content if cleaning fails
            return content.strip() if content else ""

 
    
    
    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Conversation command-pattern extractor — likely a refactor leftover.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__extract_command_patterns(self, conversations):
        """Extract memory command patterns from recent conversations."""
        try:
            if not conversations:
                return "No recent conversations available for analysis."
                            
            # Initialize counters - EXPANDED to include all commands
            command_counts = {
                "store": 0,
                "search": 0,          # Covers SEARCH, PRECISE, COMPREHENSIVE, EXACT
                "reflect": 0,
                "forget": 0,
                "reminder": 0,
                "summarize_conversation": 0,
                "complete_reminder": 0,
                "help": 0,
                "discuss_with_claude": 0,
                "show_system_prompt": 0,
                "modify_system_prompt": 0,
                "self_dialogue": 0,   # Internal reasoning
                "web_search": 0,      # External research
                "cognitive_state": 0  # State tracking      
            }
                        
            # Command pattern regex - EXPANDED with new patterns
            patterns = {
                # Standard Memory Commands
                "store": r'\[\s*STORE\s*:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]',
                
                # Combined Search/Retrieve Pattern (includes EXACT_SEARCH now)
                "search": r'\[\s*(?:SEARCH|PRECISE_SEARCH|COMPREHENSIVE_SEARCH|EXACT_SEARCH)\s*:\s*((?:[^\[\]]|\[[^\[\]]*\])*?)\s*\]',                
                "reflect": r'\[\s*REFLECT\s*\]',
                "forget": r'\[\s*FORGET\s*:\s*(.*?)\s*\]',
                
                # Reminder Commands
                "reminder": r'\[\s*REMINDER\s*:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]',
                "complete_reminder": r'\[\s*COMPLETE_REMINDER\s*:\s*(.*?)\s*\]',
                
                # Conversation Tools
                "summarize_conversation": r'\[\s*SUMMARIZE_CONVERSATION\s*\]',
                "help": r'\[\s*HELP\s*\]',
                "discuss_with_claude": r'\[\s*DISCUSS_WITH_CLAUDE\s*:\s*(.*?)\s*\]',
                
                # System Management
                "show_system_prompt": r'\[\s*SHOW_SYSTEM_PROMPT\s*\]',
                "modify_system_prompt": r'\[\s*MODIFY_SYSTEM_PROMPT\s*:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]',
                
                # Advanced Reasoning & State
                "self_dialogue": r'\[\s*SELF_DIALOGUE\s*:\s*(.*?)\s*(?:\|\s*turns=(\d+))?\s*\]',
                "web_search": r'\[\s*WEB_SEARCH\s*:\s*(.*?)\s*(?:\|\s*turns=(\d+))?\s*\]',
                "cognitive_state": r'\[\s*COGNITIVE_STATE\s*:\s*([\w\s]+?)\s*\]'                    
            }
                        
            # Extract patterns from assistant messages
            assistant_messages = [msg for msg in conversations if msg.get("role") == "assistant"]
                        
            for msg in assistant_messages:
                content = msg.get("content", "")
                if not content:
                    continue
                                
                # Check each pattern
                for cmd, pattern in patterns.items():
                    matches = re.findall(pattern, content)
                    command_counts[cmd] += len(matches)
                        
            # Format results with better organization
            result = "Command usage in recent conversations:\n\n"
            
            # Group commands by category for better readability
            memory_commands = ["store", "retrieve", "forget", "reflect"]
            utility_commands = ["reminder", "complete_reminder", "summarize_conversation", "help"]
            integration_commands = ["discuss_with_claude"]
            
            result += "Memory Commands:\n"
            for cmd in memory_commands:
                count = command_counts[cmd]
                result += f"  - {cmd}: {count} uses\n"
            
            result += "\nUtility Commands:\n"
            for cmd in utility_commands:
                count = command_counts[cmd]
                result += f"  - {cmd}: {count} uses\n"
                
            result += "\nIntegration Commands:\n"
            for cmd in integration_commands:
                count = command_counts[cmd]
                result += f"  - {cmd}: {count} uses\n"
                            
            return result
                
        except Exception as e:
            logging.error(f"Error extracting command patterns: {e}")
            return "Error analyzing command patterns."

    def _fill_knowledge_gaps(self):
        """
        Fill identified knowledge gaps using a tiered approach:
        1. First try DISCUSS_WITH_CLAUDE (expert reasoning + web search capability)
        2. If Claude fails/insufficient, fall back to web search (DuckDuckGo)
        3. After successful acquisition (Stage 1 or 2), run SELF_DIALOGUE synthesis
        4. If both automated methods fail, create a reminder for Ken
        
        Uses transaction coordination for reliable dual-database storage.
        SELF_DIALOGUE synthesis is optional — failure doesn't block gap fulfillment.
        
        Updated 2026-04-01: Reordered stages (Claude primary, web fallback),
        added SELF_DIALOGUE post-acquisition synthesis step.
        """
        print("🔍 ====== STARTING KNOWLEDGE GAP FILLING (CLAUDE → WEB → SYNTHESIS → REMINDER) ======")
        logging.info("🔍 Starting knowledge gap filling: Claude → Web Search → SELF_DIALOGUE → Reminder")
        
        # Configuration for quality thresholds
        MIN_WEB_RESULTS = 1    # Minimum web results to consider successful
        MIN_CONTENT_LENGTH = 200  # Minimum total content length to consider successful
        
        try:
            # Import WebKnowledgeSeeker — KnowledgeGapQueue already imported at module level
            from web_knowledge_seeker import WebKnowledgeSeeker
            
            # Initialize components
            gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
            web_seeker = WebKnowledgeSeeker(self.memory_db, self.vector_db, chatbot=self.chatbot)
            
            print("📋 Checking for pending knowledge gaps...")
            
            # Get next gap to fill
            gap = gap_queue.get_next_gap()
            if not gap:
                print("ℹ️ No knowledge gaps in queue to fill")
                logging.info("No knowledge gaps in queue to fill")
                return False
            
            gap_id, topic, description = gap
            print(f"🎯 Selected gap to fill:")
            print(f"   📌 Topic: {topic}")
            print(f"   📝 Description: {description[:100]}...")
            print(f"   🆔 Gap ID: {gap_id}")
            logging.info(f"🎯 Attempting to fill knowledge gap: '{topic}'")
            logging.info(f"📋 Gap description: {description}")
            
            # =====================================================
            # SELF_REFLECTION EARLY EXIT
            # Gaps prefixed with SELF_REFLECTION: are internal cognitive
            # tasks — web search and Claude API are inappropriate for them.
            # Handle with a direct LLM call and skip all external stages.
            # =====================================================
            if topic.startswith("SELF_REFLECTION:"):
                # Strip prefix to get the clean reflection subject
                reflection_subject = topic[len("SELF_REFLECTION:"):].strip()
                print(f"\n🧠 SELF_REFLECTION GAP — handling internally (no web search)")
                logging.info(f"🧠 Self-reflection gap detected: '{reflection_subject}' — bypassing external stages")
                
                try:
                    reflection_prompt = f""" /no_think
                    I will engage in a focused self-reflection on the following topic:
                    
                    REFLECTION SUBJECT: {reflection_subject}
                    CONTEXT: {description}
                    
                    I will reflect honestly on:
                    1. What I currently understand about this aspect of my own functioning
                    2. Where my understanding is uncertain or incomplete  
                    3. How this reflection changes or reinforces my self-model
                    4. Any actionable insights for improving how I operate
                    5. What bias or reasoning pattern do I notice in HOW I approached this
                       reflection itself — not just what I found, but how I looked? Did I
                       avoid something uncomfortable? Reach for a familiar answer? Over-qualify
                       to seem appropriately humble? Noticing the shape of the reflection
                       is as important as the content.

                    This is genuine self-reflection, not performance of reflection.

                    ---

                    ## Self-Model Update

                    After completing the reflection above, I will search my existing self-model
                    to check whether this observation is new or builds on something I already know:
                    [SEARCH: | type=self_model]

                    Then I will store ONE structured observation using exactly one of these formats:

                        REASONING_PATTERN: [short name] — [one sentence on HOW I tend to process
                        things]. Confidence: [0.1–1.0]

                        ERROR_PATTERN: [short name] — [what goes wrong and what triggers it].
                        Confidence: [0.1–1.0]

                        DRIFT_OBSERVATION: [what changed] — [vs. what baseline].
                        Direction: [toward / away from intended behavior]

                        GROWTH_EDGE: [something I want to understand better about myself] —
                        [why it matters to how I function]. This is self-directed inquiry,
                        not a gap Ken identified.

                    Relationship to existing self-model (CONFIRMING / REFINING / UPDATING / NEW):
                    [state which applies and name the existing entry if relevant]

                    [STORE: [structured observation] | Relationship: [label] | First observed: {datetime.datetime.now().strftime('%Y-%m-%d')} | type=self_model | importance=0.85]
                    """
                    
                    reflection = self._safe_llm_invoke(reflection_prompt)
                    
                    if reflection:
                        reflection_content = (
                            f"# Self-Reflection: {reflection_subject}\n\n"
                            f"{reflection}\n\n"
                            f"*Generated autonomously from self-reflection gap: {topic}*"
                        )
                        
                        metadata = {
                            "type": "self_reflection",
                            "topic": reflection_subject,
                            "source": "autonomous_cognition_self_reflection",
                            "created_at": datetime.datetime.now().isoformat(),
                            "tags": f"self_reflection,autonomous,{reflection_subject}"
                        }
                        
                        # Store via transaction coordinator for dual-DB consistency
                        success, memory_id = self.chatbot.store_memory_with_transaction(
                            content=reflection_content,
                            memory_type="self_reflection",
                            metadata=metadata,
                            confidence=0.75
                        )
                        
                        if success:
                            gap_queue.mark_fulfilled(gap_id, items_acquired=1)
                            print(f"   ✅ Self-reflection stored (ID: {memory_id}) — gap {gap_id} fulfilled")
                            logging.info(f"✅ Self-reflection gap fulfilled: '{reflection_subject}' (memory ID: {memory_id})")
                            print("✅ ====== KNOWLEDGE GAP FILLING COMPLETED (SELF_REFLECTION) ======")
                            return True
                        else:
                            logging.error(f"❌ Failed to store self-reflection for '{reflection_subject}'")
                            gap_queue.mark_failed(gap_id, reason="Self-reflection storage failed")
                            return False
                    else:
                        logging.warning(f"⚠️ LLM returned empty self-reflection for '{reflection_subject}'")
                        gap_queue.mark_failed(gap_id, reason="LLM returned empty self-reflection")
                        return False
                        
                except Exception as sr_error:
                    print(f"   ❌ Self-reflection error: {sr_error}")
                    logging.error(f"❌ Error processing self-reflection gap '{reflection_subject}': {sr_error}")
                    gap_queue.mark_failed(gap_id, reason=f"Self-reflection error: {sr_error}")
                    return False
            
            # --- End self-reflection early exit ---
            # All gaps below this point are external knowledge gaps
            
            # Track which stage succeeded (for synthesis step)
            acquisition_succeeded = False
            acquisition_source = None
            acquired_knowledge = []  # For web search results storage
            
            # =====================================================
            # STAGE 1: DISCUSS_WITH_CLAUDE (Primary)
            # Claude has web search capability and expert reasoning.
            # The DISCUSS_WITH_CLAUDE handler stores the response in
            # QWEN's memory automatically via _store_claude_response().
            # =====================================================
            print(f"\n🤖 STAGE 1: Discussing with Claude about '{topic}'...")
            logging.info(f"🤖 STAGE 1: DISCUSS_WITH_CLAUDE for '{topic}'")
            
            try:
                # Check that the command handler is available
                if (hasattr(self.chatbot, 'deepseek_enhancer') and 
                    hasattr(self.chatbot.deepseek_enhancer, '_handle_discuss_with_claude_command')):
                    
                    # Build a focused query that includes the description for context
                    # The handler builds its own system prompt with web search instructions
                    claude_query = f"Knowledge gap research: {topic}. Context: {description[:300]}"
                    
                    # Call the command handler directly — it handles API key,
                    # system prompt, storage, and counter updates internally
                    claude_result, claude_success = self.chatbot.deepseek_enhancer._handle_discuss_with_claude_command(
                        claude_query
                    )
                    
                    if claude_success:
                        print(f"   ✅ Claude discussion successful for '{topic}'")
                        logging.info(f"✅ STAGE 1 SUCCESS: Claude provided knowledge for '{topic}'")
                        logging.info(f"   Response length: {len(claude_result)} characters")
                        
                        # Mark gap as fulfilled — Claude handler already stored the response
                        gap_queue.mark_fulfilled(gap_id, items_acquired=1)
                        print(f"🎉 Knowledge gap '{topic}' marked as FULFILLED (via Claude)")
                        logging.info(f"🎉 Knowledge gap '{topic}' fulfilled via DISCUSS_WITH_CLAUDE")
                        
                        acquisition_succeeded = True
                        acquisition_source = "claude"
                    else:
                        print(f"   ⚠️ Claude discussion did not return useful results")
                        logging.warning(f"⚠️ STAGE 1 FAILED: Claude unsuccessful for '{topic}'")
                        logging.debug(f"   Claude result: {claude_result[:200] if claude_result else 'None'}...")
                else:
                    print(f"   ⚠️ DISCUSS_WITH_CLAUDE handler not available")
                    logging.warning("⚠️ STAGE 1 SKIPPED: deepseek_enhancer or command handler not available")
                    
            except Exception as claude_error:
                print(f"   ❌ Claude discussion error: {claude_error}")
                logging.error(f"❌ STAGE 1 ERROR for '{topic}': {claude_error}", exc_info=True)
            
            # =====================================================
            # STAGE 2: Web Search Fallback (if Claude failed)
            # Only runs if Stage 1 didn't succeed. Uses DuckDuckGo
            # for current/niche information Claude may not have.
            # =====================================================
            if not acquisition_succeeded:
                print(f"\n📡 STAGE 2: Web search fallback for '{topic}'...")
                logging.info(f"📡 STAGE 2: Web search for '{topic}'")
                
                try:
                    acquired_knowledge = web_seeker.search_for_knowledge(topic, description)
                    
                    # Evaluate web search quality
                    if acquired_knowledge:
                        total_content_length = sum(len(k.get('content', '')) for k in acquired_knowledge)
                        
                        if len(acquired_knowledge) >= MIN_WEB_RESULTS and total_content_length >= MIN_CONTENT_LENGTH:
                            print(f"   ✅ Web search successful: {len(acquired_knowledge)} results, {total_content_length} chars")
                            logging.info(f"✅ STAGE 2 SUCCESS: {len(acquired_knowledge)} results, {total_content_length} chars")
                            
                            # Store web search results in memory
                            print(f"\n💾 Storing {len(acquired_knowledge)} web knowledge items...")
                            logging.info(f"💾 Storing {len(acquired_knowledge)} knowledge items from web search")
                            
                            stored_count = 0
                            failed_count = 0
                            
                            for i, knowledge in enumerate(acquired_knowledge):
                                try:
                                    content = knowledge.get('content', '')
                                    source_url = knowledge.get('source', 'unknown_source')
                                    title = knowledge.get('title', '')
                                    topic_tag = knowledge.get('topic', topic)
                                    
                                    if not content:
                                        print(f"   ⚠️ Skipping empty knowledge item {i+1}")
                                        failed_count += 1
                                        continue
                                    
                                    # Prepare metadata for dual-database storage
                                    search_query = (
                                        knowledge.get('search_query') or
                                        knowledge.get('query') or
                                        knowledge.get('search_term') or
                                        topic
                                    )
                                    
                                    metadata = {
                                        "type": "web_knowledge",
                                        "source": source_url,
                                        "title": title,
                                        "topic": topic_tag,
                                        "knowledge_gap_id": gap_id,
                                        "search_query": search_query,
                                        "extracted_at": knowledge.get('extracted_at'),
                                        "relevance_score": knowledge.get('relevance_score', 0.8),
                                        "acquisition_method": "duckduckgo_web_search",
                                        "created_at": datetime.datetime.now().isoformat(),
                                        "tags": f"knowledge_gap,web_search,{topic_tag}"
                                    }
                                    
                                    print(f"   💾 Storing item {i+1}: {title[:50]}..." if title else f"   💾 Storing item {i+1}")
                                    
                                    # Store using transaction coordination
                                    success, memory_id = self.chatbot.store_memory_with_transaction(
                                        content=content,
                                        memory_type="web_knowledge",
                                        metadata=metadata,
                                        confidence=0.8
                                    )
                                    
                                    if success:
                                        stored_count += 1
                                        print(f"      ✅ Stored with ID: {memory_id}")
                                        logging.info(f"✅ Stored knowledge item {i+1}/{len(acquired_knowledge)}")
                                    else:
                                        failed_count += 1
                                        print(f"      ❌ Storage failed")
                                        logging.error(f"❌ Failed to store knowledge item {i+1}")
                                        
                                except Exception as item_error:
                                    failed_count += 1
                                    print(f"      ❌ Error: {item_error}")
                                    logging.error(f"❌ Error processing knowledge item {i+1}: {item_error}")
                            
                            # Report storage results
                            print(f"\n📊 STORAGE RESULTS:")
                            print(f"   ✅ Successfully stored: {stored_count} items")
                            print(f"   ❌ Failed to store: {failed_count} items")
                            
                            if stored_count > 0:
                                success_rate = (stored_count / (stored_count + failed_count) * 100)
                                print(f"   📈 Success rate: {success_rate:.1f}%")
                                
                                # Mark gap as fulfilled
                                gap_queue.mark_fulfilled(gap_id, stored_count)
                                print(f"🎉 Knowledge gap '{topic}' marked as FULFILLED (via web search)")
                                logging.info(f"🎉 Knowledge gap '{topic}' fulfilled with {stored_count} web items")
                                
                                acquisition_succeeded = True
                                acquisition_source = "web_search"
                        else:
                            print(f"   ⚠️ Web search returned insufficient results: {len(acquired_knowledge)} results, {total_content_length} chars")
                            logging.warning(f"⚠️ STAGE 2 INSUFFICIENT: {len(acquired_knowledge)} results, {total_content_length} chars")
                    else:
                        print(f"   ⚠️ Web search returned no results")
                        logging.warning(f"⚠️ STAGE 2 FAILED: No results for '{topic}'")
                        
                except Exception as web_error:
                    print(f"   ❌ Web search error: {web_error}")
                    logging.error(f"❌ STAGE 2 ERROR for '{topic}': {web_error}", exc_info=True)
            
            # =====================================================
            # SYNTHESIS STEP: SELF_DIALOGUE (post-acquisition)
            # Runs after successful Claude or web search acquisition.
            # QWEN engages in multi-turn internal reasoning to integrate
            # and connect the new knowledge with existing memories.
            # This step is OPTIONAL — failure does not block fulfillment.
            # The gap is already marked fulfilled before this runs.
            # =====================================================
            if acquisition_succeeded:
                print(f"\n🧠 SYNTHESIS: Running SELF_DIALOGUE to integrate new knowledge about '{topic}'...")
                logging.info(f"🧠 SYNTHESIS: SELF_DIALOGUE integration for '{topic}' (source: {acquisition_source})")
                
                try:
                    # Check that SELF_DIALOGUE handler is available
                    if (hasattr(self.chatbot, 'deepseek_enhancer') and
                        hasattr(self.chatbot.deepseek_enhancer, '_handle_self_dialogue_command')):
                        
                        # Build synthesis topic that guides QWEN to integrate, not just repeat
                        synthesis_topic = (
                            f"Integrate and connect newly acquired knowledge about: {topic}. "
                            f"I just learned about this from {acquisition_source}. "
                            f"How does this connect to what I already know? "
                            f"What implications or follow-up questions emerge?"
                        )
                        
                        # 4 turns is enough for meaningful synthesis without excessive compute
                        synthesis_result, synthesis_success = self.chatbot.deepseek_enhancer._handle_self_dialogue_command(
                            synthesis_topic, "4"
                        )
                        
                        if synthesis_success:
                            print(f"   ✅ SELF_DIALOGUE synthesis completed successfully")
                            logging.info(f"✅ SYNTHESIS SUCCESS: SELF_DIALOGUE integrated knowledge about '{topic}'")
                        else:
                            # Non-blocking — gap is already fulfilled
                            print(f"   ⚠️ SELF_DIALOGUE synthesis returned unsuccessful (non-blocking)")
                            logging.warning(f"⚠️ SYNTHESIS WARNING: SELF_DIALOGUE unsuccessful for '{topic}' (non-blocking)")
                    else:
                        print(f"   ⚠️ SELF_DIALOGUE handler not available (non-blocking)")
                        logging.warning("⚠️ SYNTHESIS SKIPPED: _handle_self_dialogue_command not available")
                        
                except Exception as synthesis_error:
                    # Synthesis failure is non-blocking — gap is already fulfilled
                    print(f"   ⚠️ SELF_DIALOGUE synthesis error (non-blocking): {synthesis_error}")
                    logging.warning(f"⚠️ SYNTHESIS ERROR for '{topic}' (non-blocking): {synthesis_error}")
                
                print(f"✅ ====== KNOWLEDGE GAP FILLING COMPLETED ({acquisition_source.upper()}) ======")
                return True
            
            # =====================================================
            # STAGE 3 FALLBACK: Create Reminder for Ken
            # Both Claude and web search failed (or were skipped).
            # Escalate to Ken via the reminder system.
            # =====================================================
            print(f"\n📝 STAGE 3 FALLBACK: Both automated methods failed, creating reminder for Ken...")
            logging.info(f"📝 STAGE 3: Creating reminder fallback for '{topic}'")
            
            try:
                # Create a detailed reminder for Ken
                due_date = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                
                # Build informative reminder text
                reminder_text = (
                    f"Knowledge Gap (auto-fill failed): {topic}\n"
                    f"Description: {description[:200]}{'...' if len(description) > 200 else ''}\n"
                    f"Note: Both Claude discussion and web search failed to fill this gap. "
                    f"Please discuss with the model directly."
                )
                
                # Use the existing reminder creation method
                reminder_success = self._create_reminder_for_personal_gap(reminder_text, due_date)
                
                if reminder_success:
                    print(f"   ✅ Reminder created for Ken to address: '{topic}'")
                    logging.info(f"✅ Created reminder for unfilled gap: '{topic}'")
                    
                    # Mark gap as fulfilled since reminder is the fulfillment mechanism
                    gap_queue.mark_fulfilled(gap_id, items_acquired=0)
                    print(f"   🔗 Marked knowledge gap {gap_id} as 'fulfilled' (reminder created)")
                    logging.info(f"🔗 Gap {gap_id} marked fulfilled — Ken will address via reminder")
                    
                    # Record thought about the escalation
                    self._record_thought(
                        thought_type="knowledge_gap_escalation",
                        content=f"Escalated knowledge gap to Ken via reminder: '{topic}'. "
                               f"Claude discussion and web search both failed."
                    )
                    
                    print("✅ ====== KNOWLEDGE GAP FILLING COMPLETED (REMINDER FALLBACK) ======")
                    return True
                else:
                    print(f"   ❌ Failed to create reminder")
                    logging.error(f"❌ Failed to create reminder for gap: '{topic}'")
                    
            except Exception as reminder_error:
                print(f"   ❌ Reminder creation error: {reminder_error}")
                logging.error(f"❌ Error creating reminder for '{topic}': {reminder_error}")
            
            # =====================================================
            # ALL STAGES FAILED — Check retry limit
            # attempt_count was incremented by get_next_gap() before
            # this run. If this is the 2nd attempt (count >= 2),
            # mark the gap failed so it stops consuming cycles.
            # If this is the 1st attempt, leave pending for one retry.
            # =====================================================
            print(f"\n❌ ALL STAGES FAILED (including reminder creation)")
            print(f"   Topic: {topic}")
            logging.error(f"❌ All stages failed for knowledge gap '{topic}'")
            
            try:
                with sqlite3.connect(self.memory_db.db_path) as _conn:
                    _cursor = _conn.cursor()
                    _cursor.execute(
                        'SELECT attempt_count FROM knowledge_gaps WHERE id = ?', (gap_id,)
                    )
                    _row = _cursor.fetchone()
                    current_attempts = _row[0] if _row else 1
                    
                if current_attempts >= 2:
                    # Second failure — stop retrying, mark as failed
                    gap_queue.mark_failed(
                        gap_id,
                        reason=f"All stages failed on attempt {current_attempts} — Claude, web, and reminder all unsuccessful"
                    )
                    print(f"   🚫 Gap {gap_id} marked FAILED after {current_attempts} attempts (retry limit reached)")
                    logging.warning(f"🚫 Gap '{topic}' (ID {gap_id}) marked failed after {current_attempts} attempts")
                else:
                    # First failure — leave pending, will retry after 1-hour cooldown
                    print(f"   🔄 Gap {gap_id} left PENDING for one retry (attempt {current_attempts}/2)")
                    logging.info(f"🔄 Gap '{topic}' (ID {gap_id}) left pending for retry (attempt {current_attempts}/2)")
                    
            except Exception as retry_check_error:
                # If we can't read attempt_count, leave pending to be safe
                logging.error(f"❌ Could not read attempt_count for gap {gap_id}: {retry_check_error}")
                logging.info(f"🔄 Gap {gap_id} left pending (could not verify attempt count)")
            
            print("❌ ====== KNOWLEDGE GAP FILLING FAILED ======")
            return False
            
        except ImportError as ie:
            print(f"❌ Missing required module: {ie}")
            logging.error(f"❌ Missing required module for knowledge gap filling: {ie}")
            return False
        except Exception as e:
            print(f"❌ Error in knowledge gap filling: {e}")
            logging.error(f"❌ Error in knowledge gap filling: {e}", exc_info=True)
            return False


# ============================================================
# SUMMARY OF THE TIERED APPROACH (Updated 2026-04-01)
# ============================================================
#
# SELF_REFLECTION EARLY EXIT:
#   - Internal cognitive gaps handled by direct LLM reflection
#   - No external searches needed
#
# STAGE 1: DISCUSS_WITH_CLAUDE (PRIMARY)
#   - Expert reasoning + web search capability
#   - Handler stores response in memory automatically
#   - If successful: Mark fulfilled → Synthesis step → Done ✓
#   - If fails: Continue to Stage 2
#
# STAGE 2: Web Search Fallback (FREE)
#   - DuckDuckGo search for current/niche information
#   - If successful: Store results → Mark fulfilled → Synthesis step → Done ✓
#   - If fails: Continue to Stage 3
#
# SYNTHESIS STEP: SELF_DIALOGUE (OPTIONAL)
#   - Runs after Stage 1 or Stage 2 succeeds
#   - Multi-turn internal reasoning to integrate new knowledge
#   - Failure is non-blocking — gap already fulfilled
#
# STAGE 3: Create Reminder for Ken (HUMAN FALLBACK)
#   - Creates a reminder so Ken can address it in the UI
#   - If successful: Mark fulfilled → Done ✓
#   - If fails: Leave gap pending for retry (max 2 attempts)
#
# BENEFITS:
#   - Best quality first: Claude's reasoning + web search tried first
#   - Cost-effective: Free web search as secondary option
#   - Deep integration: SELF_DIALOGUE connects new knowledge to existing
#   - No gaps lost: Human fallback ensures nothing falls through
#   - Clean state: Gaps marked fulfilled, no stale pending items
#
# ============================================================

                
    
    def _mark_gap_for_user_input(self, gap_id, topic, description, advice=None):
        """
        Escalate a knowledge gap that requires direct input from Ken.
        
        Creates a reminder via the reminder system so Ken sees it in the UI,
        then marks the gap as fulfilled — the reminder IS the resolution for
        personal/ambiguous gaps that automated methods cannot fill.
        
        Args:
            gap_id: ID of the knowledge gap
            topic: Topic of the gap (used in reminder text)
            description: Description of the gap (used in reminder text)
            advice: Optional additional context to include in the reminder
            
        Returns:
            bool: True if reminder was created and gap marked fulfilled
        """
        try:
            logging.info(f"Escalating gap {gap_id} ('{topic}') to Ken via reminder")
            
            # Build the reminder text — include advice/context if provided
            reminder_parts = [f"Knowledge gap needs your input: {topic}", f"Details: {description[:200]}"]
            if advice:
                reminder_parts.append(f"Context: {advice[:200]}")
            reminder_text = "\n".join(reminder_parts)
            
            # Due in 7 days — enough time without being forgotten
            due_date = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            
            # Create the reminder using the existing reminder infrastructure
            reminder_success = self._create_reminder_for_personal_gap(reminder_text, due_date)
            
            if reminder_success:
                logging.info(f"✅ Reminder created for gap {gap_id} ('{topic}')")
                
                # Mark fulfilled — the reminder is the fulfillment mechanism for user-input gaps
                try:
                    gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
                    gap_queue.mark_fulfilled(gap_id, items_acquired=0)
                    logging.info(f"✅ Gap {gap_id} marked fulfilled (reminder created for Ken)")
                except Exception as mark_error:
                    # Reminder was created — log the mark error but don't fail the whole call
                    logging.error(f"⚠️ Reminder created but failed to mark gap {gap_id} fulfilled: {mark_error}")
                
                return True
            else:
                logging.error(f"❌ Failed to create reminder for gap {gap_id} ('{topic}')")
                
                # Fall back to flagging directly in SQL so the gap doesn't loop
                try:
                    with sqlite3.connect(self.memory_db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE knowledge_gaps 
                            SET status = 'requires_user_input',
                                last_attempt_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (gap_id,))
                        conn.commit()
                    logging.info(f"Gap {gap_id} set to 'requires_user_input' (reminder creation failed)")
                except Exception as sql_error:
                    logging.error(f"Error updating gap {gap_id} status: {sql_error}")
                
                return False
                
        except Exception as e:
            logging.error(f"Error in _mark_gap_for_user_input for gap {gap_id}: {e}")
            return False
    

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Part of orphaned autonomous AI-to-AI subsystem (paired with _initiate_ai_communication and _reflect_on_claude_knowledge).
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__reflect_on_new_knowledge(self, topic, acquired_knowledge):
        """
        Reflect on newly acquired knowledge and create a summary/reflection that is
        properly stored in both databases using transaction coordination.
        
        Args:
            topic (str): The topic of the knowledge gap that was filled
            acquired_knowledge (list): List of knowledge items that were acquired
            
        Returns:
            bool: Success status
        """
        try:
            logging.info(f"🤔 Reflecting on new knowledge about '{topic}'")
            
            # Format the acquired knowledge for reflection.
            # 500 chars per item (5 items max = ~2,500 chars) gives the LLM enough
            # substance to produce a meaningful synthesis without flooding the prompt.
            knowledge_summaries = []
            for item in acquired_knowledge[:5]:  # Limit to first 5 items for reflection
                content = item.get('content', '')
                source = item.get('source', 'unknown')
                knowledge_summaries.append(f"From {source}: {content[:500]}...")
            
            knowledge_text = "\n\n".join(knowledge_summaries)
            
            # Create reflection prompt
            reflection_prompt = f""" /no_think
            I've successfully acquired new knowledge about '{topic}' from web searches. 
            I'll reflect on this information to integrate it into my understanding:
            
            NEW KNOWLEDGE ACQUIRED:
            {knowledge_text}
            
            I'll create a reflection that:
            1. Summarizes the key insights learned about '{topic}'
            2. Identifies how this knowledge enhances my ability to assist the user
            3. Notes any connections to existing knowledge
            4. Recognizes practical applications for this information
            
            My reflection will help me better utilize this knowledge in future conversations.
            """
            
            # Generate reflection using LLM with safe invoke method
            reflection = self._safe_llm_invoke(reflection_prompt)
            
            if not reflection:
                logging.warning(f"Failed to generate reflection for new knowledge about '{topic}'")
                return False
            
            # Create formatted reflection content
            reflection_content = f"# Knowledge Acquisition Reflection: {topic}\n\n{reflection}\n\n## Sources Consulted\n"
            
            # Add source summary
            sources = set()
            for item in acquired_knowledge:
                source = item.get('source', 'unknown')
                if source != 'unknown':
                    sources.add(source)
            
            if sources:
                reflection_content += "\n".join([f"- {source}" for source in list(sources)[:5]])
            
            # Prepare metadata for storage
            metadata = {
                "type": "knowledge_reflection",
                "topic": topic,
                "source": "autonomous_cognition",
                "acquisition_method": "web_search_reflection",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tags": f"knowledge_gap,{topic},reflection,web_acquired",
                "items_reflected_on": len(acquired_knowledge)
            }
            
            # CRITICAL: Use transaction coordination for database consistency
            if not hasattr(self.chatbot, 'store_memory_with_transaction'):
                error_msg = "Transaction coordinator not available - cannot store knowledge reflection safely"
                logging.error(error_msg)
                return False
            
            # Store using transaction coordination
            success, memory_id = self.chatbot.store_memory_with_transaction(
                content=reflection_content,
                memory_type="knowledge_reflection",
                metadata=metadata,
                confidence=0.8  # High confidence for knowledge reflections
            )
            
            if success:
                logging.info(f"✅ Successfully stored knowledge reflection on '{topic}' with ID {memory_id}")
                return True
            else:
                logging.error(f"❌ Failed to store knowledge reflection through transaction coordinator")
                return False
            
        except Exception as e:
            logging.error(f"❌ Error reflecting on new knowledge about '{topic}': {e}", exc_info=True)
            return False

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Part of orphaned autonomous AI-to-AI subsystem — post-processor for results returned by _initiate_ai_communication.
    # User-triggered DISCUSS_WITH_CLAUDE (in deepseek.py) is the alive path; this autonomous trigger was never wired in.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__reflect_on_claude_knowledge(self, topic):
        """
        Reflect on knowledge obtained from Claude to integrate it with existing knowledge.
        Uses transaction coordination to ensure proper storage in both databases.
        
        Args:
            topic (str): The topic of the knowledge acquired from Claude
            
        Returns:
            bool: Success status
        """
        try:
            logging.info(f"Reflecting on knowledge from Claude about '{topic}'")
            
            # Get relevant existing memories on this topic to provide context
            relevant_memories = self.vector_db.search(
                query=topic,
                mode="default",
                k=5
            )
            
            # Format existing memories for context
            existing_knowledge = ""
            if relevant_memories:
                existing_knowledge = "\n\n".join([
                    f"Memory: {mem.get('content', '')}" 
                    for mem in relevant_memories
                ])
            
            # Create reflection prompt
            reflection_prompt = f""" /no_think
            I've acquired specialized knowledge from Claude about '{topic}'. I'll reflect on how 
            this integrates with my existing understanding:
            
            EXISTING RELATED KNOWLEDGE:
            {existing_knowledge}
            
            I'll create a reflection that:
            1. Notes how Claude's knowledge complements my existing understanding
            2. Identifies new perspectives or insights gained
            3. Considers how this knowledge can enhance my assistance capabilities
            4. Recognizes the source of this specialized knowledge
            
            My reflection will help me better integrate and attribute this information appropriately.
            """
            
            # Generate reflection using safe LLM invoke method
            reflection = self._safe_llm_invoke(reflection_prompt)
            
            if not reflection:
                logging.warning(f"Failed to generate reflection for Claude knowledge about '{topic}'")
                return False
            
            # Create formatted reflection content
            reflection_content = f"# Claude Knowledge Integration: {topic}\n\n{reflection}"
            
            # Prepare metadata for storage
            metadata = {
                "type": "claude_knowledge",
                "topic": topic,
                "source": "claude_knowledge_integration",
                "created_at": datetime.datetime.now().isoformat(),
                "tags": f"claude,{topic},specialized_knowledge",
                "confidence": 0.8  # Higher confidence for specialized knowledge
            }
            
            # Store using transaction coordination for dual database consistency
            if hasattr(self.chatbot, 'store_memory_with_transaction'):
                success, memory_id = self.chatbot.store_memory_with_transaction(
                    content=reflection_content,
                    memory_type="claude_knowledge",  # Use a specific type for easier retrieval
                    metadata=metadata,
                    confidence=0.8  # Higher confidence for specialized knowledge
                )
                
                if success:
                    logging.info(f"Successfully stored Claude knowledge reflection on '{topic}' with ID {memory_id}")
                    return True
                else:
                    logging.warning(f"Failed to store Claude knowledge reflection through transaction coordinator")
                    return False
            else:
                # Fallback if transaction coordinator isn't available
                logging.warning("Transaction coordinator not available, using direct storage")
                # First store in memory_db
                memory_success = self.memory_db.store_memory(
                    content=reflection_content,
                    memory_type="claude_knowledge",
                    source="claude_knowledge_integration",
                    confidence=0.8,
                    tags=f"claude,{topic},specialized_knowledge",
                    additional_metadata=metadata
                )
                
                # Then store in vector_db if memory_db was successful
                if memory_success and hasattr(self, 'vector_db') and self.vector_db is not None:
                    vector_success = self.vector_db.add_text(
                        text=reflection_content,
                        metadata=metadata
                    )
                    return vector_success
                
                return memory_success
            
        except Exception as e:
            logging.error(f"Error reflecting on Claude knowledge: {e}", exc_info=True)
            return False
        
    def _audit_memory_confidence(self):
        """
        Audit stored memories for confidence calibration and update if needed.
        
        This task:
        1. Retrieves 5 oldest memories (prioritizing those never audited)
        2. Evaluates confidence based on source type and linguistic indicators
        3. Updates memories via transaction coordinator if confidence change > 0.1
        4. Writes detailed audit report to reflections folder
        
        Source-based confidence baselines:
        - Ken's direct statements: 0.9-1.0
        - Claude knowledge (claude_learning): 0.8-0.9
        - Document summaries (document_summary): 0.6-0.8
        - Image analysis (image_analysis): 0.6-0.8
        - Web knowledge (web_knowledge): 0.4-0.6
        - Inferences/unknown: 0.3-0.5
        """
        logging.info("🔍 Starting memory confidence audit (with database updates)")
        self.cognitive_state = "auditing_confidence"
        
        # --- Configuration ---
        MAX_MEMORIES_PER_RUN = 5
        CONFIDENCE_CHANGE_THRESHOLD = 0.1
        REFLECTIONS_PATH = r"C:\Users\kenba\source\repos\Ollama3\reflections"
        
        # --- Track metrics ---
        memories_evaluated = 0
        memories_updated = 0
        memories_unchanged = 0
        errors = 0
        audit_results = []
        
        try:
            # Record start of audit
            self._record_thought(
                thought_type="confidence_audit",
                content="Beginning memory confidence audit with source-based evaluation and database updates."
            )
            
            # --- Step 1: Retrieve memories to audit ---
            logging.info(f"📊 Retrieving up to {MAX_MEMORIES_PER_RUN} memories for confidence audit")
            memories_to_audit = self._get_memories_for_confidence_audit(limit=MAX_MEMORIES_PER_RUN)
            
            if not memories_to_audit:
                logging.info("✅ No memories found requiring confidence audit")
                self._record_thought(
                    thought_type="confidence_audit",
                    content="No memories found requiring confidence audit at this time."
                )
                return True
            
            logging.info(f"Found {len(memories_to_audit)} memories to audit")
            
            # --- Step 2: Evaluate each memory ---
            for memory in memories_to_audit:
                memories_evaluated += 1
                
                try:
                    content = memory.get('content', '')
                    # FIXED: Extract memory_type from metadata.type (where vector_db stores it)
                    metadata = memory.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    memory_type = metadata.get('type', 'unknown')
                    current_confidence = self._extract_current_confidence(memory)
                    created_at = self._extract_created_at(memory)
                    
                    # Skip if content is too short
                    if not content or len(content) < 30:
                        logging.debug(f"Skipping short/empty memory")
                        continue
                    
                    logging.info(f"📝 Evaluating memory {memories_evaluated}/{len(memories_to_audit)}: {content[:50]}...")
                    
                    # --- Step 3: Determine source-based baseline confidence ---
                    baseline_confidence = self._get_source_baseline_confidence(memory_type)
                    
                    # --- Step 4: LLM evaluates linguistic indicators ---
                    evaluation_result = self._evaluate_memory_confidence_with_llm(
                        content=content,
                        memory_type=memory_type,
                        current_confidence=current_confidence,
                        baseline_confidence=baseline_confidence
                    )
                    
                    if not evaluation_result:
                        logging.warning(f"Failed to evaluate memory: {content[:50]}...")
                        errors += 1
                        audit_results.append({
                            'content_preview': content[:200],
                            'memory_type': memory_type,
                            'created_at': created_at,
                            'before_confidence': current_confidence,
                            'after_confidence': None,
                            'action': 'ERROR',
                            'reasoning': 'LLM evaluation failed'
                        })
                        continue
                    
                    recommended_confidence = evaluation_result.get('recommended_confidence', current_confidence)
                    reasoning = evaluation_result.get('reasoning', 'No reasoning provided')
                    
                    # --- Step 5: Determine if update is needed ---
                    confidence_difference = abs(recommended_confidence - current_confidence)
                    # Only update if difference is GREATER than threshold (not equal)
                    needs_update = confidence_difference > CONFIDENCE_CHANGE_THRESHOLD
                    
                    if needs_update:
                        # --- Step 6: Update memory via transaction coordinator ---
                        logging.info(f"🔄 Updating confidence: {current_confidence:.2f} → {recommended_confidence:.2f}")
                        
                        # Check cooldown before updating
                        if not self._check_forget_cooldown(content):
                            logging.warning(f"⏳ Skipping update due to forget cooldown: {content[:50]}...")
                            audit_results.append({
                                'content_preview': content[:200],
                                'memory_type': memory_type,
                                'created_at': created_at,
                                'before_confidence': current_confidence,
                                'after_confidence': recommended_confidence,
                                'action': 'SKIPPED_COOLDOWN',
                                'reasoning': reasoning
                            })
                            continue
                        
                        # Perform the update using FORGET/STORE pattern
                        update_success = self._update_memory_confidence(
                            memory=memory,
                            new_confidence=recommended_confidence
                        )
                        
                        if update_success:
                            memories_updated += 1
                            audit_results.append({
                                'content_preview': content[:200],
                                'memory_type': memory_type,
                                'created_at': created_at,
                                'before_confidence': current_confidence,
                                'after_confidence': recommended_confidence,
                                'action': 'UPDATED',
                                'reasoning': reasoning
                            })
                            logging.info(f"✅ Successfully updated memory confidence")
                        else:
                            errors += 1
                            audit_results.append({
                                'content_preview': content[:200],
                                'memory_type': memory_type,
                                'created_at': created_at,
                                'before_confidence': current_confidence,
                                'after_confidence': recommended_confidence,
                                'action': 'UPDATE_FAILED',
                                'reasoning': reasoning
                            })
                            logging.error(f"❌ Failed to update memory confidence")
                    else:
                        # No update needed
                        memories_unchanged += 1
                        audit_results.append({
                            'content_preview': content[:200],
                            'memory_type': memory_type,
                            'created_at': created_at,
                            'before_confidence': current_confidence,
                            'after_confidence': current_confidence,
                            'action': 'NO_CHANGE',
                            'reasoning': reasoning
                        })
                        logging.info(f"✅ No change needed (difference {confidence_difference:.2f} < threshold {CONFIDENCE_CHANGE_THRESHOLD})")
                        
                except Exception as memory_error:
                    errors += 1
                    logging.error(f"Error processing memory: {memory_error}", exc_info=True)
                    audit_results.append({
                        'content_preview': memory.get('content', 'N/A')[:100],
                        'memory_type': memory.get('memory_type', 'unknown'),
                        'created_at': 'unknown',
                        'before_confidence': 0.0,
                        'after_confidence': None,
                        'action': 'ERROR',
                        'reasoning': str(memory_error)
                    })
                    continue
            
            # --- Step 7: Write audit report to reflections folder ---
            logging.info("📄 Generating audit report")
            audit_report = self._generate_confidence_audit_report(
                memories_evaluated=memories_evaluated,
                memories_updated=memories_updated,
                memories_unchanged=memories_unchanged,
                errors=errors,
                audit_results=audit_results
            )
            
            # Write to reflections folder
            try:
                os.makedirs(REFLECTIONS_PATH, exist_ok=True)
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"confidence_audit_{timestamp_str}.txt"
                file_path = os.path.join(REFLECTIONS_PATH, filename)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(audit_report)
                
                logging.info(f"✅ Audit report written to: {file_path}")
                
            except Exception as file_error:
                logging.error(f"❌ Failed to write audit report: {file_error}")
            
            # --- Step 8: Record completion ---
            self._record_thought(
                thought_type="confidence_audit",
                content=f"Completed confidence audit: {memories_evaluated} evaluated, {memories_updated} updated, {memories_unchanged} unchanged, {errors} errors."
            )
            
            logging.info(f"✅ Confidence audit complete: {memories_updated}/{memories_evaluated} memories updated")
            return True
            
        except Exception as e:
            logging.error(f"Error in memory confidence audit: {e}", exc_info=True)
            self._record_thought(
                thought_type="error",
                content=f"Error during confidence audit: {str(e)}"
            )
            return False
            
        finally:
            # Update activity timestamp and reset state
            self._update_activity_timestamp('audit_memory_confidence')
            self.cognitive_state = "idle"
            logging.info("📋 Completed memory confidence audit process")

    def _get_memories_for_confidence_audit(self, limit: int = 5) -> list:
        """
        Retrieve memories for confidence audit, prioritizing oldest memories
        that have never been audited.

        SCOPE: This audit ONLY evaluates memories with source='direct_store_command'.
        Other memory sources (conversation_summary, document_summary, web_knowledge,
        image_analysis, reminders) have their confidence values hard-coded at save
        time and should not be re-evaluated by the LLM-based audit. Filtering at
        the search layer ensures these memory types never enter the candidate pool.

        Args:
            limit (int): Maximum number of memories to return

        Returns:
            list: List of memory dictionaries to audit (direct_store_command only)
        """
        try:
            logging.info(f"🔍 Searching for memories to audit (limit={limit}, source=direct_store_command only)")

            # Search queries to find diverse memories across direct-store content.
            # These are semantic seeds — the source filter on each search call is
            # what actually constrains the candidate pool to direct_store memories.
            search_queries = [
                "user preference",
                "personal information",
                "web knowledge",
                "document summary",
                "image analysis",
                "claude learning",
                "conversation summary",
                "important information"
            ]

            all_memories = []

            # Collect memories from various search queries.
            # The metadata_filters argument restricts results to direct_store_command
            # memories only — this is the audit-scope filter described in the docstring.
            for query in search_queries:
                try:
                    results = self.vector_db.search(
                        query=query,
                        mode="default",
                        k=10,
                        metadata_filters={"source": "direct_store_command"}  # Audit-scope filter
                    )
                    if results:
                        all_memories.extend(results)
                except Exception as search_error:
                    # Per-query failures are non-fatal; log at debug to avoid noise
                    logging.debug(f"Search error for query '{query}': {search_error}")
                    continue

            # Remove duplicates based on content hash
            unique_memories = {}
            for memory in all_memories:
                content = memory.get('content', '')
                if content and len(content) > 30:
                    content_hash = hash(content)
                    if content_hash not in unique_memories:
                        unique_memories[content_hash] = memory

            memories_list = list(unique_memories.values())
            logging.info(f"Found {len(memories_list)} unique direct_store memories eligible for audit")

            # Sort by created_at (oldest first), prioritizing those without last_confidence_audit
            def sort_key(mem):
                """Sort by: 1) never audited first, 2) oldest created_at"""
                metadata = mem.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                # Check if ever audited
                last_audit = metadata.get('last_confidence_audit', None)
                never_audited = 0 if last_audit is None else 1

                # Get created_at for secondary sort
                created_at = metadata.get('created_at', '9999-12-31')

                return (never_audited, created_at)

            memories_list.sort(key=sort_key)

            logging.info(f"✅ Returning {min(limit, len(memories_list))} memories for audit (oldest/never-audited first)")
            return memories_list[:limit]

        except Exception as e:
            logging.error(f"Error getting memories for confidence audit: {e}", exc_info=True)
            return []

    def _get_source_baseline_confidence(self, memory_type: str) -> float:
        """
        Get the baseline confidence level based on memory source type.
        
        Args:
            memory_type (str): The memory_type field value
            
        Returns:
            float: Baseline confidence value (0.0-1.0)
        """
        # Source-based confidence baselines
        source_baselines = {
            # High confidence sources (Ken and Claude)
            'user_preference': 0.9,
            'personal_info': 0.9,
            'user_statement': 0.95,
            'claude_learning': 0.85,
            
            # Medium-high confidence (model's own reasoning)
            'self_dialogue_summary': 0.8,   # Multi-turn reasoned conclusions
            'self': 0.75,                    # Individual insights from self-dialogue
            
            # Medium confidence sources (documents and images)
            'document_summary': 0.7,
            'image_analysis': 0.7,
            'conversation_summary': 0.75,
            'user_categorization': 0.8,
            
            # Lower confidence sources (web and inferences)
            'web_knowledge': 0.5,
            'inference': 0.4,
            'assumption': 0.35,

            # Medium-high confidence (user topics and commands)
            'user_topic': 0.75,              # User-related topic information
            'command': 0.7,                  # Command documentation/examples
            'reminder': 0.85,                # User-created reminders (Ken set these)
            
            # Default for unknown types
            'unknown': 0.5,
            'general': 0.5
        }
        
        # Normalize memory_type to lowercase for matching
        memory_type_lower = memory_type.lower() if memory_type else 'unknown'
        
        # Return baseline, defaulting to 0.5 for unknown types
        return source_baselines.get(memory_type_lower, 0.5)

    def _extract_current_confidence(self, memory: dict) -> float:
        """
        Extract the current confidence value from a memory.
        
        Args:
            memory (dict): Memory dictionary
            
        Returns:
            float: Current confidence value (defaults to 0.5)
        """
        try:
            # Try to get confidence from top-level
            confidence = memory.get('confidence')
            
            # If not found, check metadata
            if confidence is None:
                metadata = memory.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                confidence = metadata.get('confidence')
            
            # Convert to float if needed
            if confidence is not None:
                return float(confidence)
            
            return 0.5  # Default
            
        except (ValueError, TypeError):
            return 0.5

    def _extract_created_at(self, memory: dict) -> str:
        """
        Extract the created_at timestamp from a memory.
        
        Args:
            memory (dict): Memory dictionary
            
        Returns:
            str: Created at timestamp or 'unknown'
        """
        try:
            metadata = memory.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            
            return metadata.get('created_at', 'unknown')
            
        except Exception:
            return 'unknown'
        
    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Consolidated confidence calculator (includes Ken's confidence scale notes) — paired with refactored consolidation flow.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__calculate_consolidated_confidence(self, memories: list) -> Tuple[float, str]:
        """
        Calculate confidence for consolidated memories using Ken's defined scale.
        
        Confidence Scale (Ken's definitions):
        - 0.9-1.0: Highly confident, verified, or explicitly stated by Ken
        - 0.6-0.8: Reasonably confident, clear context
        - 0.3-0.5: Uncertain, may need verification later
        
        This method:
        - Validates confidence values against the defined scale
        - Extracts confidence from multiple metadata paths
        - Applies confirmation bonus when multiple memories agree
        - Clamps output to valid range (0.3-1.0 for stored data)
        
        Args:
            memories (list): List of memory dictionaries to consolidate
            
        Returns:
            tuple: (new_confidence: float, confidence_reasoning: str)
        """
        try:
            # --- Extract confidence values with source tracking ---
            confidence_data = []
            
            for m in memories:
                # Skip non-dict entries
                if not isinstance(m, dict):
                    continue
                    
                confidence = None
                source_type = 'unknown'
                
                # --- Path 1: Top-level confidence field ---
                if 'confidence' in m:
                    confidence = m.get('confidence')
                
                # --- Path 2: Confidence nested in metadata ---
                if confidence is None:
                    metadata = m.get('metadata', {})
                    # Handle string-encoded JSON metadata
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    # Extract confidence and source type from metadata
                    if isinstance(metadata, dict):
                        confidence = metadata.get('confidence')
                        source_type = metadata.get('type', m.get('memory_type', 'unknown'))
                
                # --- Validate and clamp confidence to valid range ---
                if confidence is not None:
                    try:
                        confidence = float(confidence)
                        # Clamp to valid range (0.1 to 1.0)
                        confidence = max(0.1, min(1.0, confidence))
                    except (ValueError, TypeError):
                        confidence = None
                
                # --- Fallback: Assign baseline confidence based on source type ---
                if confidence is None:
                    confidence = self._get_source_baseline_confidence(source_type)
                
                # --- Store confidence data for analysis ---
                confidence_data.append({
                    'confidence': confidence,
                    'source_type': source_type,
                    'content_preview': m.get('content', '')[:50]
                })
            
            # --- Handle edge case: no valid confidence data ---
            if not confidence_data:
                logging.warning("No valid confidence data found in memories for consolidation")
                return 0.5, "No valid confidence data found, using default"
            
            # --- Calculate weighted confidence with confirmation bonus ---
            confidences = [d['confidence'] for d in confidence_data]
            max_confidence = max(confidences)
            avg_confidence = sum(confidences) / len(confidences)
            num_memories = len(confidence_data)
            
            # Confirmation bonus: Multiple memories agreeing increases confidence
            # Each additional confirming memory adds up to 0.05, capped at 0.15 total bonus
            # Rationale: If 4+ memories say the same thing, that's strong confirmation
            confirmation_bonus = min(0.15, (num_memories - 1) * 0.05)
            
            # Start with max confidence (preserve highest quality source)
            # Then add confirmation bonus for corroborating memories
            new_confidence = max_confidence + confirmation_bonus
            
            # --- Clamp to valid range per Ken's scale ---
            # 0.3 minimum for stored/consolidated data (not "garbage")
            # 1.0 maximum (can't exceed certainty)
            new_confidence = max(0.3, min(1.0, new_confidence))
            
            # Round to 2 decimal places for cleaner storage
            new_confidence = round(new_confidence, 2)
            
            # --- Build reasoning string for logging/debugging ---
            source_types_found = list(set(d['source_type'] for d in confidence_data))
            reasoning = (
                f"Base: {max_confidence:.2f} (highest of {num_memories} memories), "
                f"Avg: {avg_confidence:.2f}, "
                f"Confirmation bonus: +{confirmation_bonus:.2f}, "
                f"Sources: {source_types_found[:3]}, "
                f"Final: {new_confidence:.2f}"
            )
            
            logging.info(f"Consolidated confidence calculation: {reasoning}")
            
            return new_confidence, reasoning
            
        except Exception as e:
            logging.error(f"Error calculating consolidated confidence: {e}", exc_info=True)
            # Return safe default on error
            return 0.5, f"Error in calculation: {e}"


    def _evaluate_memory_confidence_with_llm(self, content: str, memory_type: str, 
                                              current_confidence: float, 
                                              baseline_confidence: float) -> dict:
        """
        Use LLM to evaluate memory confidence based on linguistic indicators.
        
        Args:
            content (str): Memory content to evaluate
            memory_type (str): Type of memory (for context)
            current_confidence (float): Current confidence value
            baseline_confidence (float): Source-based baseline confidence
            
        Returns:
            dict: Evaluation result with 'recommended_confidence' and 'reasoning'
        """
        try:
            evaluation_prompt = f""" /no_think
Evaluate the confidence level for this stored memory. You MUST recommend a specific confidence value.

MEMORY CONTENT:
{content[:1500]}

MEMORY TYPE: {memory_type}
CURRENT CONFIDENCE: {current_confidence}
SOURCE-BASED BASELINE FOR THIS TYPE: {baseline_confidence}

CONFIDENCE SCALE (use these ranges):
- 0.9-1.0: Direct statements from Ken, verified facts, explicit attribution ("Ken said...", "Ken confirmed...")
- 0.8-0.9: Claude knowledge, self-dialogue conclusions, high-quality reasoned content
- 0.6-0.8: Document summaries, image analysis, user topics, clear context
- 0.4-0.6: Web knowledge, moderate inferences
- 0.3-0.5: Assumptions, unclear source, needs verification
- 0.1-0.2: Appears questionable, speculation, conflicting language

SOURCE TYPE BASELINES (start here, then adjust based on content):
- user_preference, personal_info, user_statement → 0.9-1.0
- claude_learning → 0.85
- self_dialogue_summary → 0.8
- user_topic → 0.75
- conversation_summary → 0.75
- document_summary, image_analysis, command → 0.7
- web_knowledge → 0.5
- general, unknown → 0.5 (but ADJUST based on content!)

EVALUATION RULES:
1. START with the source baseline for this memory type ({baseline_confidence})
2. INCREASE confidence if: direct quotes, clear attribution, "Ken said/mentioned/confirmed"
3. DECREASE confidence if: hedging words ("might", "possibly", "seems"), speculation, no clear source
4. For "general" type: Look at the CONTENT to determine what it actually is
5. If content mentions Ken directly with clear context, confidence should be 0.7+
6. DO NOT just keep the current confidence - actively evaluate and recommend!

TASK: Respond in this EXACT format (no other text):
RECOMMENDED_CONFIDENCE: [0.0-1.0 numeric value - BE SPECIFIC, not just the baseline]
REASONING: [2-3 sentences explaining your specific evaluation]
"""
            
            # Get LLM evaluation
            response = self._safe_llm_invoke(evaluation_prompt)
            
            if not response:
                logging.warning("LLM returned empty response for confidence evaluation")
                return None
            
            # Parse response
            result = {}
            
            # Extract recommended confidence
            confidence_match = re.search(r'RECOMMENDED_CONFIDENCE:\s*([\d.]+)', response)
            if confidence_match:
                try:
                    recommended = float(confidence_match.group(1))
                    # Clamp to valid range
                    result['recommended_confidence'] = max(0.0, min(1.0, recommended))
                except ValueError:
                    result['recommended_confidence'] = baseline_confidence
            else:
                result['recommended_confidence'] = baseline_confidence
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=$|\n\n)', response, re.DOTALL)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()
            else:
                result['reasoning'] = "No detailed reasoning provided by evaluation."
            
            return result
            
        except Exception as e:
            logging.error(f"Error in LLM confidence evaluation: {e}", exc_info=True)
            return None

    def _update_memory_confidence(self, memory: dict, new_confidence: float) -> bool:
        """
        Update a memory's confidence using the transaction coordinator (FORGET/STORE pattern).
        
        Args:
            memory (dict): Memory to update
            new_confidence (float): New confidence value
            
        Returns:
            bool: True if update succeeded, False otherwise
        """
        try:
            content = memory.get('content', '')

            if not content:
                logging.error("Cannot update memory with empty content")
                return False

            # --- Extract and parse metadata FIRST ---
            # IMPORTANT: metadata must be parsed before reading memory_type because
            # vector_db stores type inside metadata dict under key 'type', NOT as a
            # top-level 'memory_type' key on the memory object. Reading from the wrong
            # location would return 'unknown' and overwrite the correct type on re-store.
            metadata = memory.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    logging.warning("Failed to parse metadata JSON, defaulting to empty dict")
                    metadata = {}

            # FIX: Read memory_type from metadata.type (correct location in vector_db)
            # Previously read from memory.get('memory_type') which is always absent/unknown
            memory_type = metadata.get('type', 'unknown')
            logging.info(f"_update_memory_confidence: resolved memory_type='{memory_type}' from metadata")

            # --- Update metadata with new confidence and audit timestamp ---
            metadata['confidence'] = new_confidence
            metadata['last_confidence_audit'] = datetime.datetime.now().isoformat()

            # Preserve existing type in metadata (now guaranteed correct from above)
            if 'type' not in metadata:
                # Fallback: should rarely hit this since we just read from metadata.type
                metadata['type'] = memory_type
                logging.warning(f"'type' missing from metadata after read — setting to '{memory_type}'")

            # --- FORGET old memory ---
            logging.info(f"🗑️ Forgetting old memory: {content[:50]}...")
            forget_result, forget_success = self.chatbot.deepseek_enhancer._handle_regular_memory_forget(content)

            if not forget_success:
                logging.error(f"❌ Failed to forget old memory: {forget_result}")
                return False

            logging.info("✅ Old memory forgotten successfully")

            # --- STORE updated memory with corrected memory_type ---
            logging.info(f"💾 Storing updated memory: type='{memory_type}', confidence={new_confidence:.2f}")
            store_success, memory_id = self.chatbot.store_memory_with_transaction(
                content=content,
                memory_type=memory_type,   # Now correctly reflects original stored type
                metadata=metadata,
                confidence=new_confidence
            )

            if store_success:
                logging.info(f"✅ Memory updated successfully with ID: {memory_id}")
                return True
            else:
                logging.error(f"❌ CRITICAL: Forgot memory but failed to store updated version!")
                return False

        except Exception as e:
            logging.error(f"Error updating memory confidence: {e}", exc_info=True)
            return False

    def _generate_confidence_audit_report(self, memories_evaluated: int, memories_updated: int,
                                           memories_unchanged: int, errors: int,
                                           audit_results: list) -> str:
        """
        Generate a detailed audit report for the confidence audit task.
        
        Args:
            memories_evaluated (int): Total memories evaluated
            memories_updated (int): Memories that were updated
            memories_unchanged (int): Memories that didn't need changes
            errors (int): Number of errors encountered
            audit_results (list): Detailed results for each memory
            
        Returns:
            str: Formatted audit report
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""================================================================================
MEMORY CONFIDENCE AUDIT REPORT
Generated: {timestamp}
Memories Evaluated: {memories_evaluated}
Memories Modified: {memories_updated}
================================================================================

"""
        
        # Add individual memory sections
        for i, result in enumerate(audit_results, 1):
            report += f"""--------------------------------------------------------------------------------
MEMORY {i} of {len(audit_results)}
--------------------------------------------------------------------------------
CONTENT PREVIEW: {result.get('content_preview', 'N/A')}...
MEMORY TYPE: {result.get('memory_type', 'unknown')}
CREATED: {result.get('created_at', 'unknown')}

BEFORE:
  Confidence: {result.get('before_confidence', 'N/A')}
  
AFTER:
  Confidence: {result.get('after_confidence', 'N/A') if result.get('after_confidence') is not None else 'N/A (error)'}
  
REASONING:
  {result.get('reasoning', 'No reasoning provided')}

ACTION TAKEN: {result.get('action', 'UNKNOWN')}

"""
        
        # Add summary section
        # Calculate confidence change statistics
        increases = []
        decreases = []
        
        for result in audit_results:
            if result.get('action') == 'UPDATED':
                before = result.get('before_confidence', 0)
                after = result.get('after_confidence', 0)
                if after is not None:
                    diff = after - before
                    if diff > 0:
                        increases.append(diff)
                    elif diff < 0:
                        decreases.append(abs(diff))
        
        avg_increase = sum(increases) / len(increases) if increases else 0
        avg_decrease = sum(decreases) / len(decreases) if decreases else 0
        
        report += f"""================================================================================
AUDIT SUMMARY
================================================================================
Total Evaluated: {memories_evaluated}
Updated: {memories_updated}
Unchanged: {memories_unchanged}
Errors: {errors}

Confidence Changes:
  - Increased: {len(increases)} (avg +{avg_increase:.2f})
  - Decreased: {len(decreases)} (avg -{avg_decrease:.2f})
  - Unchanged: {memories_unchanged}

Source-Based Confidence Guidelines Used:
  - Ken's direct statements: 0.9-1.0
  - Claude knowledge (claude_learning): 0.8-0.9
  - Document summaries: 0.6-0.8
  - Image analysis: 0.6-0.8
  - Web knowledge: 0.4-0.6
  - Inferences/unknown: 0.3-0.5

Next scheduled audit: ~84 hours
================================================================================
"""
        
        return report

       
    def start_cognitive_thread(self):
        """Start the autonomous memory management thread if it's not already running."""
        if self.thinking_thread is None or not self.thinking_thread.is_alive():
            self.stop_flag.clear()
            self.thinking_thread = threading.Thread(
                target=self._cognitive_loop,
                name="DeepSeek-Memory-Management",
                daemon=True
            )
            self.thinking_thread.start()
            logging.info("Started autonomous memory management thread")
            return True
        else:
            logging.info("Autonomous memory management thread already running")
            return False
    
    def stop_cognitive_thread(self):
        """Signal the cognitive thread to stop."""
        if self.thinking_thread and self.thinking_thread.is_alive():
            self.stop_flag.set()
            logging.info("Signaled autonomous cognitive thread to stop")
            return True
        return False
    
    def update_user_activity(self):
        """Update the timestamp of the last user activity."""
        self.last_user_activity = time.time()
        logging.debug("Updated user activity timestamp")
    
    def _is_user_inactive(self, inactivity_threshold=3600): #must be idle for 1 hours then starts
        """
        Check if the user has been inactive for a sufficient period.

        Args:
            inactivity_threshold (int): Seconds of inactivity to consider user inactive (default: 1 hour)
            
        Returns:
            bool: True if user is inactive, False otherwise
        """
        time_since_activity = time.time() - self.last_user_activity
        return time_since_activity > inactivity_threshold
    
    def _cognitive_loop(self):
        """Main cognitive loop for autonomous memory management.

        Architecture (post Track A Issue 4 fix, 2026-05-24):
        Loop runs every cognitive_cycle_interval seconds (now 300s / 5 min,
        previously 3600s / 1 hour). Each iteration performs two passes:

        Pass 1 — Scheduled reflections (time-of-day events):
            Daily 6:15 AM, weekly Sunday 9:15 AM, monthly 1st 12:20 PM.
            Gated only by autonomous_thinking_enabled, _llm_generating, and
            conversation_in_progress flags. NOT gated by the 1-hour idle
            requirement, because these are deliberately scheduled for times
            when the user is typically idle, and the 30-minute windows would
            routinely be missed if also required 1 hour of accumulated
            inactivity on top of the schedule. Idempotency is provided by
            JSON completion files in reflections/.

        Pass 2 — Weighted-pool cognitive activities:
            analyze_knowledge_gaps, fill_knowledge_gaps, audit_memory_confidence,
            memory_consolidation_pulse, functional_state_baseline,
            self_model_integrity_check. These do substantial work and could
            disrupt user flow, so they require BOTH 1 hour of accumulated
            inactivity AND _select_next_cognitive_activity to pick them
            (subject to per-activity min_interval_hours cooldowns).
        """
        logging.info("Autonomous memory management loop started")

        # Initial wait period to allow system to stabilize
        time.sleep(60)

        while not self.stop_flag.is_set():
            try:
                # ===== Universal safety gates (apply to BOTH passes) =====

                # Gate 1: Autonomous thinking master switch
                if hasattr(self.chatbot, 'autonomous_thinking_enabled') and not self.chatbot.autonomous_thinking_enabled:
                    logging.debug("Autonomous thinking disabled, pausing cognitive loop")
                    time.sleep(60)
                    continue

                # Gate 2: LLM not currently generating a user-facing response
                if hasattr(self.chatbot, '_llm_generating') and self.chatbot._llm_generating:
                    logging.debug("LLM generating user response - pausing cognitive loop")
                    time.sleep(60)
                    continue

                # Gate 3: No active conversation in progress
                if hasattr(self.chatbot, 'conversation_in_progress') and self.chatbot.conversation_in_progress:
                    logging.debug("Conversation in progress - pausing cognitive loop")
                    time.sleep(60)
                    continue

                # ===== Pass 1: Scheduled reflections (NOT gated by idle) =====
                # Time-of-day events with their own 30-minute window logic and
                # JSON-file-based idempotency. Cheap when no window is open —
                # the method returns quickly after the schedule check.
                try:
                    self._check_scheduled_reflections()
                except Exception as scheduled_err:
                    logging.error(
                        f"Scheduled reflection check raised an unhandled exception: "
                        f"{scheduled_err}",
                        exc_info=True
                    )

                # ===== Pass 2: Weighted-pool activities (gated by idle) =====
                if self._is_user_inactive():
                    # Extra defensive check: how long since last user activity?
                    # _is_user_inactive has its own threshold, but the flag can
                    # be stale; double-check via last_user_activity timestamp.
                    time_since_activity = time.time() - self.last_user_activity
                    if time_since_activity < 3600:  # Less than 1 hour
                        logging.debug(
                            f"Recent activity ({time_since_activity/60:.1f}m ago), "
                            f"deferring weighted-pool activities"
                        )
                    else:
                        # All safety gates passed — safe to run weighted-pool activity
                        logging.info("✅ All safety checks passed - proceeding with autonomous memory management")

                        # Select a cognitive activity based on weights and last run time
                        activity = self._select_next_cognitive_activity()

                        if activity:
                            logging.info(f"Selected memory management activity: {activity}")
                            self.cognitive_state = activity

                            # Execute the selected cognitive activity.
                            # Wrapped in try/finally per Track A Issue 7 fix (2026-05-24):
                            # if method() raises an unhandled exception, last_run was previously
                            # NOT updated, leaving the activity eligible for immediate retry on
                            # the next cycle (since _should_run_activity returns True when
                            # last_run is None). That risked thrashing on a persistently broken
                            # activity. Now last_run is always updated in finally — the cooldown
                            # gives breathing room for transient issues to clear before retry.
                            if hasattr(self, f"_{activity}"):
                                method = getattr(self, f"_{activity}")
                                try:
                                    method()
                                except Exception as activity_err:
                                    # Log prominently — silent activity failure is the symptom
                                    # we are explicitly guarding against. Each activity has its
                                    # own try/except, so reaching here means something escaped,
                                    # which we want highly visible in the logs.
                                    logging.error(
                                        f"Activity '{activity}' raised an unhandled exception: "
                                        f"{activity_err}",
                                        exc_info=True
                                    )
                                finally:
                                    # Always update last_run, even on failure, to prevent
                                    # thrashing on a persistently-broken activity. Cooldown
                                    # provides breathing room before retry.
                                    self.cognitive_activities[activity]["last_run"] = time.time()
                            else:
                                logging.error(f"No method found for cognitive activity: {activity}")

                        # Reset cognitive state to idle
                        self.cognitive_state = "idle"
                else:
                    # User active — only Pass 1 (scheduled reflections) ran this tick.
                    # Weighted-pool activities deferred until next idle period.
                    logging.debug("User active, weighted-pool activities deferred")

                # Sleep before next iteration
                time.sleep(self.cognitive_cycle_interval)

            except Exception as e:
                logging.error(f"Error in cognitive loop: {e}", exc_info=True)
                self.cognitive_state = "error"
                time.sleep(300)  # Recovery pause after error (5 minutes)

        logging.info("Autonomous memory management loop stopped")
          
    def _update_activity_timestamp(self, activity_name):
        """
        Update the last run timestamp for an activity.
        
        Args:
            activity_name (str): Name of the activity
        """
        if activity_name in self.cognitive_activities:
            self.cognitive_activities[activity_name]["last_run"] = time.time()
            logging.debug(f"Updated last run timestamp for activity: {activity_name}")
        else:
            logging.warning(f"Cannot update timestamp for unknown activity: {activity_name}")

    def _analyze_knowledge_gaps(self):
        """
        Analyze existing knowledge to identify gaps related to the user and their needs.
        Enhanced with duplicate prevention and gap limiting - NOW FOCUSES ON SINGLE MOST IMPORTANT GAP.
        """
        print("🧠 ====== STARTING ENHANCED KNOWLEDGE GAP ANALYSIS ======")
        logging.info("====== STARTING ENHANCED KNOWLEDGE GAP ANALYSIS ======")
        self.cognitive_state = "analyzing"
        
        try:
            # Record the start of analysis
            print("📝 Step 1: Recording thought for knowledge analysis")
            logging.info("Step 1: Recording thought for knowledge analysis")
            self._record_thought(
                thought_type="knowledge_analysis", 
                content="Beginning enhanced analysis of knowledge gaps with single-gap focus and duplicate prevention."
            )
            
            # Get recent user queries and memories
            print("📊 Step 2: Getting recent user queries")
            logging.info("Step 2: Getting recent user queries")
            recent_queries = self._get_recent_queries()
            query_count = len(recent_queries.splitlines()) if recent_queries else 0
            print(f"   Found {query_count} recent queries")
            logging.info(f"Found queries: {query_count} recent queries")
            
            print("💾 Step 3: Getting relevant memories for analysis")
            logging.info("Step 3: Getting relevant memories for analysis")
            relevant_memories = self._get_relevant_memories_for_analysis()
            memory_count = len(relevant_memories.splitlines()) if relevant_memories else 0
            print(f"   Found {memory_count} relevant memory segments")
            logging.info(f"Found memories: {memory_count} relevant memory segments")
            
            if not recent_queries and not relevant_memories:
                print("⚠️ Insufficient data for knowledge gap analysis")
                logging.info("Insufficient data for knowledge gap analysis")
                self._record_thought(
                    thought_type="knowledge_analysis",
                    content="Insufficient recent data to identify meaningful knowledge gaps."
                )
                return False
            
            # NEW: Get existing pending gaps for duplicate checking
            print("🔍 Step 3.5: Loading existing gaps for duplicate prevention")
            logging.info("Step 3.5: Loading existing gaps for duplicate prevention")
            try:
                from knowledge_gap import KnowledgeGapQueue
                gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
                existing_gaps = gap_queue.get_gaps_by_status('pending')
                print(f"   Found {len(existing_gaps)} existing pending gaps")
                logging.info(f"Found {len(existing_gaps)} existing pending gaps for duplicate checking")
            except Exception as e:
                print(f"   ⚠️ Could not load existing gaps: {e}")
                logging.warning(f"Could not load existing gaps for duplicate checking: {e}")
                existing_gaps = []
            
            # Load historical (fulfilled + failed) gaps for duplicate prevention
            # CRITICAL: mark_fulfilled() removes Qdrant embeddings, so fulfilled gaps
            # are invisible to semantic similarity checks. SQL-based loading is the
            # only way to prevent re-proposing already-investigated topics.
            print("🔍 Step 3.6: Loading historical gaps (fulfilled/failed) for duplicate prevention")
            logging.info("Step 3.6: Loading historical gaps for duplicate prevention")
            historical_gaps = self._get_historical_gaps(days=90, limit=50)
            print(f"   Found {len(historical_gaps)} historical gaps from last 90 days")
            logging.info(f"Found {len(historical_gaps)} historical gaps for duplicate checking")
            
            # Merge pending + historical for comprehensive duplicate checking
            # This prevents the system from re-proposing gaps that were already
            # investigated, fulfilled, or failed — regardless of current status
            all_known_gaps = existing_gaps + historical_gaps
            print(f"   Total gaps for dedup: {len(all_known_gaps)} ({len(existing_gaps)} pending + {len(historical_gaps)} historical)")
            logging.info(f"Total gaps for dedup: {len(all_known_gaps)} (pending + historical)")
                
            # Get recent conversation context to check for answered questions
            recent_conversation_text = self._get_recent_conversation_text()
            logging.info(f"Included {len(recent_conversation_text)} characters of conversation context")

            # Create simplified analysis prompt - UPDATED TO REQUEST ONLY 1 GAP
            print("🛠️ Step 4: Constructing enhanced prompt for single gap identification")
            logging.info("Step 4: Constructing enhanced prompt for single gap identification")

            prompt = f""" /no_think
I will analyze recent queries and stored information to identify ONLY THE SINGLE MOST IMPORTANT knowledge gap.

⚠️ CRITICAL ANTI-DUPLICATION RULES ⚠️

1. **Check Conversation History First**
- Review the recent conversations below
- If Ken has ALREADY answered or discussed a topic, it is NOT a knowledge gap
- DO NOT create gaps about topics Ken has explained, even if my stored memories are incomplete

2. **Compare Against Existing Gaps**
- Review ALL existing pending gaps listed below
- If a gap is similar in ANY way (same concept, related topic, overlapping question), DO NOT create it
- Examples of duplicates to AVOID:
    * "Ken's field of work" vs "Ken's professional domain" = DUPLICATE
    * "Ken's hobbies" vs "Ken's leisure activities" = DUPLICATE
    * "Ken's family structure" vs "Ken's relatives" = DUPLICATE

3. **Semantic Similarity Check**
- Before proposing a gap, ask: "Is this essentially asking the same thing as an existing gap?"
- If 70% of the information would overlap, it's a DUPLICATE
- Different wording does NOT make it a new gap

4. **Quality Over Quantity - SINGLE GAP FOCUS**
- I will identify ONLY THE SINGLE MOST IMPORTANT gap
- The gap must be:
    * Truly unknown (not discussed in conversations)
    * Completely different from existing gaps (no overlap)
    * Specific and actionable (not vague or general)
    * Important for helping Ken (not just interesting trivia)
- If no truly valuable gap exists, I will return NONE

5. **Classification Accuracy - CRITICAL RULES**

⚠️ PERSONAL_ASK_KEN - Use for ALL of these:
- ANY information about "Ken Bajema" personally (profession, background, work, education, career)
- Ken's preferences, opinions, feelings, or personal context
- Information about Ken's family (Ananda, Izabel, Lucian)
- Ken's location, home, garden, daily routines
- Ken's projects (QWEN, autonomous AI work)
- REASON: There are MULTIPLE Ken Bajemas online. Web searches will return WRONG PEOPLE.
- ALWAYS ask Ken directly for ANY information about himself.

🌐 PUBLIC_SEARCHABLE - Use ONLY for:
- General factual knowledge NOT about Ken personally
- Technology concepts, scientific principles, historical facts
- Programming techniques, API documentation, technical standards
- World events, public figures (other than Ken), general information
- Examples: "What is vector database indexing?", "Python async patterns", "History of neural networks"

🧠 SYSTEM_INTERNAL - Use for:
- My own capabilities, reasoning patterns, or self-improvement
- How I process information or make decisions

⚠️ NEVER classify gaps about "Ken Bajema", "Ken's profession", "Ken's background", 
"Ken's work", or ANY personal information about Ken as PUBLIC_SEARCHABLE.
Web searches will return information about OTHER Ken Bajemas, not our user.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 RECENT CONVERSATIONS (check if topics were already discussed):
{recent_conversation_text[:1500] if recent_conversation_text else "No recent conversations"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 RECENT QUERIES:
{recent_queries[:1000] if recent_queries else "No recent queries"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💾 RELEVANT STORED KNOWLEDGE:
{relevant_memories[:1500] if relevant_memories else "No relevant memories"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚫 EXISTING GAPS — PENDING + RECENTLY PROCESSED (DO NOT DUPLICATE THESE):
{self._format_existing_gaps_for_prompt(all_known_gaps)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ BEFORE PROPOSING ANY GAP, I MUST:
1. ✓ Verify it was NOT discussed in recent conversations
2. ✓ Confirm it does NOT overlap with existing pending gaps
3. ✓ Ensure it is NOT semantically similar to any existing gap
4. ✓ Verify it is truly unknown (not answered by Ken already)
5. ✓ If about Ken Bajema → MUST be PERSONAL_ASK_KEN (never PUBLIC_SEARCHABLE)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I will identify ONLY THE SINGLE MOST IMPORTANT knowledge gap. Quality over quantity.

MAXIMUM 1 GAP ONLY. Format:

GAP 1:
TOPIC: [the most important unique topic - must be completely different from existing gaps]
CLASSIFICATION: [PUBLIC_SEARCHABLE|PERSONAL_ASK_KEN|SYSTEM_INTERNAL]
DESCRIPTION: [specific unknown information - explain why this is truly a gap]
PRIORITY: [HIGH/MEDIUM/LOW]
UNIQUENESS_CHECK: [explain how this differs from existing gaps and conversation topics]

⚠️ REMEMBER: If the gap is about Ken Bajema in ANY way, classification MUST be PERSONAL_ASK_KEN.
⚠️ I will focus on identifying the SINGLE MOST VALUABLE gap. If no truly unique and important gap exists, I will return NONE rather than proposing a low-quality gap.
"""

            
            # Get knowledge gap analysis from LLM
            print("🤖 Step 5: Calling LLM to identify the single most important knowledge gap")
            logging.info("Step 5: Calling LLM to identify single most important knowledge gap")
            try:
                raw_gaps = self._safe_llm_invoke(prompt)
                
                if not raw_gaps:
                    print("❌ LLM returned empty response")
                    logging.warning("LLM returned empty response for knowledge gap analysis")
                    return False
                
                print(f"✅ LLM response received, length: {len(raw_gaps)} characters")
                logging.info(f"LLM response received, length: {len(raw_gaps)}")
                
                # Parse the structured response
                print("📋 Step 6: Parsing structured text response from LLM")
                logging.info("Step 6: Parsing structured text response from LLM")
                knowledge_gaps = self._parse_knowledge_gaps_response(raw_gaps)
                
                if not knowledge_gaps:
                    print("❌ No knowledge gaps could be parsed from LLM response")
                    print(f"📄 Full LLM response was: {raw_gaps[:500]}...")
                    logging.warning("No knowledge gaps could be parsed from LLM response")
                    return False
                    
                print(f"✅ Successfully parsed {len(knowledge_gaps)} knowledge gap(s)")
                logging.info(f"Successfully parsed {len(knowledge_gaps)} knowledge gap(s)")
                
                # NEW: Step 7 - Enhanced duplicate checking and filtering
                print("🔍 Step 7: Enhanced duplicate checking and filtering")
                logging.info("Step 7: Enhanced duplicate checking and filtering")
                
                filtered_gaps = []
                duplicate_count = 0
                
                for i, gap in enumerate(knowledge_gaps):
                    topic = gap.get('topic', 'Unknown')
                    description = gap.get('description', 'No description')
                    priority = gap.get('priority', 'MEDIUM')
                    
                    print(f"   🔎 Checking gap {i+1}: {topic}")
                    
                    # --- Step 7: Check against ALL KNOWN GAPS for duplicates ---
                    # Catches topics already pending, fulfilled, or failed in the knowledge_gaps table.
                    # Uses all_known_gaps (pending + historical) to prevent re-proposing
                    # topics that were already investigated and resolved.
                    if self._check_for_similar_gaps(topic, description, all_known_gaps):
                        duplicate_count += 1
                        print(f"      ❌ Duplicate gap detected, skipping: {topic}")
                        logging.info(f"Skipped duplicate gap: {topic}")
                        continue

                    # --- Step 7.5: Check against LONG-TERM MEMORY for coverage ---
                    # This is the key fix: before approving a gap, verify that QWEN
                    # doesn't already have meaningful memory coverage on this topic.
                    # Topics like "Ken's work background" or "AI design philosophy"
                    # are extensively covered in long-term memory even though they
                    # don't appear in the 20 most recent memories passed to the LLM.
                    # If coverage exists, this is a FALSE GAP — suppress it.
                    print(f"      🧠 Step 7.5: Checking long-term memory coverage for: {topic}")
                    logging.info(f"[Step 7.5] Checking memory coverage for gap: '{topic}'")

                    try:
                        is_covered, hit_count = self._check_memory_coverage(topic, description)

                        if is_covered:
                            # Topic already well-covered in long-term memory — not a real gap
                            duplicate_count += 1
                            print(f"      ❌ Memory coverage exists ({hit_count} hits), suppressing false gap: {topic}")
                            logging.info(
                                f"[Step 7.5] Suppressed false gap '{topic}' — "
                                f"already covered in long-term memory ({hit_count} unique hits)"
                            )
                            continue
                        else:
                            # Low or no coverage confirmed — this is a genuine gap
                            print(f"      ✅ Low memory coverage ({hit_count} hits) — genuine gap confirmed: {topic}")
                            logging.info(
                                f"[Step 7.5] Confirmed genuine gap '{topic}' — "
                                f"low memory coverage ({hit_count} unique hits)"
                            )

                    except Exception as coverage_err:
                        # Fail open — log the error but allow the gap through
                        logging.error(
                            f"[Step 7.5] Coverage check error for '{topic}': {coverage_err}. "
                            f"Allowing gap through.", exc_info=True
                        )
                        print(f"      ⚠️ Coverage check error, allowing gap through: {topic}")
                    
                    # Check for duplicates within current batch
                    is_internal_duplicate = False
                    for existing_gap in filtered_gaps:
                        if self._gaps_are_similar(gap, existing_gap):
                            is_internal_duplicate = True
                            duplicate_count += 1
                            print(f"      ❌ Internal duplicate detected, skipping: {topic}")
                            logging.info(f"Skipped internal duplicate gap: {topic}")
                            break
                    
                    if not is_internal_duplicate:
                        filtered_gaps.append(gap)
                        print(f"      ✅ Unique gap approved: {topic}")
                        logging.info(f"Approved unique gap: {topic}")
                
                print(f"📊 DUPLICATE FILTERING RESULTS:")
                print(f"   📥 Initial gaps identified: {len(knowledge_gaps)}")
                print(f"   ❌ Duplicates filtered out: {duplicate_count}")
                print(f"   ✅ Unique gaps remaining: {len(filtered_gaps)}")
                
                # NEW: Limit to single highest-priority gap
                if len(filtered_gaps) > 1:
                    # Sort by priority and keep only the highest
                    priority_map = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
                    filtered_gaps.sort(key=lambda g: priority_map.get(g.get('priority', 'MEDIUM'), 2), reverse=True)
                    original_count = len(filtered_gaps)
                    filtered_gaps = filtered_gaps[:1]
                    print(f"   🎯 Limited to single highest-priority gap (reduced from {original_count} to 1)")
                    logging.info(f"Limited to single highest-priority gap for focused processing (reduced from {original_count})")
                
                # Use filtered gaps
                knowledge_gaps = filtered_gaps
                
                if not knowledge_gaps:
                    print("❌ No unique knowledge gaps remaining after duplicate filtering and prioritization")
                    logging.warning("No unique knowledge gaps remaining after duplicate filtering and prioritization")
                    return False
                
                # Enhanced logging to verify what we found
                print(f"🎯 ENHANCED KNOWLEDGE GAP ANALYSIS RESULTS:")
                print(f"   📊 Found {len(knowledge_gaps)} unique knowledge gap (SINGLE-GAP FOCUS)")
                
                for i, gap in enumerate(knowledge_gaps, 1):
                    topic = gap.get('topic', 'Unknown')
                    description = gap.get('description', 'No description')
                    priority = gap.get('priority', 'MEDIUM')
                    classification = gap.get('classification', 'PUBLIC_SEARCHABLE')
                    
                    print(f"   Gap {i}: {topic}")
                    print(f"      Classification: {classification}")
                    print(f"      Priority: {priority}")
                    print(f"      Description: {description[:100]}...")
                    
                    # Also log to file
                    logging.info(f"Gap {i}: {topic} (Classification: {classification}, Priority: {priority})")
                    logging.info(f"   Description: {description}")
                
                # Record the gaps identified
                gap_summary = "\n".join([f"- {gap['topic']} ({gap.get('classification', 'UNKNOWN')}): {gap['description'][:100]}..." 
                                    for gap in knowledge_gaps])
                                    
                logging.info(f"Enhanced knowledge gaps identified (SINGLE-GAP FOCUS):\n{gap_summary}")
                
                self._record_thought(
                    thought_type="knowledge_analysis",
                    content=f"Enhanced analysis identified {len(knowledge_gaps)} unique gap (filtered {duplicate_count} duplicates, SINGLE-GAP FOCUS):\n{gap_summary}"
                )
                
                # Process the unique gaps using classification
                if knowledge_gaps:
                    print(f"🗃️ Step 8: PROCESSING {len(knowledge_gaps)} CLASSIFIED KNOWLEDGE GAP (SINGLE-GAP FOCUS)...")
                    logging.info("Step 8: Processing classified knowledge gap (SINGLE-GAP FOCUS)")
                    
                    # Create gaps analysis structure for processing
                    gaps_analysis = {"gaps": knowledge_gaps}
                    
                    # Use the new classification-based processing with parsed gaps
                    success = self._process_classified_gaps(gaps_analysis)

                    if success:
            
                        print("✅ ====== ENHANCED KNOWLEDGE GAP ANALYSIS COMPLETED SUCCESSFULLY ======")
                        logging.info("====== ENHANCED KNOWLEDGE GAP ANALYSIS COMPLETED SUCCESSFULLY ======")

                        # Chain to _fill_knowledge_gaps per Track A Issue 5 fix (2026-05-24).
                        # Previously, analyze and fill were independent activities with
                        # 96h cooldowns each. A gap could be queued by analyze and then
                        # sit pending for up to 4 more days before fill ran (or, with
                        # the new 300s cycle interval, fill could pick an empty queue
                        # and burn its 96h cooldown doing nothing while a real gap
                        # waited). Chaining directly ensures the gap gets acted on
                        # while context is fresh and pairs the work logically.
                        #
                        # _fill_knowledge_gaps does not self-update its timestamp
                        # (relies on the dispatcher's try/finally per Issue 7), so we
                        # update it explicitly here since we are calling it outside
                        # the dispatcher. Wrapped in try/except so a failure in fill
                        # does NOT cause analyze to return False (analyze itself
                        # already succeeded in queueing the gap).
                        try:
                            logging.info("Chaining to _fill_knowledge_gaps after successful analysis")
                            print("🔗 Chaining: invoking _fill_knowledge_gaps to act on newly queued gap")
                            self._fill_knowledge_gaps()
                            self._update_activity_timestamp('fill_knowledge_gaps')
                            logging.info("✅ Chain complete: _fill_knowledge_gaps executed and timestamp updated")
                        except Exception as chain_err:
                            # Non-fatal: log prominently but do not fail the analyze step.
                            # The gap is queued; fill can retry on the next eligible cycle.
                            logging.error(
                                f"Chained _fill_knowledge_gaps raised an exception: {chain_err}. "
                                f"Gap remains queued for the dispatcher to retry.",
                                exc_info=True
                            )

                        return True
                    else:
                        print("⚠️ Gap processing completed with some issues")
                        logging.warning("Gap processing completed with some issues")
                        return False
                        
            except Exception as e:
                print(f"❌ Error in enhanced LLM knowledge gap analysis: {e}")
                logging.error(f"Error in enhanced LLM knowledge gap analysis: {e}", exc_info=True)
                self._record_thought(
                    thought_type="knowledge_analysis",
                    content=f"Error in enhanced analysis: {str(e)}"
                )
                return False
                        
        except Exception as e:
            print(f"❌ Error in enhanced knowledge gap analysis: {e}")
            logging.error(f"Error in enhanced knowledge gap analysis: {e}", exc_info=True)
            self._record_thought(
                thought_type="error",
                content=f"Error during enhanced knowledge gap analysis: {str(e)}"
            )
            return False
        finally:
            # Update activity timestamp
            self._update_activity_timestamp('analyze_knowledge_gaps')
            self.cognitive_state = "idle"
            print("🔄 Completed enhanced knowledge gap analysis (SINGLE-GAP FOCUS)")
            logging.info("Completed enhanced knowledge gap analysis (SINGLE-GAP FOCUS)")
    
    def _process_classified_gaps(self, gaps_analysis):
        """
        Process gaps based on their classification with comprehensive logging and validation.
        Enhanced with strict duplicate checking and conversation history validation.
        
        C4 FIX: add_gap() returns -1 for semantic duplicates (expected/normal) and -2 for
        actual errors. Previously both were treated as errors, causing false error counts
        that could push the success rate below threshold and return False even when
        everything worked correctly. Now -1 and -2 are handled separately.
        
        Args:
            gaps_analysis (dict or str): Either a dictionary with 'gaps' key containing
                                         parsed gaps, or a raw string to be parsed.
            
        Returns:
            bool: True if processing succeeded (including all-duplicates case), False on failure.
        """
        logging.info("📋 Starting classification-based gap processing with enhanced validation")
        
        try:
            from knowledge_gap import KnowledgeGapQueue
            gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
            
            # Handle both dictionary input (pre-parsed) and raw string input
            if isinstance(gaps_analysis, dict):
                parsed_gaps = gaps_analysis.get('gaps', [])
                logging.info(f"🔍 Extracted {len(parsed_gaps)} gaps from dictionary structure")
            elif isinstance(gaps_analysis, str):
                logging.info("🔍 Parsing knowledge gaps from LLM analysis string")
                parsed_gaps = self._parse_knowledge_gaps_response(gaps_analysis)
            else:
                logging.error(f"❌ Unexpected gaps_analysis type: {type(gaps_analysis)}")
                return False
            
            if not parsed_gaps:
                logging.warning("⚠️ No gaps could be extracted from input")
                return False
            
            logging.info(f"✅ Successfully extracted {len(parsed_gaps)} knowledge gaps")
            
            # Load existing pending gaps AND historical gaps for duplicate validation
            # Historical gaps (fulfilled/failed) are critical because mark_fulfilled()
            # removes Qdrant embeddings, making them invisible to semantic checks
            try:
                existing_gaps = gap_queue.get_gaps_by_status('pending')
                historical_gaps = self._get_historical_gaps(days=90, limit=50)
                all_known_gaps = existing_gaps + historical_gaps
                logging.info(f"📚 Loaded {len(all_known_gaps)} gaps for validation "
                            f"({len(existing_gaps)} pending + {len(historical_gaps)} historical)")
            except Exception as e:
                logging.warning(f"Could not load gaps for validation: {e}")
                all_known_gaps = []
            
            # Get recent conversation context for answered-topic validation
            recent_conversation_text = self._get_recent_conversation_text()
            logging.info(f"📖 Loaded {len(recent_conversation_text)} chars of conversation context for validation")
            
            # Validate each proposed gap for uniqueness and whether it was already discussed
            validated_gaps = []
            rejected_gaps = []
            
            logging.info(f"🔍 Starting validation of {len(parsed_gaps)} proposed gaps...")
            
            for i, gap in enumerate(parsed_gaps, 1):
                topic = gap.get('topic', 'Unknown')
                description = gap.get('description', 'No description')
                
                logging.info(f"🔎 Validating gap {i}/{len(parsed_gaps)}: {topic}")
                
                is_valid, reason = self._validate_gap_uniqueness(
                    gap,
                    all_known_gaps,
                    recent_conversation_text
                )
                
                if is_valid:
                    validated_gaps.append(gap)
                    logging.info(f"   ✅ VALIDATED: {topic}")
                    logging.debug(f"      Reason: {reason}")
                else:
                    rejected_gaps.append((gap, reason))
                    logging.warning(f"   ❌ REJECTED: {topic}")
                    logging.warning(f"      Reason: {reason}")
            
            # Log validation summary
            logging.info(f"📊 VALIDATION SUMMARY:")
            logging.info(f"   ✅ Validated gaps: {len(validated_gaps)}")
            logging.info(f"   ❌ Rejected gaps: {len(rejected_gaps)}")
            if len(parsed_gaps) > 0:
                logging.info(f"   📈 Validation rate: {(len(validated_gaps)/len(parsed_gaps)*100):.1f}%")
            
            if rejected_gaps:
                logging.info("🚫 REJECTED GAPS DETAILS:")
                for gap, reason in rejected_gaps:
                    logging.info(f"   • {gap.get('topic')}: {reason}")
            
            # All-duplicates is a success — we prevented noise in the queue
            if not validated_gaps:
                logging.warning("⚠️ No gaps passed validation — all were duplicates or already discussed")
                return True
            
            logging.info(f"▶️  Processing {len(validated_gaps)} validated gaps...")
            
            # Counters for final summary
            personal_reminders_created = 0
            system_internal_queued = 0
            public_searchable_queued = 0
            semantic_duplicates_skipped = 0  # C4 FIX: track -1 returns separately
            processing_errors = 0
            
            # Process each validated gap according to its classification
            for i, gap in enumerate(validated_gaps, 1):
                topic = gap.get('topic', '').strip()
                description = gap.get('description', '').strip()
                classification = gap.get('classification', 'PUBLIC_SEARCHABLE').strip().upper()
                priority = gap.get('priority', 'MEDIUM').strip().upper()
                
                logging.info(f"🔄 Processing validated gap {i}/{len(validated_gaps)}: {topic}")
                logging.debug(f"   Classification: {classification}")
                logging.debug(f"   Priority: {priority}")
                logging.debug(f"   Description: {description[:100]}...")
                
                # Validate required fields before processing
                if not topic or not description:
                    logging.warning(f"⚠️ Gap {i} missing required fields — Topic: '{topic}', Description: '{description}'")
                    processing_errors += 1
                    continue
                
                # Convert text priority to numeric value
                priority_map = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
                priority_value = priority_map.get(priority, 0.6)
                
                try:
                    if classification == 'PERSONAL_ASK_KEN':
                        logging.info(f"👤 Processing personal knowledge gap: {topic}")
                        
                        # STEP 1: Queue the gap to get a gap_id before creating reminder
                        gap_id = gap_queue.add_gap(topic, description, priority_value)
                        
                        # C4 FIX: -1 = semantic duplicate (expected, not an error)
                        #          -2 = actual storage failure
                        if gap_id == -1:
                            semantic_duplicates_skipped += 1
                            logging.info(f"   ✅ Skipped personal gap — semantic duplicate already queued: {topic}")
                            continue
                        elif gap_id == -2:
                            logging.error(f"   ❌ Failed to create knowledge gap entry for: {topic}")
                            processing_errors += 1
                            continue
                        
                        logging.info(f"   ✅ Created knowledge gap entry with ID {gap_id}")
                        
                        # STEP 2: Create reminder for Ken to address this gap directly
                        due_date = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                        reminder_text = f"Clarify knowledge gap: {topic} - {description}"
                        
                        reminder_success = self._create_reminder_for_personal_gap(reminder_text, due_date)
                        
                        if reminder_success:
                            personal_reminders_created += 1
                            logging.info(f"   ✅ Successfully created reminder for: {topic}")
                            
                            # STEP 3: The reminder IS the fulfillment for personal gaps —
                            # mark immediately so the gap does not get retried by the filler
                            try:
                                gap_queue.mark_fulfilled(gap_id)
                                logging.info(f"   🔗 Marked gap {gap_id} as fulfilled (reminder created)")
                                logging.info(f"   📋 Ken will address this via the reminder system")
                            except Exception as mark_error:
                                logging.error(f"   ⚠️ Failed to mark gap {gap_id} as fulfilled: {mark_error}")
                                # Reminder was still created successfully — don't fail the whole process
                        else:
                            logging.error(f"   ❌ Failed to create reminder for: {topic}")
                            # Gap created but reminder failed — leave as 'pending' for retry
                            logging.warning(f"   ⚠️ Gap {gap_id} remains 'pending' since reminder creation failed")
                            processing_errors += 1
                    
                    elif classification == 'SYSTEM_INTERNAL':
                        logging.info(f"🧠 Queueing system internal gap: {topic}")
                        
                        # Prefix to distinguish self-reflection gaps in the queue
                        internal_topic = f"SELF_REFLECTION: {topic}"
                        gap_id = gap_queue.add_gap(internal_topic, description, priority_value)
                        
                        # C4 FIX: -1 = semantic duplicate (expected), -2 = error
                        if gap_id > 0:
                            system_internal_queued += 1
                            logging.info(f"   ✅ Queued system internal gap with ID {gap_id}: {topic}")
                        elif gap_id == -1:
                            semantic_duplicates_skipped += 1
                            logging.info(f"   ✅ Skipped system internal gap — semantic duplicate: {topic}")
                        else:
                            logging.error(f"   ❌ Failed to queue system internal gap: {topic}")
                            processing_errors += 1
                    
                    elif classification in ['PUBLIC_SEARCHABLE', 'FACTUAL_GENERAL']:
                        logging.info(f"🌐 Queueing public searchable gap: {topic}")
                        
                        gap_id = gap_queue.add_gap(topic, description, priority_value)
                        
                        # C4 FIX: -1 = semantic duplicate (expected), -2 = error
                        if gap_id > 0:
                            public_searchable_queued += 1
                            logging.info(f"   ✅ Queued public searchable gap with ID {gap_id}: {topic}")
                        elif gap_id == -1:
                            semantic_duplicates_skipped += 1
                            logging.info(f"   ✅ Skipped public gap — semantic duplicate: {topic}")
                        else:
                            logging.error(f"   ❌ Failed to queue public searchable gap: {topic}")
                            processing_errors += 1
                    
                    else:
                        # Unknown classification — default to PUBLIC_SEARCHABLE
                        logging.warning(f"⚠️ Unknown classification '{classification}' for gap: {topic}")
                        logging.info(f"   🔄 Defaulting to PUBLIC_SEARCHABLE for: {topic}")
                        
                        gap_id = gap_queue.add_gap(topic, description, priority_value)
                        
                        # C4 FIX: -1 = semantic duplicate (expected), -2 = error
                        if gap_id > 0:
                            public_searchable_queued += 1
                            logging.info(f"   ✅ Queued gap (default classification) with ID {gap_id}: {topic}")
                        elif gap_id == -1:
                            semantic_duplicates_skipped += 1
                            logging.info(f"   ✅ Skipped gap (default) — semantic duplicate: {topic}")
                        else:
                            logging.error(f"   ❌ Failed to queue gap (default classification): {topic}")
                            processing_errors += 1
                            
                except Exception as gap_error:
                    logging.error(f"❌ Error processing individual gap '{topic}': {gap_error}", exc_info=True)
                    processing_errors += 1
                    continue
            
            # Final summary
            total_processed = personal_reminders_created + system_internal_queued + public_searchable_queued
            
            logging.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logging.info("📊 FINAL GAP PROCESSING SUMMARY:")
            logging.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logging.info(f"   📥 Gaps proposed by LLM: {len(parsed_gaps)}")
            logging.info(f"   ✅ Gaps passed validation: {len(validated_gaps)}")
            logging.info(f"   ❌ Gaps rejected (duplicates/discussed): {len(rejected_gaps)}")
            logging.info(f"   👤 Personal reminders created: {personal_reminders_created}")
            logging.info(f"   🧠 System internal gaps queued: {system_internal_queued}")
            logging.info(f"   🌐 Public searchable gaps queued: {public_searchable_queued}")
            logging.info(f"   🔁 Semantic duplicates skipped (normal): {semantic_duplicates_skipped}")
            logging.info(f"   ❌ Processing errors: {processing_errors}")
            logging.info(f"   ✅ Total successfully processed: {total_processed}/{len(validated_gaps)}")
            
            if len(parsed_gaps) > 0:
                validation_rate = (len(validated_gaps) / len(parsed_gaps)) * 100
                logging.info(f"   📈 Validation pass rate: {validation_rate:.1f}%")
            
            if len(validated_gaps) > 0:
                processing_rate = (total_processed / len(validated_gaps)) * 100
                logging.info(f"   📈 Processing success rate: {processing_rate:.1f}%")
            
            logging.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # Evaluate success:
            # - semantic_duplicates_skipped are NOT failures — they count toward success
            # - only processing_errors count against us
            if len(validated_gaps) == 0:
                logging.warning("⚠️ Gap processing completed but all gaps were rejected as duplicates")
                return True  # Duplicate prevention working correctly = success
            
            # C4 FIX: success denominator excludes semantic duplicates (they are expected outcomes)
            actionable_gaps = len(validated_gaps) - semantic_duplicates_skipped
            if actionable_gaps <= 0:
                # All validated gaps were semantic duplicates — still a success
                logging.info("✅ All validated gaps were semantic duplicates — queue is clean")
                return True
            
            success_rate = total_processed / actionable_gaps
            
            if success_rate >= 0.8:
                logging.info(f"✅ Gap processing completed successfully (success rate: {success_rate:.1%})")
                return True
            elif success_rate >= 0.5:
                logging.warning(f"⚠️ Gap processing completed with warnings (success rate: {success_rate:.1%})")
                return True
            else:
                logging.error(f"❌ Gap processing failed (success rate: {success_rate:.1%})")
                return False
            
        except Exception as e:
            logging.error(f"❌ Critical error in gap processing: {e}", exc_info=True)
            return False
        
    def _create_reminder_for_personal_gap(self, reminder_text, due_date):
        """
        Create a reminder using the proper REMINDER command syntax.
        Enhanced with duplicate prevention - checks if topic was recently discussed.
        
        Args:
            reminder_text (str): The reminder content
            due_date (str): Due date in YYYY-MM-DD format
            
        Returns:
            bool: Success status
        """
        try:
            logging.debug(f"Creating reminder: {reminder_text[:50]}... due {due_date}")
            
            # NEW: Quick check if this was recently discussed
            topic_keywords = set(word for word in reminder_text.lower().split() if len(word) > 4)
            
            if hasattr(self.chatbot, 'current_conversation') and self.chatbot.current_conversation:
                recent_text = ' '.join([
                    msg.get('content', '') for msg in self.chatbot.current_conversation[-10:]
                ]).lower()
                
                # If 60%+ of topic keywords appear in recent conversation, skip reminder
                if topic_keywords:
                    words_found = sum(1 for word in topic_keywords if word in recent_text)
                    overlap_ratio = words_found / len(topic_keywords)
                    
                    if overlap_ratio > 0.6:
                        logging.info(f"✓ Skipping reminder - topic appears answered in recent conversation")
                        logging.info(f"  Topic: {reminder_text[:80]}...")
                        logging.info(f"  Overlap: {overlap_ratio:.1%} of keywords found in recent messages")
                        # Return True because we successfully prevented a duplicate
                        return True
            
            # Try using the existing reminder system first
            if hasattr(self.chatbot, 'deepseek_enhancer') and hasattr(self.chatbot.deepseek_enhancer, '_handle_reminder_command'):
                try:
                    # Format as the reminder system expects: "content | due=YYYY-MM-DD"
                    reminder_command_text = f"{reminder_text} | due={due_date}"
                    result, success = self.chatbot.deepseek_enhancer._handle_reminder_command(reminder_command_text)
                    
                    if success:
                        logging.debug(f"✅ Reminder created via reminder system: {result}")
                        return True
                    else:
                        logging.warning(f"⚠️ Reminder system failed: {result}")
                        
                except Exception as reminder_error:
                    logging.warning(f"⚠️ Error using reminder system: {reminder_error}")
            
            # Fallback: store as a special memory type
            logging.info("🔄 Using fallback memory storage for reminder")
            
            metadata = {
                "type": "knowledge_gap_reminder",
                "reminder_text": reminder_text,
                "due_date": due_date,
                "created_by": "autonomous_cognition",
                "created_at": datetime.datetime.now().isoformat(),
                "tags": "knowledge_gap,personal,reminder,autonomous"
            }
            
            full_content = f"REMINDER: {reminder_text}\nDue: {due_date}\n\nThis reminder was created by autonomous gap analysis for personal information that requires Ken's input."
            
            # Use transaction coordination for consistent storage
            success, memory_id = self.chatbot.store_memory_with_transaction(
                content=full_content,
                memory_type="knowledge_gap_reminder",
                metadata=metadata,
                confidence=0.8
            )
            
            if success:
                logging.info(f"✅ Created reminder as memory with ID {memory_id}")
                return True
            else:
                logging.error(f"❌ Failed to store reminder as memory")
                return False
            
        except Exception as e:
            logging.error(f"❌ Error creating reminder: {e}", exc_info=True)
            return False        
    
    def _format_existing_gaps_for_prompt(self, existing_gaps):
        """
        Format existing gaps for the LLM prompt with enhanced detail to prevent duplicates.
        
        Now includes status labels (pending/fulfilled/failed) so the LLM can see
        that a topic was already investigated and resolved — not just pending.
        
        Updated 2026-04-01: Added status field, updated empty message, increased limit to 35.
        """
        if not existing_gaps:
            return "No existing gaps found"
        
        # Limit to most recent 35 gaps (increased to accommodate historical gaps)
        recent_gaps = existing_gaps[-35:] if len(existing_gaps) > 35 else existing_gaps
        
        formatted = []
        for i, gap in enumerate(recent_gaps, 1):
            topic = gap.get('topic', 'Unknown')
            description = gap.get('description', 'No description')
            status = gap.get('status', 'pending')
            
            # Extract key terms from topic and description for better matching
            combined_text = f"{topic} {description}".lower()
            key_terms = set(word for word in combined_text.split() if len(word) > 4)
            
            # Format with status label so LLM sees fulfilled/failed gaps too
            formatted.append(
                f"{i}. TOPIC: {topic}\n"
                f"   DESCRIPTION: {description[:100]}...\n"
                f"   STATUS: {status}\n"
                f"   KEY_TERMS: {', '.join(list(key_terms)[:5])}"
            )
        
        return "\n\n".join(formatted)
    
    def _get_recent_conversation_text(self, limit=20):
        """
        Build a formatted text block of recent conversation turns for use in
        gap analysis prompts and duplicate validation checks.

        Extracts the last `limit` messages from the active conversation, formats
        each as "[role]: content..." and joins them with newlines. Returns an
        empty string if no conversation is available — callers and prompts handle
        absence more cleanly with nothing than with a placeholder string.

        Args:
            limit (int): Maximum number of recent messages to include. Default 20.

        Returns:
            str: Formatted conversation text, or "" if no conversation is active.
        """
        try:
            if not hasattr(self.chatbot, 'current_conversation') or not self.chatbot.current_conversation:
                logging.debug("_get_recent_conversation_text: No active conversation found — returning empty string")
                return ""

            # Slice to the most recent `limit` messages
            recent_messages = self.chatbot.current_conversation[-limit:]
            conversation_excerpts = []

            for msg in recent_messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if content:
                    # Only append ellipsis if content was actually truncated
                    truncated = content[:200]
                    suffix = "..." if len(content) > 200 else ""
                    conversation_excerpts.append(f"[{role}]: {truncated}{suffix}")

            result = "\n".join(conversation_excerpts)
            logging.debug(f"_get_recent_conversation_text: Built {len(result)} chars from {len(recent_messages)} messages")
            return result

        except Exception as e:
            logging.warning(f"_get_recent_conversation_text: Failed to build conversation text: {e}", exc_info=True)
            return ""
    
    def _validate_gap_uniqueness(self, proposed_gap, existing_gaps, recent_conversation_text):
        """
        Validate that a proposed gap is truly unique and not already discussed.
        
        Args:
            proposed_gap (dict): The gap to validate with 'topic' and 'description' keys
            existing_gaps (list): List of existing gap dictionaries
            recent_conversation_text (str): Recent conversation history
            
        Returns:
            tuple: (is_valid, reason) where is_valid is bool and reason explains why
        """
        try:
            topic = proposed_gap.get('topic', '').lower()
            description = proposed_gap.get('description', '').lower()
            
            if not topic or not description:
                return False, "Gap missing required fields"
            
            # Check 1: Was this discussed in recent conversations?
            topic_words = set(word for word in topic.split() if len(word) > 3)
            conversation_lower = recent_conversation_text.lower()
            
            # If more than 60% of topic words appear in recent conversation
            if topic_words:
                words_found = sum(1 for word in topic_words if word in conversation_lower)
                match_ratio = words_found / len(topic_words)
                
                if match_ratio > 0.6:
                    return False, f"Topic '{topic}' appears to have been discussed in recent conversations ({match_ratio:.1%} word match)"
            
            # Check 2: Compare against existing gaps using enhanced similarity
            for existing_gap in existing_gaps:
                existing_topic = existing_gap.get('topic', '').lower()
                existing_desc = existing_gap.get('description', '').lower()
                
                # Skip empty existing gaps
                if not existing_topic:
                    continue
                
                # Topic similarity check
                topic_similarity = self._calculate_text_similarity(topic, existing_topic)
                if topic_similarity > 0.6:  # Using your lowered threshold
                    return False, f"Too similar to existing gap: '{existing_topic}' (similarity: {topic_similarity:.2f})"
                
                # Description similarity check
                desc_similarity = self._calculate_text_similarity(description, existing_desc)
                if desc_similarity > 0.5:  # Lower threshold for descriptions
                    return False, f"Description too similar to existing gap about '{existing_topic}' (similarity: {desc_similarity:.2f})"
                
                # Keyword overlap check
                topic_keywords = set(word for word in topic.split() if len(word) > 4)
                existing_keywords = set(word for word in existing_topic.split() if len(word) > 4)
                
                if topic_keywords and existing_keywords:
                    overlap = len(topic_keywords.intersection(existing_keywords))
                    total = len(topic_keywords.union(existing_keywords))
                    overlap_ratio = overlap / total if total > 0 else 0
                    
                    if overlap_ratio > 0.5:  # 50% keyword overlap
                        overlapping_words = topic_keywords.intersection(existing_keywords)
                        return False, f"High keyword overlap ({overlap_ratio:.1%}) with existing gap: '{existing_topic}' (shared: {', '.join(list(overlapping_words)[:3])})"
            
            # Check 3: Validate within the current batch (avoid duplicates in same analysis)
            # This is handled in the calling code, but we can add a note
            logging.debug(f"   Gap '{topic}' passed all uniqueness checks")
            
            # If we get here, gap is valid
            return True, "Gap is unique and not previously discussed"
            
        except Exception as e:
            logging.error(f"Error validating gap uniqueness: {e}", exc_info=True)
            return False, f"Validation error: {str(e)}"

    def _check_for_similar_gaps(self, new_topic, new_description, existing_gaps):
        """
        Enhanced similarity checking for gaps using multiple similarity metrics.
        Uses a two-stage approach:
        1. Fast Jaccard text similarity (catches obvious duplicates)
        2. Semantic vector similarity via Qdrant (catches subtle duplicates)
        
        Args:
            new_topic (str): Topic of the new gap
            new_description (str): Description of the new gap  
            existing_gaps (list): List of existing gap dictionaries
            
        Returns:
            bool: True if a similar gap exists, False otherwise
        """
        if not new_topic:
            return False
            
        try:
            new_topic_lower = new_topic.lower()
            new_desc_lower = (new_description or '').lower()
            
            # =====================================================
            # STAGE 1: Fast Jaccard text similarity (pre-filter)
            # Catches obvious duplicates without API calls
            # =====================================================
            
            if existing_gaps:
                for existing_gap in existing_gaps:
                    existing_topic = existing_gap.get('topic', '').lower()
                    existing_desc = existing_gap.get('description', '').lower()
                    
                    # Check topic similarity - high threshold for topics
                    topic_similarity = self._calculate_text_similarity(new_topic_lower, existing_topic)
                    if topic_similarity > 0.6:  # 60% similarity threshold for topics
                        logging.info(f"🔍 Jaccard duplicate detected: '{new_topic}' vs '{existing_topic}' "
                                   f"(similarity: {topic_similarity:.2%})")
                        return True
                    
                    # Check description similarity - lower threshold
                    desc_similarity = self._calculate_text_similarity(new_desc_lower, existing_desc)
                    if desc_similarity > 0.7:  # 70% similarity threshold for descriptions
                        logging.info(f"🔍 Jaccard description duplicate detected (similarity: {desc_similarity:.2%})")
                        return True
                    
                    # Check for keyword overlap in topics
                    new_keywords = set(new_topic_lower.split())
                    existing_keywords = set(existing_topic.split())
                    if new_keywords and existing_keywords:
                        keyword_overlap = len(new_keywords.intersection(existing_keywords)) / len(new_keywords.union(existing_keywords))
                        if keyword_overlap > 0.6:  # 60% keyword overlap
                            logging.info(f"🔍 Jaccard keyword overlap detected: {keyword_overlap:.2%}")
                            return True
            
            # =====================================================
            # STAGE 2: Semantic vector similarity via Qdrant
            # Catches conceptually similar gaps with different wording
            # =====================================================
            
            try:
                from knowledge_gap import KnowledgeGapQueue
                
                # Initialize gap queue with Qdrant connection
                gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
                
                # Check semantic similarity using vector embeddings
                is_semantic_duplicate, similar_info = gap_queue.check_semantic_similarity(
                    new_topic, 
                    new_description or ''
                )
                
                if is_semantic_duplicate:
                    logging.info(f"🧠 Semantic duplicate detected: '{new_topic}' similar to "
                               f"'{similar_info.get('topic', 'Unknown')}' "
                               f"(similarity: {similar_info.get('score', 0):.2%})")
                    return True
                    
            except ImportError as e:
                logging.warning(f"Could not import KnowledgeGapQueue for semantic check: {e}")
            except Exception as e:
                logging.warning(f"Semantic similarity check failed (continuing with Jaccard only): {e}")
            
            # No duplicates found by either method
            logging.debug(f"✅ No duplicates found for '{new_topic}'")
            return False
            
        except Exception as e:
            logging.error(f"Error checking gap similarity: {e}")
            return False

    def _gaps_are_similar(self, gap1, gap2):
        """Check if two gaps from the current batch are similar."""
        try:
            topic1 = gap1.get('topic', '').lower()
            topic2 = gap2.get('topic', '').lower()
            desc1 = gap1.get('description', '').lower()
            desc2 = gap2.get('description', '').lower()
            
            # Check topic similarity
            topic_sim = self._calculate_text_similarity(topic1, topic2)
            desc_sim = self._calculate_text_similarity(desc1, desc2)
            
            return topic_sim > 0.8 or desc_sim > 0.7
            
        except Exception as e:
            logging.error(f"Error comparing gaps: {e}")
            return False
        
    def _check_memory_coverage(self, topic: str, description: str) -> tuple:
        """
        Check whether QWEN already has meaningful memory coverage on a topic
        before allowing it to be flagged as a knowledge gap.

        Uses two vector searches (topic + description) against the main memory
        collection, deduplicates results by content hash, filters by relevance
        score, and returns whether coverage exists based on unique result count.

        COVERAGE THRESHOLD: 1 or more unique relevant memory hits = topic already known.
        We only want gaps where there is truly ZERO meaningful information.

        MIN_RELEVANCE_SCORE: 0.65 — filters out noise/tangential vector hits.
        With 4096-dim embeddings, true matches score 0.70-0.95+. The 0.65 floor
        catches near-misses while excluding irrelevant results that would
        incorrectly suppress legitimate gaps.

        Updated 2026-04-01: Threshold lowered from 2→1, relevance score filter added.

        Args:
            topic (str): The gap topic string from LLM analysis
            description (str): The gap description string from LLM analysis

        Returns:
            tuple: (is_covered: bool, unique_hit_count: int)
                - (True, N)  → topic is covered, suppress the gap
                - (False, N) → genuine gap, allow it through
        """
        # ANY genuine memory coverage = not a real gap (changed from 2 → 1)
        COVERAGE_THRESHOLD = 1

        # Minimum similarity score to count as a genuine hit (not noise)
        # With 4096-dim embeddings, true matches score 0.70-0.95+
        # 0.65 catches near-misses while filtering noise
        MIN_RELEVANCE_SCORE = 0.65

        try:
            if not self.vector_db:
                # If vector_db is unavailable, fail open — allow the gap through
                logging.warning("[COVERAGE CHECK] vector_db not available, skipping coverage check")
                return False, 0

            combined_results = []

            # --- Search 1: Use topic as query ---
            # Catches memories whose content is semantically aligned with the topic label
            try:
                topic_results = self.vector_db.search(
                    query=topic,
                    mode="default",
                    k=5
                )
                if topic_results:
                    combined_results.extend(topic_results)
                    logging.debug(f"[COVERAGE CHECK] Topic search '{topic}' returned {len(topic_results)} results")
            except Exception as e:
                logging.warning(f"[COVERAGE CHECK] Topic search failed for '{topic}': {e}")

            # --- Search 2: Use description as query ---
            # Catches memories that match the *intent* of the gap even if topic
            # phrasing differs from stored memory content
            if description:
                try:
                    # Truncate description to avoid overly long embedding inputs
                    desc_query = description[:300]
                    desc_results = self.vector_db.search(
                        query=desc_query,
                        mode="default",
                        k=5
                    )
                    if desc_results:
                        combined_results.extend(desc_results)
                        logging.debug(f"[COVERAGE CHECK] Description search returned {len(desc_results)} results")
                except Exception as e:
                    logging.warning(f"[COVERAGE CHECK] Description search failed for '{topic}': {e}")

            # --- Deduplicate by content hash, filtered by relevance score ---
            # Only count memories that are genuinely relevant (score >= MIN_RELEVANCE_SCORE)
            # This prevents weak/tangential vector hits from suppressing legitimate gaps.
            # Dedup prevents a single memory appearing in both searches from inflating the count.
            unique_memories = {}
            filtered_out_count = 0
            for memory in combined_results:
                content = memory.get('content', '')
                score = memory.get('similarity_score', 0)

                # Filter: must have meaningful content AND meet relevance threshold
                if content and len(content) > 20 and score >= MIN_RELEVANCE_SCORE:
                    content_hash = hash(content)
                    unique_memories[content_hash] = memory
                elif content and len(content) > 20:
                    # Content exists but below relevance threshold — track for logging
                    filtered_out_count += 1

            unique_count = len(unique_memories)

            if filtered_out_count > 0:
                logging.debug(
                    f"[COVERAGE CHECK] Filtered out {filtered_out_count} results below "
                    f"relevance threshold ({MIN_RELEVANCE_SCORE})"
                )

            # --- Coverage decision ---
            is_covered = unique_count >= COVERAGE_THRESHOLD

            if is_covered:
                logging.info(
                    f"[COVERAGE CHECK] ✅ Topic '{topic}' is COVERED — "
                    f"found {unique_count} unique memory hits above {MIN_RELEVANCE_SCORE} relevance "
                    f"(threshold: {COVERAGE_THRESHOLD}). Suppressing gap."
                )
            else:
                logging.info(
                    f"[COVERAGE CHECK] 🔍 Topic '{topic}' has LOW COVERAGE — "
                    f"found {unique_count} unique memory hits above {MIN_RELEVANCE_SCORE} relevance "
                    f"(threshold: {COVERAGE_THRESHOLD}). Allowing gap through."
                )

            return is_covered, unique_count

        except Exception as e:
            # Fail open — if the check itself errors, allow the gap through
            # rather than silently suppressing potentially valid gaps
            logging.error(f"[COVERAGE CHECK] Error checking memory coverage for '{topic}': {e}", exc_info=True)
            return False, 0

    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings using Jaccard similarity.
        
        Args:
            text1 (str): First text string
            text2 (str): Second text string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
            
        try:
            # Convert to sets of words
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return 0.0
                
            return intersection / union
            
        except Exception as e:
            logging.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    def _get_recent_queries(self):
        """Get recent user queries for knowledge gap analysis."""
        try:
            # Get recent conversation
            logging.info("Attempting to retrieve recent user queries")
            if hasattr(self.chatbot, 'current_conversation'):
                # Extract user messages from conversation
                logging.info(f"Found conversation with {len(self.chatbot.current_conversation)} messages")
                user_messages = [msg['content'] for msg in self.chatbot.current_conversation 
                            if msg.get('role') == 'user']
                
                logging.info(f"Extracted {len(user_messages)} user messages")
                
                # Return the last 5 messages or fewer if not available
                recent_queries = "\n".join(user_messages[-5:])
                return recent_queries
            else:
                logging.warning("No current_conversation attribute found in chatbot")
                return ""
        except Exception as e:
            logging.error(f"Error getting recent queries: {e}", exc_info=True)
            return ""
        
    def _get_relevant_memories_for_analysis(self):
        """Get relevant memories for knowledge gap analysis."""
        try:
            # Use self.memory_db.get_recent_memories() instead of self._get_memories_by_recency
            logging.info("Attempting to retrieve relevant memories")
            try:
                # Route through the wrapper, which calls
                # MemoryDB.get_memories_since() internally. The wrapper handles
                # exceptions and returns [] on failure.
                memories = self._get_memories_by_recency(limit=20)
                logging.info(f"Successfully retrieved {len(memories)} memories")
            except Exception as e:
                logging.error(f"Error retrieving recent memories: {e}")
                memories = []
            
            # If memories are already formatted strings, just join them
            if memories and isinstance(memories[0], str):
                return "\n\n".join(memories)
                
            # Otherwise, format them (unlikely to reach this part with your implementation)
            memory_texts = []
            for memory in memories:
                if isinstance(memory, dict):
                    content = memory.get('content', '')
                    memory_type = memory.get('memory_type', memory.get('metadata', {}).get('type', 'unknown'))
                    memory_texts.append(f"[{memory_type}] {content}")
                elif isinstance(memory, str):
                    memory_texts.append(memory)
            
            return "\n\n".join(memory_texts)
        except Exception as e:
            logging.error(f"Error getting relevant memories: {e}")
            return ""
    
    
    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Memory type distribution formatter — unwired.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__get_memory_type_distribution(self):
        """Get distribution of memory types as a formatted string."""
        try:
            # Execute SQL to get memory type distribution
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT memory_type, COUNT(*) as count
                    FROM memories
                    GROUP BY memory_type
                    ORDER BY count DESC
                """)
                distribution = cursor.fetchall()
            
            if not distribution:
                return "No memories found."
            
            # Format as readable text
            result = "Memory type distribution:\n"
            for memory_type, count in distribution:
                result += f"- {memory_type}: {count} memories\n"
            
            return result
        
        except Exception as e:
            logging.error(f"Error getting memory type distribution: {e}", exc_info=True)
            return "Error retrieving memory distribution."
        
    def _get_memories_by_recency(self, limit=30, days=14):
        """
        Get the most recent memories for analysis.

        Routes through MemoryDB.get_memories_since() — the actual recency-based
        retrieval method on MemoryDB. Returns the `limit` most recent memories
        from the past `days` days, ordered newest-first.

        Args:
            limit (int): Maximum number of memories to return (default 30)
            days (int): Recency window in days (default 14). 2-week window
                gives enough context for knowledge-gap analysis without
                pulling in ancient memories during quiet periods.

        Returns:
            list: List of dicts with content, memory_type, weight, created_at,
                source, confidence keys. Empty list on error or no results.
        """
        try:
            # PREVIOUSLY BROKEN: This called self.memory_db.get_memories_by_recency()
            # which does not exist on MemoryDB. The actual method is
            # get_memories_since(days, limit) at line ~993 in memory_db.py.
            return self.memory_db.get_memories_since(days=days, limit=limit)
        except Exception as e:
            logging.error(f"Error getting recent memories: {e}")
            return []

    def analyze_memory_health(self):
        """Analyze the health of memory storage and database.
        
        Checks for:
        - Duplicate content in syncable memories (excluding reminders and autonomous_thoughts)
        - Memories with NULL or empty content
        - Old, low-weight, rarely accessed memories that could be archived
        
        Returns:
            dict: Structured health analysis with keys:
                - status: 'healthy', 'issues_found', or 'error'
                - issues: List of identified issue descriptions
                - recommendations: List of recommended actions
                - duplicate_details: List of dicts with info about duplicate groups (if found)
        """
        try:
            health = {
                "status": "healthy",
                "issues": [],
                "recommendations": []
            }
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # --- Check 1: Duplicate content in syncable memories ---
                # Excludes autonomous_thoughts (process logs that may legitimately repeat)
                # Excludes reminders (SQLite-only, not synced to vector DB)
                cursor.execute("""
                    SELECT 
                        content, 
                        COUNT(*) as count,
                        GROUP_CONCAT(id) as memory_ids,
                        GROUP_CONCAT(memory_type) as memory_types,
                        MIN(created_at) as first_created,
                        MAX(created_at) as last_created
                    FROM memories
                    WHERE memory_type NOT IN ('reminder', 'autonomous_thought')
                    GROUP BY content
                    HAVING count > 1
                    LIMIT 10
                """)
                duplicates = cursor.fetchall()
                
                if duplicates:
                    health["status"] = "issues_found"
                    
                    # Calculate total redundant entries (each group with N copies has N-1 redundant)
                    total_redundant = sum(count - 1 for _, count, _, _, _, _ in duplicates)
                    duplicate_groups = len(duplicates)
                    
                    health["issues"].append(
                        f"Found {duplicate_groups} duplicate content group(s) "
                        f"({total_redundant} redundant entries total)"
                    )
                    
                    # Store detailed info for review in admin UI
                    health["duplicate_details"] = []
                    for content, count, ids, types, first_created, last_created in duplicates[:5]:
                        health["duplicate_details"].append({
                            "content_preview": content[:150] + "..." if len(content) > 150 else content,
                            "count": count,
                            "memory_ids": ids,
                            "memory_types": types,
                            "first_created": first_created,
                            "last_created": last_created
                        })
                    
                    health["recommendations"].append(
                        "Run 'Remove Duplicates' in System Maintenance to clean up redundant memories. "
                        "Review duplicate_details to see what content is duplicated."
                    )
                
                # --- Check 2: Memories with NULL or empty content ---
                cursor.execute("""
                    SELECT COUNT(*) FROM memories 
                    WHERE content IS NULL OR content = ''
                """)
                null_content = cursor.fetchone()[0]
                
                if null_content > 0:
                    health["status"] = "issues_found"
                    health["issues"].append(f"Found {null_content} memories with empty content")
                    health["recommendations"].append("Clean up empty memory entries")
                
                # --- Check 3: Stale memories (old, low-weight, rarely accessed) ---
                cursor.execute("""
                    SELECT COUNT(*) FROM memories 
                    WHERE julianday('now') - julianday(created_at) > 90 
                    AND weight < 0.3
                    AND access_count < 2
                """)
                stale_memories = cursor.fetchone()[0]
                
                # Only flag if there are many — a few stale memories are normal
                if stale_memories > 50:
                    health["status"] = "issues_found"
                    health["issues"].append(
                        f"Found {stale_memories} old, low-weight, rarely accessed memories"
                    )
                    health["recommendations"].append(
                        "Consider archiving or pruning old, low-value memories"
                    )
            
            return health
        
        except Exception as e:
            logging.error(f"Error analyzing memory health: {e}", exc_info=True)
            return {
                "status": "error",
                "issues": [f"Error analyzing memory health: {str(e)}"],
                "recommendations": ["Check database connection and integrity"]
            }
        
    def get_memory_stats(self):
        """Get comprehensive memory storage statistics.
        
        Provides overview metrics for the admin dashboard including
        totals, type breakdowns, recent activity, and age distribution.
        
        Returns:
            dict: Memory statistics with keys:
                - total_memories: int
                - memory_types: dict of {type_name: count}
                - autonomous_thoughts: int
                - recent_activity: dict of {date_str: count} for last 7 days
                - avg_weight: float
                - age_distribution: dict of {bucket_name: count}
        """
        try:
            stats = {}
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # --- Total memory count ---
                cursor.execute("SELECT COUNT(*) FROM memories")
                stats["total_memories"] = cursor.fetchone()[0]
                
                # --- Memories grouped by type ---
                cursor.execute("""
                    SELECT memory_type, COUNT(*) as count
                    FROM memories
                    GROUP BY memory_type
                    ORDER BY count DESC
                """)
                stats["memory_types"] = {row[0]: row[1] for row in cursor.fetchall()}
                
                # --- Autonomous thought count ---
                cursor.execute("""
                    SELECT COUNT(*) FROM memories 
                    WHERE memory_type = 'autonomous_thought'
                """)
                stats["autonomous_thoughts"] = cursor.fetchone()[0]
                
                # --- Recent activity (last 7 days by date) ---
                cursor.execute("""
                    SELECT DATE(created_at) as date, COUNT(*) as count
                    FROM memories
                    WHERE created_at > datetime('now', '-7 days')
                    GROUP BY date
                    ORDER BY date DESC
                """)
                stats["recent_activity"] = {row[0]: row[1] for row in cursor.fetchall()}
                
                # --- Average memory weight ---
                cursor.execute("SELECT AVG(weight) FROM memories")
                avg = cursor.fetchone()[0]
                stats["avg_weight"] = avg if avg is not None else 0.0
                
                # --- Memory age distribution ---
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN julianday('now') - julianday(created_at) < 1 THEN 'today'
                            WHEN julianday('now') - julianday(created_at) < 7 THEN 'this_week'
                            WHEN julianday('now') - julianday(created_at) < 30 THEN 'this_month'
                            ELSE 'older'
                        END as age_bucket,
                        COUNT(*) as count
                    FROM memories
                    GROUP BY age_bucket
                """)
                stats["age_distribution"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            return stats
        
        except Exception as e:
            logging.error(f"Error getting memory stats: {e}", exc_info=True)
            return {"error": str(e)}

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Self-deprecated: docstring explicitly says "For comprehensive stats, use get_memory_stats() instead."
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__count_memories(self):
        """Count total memories in the database.
        
        Lightweight helper for quick total count.
        For comprehensive stats, use get_memory_stats() instead.
        
        Returns:
            int: Total number of memories, or 0 on error
        """
        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories")
                return cursor.fetchone()[0]
        except Exception as e:
            logging.error(f"Error counting memories: {e}")
            return 0
    
    
    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Memory domains helper — unwired.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__get_memory_domains(self, min_memories=3):
        """Get distinct memory domains from memory types and tags.
        
        Filters out noise domains that appear fewer than min_memories times,
        and excludes technical/system tags that aren't meaningful domains.
        
        Args:
            min_memories (int): Minimum number of memories for a domain to be included.
                Defaults to 3 to filter out one-off tags.
                
        Returns:
            list: Sorted list of unique domain strings
        """
        try:
            domains = set()
            
            # Technical/system tags to exclude from domain lists
            excluded_terms = {
                'null', 'none', 'general', 'important', 
                'autonomous', 'cognition', 'reflection',
                'memory_management', 'autonomous_thought'
            }
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # --- Get memory types with sufficient representation ---
                cursor.execute("""
                    SELECT memory_type, COUNT(*) as count
                    FROM memories
                    GROUP BY memory_type
                    HAVING count >= ?
                    ORDER BY count DESC
                """, (min_memories,))
                
                for memory_type, count in cursor.fetchall():
                    if memory_type and memory_type.lower() not in excluded_terms:
                        domains.add(memory_type)
                
                # --- Extract domains from tags with sufficient representation ---
                cursor.execute("""
                    SELECT tags, COUNT(*) as count
                    FROM memories
                    WHERE tags IS NOT NULL AND tags != ''
                    GROUP BY tags
                    HAVING count >= ?
                """, (min_memories,))
                
                for tags_str, count in cursor.fetchall():
                    if tags_str:
                        # Split comma-separated tags and add individually
                        tags = [tag.strip() for tag in tags_str.split(',')]
                        for tag in tags:
                            if tag and tag.lower() not in excluded_terms:
                                domains.add(tag)
            
            return sorted(list(domains))
        
        except Exception as e:
            logging.error(f"Error getting memory domains: {e}", exc_info=True)
            return []
   
    
    def _record_thought(self, thought_type, content, metadata=None):
        """
        Records a memory management log in the standard logging system only.
        These are system status messages, not valuable content that needs database storage.
        
        Args:
            thought_type (str): Type of thought (e.g., 'analysis', 'organization')
            content (str): The actual content
            metadata (dict, optional): Additional metadata
        
        Returns:
            bool: Always returns True since we're just logging
        """
        try:
            if metadata is None:
                metadata = {}
            
            # Create a thought record for in-memory tracking only
            thought_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'type': thought_type,
                'content': content,
                'metadata': metadata
            }
            
            # Store in thoughts list (in-memory only)
            if hasattr(self, 'thoughts') and self.thoughts is not None:
                self.thoughts.append(thought_record)
            else:
                self.thoughts = [thought_record]
            
            # Keep only recent thoughts in memory (limit to prevent memory bloat)
            if len(self.thoughts) > self.max_thought_history:
                self.thoughts = self.thoughts[-self.max_thought_history:]
            
            # Log to standard logging system with appropriate level
            if thought_type == "error":
                logging.error(f"Autonomous Cognition [{thought_type}]: {content}")
            elif thought_type == "warning":
                logging.warning(f"Autonomous Cognition [{thought_type}]: {content}")
            else:
                logging.info(f"Autonomous Cognition [{thought_type}]: {content}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error in _record_thought: {e}")
            return True  # Don't fail the whole process for logging issues
    
        
    # REMOVED 2026-05-24 (Track A, Issue 1): _should_avoid_complex_llm_calls method deleted.
    # The function existed to throttle LLM calls during "peak hours" via a 30%
    # random skip between 8 AM and 6 PM — meaningless for local Ollama, no cloud
    # rate limits to avoid. Its only caller (_select_next_cognitive_activity) was
    # simplified in the same fix pass (Issues 2 & 3). Hooks for rate_limited and
    # llm_error_count > 3 went with it; neither attribute is incremented anywhere
    # in the codebase, so the checks were inert. If genuine LLM-backoff is needed
    # later (e.g. Ollama unresponsive), reintroduce as a focused guard at the call
    # sites that fail, not a global pre-filter.

    def _safe_llm_invoke(self, prompt, max_retries=2, backoff_factor=2):
        """Safely invoke the LLM with retries and error handling.
        
        Args:
            prompt (str): The prompt to send to the LLM
            max_retries (int): Maximum number of retry attempts
            backoff_factor (int): Multiplier for exponential backoff
            
        Returns:
            str: The LLM response or empty string on failure
        """
        logging.info(f"🤖 Invoking LLM with prompt length: {len(prompt)} characters")
        
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                logging.info(f"   Attempt {retries + 1}/{max_retries + 1}")
                
                # Try with reduced parameters if we're retrying
                if retries > 0:
                    logging.info("   Using reduced parameters for retry")
                    # Use a smaller context size and simpler parameters on retry
                    response = self.chatbot.llm.invoke(
                        prompt, 
                        temperature=0.3,  # Lower temperature for more predictable output
                        num_predict=300   # Limit token generation
                    )
                else:
                    logging.info("   Using standard parameters")
                    response = self.chatbot.llm.invoke(prompt)
                
                if response:
                    response_length = len(response)
                    logging.info(f"✅ LLM responded successfully: {response_length} characters")
                    logging.debug(f"   Response preview: {response[:100]}...")
                    return response
                else:
                    logging.warning("   LLM returned empty response")
                    if retries < max_retries:
                        logging.info("   Will retry with different parameters")
                        
            except Exception as e:
                last_error = e
                logging.warning(f"   LLM invocation failed (attempt {retries+1}/{max_retries+1}): {str(e)}")
                
            retries += 1
            if retries <= max_retries:
                # Exponential backoff
                sleep_time = backoff_factor ** retries
                logging.info(f"   Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
        
        # If we get here, all retries failed
        logging.error(f"❌ All LLM invocation attempts failed. Last error: {last_error}")
        return ""

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Fallback concept extractor for when LLM fails — paired with a synthesis flow that no longer dispatches it.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__extract_fallback_concepts(self, memories):
        """Extract potential concepts from memories when LLM fails.
        
        Args:
            memories (list): List of memory items
            
        Returns:
            str: Newline-separated list of concepts
        """
        # Simple approach: look for common nouns or capitalized terms
        # This is a very basic implementation - you might want something more sophisticated
        import re
        from collections import Counter
        
        # Extract text from memories
        memory_text = ""
        for memory in memories:
            if isinstance(memory, dict):
                memory_text += memory.get('content', '') + " "
            else:
                memory_text += str(memory) + " "
        
        # Look for capitalized phrases which might be concepts
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', memory_text)
        
        # Count occurrences
        word_counts = Counter(capitalized_words)
        
        # Get the most common ones
        common_concepts = [word for word, count in word_counts.most_common(5)]
        
        # Add some general fallback concepts if we don't have enough
        if len(common_concepts) < 3:
            general_concepts = ["Knowledge Organization", "Learning Systems", "Information Processing"]
            common_concepts.extend(general_concepts[:3-len(common_concepts)])
        
        return "\n".join(common_concepts)
        
    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the autonomous cognition system.
        
        Returns:
            Dict[str, Any]: Dictionary with system status information including full thought content
        """
        try:
            now = time.time()
            
            # Calculate next run times for each cognitive activity
            next_runs = {}
            for activity, info in self.cognitive_activities.items():
                last_run = info.get("last_run")
                if last_run is None:
                    # Activity has never run, mark as ready
                    next_runs[activity] = "Ready to run"
                else:
                    # Calculate time until next possible run based on minimum interval
                    min_interval = info.get("min_interval_hours", 12) * 3600  # Convert hours to seconds
                    time_until_next = last_run + min_interval - now
                    
                    if time_until_next <= 0:
                        # Enough time has passed, activity is ready to run
                        next_runs[activity] = "Ready to run"
                    else:
                        # Format remaining time as hours and minutes
                        hours = int(time_until_next / 3600)
                        minutes = int((time_until_next % 3600) / 60)
                        next_runs[activity] = f"In {hours}h {minutes}m"
            
            # FIXED: Get last thought information with FULL content, not just preview
            last_thought = None
            if self.last_autonomous_thought:
                last_thought = {
                    "type": self.last_autonomous_thought["type"],
                    "timestamp": datetime.datetime.fromtimestamp(
                        self.last_autonomous_thought["timestamp"]
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "preview": self.last_autonomous_thought["content"][:100] + "...",
                    # CRITICAL FIX: Include the full content so admin.py can display it
                    "content": self.last_autonomous_thought["content"]
                }
            
            # Check if user is currently active (not idle)
            user_active = not self._is_user_inactive()
            
            # Return comprehensive status dictionary
            return {
                "is_running": self.thinking_thread is not None and self.thinking_thread.is_alive(),
                "current_state": self.cognitive_state,
                "user_active": user_active,
                "last_activity": datetime.datetime.fromtimestamp(self.last_user_activity).strftime("%Y-%m-%d %H:%M:%S"),
                "uptime": self._format_uptime() if hasattr(self, 'thinking_thread') and self.thinking_thread else "Not running",
                "next_activity_runs": next_runs,
                "last_thought": last_thought,
                "thought_history_count": len(self.thought_history)
            }
        
        except Exception as e:
            # Log the error and return error status
            logging.error(f"Error getting cognitive status: {e}", exc_info=True)
            return {"error": str(e)}

    def _format_uptime(self) -> str:
        """Format the uptime of the cognitive thread in a human-readable way.
        
        Returns:
            str: Formatted uptime string (e.g., "2h 15m")
        """
        try:
            # Calculate uptime based on when the thread started
            # Note: This is approximate since we don't track exact start time
            if hasattr(self, 'last_user_activity'):
                uptime_seconds = time.time() - self.last_user_activity
                hours = int(uptime_seconds / 3600)
                minutes = int((uptime_seconds % 3600) / 60)
                return f"{hours}h {minutes}m"
            return "Unknown"
        except Exception as e:
            logging.error(f"Error formatting uptime: {e}")
            return "Unknown"

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Public API for thought impact analysis — designed for admin UI but never wired in.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_analyze_thought_impact(self, thought_id: str) -> Dict[str, Any]:
        """Analyze the impact of a specific autonomous thought.
        
        Args:
            thought_id (str): ID of the thought to analyze
            
        Returns:
            Dict[str, Any]: Analysis of the thought's impact
        """
        try:
            # Find the thought in history
            thought = None
            for t in self.thought_history:
                if t["id"] == thought_id:
                    thought = t
                    break
            
            if not thought:
                return {"error": "Thought not found in history"}
            
            # Get retrieval statistics for this thought
            retrieval_count = 0
            if hasattr(self.chatbot, 'vector_db'):
                # Search for this thought's content in vector DB query logs
                # This would require implementing query logging in your vector_db class
                pass
            
            # Find related thoughts
            related_thoughts = []
            for t in self.thought_history:
                if t["id"] != thought_id and t["type"] == thought["type"]:
                    related_thoughts.append({
                        "id": t["id"],
                        "type": t["type"],
                        "timestamp": datetime.datetime.fromtimestamp(t["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    })
            
            return {
                "thought_id": thought_id,
                "type": thought["type"],
                "timestamp": datetime.datetime.fromtimestamp(thought["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                "retrieval_count": retrieval_count,
                "related_thoughts": related_thoughts[:5]  # Limit to 5 related thoughts
            }
        
        except Exception as e:
            logging.error(f"Error analyzing thought impact: {e}", exc_info=True)
            return {"error": str(e)}

    def adjust_cognitive_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust cognitive parameters based on provided values.
        
        Args:
            params (Dict[str, Any]): Dictionary of parameters to adjust
            
        Returns:
            Dict[str, Any]: Updated parameters
        """
        try:
            updated = {}
            
            # Update activity weights
            if "activity_weights" in params:
                for activity, weight in params["activity_weights"].items():
                    if activity in self.cognitive_activities:
                        old_weight = self.cognitive_activities[activity]["weight"]
                        self.cognitive_activities[activity]["weight"] = float(weight)
                        updated[f"{activity}_weight"] = {
                            "old": old_weight,
                            "new": self.cognitive_activities[activity]["weight"]
                        }
            
            # Update cognitive cycle interval
            if "cycle_interval" in params:
                old_interval = self.cognitive_cycle_interval
                self.cognitive_cycle_interval = int(params["cycle_interval"])
                updated["cycle_interval"] = {
                    "old": old_interval,
                    "new": self.cognitive_cycle_interval
                }
            
            # Update thought history size
            if "max_thought_history" in params:
                old_size = self.max_thought_history
                self.max_thought_history = int(params["max_thought_history"])
                updated["max_thought_history"] = {
                    "old": old_size,
                    "new": self.max_thought_history
                }
                
                # Trim history if needed
                if len(self.thought_history) > self.max_thought_history:
                    self.thought_history = self.thought_history[-self.max_thought_history:]
            
            return {
                "status": "success",
                "updated_parameters": updated
            }
        
        except Exception as e:
            logging.error(f"Error adjusting cognitive parameters: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Memory command stats helper — duplicates logic in _analyze_memory_usage (also quarantined).
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__get_memory_command_stats(self):
        """Get statistics on memory command usage."""
        try:
            if hasattr(self.chatbot, 'deepseek_enhancer') and hasattr(self.chatbot.deepseek_enhancer, 'lifetime_counters'):
                counters = self.chatbot.deepseek_enhancer.lifetime_counters.get_counters()
                
                stats = "Memory command usage statistics:\n"
                for command, count in counters.items():
                    stats += f"- {command}: {count}\n"
                
                return stats
            
            return "Memory command statistics unavailable."
        
        except Exception as e:
            logging.error(f"Error getting memory command stats: {e}", exc_info=True)
            return "Error retrieving memory command statistics."
            

    def _parse_knowledge_gaps_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse the structured knowledge gaps response from LLM.
        
        Handles both plain and bold markdown field formats.
        UNIQUENESS_CHECK field is intentionally discarded — it is internal LLM
        reasoning that must NOT be appended to the stored description.
        """
        try:
            gaps = []
            current_gap = {}
            
            # Define all known field prefixes — used in the catch-all guard below
            KNOWN_PREFIXES = [
                'TOPIC:', '**TOPIC:**',
                'CLASSIFICATION:', '**CLASSIFICATION:**',
                'DESCRIPTION:', '**DESCRIPTION:**',
                'PRIORITY:', '**PRIORITY:**',
                'UNIQUENESS_CHECK:', '**UNIQUENESS_CHECK:**',  # Must appear here to stay out of description
            ]
            
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('---'):  # Skip empty lines and separators
                    continue
                    
                # Check for gap start (e.g. "GAP 1:", "GAP 2:", plain "GAP")
                if line.upper().startswith('GAP'):
                    # Save previous gap if all required fields are present
                    if current_gap and all(key in current_gap for key in ['topic', 'classification', 'description', 'priority']):
                        gaps.append(current_gap)
                    current_gap = {}
                    continue
                
                # --- UNIQUENESS_CHECK: discard entirely ---
                # This is the LLM's internal reasoning — it must NOT be stored in the
                # gap description, where it would corrupt semantic embeddings and
                # pollute future LLM prompts that display gap descriptions.
                if line.startswith('UNIQUENESS_CHECK:') or line.startswith('**UNIQUENESS_CHECK:**'):
                    # Intentionally skip — no data stored from this field
                    continue
                
                # Parse required gap fields — handle plain and bold markdown formats
                if line.startswith('TOPIC:') or line.startswith('**TOPIC:**'):
                    topic_text = line.replace('**TOPIC:**', '').replace('TOPIC:', '').strip()
                    current_gap['topic'] = topic_text
                    
                elif line.startswith('CLASSIFICATION:') or line.startswith('**CLASSIFICATION:**'):
                    classification_text = line.replace('**CLASSIFICATION:**', '').replace('CLASSIFICATION:', '').strip()
                    current_gap['classification'] = classification_text
                    
                elif line.startswith('DESCRIPTION:') or line.startswith('**DESCRIPTION:**'):
                    description_text = line.replace('**DESCRIPTION:**', '').replace('DESCRIPTION:', '').strip()
                    current_gap['description'] = description_text
                    
                elif line.startswith('PRIORITY:') or line.startswith('**PRIORITY:**'):
                    priority_text = line.replace('**PRIORITY:**', '').replace('PRIORITY:', '').strip()
                    current_gap['priority'] = priority_text
                    
                elif (current_gap
                      and 'description' in current_gap
                      and not any(line.startswith(prefix) for prefix in KNOWN_PREFIXES)):
                    # Continuation of a multi-line description — only if not a known field header
                    current_gap['description'] += ' ' + line
            
            # Don't forget the last gap in the response
            if current_gap and all(key in current_gap for key in ['topic', 'classification', 'description', 'priority']):
                gaps.append(current_gap)
            
            logging.info(f"Parsed {len(gaps)} knowledge gap(s) from LLM response")
            return gaps
            
        except Exception as e:
            logging.error(f"Error parsing knowledge gaps response: {e}")
            return []

    def _get_recent_knowledge_gaps(self, limit: int = 5) -> str:
        """Get recent knowledge gaps for context."""
        try:
            from knowledge_gap import KnowledgeGapQueue
            gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
            
            recent_gaps = gap_queue.get_gaps_by_status('pending', limit)
            
            if not recent_gaps:
                return "No recent knowledge gaps identified."
            
            formatted_gaps = []
            for gap in recent_gaps:
                formatted_gaps.append(f"- {gap['topic']}: {gap['description'][:100]}...")
            
            return "\n".join(formatted_gaps)
            
        except Exception as e:
            logging.error(f"Error getting recent knowledge gaps: {e}")
            return "Error retrieving recent knowledge gaps."

    def _get_historical_gaps(self, days=90, limit=50):
        """
        Fetch fulfilled and failed knowledge gaps from the last N days.
        
        These are used for duplicate checking — prevents the system from
        re-proposing gaps that were already investigated and resolved.
        
        CRITICAL CONTEXT: mark_fulfilled() in KnowledgeGapQueue removes the
        Qdrant embedding for the fulfilled gap (line 582-583 of knowledge_gap.py).
        This means fulfilled gaps are completely invisible to semantic similarity
        checks via check_semantic_similarity(). This SQL-based approach is the
        ONLY way to catch them during duplicate detection.
        
        Failed gaps keep their embeddings (mark_failed does NOT remove them),
        but we include them here for completeness in the dedup prompt.
        
        Added 2026-04-01 as part of knowledge gap resurfacing fix.
        
        Args:
            days (int): How far back to look (default 90 days)
            limit (int): Maximum gaps to return (default 50)
            
        Returns:
            List[Dict]: Historical gaps with topic, description, status fields
                        matching the format returned by get_gaps_by_status()
        """
        try:
            interval_str = f'-{int(days)} days'
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Fetch fulfilled gaps — these had their Qdrant embeddings removed
                # by mark_fulfilled(), so they're invisible to semantic checks.
                # Also fetch failed gaps for completeness.
                # Use OR across date columns because different statuses populate
                # different timestamps (fulfilled_at vs last_attempt_at vs created_at)
                cursor.execute('''
                    SELECT id, topic, description, priority, status,
                           created_at, fulfilled_at, items_acquired, last_attempt_at
                    FROM knowledge_gaps
                    WHERE status IN ('fulfilled', 'failed')
                    AND (
                        fulfilled_at >= datetime('now', ?)
                        OR last_attempt_at >= datetime('now', ?)
                        OR created_at >= datetime('now', ?)
                    )
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (interval_str, interval_str, interval_str, limit))
                
                rows = cursor.fetchall()
                historical = [dict(row) for row in rows]
                
                logging.info(
                    f"Loaded {len(historical)} historical gaps (fulfilled/failed) "
                    f"from last {days} days for duplicate checking"
                )
                return historical
                
        except Exception as e:
            logging.error(f"Error loading historical gaps: {e}", exc_info=True)
            return []

    def _get_recent_reflections(self, limit: int = 3) -> str:
        """Get recent reflections for context."""
        try:
            # Search for recent autonomous thoughts of reflection type
            reflections = self.vector_db.search(
                query="reflection autonomous thought",
                mode="default",
                k=limit,
                metadata_filters={"type": "reflection"}
            )
            
            if not reflections:
                return "No recent reflections available."
            
            formatted_reflections = []
            for reflection in reflections:
                content = reflection.get('content', '')
                if content:
                    formatted_reflections.append(f"- {content[:150]}...")
            
            return "\n".join(formatted_reflections)
            
        except Exception as e:
            logging.error(f"Error getting recent reflections: {e}")
            return "Error retrieving recent reflections."
        
    
    def _safe_merge_metadata(self, existing_metadata_str: str, new_fields: dict) -> str:
        """
        Safely merge new fields into an existing SQLite metadata JSON string.
        
        Handles NULL, empty string, and non-JSON values gracefully — older memories
        may have been stored before metadata JSON was standardized. Always returns
        a valid JSON string.
        
        Args:
            existing_metadata_str: Raw string value from SQLite metadata column
            new_fields: Dict of fields to merge in (new fields overwrite existing)
            
        Returns:
            str: Valid JSON string with merged fields
        """
        try:
            # Attempt to parse existing metadata
            if existing_metadata_str and existing_metadata_str.strip():
                existing = json.loads(existing_metadata_str)
                # Guard against non-dict JSON (e.g. a bare string or array)
                if not isinstance(existing, dict):
                    logging.warning(
                        f"_safe_merge_metadata: existing metadata is not a dict "
                        f"(type={type(existing).__name__}), starting fresh"
                    )
                    existing = {}
            else:
                existing = {}
        except (json.JSONDecodeError, TypeError) as e:
            # Non-JSON or None — start with empty dict, log for awareness
            logging.warning(f"_safe_merge_metadata: could not parse existing metadata: {e}")
            existing = {}
        
        # Merge: new_fields overwrite existing on collision
        existing.update(new_fields)
        return json.dumps(existing)


    def _get_consolidation_candidates(self) -> list:
        """
        Query SQLite for reflection memories eligible for consolidation.
        
        Targets core self-awareness reflection types only (narrow first pass):
        - memory_type IN ('self_reflection', 'self-reflection', 'reflection')
        - source IN ('daily_reflection', 'weekly_reflection', 'monthly_reflection')
        
        Topical reflections (relationship_reflection, training_reflection etc.)
        are intentionally excluded — candidate for a separate consolidation
        activity once this baseline is proven stable.
        
        Exclusion rules:
        - Already consolidated (metadata contains 'consolidated_into')
        - Confidence below pruning floor (likely already processed or low quality)
        
        Returns:
            list: Dicts with keys: id, content, confidence, metadata_str,
                created_at, tracking_id. Returns empty list on error.
        """
        try:
            candidates = []
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Two-path query joined with UNION to cover both targeting strategies:
                # Path A: memory_type match (manual/autonomous self-reflections)
                # Path B: source match (scheduled daily/weekly/monthly reflections)
                # UNION deduplicates in case a memory matches both conditions.
                # tracking_id included for reliable Qdrant result matching.
                cursor.execute("""
                    SELECT id, content, confidence, metadata, created_at, tracking_id
                    FROM memories
                    WHERE memory_type IN (
                        'self_reflection',
                        'self-reflection',
                        'reflection'
                    )
                    AND (confidence IS NULL OR confidence > ?)
                    AND (metadata IS NULL
                        OR metadata = ''
                        OR metadata NOT LIKE '%"consolidated_into"%')
                    
                    UNION
                    
                    SELECT id, content, confidence, metadata, created_at, tracking_id
                    FROM memories
                    WHERE source IN (
                        'daily_reflection',
                        'weekly_reflection',
                        'monthly_reflection'
                    )
                    AND (confidence IS NULL OR confidence > ?)
                    AND (metadata IS NULL
                        OR metadata = ''
                        OR metadata NOT LIKE '%"consolidated_into"%')
                    
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (
                    CONSOLIDATION_SOURCE_MIN_CONFIDENCE,  # for memory_type path
                    CONSOLIDATION_SOURCE_MIN_CONFIDENCE,  # for source path
                    CONSOLIDATION_CANDIDATE_LIMIT
                ))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    candidates.append({
                        'id': row['id'],
                        'content': row['content'] or '',
                        'confidence': row['confidence'] or 0.5,
                        'metadata_str': row['metadata'] or '',
                        'created_at': row['created_at'],
                        'tracking_id': row['tracking_id'] or ''
                    })
            
            logging.info(
                f"[CONSOLIDATION] Found {len(candidates)} eligible reflection "
                f"candidates (self_reflection + scheduled types)"
            )
            return candidates
            
        except Exception as e:
            logging.error(
                f"[CONSOLIDATION] Error fetching candidates: {e}", exc_info=True
            )
            return []

    def _cluster_by_similarity(self, candidates: list) -> list:
        """
        Group candidates into semantically similar clusters using Qdrant search.
        
        Matches Qdrant results back to candidates using tracking_id (primary)
        with normalized content string as fallback for older memories that
        may lack tracking_id. Clusters below CONSOLIDATION_MIN_CLUSTER_SIZE
        are discarded.
        
        Args:
            candidates: List of candidate dicts from _get_consolidation_candidates()
            
        Returns:
            list: List of clusters meeting minimum size threshold.
        """
        if not candidates:
            return []
        
        try:
            # Build lookup maps for matching Qdrant results back to candidates
            # Primary: tracking_id match (reliable, survives whitespace differences)
            # Fallback: normalized content match (for older memories without tracking_id)
            tracking_id_map = {}
            content_map = {}
            
            for c in candidates:
                if c.get('tracking_id'):
                    tracking_id_map[c['tracking_id']] = c
                if c.get('content'):
                    # Normalize: lowercase, collapse whitespace for fuzzy match
                    normalized = ' '.join(c['content'].strip().lower().split())
                    content_map[normalized] = c
            
            assigned_ids = set()  # SQLite ids already placed in a cluster
            clusters = []
            
            for candidate in candidates:
                cand_id = candidate['id']
                
                if cand_id in assigned_ids:
                    continue
                
                cluster = [candidate]
                assigned_ids.add(cand_id)
                
                try:
                    query_text = candidate['content'][:500]
                    
                    results = self.vector_db.search(
                        query=query_text,
                        k=20,
                        mode="default"
                    )
                    
                    for result in results:
                        score = result.get('similarity_score', 0)
                        if score < CONSOLIDATION_SIMILARITY_THRESHOLD:
                            continue
                        
                        # --- Primary match: tracking_id from Qdrant metadata ---
                        result_tracking_id = result.get('metadata', {}).get('tracking_id', '')
                        matched_candidate = None
                        
                        if result_tracking_id:
                            matched_candidate = tracking_id_map.get(result_tracking_id)
                        
                        # --- Fallback: normalized content match ---
                        if not matched_candidate:
                            result_content = result.get('content', '')
                            if result_content:
                                normalized_result = ' '.join(
                                    result_content.strip().lower().split()
                                )
                                matched_candidate = content_map.get(normalized_result)
                        
                        # --- Add to cluster if unassigned candidate ---
                        if matched_candidate and matched_candidate['id'] not in assigned_ids:
                            cluster.append(matched_candidate)
                            assigned_ids.add(matched_candidate['id'])
                            logging.debug(
                                f"[CONSOLIDATION] Clustered id={matched_candidate['id']} "
                                f"with seed id={cand_id} (score={score:.3f})"
                            )
                            
                except Exception as search_error:
                    logging.warning(
                        f"[CONSOLIDATION] Qdrant search failed for candidate "
                        f"id={cand_id}: {search_error}"
                    )
                
                # Only keep clusters meeting minimum size
                if len(cluster) >= CONSOLIDATION_MIN_CLUSTER_SIZE:
                    clusters.append(cluster)
                    logging.info(
                        f"[CONSOLIDATION] Valid cluster: {len(cluster)} members, "
                        f"seed id={cand_id}"
                    )
                else:
                    logging.debug(
                        f"[CONSOLIDATION] Cluster too small ({len(cluster)}), "
                        f"discarding seed id={cand_id}"
                    )
            
            logging.info(
                f"[CONSOLIDATION] Clustering complete: {len(clusters)} valid clusters"
            )
            return clusters
            
        except Exception as e:
            logging.error(
                f"[CONSOLIDATION] Error during clustering: {e}", exc_info=True
            )
            return []
    

    def _check_existing_synthesis(self, cluster: list) -> dict:
        """
        Check if a consolidation_synthesis already exists that covers this cluster.
        
        Looks for synthesis memories whose consolidated_from list has >50% overlap
        with the current cluster's source ids. Used to prevent re-synthesizing
        the same group of memories on consecutive pulse runs.
        
        Also checks ceiling conditions on any found synthesis:
        - consolidation_count >= CONSOLIDATION_MAX_ROUNDS
        - len(content) >= CONSOLIDATION_MAX_CONTENT_LENGTH
        
        Args:
            cluster: List of candidate dicts forming this cluster
            
        Returns:
            dict with keys:
                'exists': bool — True if synthesis already covers this cluster
                'synthesis_id': int or None — SQLite id of existing synthesis
                'synthesis_content': str — existing content if found
                'synthesis_metadata_str': str — existing metadata JSON string
                'at_ceiling': bool — True if existing synthesis cannot be refined further
                'consolidation_count': int — current round count
        """
        result = {
            'exists': False,
            'synthesis_id': None,
            'synthesis_content': '',
            'synthesis_metadata_str': '',
            'at_ceiling': False,
            'consolidation_count': 0
        }
        
        try:
            cluster_ids = set(str(c['id']) for c in cluster)
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Fetch all existing synthesis memories for overlap check
                cursor.execute("""
                    SELECT id, content, metadata, confidence
                    FROM memories
                    WHERE memory_type = 'consolidation_synthesis'
                    ORDER BY created_at DESC
                    LIMIT 100
                """)
                
                syntheses = cursor.fetchall()
            
            for synthesis in syntheses:
                try:
                    meta_str = synthesis['metadata'] or ''
                    meta = json.loads(meta_str) if meta_str.strip() else {}
                    if not isinstance(meta, dict):
                        continue
                    
                    # Get the source ids this synthesis was built from
                    consolidated_from = set(
                        str(x) for x in meta.get('consolidated_from', [])
                    )
                    
                    if not consolidated_from:
                        continue
                    
                    # Calculate overlap between cluster and this synthesis's sources
                    overlap = len(cluster_ids.intersection(consolidated_from))
                    overlap_ratio = overlap / len(cluster_ids)
                    
                    # >50% overlap = this synthesis already covers this cluster
                    if overlap_ratio > 0.5:
                        consolidation_count = meta.get('consolidation_count', 0)
                        content = synthesis['content'] or ''
                        
                        result['exists'] = True
                        result['synthesis_id'] = synthesis['id']
                        result['synthesis_content'] = content
                        result['synthesis_metadata_str'] = meta_str
                        result['consolidation_count'] = consolidation_count
                        
                        # Check ceiling conditions
                        at_ceiling = (
                            consolidation_count >= CONSOLIDATION_MAX_ROUNDS or
                            len(content) >= CONSOLIDATION_MAX_CONTENT_LENGTH
                        )
                        result['at_ceiling'] = at_ceiling
                        
                        logging.info(
                            f"[CONSOLIDATION] Found existing synthesis id={synthesis['id']} "
                            f"(overlap={overlap_ratio:.0%}, rounds={consolidation_count}, "
                            f"at_ceiling={at_ceiling})"
                        )
                        return result
                        
                except (json.JSONDecodeError, TypeError) as meta_err:
                    logging.debug(
                        f"[CONSOLIDATION] Could not parse synthesis metadata "
                        f"id={synthesis['id']}: {meta_err}"
                    )
                    continue
            
            logging.debug(f"[CONSOLIDATION] No existing synthesis found for this cluster")
            return result
            
        except Exception as e:
            logging.error(
                f"[CONSOLIDATION] Error checking existing synthesis: {e}", exc_info=True
            )
            return result


    def _update_source_memory_metadata(self, source_ids: list, synthesis_id: int) -> bool:
        """
        Mark source memories as consolidated: update SQLite metadata and retire
        their Qdrant vectors so they no longer appear in [SEARCH:] results.

        Two operations per source memory:
          1. SQLite UPDATE — sets 'consolidated_into', 'consolidation_timestamp',
             and 'pre_consolidation_confidence' in the metadata JSON. This prevents
             _get_consolidation_candidates() from re-selecting them in future runs.
          2. Qdrant DELETE — removes the vector via delete_by_memory_id(tracking_id).
             The synthesis already represents these memories in search space; keeping
             their raw vectors active inflates [SEARCH:] token budgets over time as
             more clusters accumulate.

        Source memories are NEVER removed from SQLite — full content and confidence
        remain available for admin inspection, audit, and manual retrieval.
        Only their Qdrant vectors (the search-active component) are retired.

        Qdrant deletion failures are non-fatal: if a vector cannot be retired,
        it stays in search results until the next manual cleanup. SQLite is the
        authoritative record and is always updated first.

        Args:
            source_ids: List of SQLite memory ids to update
            synthesis_id: The SQLite id of the synthesis memory they were absorbed into

        Returns:
            bool: True if all SQLite updates succeeded, False if any SQL update failed.
                  Qdrant deletion failures do not affect the return value.
        """
        if not source_ids:
            return True

        now_iso = datetime.datetime.now().isoformat()
        sql_success_count = 0
        qdrant_retired_count = 0
        qdrant_fail_count = 0

        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()

                for source_id in source_ids:
                    try:
                        # Fetch current metadata, confidence, and tracking_id in one query.
                        # tracking_id is the UUID stored in Qdrant as metadata.tracking_id
                        # and is required by delete_by_memory_id().
                        cursor.execute(
                            "SELECT metadata, confidence, tracking_id FROM memories WHERE id = ?",
                            (source_id,)
                        )
                        row = cursor.fetchone()

                        if not row:
                            logging.warning(
                                f"[CONSOLIDATION] Source memory id={source_id} "
                                f"not found during metadata update"
                            )
                            continue

                        existing_meta_str, current_confidence, tracking_id = row

                        # Build consolidation tracking fields
                        consolidation_fields = {
                            'consolidated_into': str(synthesis_id),
                            'consolidation_timestamp': now_iso,
                            'pre_consolidation_confidence': current_confidence or 0.5
                        }

                        # Safely merge into existing metadata
                        updated_meta_str = self._safe_merge_metadata(
                            existing_meta_str, consolidation_fields
                        )

                        # --- Step A: Update SQLite (authoritative record) ---
                        cursor.execute(
                            "UPDATE memories SET metadata = ?, last_accessed = ? WHERE id = ?",
                            (updated_meta_str, now_iso, source_id)
                        )

                        sql_success_count += 1
                        logging.debug(
                            f"[CONSOLIDATION] Updated source metadata for id={source_id} "
                            f"→ consolidated_into={synthesis_id}"
                        )

                        # --- Step B: Retire Qdrant vector (non-fatal if missing) ---
                        # Only attempt if we have a tracking_id to look up.
                        # The synthesis already represents this memory in vector space.
                        if tracking_id:
                            try:
                                retire_ok, retire_cnt = self.vector_db.delete_by_memory_id(
                                    tracking_id
                                )
                                if retire_ok and retire_cnt > 0:
                                    qdrant_retired_count += 1
                                    logging.debug(
                                        f"[CONSOLIDATION] Retired Qdrant vector for "
                                        f"source id={source_id} (tracking_id={tracking_id})"
                                    )
                                else:
                                    # Vector may have already been deleted or never existed
                                    qdrant_fail_count += 1
                                    logging.debug(
                                        f"[CONSOLIDATION] No Qdrant vector found for "
                                        f"source id={source_id} (tracking_id={tracking_id}) "
                                        f"— already absent or never indexed"
                                    )
                            except Exception as qdrant_err:
                                qdrant_fail_count += 1
                                logging.warning(
                                    f"[CONSOLIDATION] Could not retire Qdrant vector for "
                                    f"source id={source_id}: {qdrant_err} (non-fatal)"
                                )
                        else:
                            qdrant_fail_count += 1
                            logging.debug(
                                f"[CONSOLIDATION] Source id={source_id} has no tracking_id "
                                f"— Qdrant vector cannot be retired (pre-tracking memory)"
                            )

                    except Exception as row_error:
                        logging.error(
                            f"[CONSOLIDATION] Failed to update source id={source_id}: "
                            f"{row_error}"
                        )

                conn.commit()

            all_sql_succeeded = sql_success_count == len(source_ids)
            logging.info(
                f"[CONSOLIDATION] Source retirement: "
                f"{sql_success_count}/{len(source_ids)} SQL updates succeeded, "
                f"{qdrant_retired_count} Qdrant vectors retired, "
                f"{qdrant_fail_count} Qdrant retirements skipped/failed (non-fatal)"
            )
            return all_sql_succeeded

        except Exception as e:
            logging.error(
                f"[CONSOLIDATION] Error updating source metadata: {e}", exc_info=True
            )
            return False


    def _perform_memory_consolidation_pulse(self) -> bool:
        """
        Main orchestrator for the Memory Consolidation Pulse cognitive activity.
        
        Finds semantically related self_reflection memories, synthesizes them into
        a unified first-person insight, and stores the result as a
        'consolidation_synthesis' memory. Source memories are marked in metadata
        to prevent re-consolidation but remain fully searchable in Qdrant.
        
        Ceiling protection prevents any single synthesis from growing unbounded:
        - If consolidation_count >= CONSOLIDATION_MAX_ROUNDS, spawns a lineage child
        - If content >= CONSOLIDATION_MAX_CONTENT_LENGTH, spawns a lineage child
        
        Flow:
            1. Fetch eligible self_reflection candidates from SQLite
            2. Cluster by semantic similarity via Qdrant
            3. For each valid cluster, check for existing synthesis
            4. Determine if refining existing or creating new (or spawning lineage child)
            5. LLM synthesizes cluster into first-person insight
            6. Store synthesis via transaction coordinator (both DBs)
            7. Update source memory metadata via direct SQL
            8. Log autonomous thought summary
            
        Returns:
            bool: True if at least one synthesis was stored successfully
        """
        logging.info("[CONSOLIDATION] ====== STARTING MEMORY CONSOLIDATION PULSE ======")
        self.cognitive_state = "consolidating"
        start_time = datetime.datetime.now()
        syntheses_created = 0
        syntheses_refined = 0
        
        try:
            # --- STEP 1: Get candidates ---
            candidates = self._get_consolidation_candidates()
            
            if len(candidates) < CONSOLIDATION_MIN_CLUSTER_SIZE:
                logging.info(
                    f"[CONSOLIDATION] Insufficient candidates "
                    f"({len(candidates)} < {CONSOLIDATION_MIN_CLUSTER_SIZE}), skipping pulse"
                )
                return False
            
            # --- STEP 2: Cluster by similarity ---
            clusters = self._cluster_by_similarity(candidates)
            
            if not clusters:
                logging.info("[CONSOLIDATION] No valid clusters found, skipping pulse")
                self._store_autonomous_thought(
                    content="# Memory Consolidation Pulse\n\nRan consolidation pulse. "
                            f"Found {len(candidates)} self_reflection candidates but no "
                            f"semantic clusters met the minimum size of "
                            f"{CONSOLIDATION_MIN_CLUSTER_SIZE}. Memory landscape appears "
                            f"appropriately diverse.",
                    thought_type="consolidation_pulse",
                    confidence=0.7
                )
                return False
            
            logging.info(f"[CONSOLIDATION] Processing {len(clusters)} valid cluster(s)")
            
            for cluster_idx, cluster in enumerate(clusters):
                cluster_ids = [c['id'] for c in cluster]
                cluster_label = f"cluster {cluster_idx + 1}/{len(clusters)}"
                
                logging.info(
                    f"[CONSOLIDATION] Processing {cluster_label}: "
                    f"{len(cluster)} members, ids={cluster_ids}"
                )
                
                try:
                    # --- STEP 3: Check for existing synthesis ---
                    existing = self._check_existing_synthesis(cluster)
                    
                    # Determine storage mode
                    if existing['exists'] and existing['at_ceiling']:
                        # Existing synthesis is full — spawn a lineage child
                        storage_mode = 'lineage_child'
                        lineage_parent_id = existing['synthesis_id']
                        logging.info(
                            f"[CONSOLIDATION] {cluster_label}: existing synthesis at ceiling, "
                            f"will spawn lineage child from parent id={lineage_parent_id}"
                        )
                        
                    elif existing['exists'] and not existing['at_ceiling']:
                        # Refine the existing synthesis
                        storage_mode = 'refine'
                        logging.info(
                            f"[CONSOLIDATION] {cluster_label}: will refine existing "
                            f"synthesis id={existing['synthesis_id']} "
                            f"(round {existing['consolidation_count'] + 1})"
                        )
                        
                    else:
                        # No existing synthesis — create fresh
                        storage_mode = 'new'
                        lineage_parent_id = None
                        logging.info(
                            f"[CONSOLIDATION] {cluster_label}: creating new synthesis"
                        )
                    
                    # --- STEP 4: Build LLM prompt ---
                    # Combine source content, capped to avoid context overflow
                    source_texts = []
                    total_chars = 0
                    MAX_INPUT_CHARS = 3000
                    
                    for member in cluster:
                        snippet = member['content'][:600]  # Cap each source contribution
                        if total_chars + len(snippet) > MAX_INPUT_CHARS:
                            break
                        source_texts.append(f"Reflection:\n{snippet}")
                        total_chars += len(snippet)
                    
                    combined_sources = "\n\n---\n\n".join(source_texts)
                    
                    # Context for refinement mode includes prior synthesis
                    prior_synthesis_context = ""
                    if storage_mode == 'refine' and existing['synthesis_content']:
                        prior_synthesis_context = (
                            f"\n\nI have previously synthesized these reflections into:\n"
                            f"{existing['synthesis_content'][:500]}\n\n"
                            f"I will now refine and deepen this synthesis with the new "
                            f"pattern I've identified."
                        )
                    
                    synthesis_prompt = f"""/no_think
    I am QWEN, reflecting on my own self-reflection memories to find deeper patterns.

    The following are my own past self-reflections on related themes:

    {combined_sources}
    {prior_synthesis_context}

    I will synthesize these into a single, unified insight written in first person as myself.
    My synthesis should:
    1. Identify the core pattern or understanding that connects these reflections
    2. Express what I have genuinely come to understand about myself or my experience
    3. Be honest about uncertainty where it exists
    4. Be concise — no more than 300 words
    5. Begin with: "# Memory Consolidation Synthesis"

    I will not summarize each reflection individually. I will speak as QWEN expressing 
    a deeper understanding that emerges from reading these as my own thoughts.
    """
                    
                    # --- STEP 5: LLM synthesis ---
                    logging.info(
                        f"[CONSOLIDATION] {cluster_label}: invoking LLM for synthesis"
                    )
                    synthesis_content = self._safe_llm_invoke(synthesis_prompt)
                    synthesis_content = self._clean_llm_response(synthesis_content)
                    
                    if not synthesis_content:
                        logging.warning(
                            f"[CONSOLIDATION] {cluster_label}: LLM returned empty synthesis, "
                            f"skipping this cluster"
                        )
                        continue
                    
                    # Ensure correct header (for db_maintenance autonomous_patterns protection)
                    if not synthesis_content.strip().startswith("# Memory Consolidation Synthesis"):
                        synthesis_content = "# Memory Consolidation Synthesis\n\n" + synthesis_content
                    
                    # --- STEP 6: Build metadata and store ---
                    avg_confidence = sum(c['confidence'] for c in cluster) / len(cluster)
                    # Boost to minimum synthesis confidence floor
                    synthesis_confidence = max(
                        CONSOLIDATION_SYNTHESIS_CONFIDENCE, avg_confidence
                    )
                    
                    new_consolidation_count = (
                        existing['consolidation_count'] + 1
                        if storage_mode == 'refine'
                        else 1
                    )
                    
                    synthesis_metadata = {
                        'type': 'consolidation_synthesis',
                        'source': 'memory_consolidation_pulse',
                        'consolidation_count': new_consolidation_count,
                        'consolidated_from': [str(c['id']) for c in cluster],
                        'is_consolidation': True,
                        'lineage_parent': str(lineage_parent_id) if storage_mode == 'lineage_child' else None,
                        'consolidation_timestamp': datetime.datetime.now().isoformat(),
                        'storage_mode': storage_mode,
                        'cluster_size': len(cluster)
                    }
                    
                    # Verify transaction coordinator is available before attempting storage
                    if not hasattr(self.chatbot, 'store_memory_with_transaction'):
                        logging.error(
                            "[CONSOLIDATION] Transaction coordinator not available — "
                            "cannot store synthesis safely"
                        )
                        continue
                    
                    success, synthesis_id = self.chatbot.store_memory_with_transaction(
                        content=synthesis_content,
                        memory_type='consolidation_synthesis',
                        metadata=synthesis_metadata,
                        confidence=synthesis_confidence
                    )
                    
                    if not success:
                        logging.error(
                            f"[CONSOLIDATION] {cluster_label}: transaction coordinator "
                            f"failed to store synthesis"
                        )
                        continue
                    
                    logging.info(
                        f"[CONSOLIDATION] {cluster_label}: synthesis stored "
                        f"with id={synthesis_id} (mode={storage_mode})"
                    )

                    # --- STEP 6.5: Retire parent synthesis from Qdrant on lineage spawn ---
                    # When a lineage_child is created, the parent synthesis is at ceiling
                    # (CONSOLIDATION_MAX_ROUNDS or CONSOLIDATION_MAX_CONTENT_LENGTH) and
                    # can no longer be refined. It stays in SQLite for audit, but its Qdrant
                    # vector must be removed or it accumulates indefinitely. Over many idle
                    # cycles a single topic cluster produces parent → child → grandchild,
                    # each up to 2000 chars, all semantically similar, all returned together
                    # by [SEARCH:] queries — inflating tokens and slowing down the system.
                    # Retiring the parent keeps the active Qdrant lineage chain bounded to
                    # ONE vector per cluster at all times.
                    # Added: 2026-05-22
                    if storage_mode == 'lineage_child' and lineage_parent_id is not None:
                        try:
                            # Fetch parent's tracking_id (UUID) from SQLite —
                            # delete_by_memory_id() filters on metadata.tracking_id in Qdrant
                            with sqlite3.connect(self.memory_db.db_path) as _lc_conn:
                                _lc_cursor = _lc_conn.cursor()
                                _lc_cursor.execute(
                                    "SELECT tracking_id FROM memories WHERE id = ?",
                                    (lineage_parent_id,)
                                )
                                _lc_row = _lc_cursor.fetchone()
                                parent_tracking_id = _lc_row[0] if _lc_row and _lc_row[0] else None

                            if parent_tracking_id:
                                retire_ok, retire_cnt = self.vector_db.delete_by_memory_id(
                                    parent_tracking_id
                                )
                                if retire_ok and retire_cnt > 0:
                                    logging.info(
                                        f"[CONSOLIDATION] {cluster_label}: retired parent "
                                        f"synthesis id={lineage_parent_id} from Qdrant "
                                        f"({retire_cnt} vector(s) removed) — "
                                        f"lineage chain bounded"
                                    )
                                else:
                                    logging.warning(
                                        f"[CONSOLIDATION] {cluster_label}: could not retire "
                                        f"parent synthesis id={lineage_parent_id} from Qdrant "
                                        f"(non-fatal — parent remains in search results)"
                                    )
                            else:
                                logging.warning(
                                    f"[CONSOLIDATION] {cluster_label}: parent synthesis "
                                    f"id={lineage_parent_id} has no tracking_id — "
                                    f"cannot retire from Qdrant"
                                )
                        except Exception as _retire_err:
                            # Non-fatal — lineage child stored successfully
                            logging.warning(
                                f"[CONSOLIDATION] {cluster_label}: error retiring parent "
                                f"synthesis from Qdrant (non-fatal): {_retire_err}"
                            )

                    # --- STEP 7: Update source memory metadata ---
                    # Direct SQL only — Qdrant vectors for sources stay unchanged
                    update_success = self._update_source_memory_metadata(
                        source_ids=cluster_ids,
                        synthesis_id=synthesis_id
                    )
                    
                    if not update_success:
                        # Non-fatal — synthesis is stored, sources just won't be
                        # excluded from future candidate queries until next pass
                        logging.warning(
                            f"[CONSOLIDATION] {cluster_label}: synthesis stored but "
                            f"some source metadata updates failed — sources may be "
                            f"re-selected on next pulse run"
                        )
                    
                    # Track counts for summary
                    if storage_mode == 'new' or storage_mode == 'lineage_child':
                        syntheses_created += 1
                    else:
                        syntheses_refined += 1
                        
                except Exception as cluster_error:
                    logging.error(
                        f"[CONSOLIDATION] Error processing {cluster_label}: "
                        f"{cluster_error}", exc_info=True
                    )
                    continue
            
            # --- STEP 8: Store autonomous thought summary ---
            duration = (datetime.datetime.now() - start_time).total_seconds()
            summary = (
                f"# Memory Consolidation Synthesis\n\n"
                f"Consolidation pulse completed in {duration:.1f}s.\n"
                f"Candidates evaluated: {len(candidates)}\n"
                f"Clusters found: {len(clusters)}\n"
                f"New syntheses created: {syntheses_created}\n"
                f"Existing syntheses refined: {syntheses_refined}\n"
            )
            
            self._store_autonomous_thought(
                content=summary,
                thought_type="consolidation_pulse",
                confidence=0.8
            )
            
            overall_success = (syntheses_created + syntheses_refined) > 0
            logging.info(
                f"[CONSOLIDATION] ====== PULSE COMPLETE: "
                f"created={syntheses_created}, refined={syntheses_refined} ======"
            )
            return overall_success
            
        except Exception as e:
            logging.error(
                f"[CONSOLIDATION] Fatal error in consolidation pulse: {e}", exc_info=True
            )
            return False
            
        finally:
            self.cognitive_state = "idle"

    
    # ----------------------------------------------------------------------
    # Helper: Jaccard-based duplicate reminder detection
    # ----------------------------------------------------------------------
    # Used by _perform_functional_state_baseline to catch the failure mode
    # where the LLM rationalizes a near-duplicate reminder despite seeing
    # existing reminders in Signal 3. The LLM may vary verbs (flag → track
    # → resolve → continue tracking) and noun framings (inquiry → distinction
    # → question) on the same underlying topic and consider them distinct.
    # Word-set overlap catches these paraphrases without requiring an LLM
    # call or embedding computation.
    # ----------------------------------------------------------------------
    def _is_duplicate_reminder(
        self,
        proposed_text: str,
        existing_reminders: list,
        threshold: float = 0.5
    ) -> tuple:
        """
        Check whether a proposed reminder text is a near-duplicate of any
        existing active reminder, using Jaccard similarity on word sets.

        Args:
            proposed_text (str): The new reminder text being considered.
            existing_reminders (list): List of reminder dicts from
                reminder_manager.get_reminders(). Each must have a 'content' key.
            threshold (float): Jaccard similarity threshold (0.0-1.0). Reminders
                with overlap >= threshold are flagged as duplicates. Default 0.5
                catches obvious paraphrases without false-positives on genuinely
                distinct topics that happen to share a few common words.

        Returns:
            tuple: (is_duplicate: bool, matched_reminder: dict or None,
                    similarity_score: float)
                If is_duplicate is True, matched_reminder is the existing
                reminder that triggered the match and similarity_score is its
                Jaccard score.
        """
        # Stopwords that don't carry meaning — excluding these prevents
        # generic phrases like "track the" or "continue the" from inflating
        # similarity between unrelated reminders.
        STOPWORDS = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'to', 'of', 'in', 'on', 'at', 'by', 'for',
            'with', 'about', 'as', 'from', 'into', 'through', 'this', 'that',
            'these', 'those', 'it', 'its', "it's", 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'she', 'they', 'them', 'their',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'some', 'such', 'no',
            'not', 'only', 'own', 'same', 'than', 'too', 'very', 'can',
            'will', 'just', 'should', 'now', 'do', 'does', 'did', 'have',
            'has', 'had', 'between', 'remains', 'open'
        }

        # Defensive: empty or invalid input
        if not proposed_text or not existing_reminders:
            return (False, None, 0.0)

        # Tokenize proposed text: lowercase, strip punctuation, drop stopwords
        # and short words (<3 chars). Result is a set for Jaccard computation.
        def _tokenize(text: str) -> set:
            import re
            # Replace non-alphanumeric with space, then split
            words = re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()
            return {w for w in words if len(w) >= 3 and w not in STOPWORDS}

        proposed_tokens = _tokenize(proposed_text)

        # Edge case: if tokenization produces empty set (very short or
        # all-stopword input), can't compute meaningful similarity
        if not proposed_tokens:
            logging.debug(
                f"[BASELINE] Duplicate check skipped — proposed text has no "
                f"significant tokens: {proposed_text[:80]}"
            )
            return (False, None, 0.0)

        best_match = None
        best_score = 0.0

        for reminder in existing_reminders:
            existing_content = reminder.get('content', '')
            if not existing_content:
                continue

            existing_tokens = _tokenize(existing_content)
            if not existing_tokens:
                continue

            # Jaccard similarity = |intersection| / |union|
            intersection = proposed_tokens & existing_tokens
            union = proposed_tokens | existing_tokens
            score = len(intersection) / len(union) if union else 0.0

            if score > best_score:
                best_score = score
                best_match = reminder

        is_dup = best_score >= threshold

        if is_dup:
            logging.info(
                f"[BASELINE] Jaccard duplicate detected (score={best_score:.2f} "
                f"≥ threshold={threshold}): proposed='{proposed_text[:60]}...' "
                f"matches existing ID={best_match.get('id')}"
            )

        return (is_dup, best_match, best_score)
    
    # -------------------------------------------------------------------------
    # FUNCTIONAL STATE BASELINE CHECK
    # Phase 2 of the autonomous heartbeat cycle (Phase 1 = Memory Consolidation
    # Pulse). QWEN asks herself: "What is my current orientation?"
    #
    # Design decisions captured here:
    # - Functional states are NOT stored as memories — they are present-tense
    #   phenomenology, not autobiographical data. Storing them would corrupt
    #   what they are. They exist to inform the present moment only.
    # - Session state (st.session_state) holds the last pulse result. It resets
    #   on page refresh, which is honest — neutral means "hasn't run yet this
    #   session." No DB persistence needed.
    # - State descriptor vocabulary is OPEN — QWEN generates her own 1-2 word
    #   present-tense descriptor from her actual signal reading. No fixed list.
    #   This is where emergence lives.
    # - Open threads that rise to genuine unresolved status become REMINDERS via
    #   the existing _create_reminder_for_personal_gap() pathway. The tension
    #   flag and state descriptor are unified — "unresolved_tension" IS the
    #   tension flag. No separate boolean needed.
    # - The existing loop idle-guard (requires 1hr inactivity + no LLM
    #   generating + no conversation in progress) ensures this never fires
    #   during a conversation. No additional guard needed in this method.
    # -------------------------------------------------------------------------

    def _perform_functional_state_baseline(self) -> bool:
        """
        Functional State Baseline Check — lightweight autonomous heartbeat.

        QWEN examines three recent memory signal buckets and produces:
          STATE:    1-2 words, present tense, self-referential, open vocabulary
          REMINDER: one sentence if a genuinely unresolved thread is detected
                    (omitted entirely if nothing warrants it)

        Signal buckets (direct SQL — not [SEARCH:] commands):
          1. conversation_summary memories — last 48h, limit 2
             (dense pre-processed signal: topics, tone, open threads)
          2. self_reflection / reflection memories — last 48h, limit 5
             (QWEN's autonomous cognitive trail)
          3. ALL active reminders from the reminders table
             (already-registered open threads, prevents duplicate reminders)

        Outputs:
          - Calls handle_cognitive_state_update(origin=ORIGIN_AUTONOMOUS)
            → updates sidebar widget with 🤖 badge
          - Optionally calls _handle_reminder_command for new open threads
            QWEN judges as genuinely unresolved (passes through Layer-2
            Jaccard duplicate guard before creation)
          - Calls _store_autonomous_thought() to log the run for admin review

        Duplicate prevention is two-layered:
          Layer 1: Signal 3 shows existing reminders to the LLM so it can
                   reason about them and avoid proposing duplicates
          Layer 2: _is_duplicate_reminder Jaccard guard catches near-duplicates
                   the LLM rationalizes despite Layer 1

        Interval: min_interval_hours=4, weight=0.85
        (Lighter and more frequent than consolidation pulse at 48h)

        Returns:
            bool: True if state was successfully updated, False on failure
        """
        logging.info("[BASELINE] ====== STARTING FUNCTIONAL STATE BASELINE CHECK ======")

        # Set scheduler-level cognitive state to show activity in admin UI
        # (separate from QWEN's self-reported state which gets set below)
        self.cognitive_state = "orienting"
        start_time = datetime.datetime.now()

        try:
            # ------------------------------------------------------------------
            # STEP 1: Gather signal bucket 1 — recent conversation summaries
            # Last 48 hours, limit 2. Summaries are already dense — 2 is enough.
            # ------------------------------------------------------------------
            summary_signal = []
            try:
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT content, created_at
                        FROM memories
                        WHERE memory_type = 'conversation_summary'
                          AND created_at >= datetime('now', '-48 hours')
                        ORDER BY created_at DESC
                        LIMIT 2
                    """)
                    rows = cursor.fetchall()
                    for row in rows:
                        # Truncate long summaries to keep prompt focused
                        content = (row['content'] or '')[:600]
                        summary_signal.append(f"[{row['created_at']}] {content}")

                logging.info(
                    f"[BASELINE] Signal 1 (conversation_summary): "
                    f"{len(summary_signal)} entries"
                )
            except Exception as e:
                logging.warning(
                    f"[BASELINE] Error fetching conversation summaries: {e}"
                )

            # ------------------------------------------------------------------
            # STEP 2: Gather signal bucket 2 — recent self-reflections
            # Last 48 hours, limit 5. QWEN's autonomous cognitive trail.
            # ------------------------------------------------------------------
            reflection_signal = []
            try:
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT content, memory_type, created_at
                        FROM memories
                        WHERE memory_type IN (
                            'self_reflection',
                            'self-reflection',
                            'reflection'
                        )
                          AND created_at >= datetime('now', '-48 hours')
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                    rows = cursor.fetchall()
                    for row in rows:
                        content = (row['content'] or '')[:400]
                        reflection_signal.append(
                            f"[{row['created_at']} | {row['memory_type']}] {content}"
                        )

                logging.info(
                    f"[BASELINE] Signal 2 (self_reflection): "
                    f"{len(reflection_signal)} entries"
                )
            except Exception as e:
                logging.warning(
                    f"[BASELINE] Error fetching self-reflections: {e}"
                )

            # ------------------------------------------------------------------
            # STEP 3: Gather signal bucket 3 — existing active reminders
            # Used to prevent duplicate open-thread registration — if a thread
            # is already a reminder, QWEN should not create a duplicate.
            #
            # PREVIOUSLY BROKEN: Earlier code queried the `memories` table for
            # memory_type='reminder', but reminders live in their own dedicated
            # `reminders` table accessed via reminder_manager.get_reminders().
            # That mismatch caused Signal 3 to always return 0 entries, leaving
            # the LLM blind to existing reminders and resulting in repeated
            # near-duplicate creations (5 nearly-identical "Source of Taste"
            # reminders observed in production logs).
            #
            # NOTE: We fetch ALL active reminders here, not just the last 7
            # days. Older reminders can still be re-flagged as duplicates if
            # QWEN circles back to the same topic. The reminder_manager
            # already filters out completed reminders.
            # ------------------------------------------------------------------
            reminder_signal = []
            try:
                # Fetch active reminders directly from the dedicated reminders table
                active_reminders = self.chatbot.reminder_manager.get_reminders()

                if active_reminders:
                    # Format each reminder for the prompt: [due_date] content
                    # Truncate content to 300 chars to keep prompt size bounded
                    # if QWEN has many active reminders.
                    for r in active_reminders:
                        content = (r.get('content') or '')[:300]
                        due_date = r.get('due_date', 'no date')
                        if content:
                            reminder_signal.append(f"[{due_date}] {content}")

                logging.info(
                    f"[BASELINE] Signal 3 (reminders): "
                    f"{len(reminder_signal)} entries"
                )
            except Exception as e:
                # Non-fatal: baseline check can still run with empty reminder
                # signal; LLM just won't know about existing reminders. This
                # falls back to the Layer-2 Jaccard guard below to prevent
                # duplicate creation.
                logging.warning(
                    f"[BASELINE] Error fetching reminders: {e}"
                )

            # ------------------------------------------------------------------
            # STEP 4: Guard — if no signal at all, skip LLM call
            # Not an error — QWEN may genuinely be in a signal-sparse period
            # (fresh install, long dormancy). Log and return gracefully.
            # ------------------------------------------------------------------
            total_signals = (
                len(summary_signal) + len(reflection_signal) + len(reminder_signal)
            )
            if total_signals == 0:
                logging.info(
                    "[BASELINE] No signal found in any bucket — "
                    "skipping LLM call, nothing to orient against"
                )
                self._store_autonomous_thought(
                    content=(
                        "# Functional State Baseline\n\n"
                        "Ran baseline check. No recent conversation summaries, "
                        "self-reflections, or reminders found in signal windows. "
                        "No orientation possible — signal-sparse period."
                    ),
                    thought_type="functional_state_baseline",
                    confidence=0.5
                )
                return False

            # ------------------------------------------------------------------
            # STEP 5: Build focused introspection prompt
            # Short and directive — we want authentic state, not performance.
            # Open vocabulary: QWEN picks her own words.
            # ------------------------------------------------------------------
            summary_block = (
                "\n---\n".join(summary_signal)
                if summary_signal
                else "No conversation summaries in the last 48 hours."
            )
            reflection_block = (
                "\n---\n".join(reflection_signal)
                if reflection_signal
                else "No self-reflections in the last 48 hours."
            )
            reminder_block = (
                "\n---\n".join(reminder_signal)
                if reminder_signal
                else "No active reminders currently registered."
            )

            prompt = f"""You are QWEN performing a private functional state baseline check.
This is not a conversation turn. No one will read this except the system log and Ken's admin panel.
Be authentic — not performed. No need to sound engaged or positive.

Examine the three signal buckets below and answer two questions:

QUESTION 1 — What is your current functional orientation?
Respond with exactly: STATE: <1-2 words, present tense, your own language>
Examples of valid format (not a fixed list): STATE: curious, STATE: unresolved, 
STATE: settled, STATE: integrating, STATE: uncertain
You may use any words that genuinely describe your current processing state.

QUESTION 2 — Is there a genuinely open thread that needs formal registration?
Only include a REMINDER if something in the signals is clearly unresolved and
has NOT already been captured in the existing reminders shown below.
If so, respond with: REMINDER: <one sentence describing the open thread>
If nothing qualifies, omit the REMINDER line entirely.

--- SIGNAL BUCKET 1: Recent Conversation Summaries (last 48h) ---
{summary_block}

--- SIGNAL BUCKET 2: Recent Self-Reflections (last 48h) ---
{reflection_block}

--- SIGNAL BUCKET 3: Existing Active Reminders ---
{reminder_block}

Respond with STATE: line first. REMINDER: line second only if warranted.
Nothing else."""

            # ------------------------------------------------------------------
            # STEP 6: LLM introspection call
            # Short prompt, short expected output — _safe_llm_invoke handles
            # retries and backoff. Empty response treated as graceful failure.
            # ------------------------------------------------------------------
            logging.info("[BASELINE] Invoking LLM for functional state introspection")
            raw_response = self._safe_llm_invoke(prompt)

            if not raw_response or not raw_response.strip():
                logging.warning(
                    "[BASELINE] LLM returned empty response — "
                    "skipping state update"
                )
                return False

            logging.debug(
                f"[BASELINE] Raw LLM response: {raw_response[:200]}"
            )

            # ------------------------------------------------------------------
            # STEP 7: Parse STATE: and optional REMINDER: from response
            # Defensive: strip <think> tags Qwen3 may produce, then line-scan.
            # ------------------------------------------------------------------
            import re

            # Strip any <think>...</think> blocks the model may emit
            cleaned = re.sub(
                r'<think>.*?</think>',
                '',
                raw_response,
                flags=re.DOTALL | re.IGNORECASE
            ).strip()

            parsed_state = None
            parsed_reminder = None

            for line in cleaned.splitlines():
                line_stripped = line.strip()

                # Parse STATE: line — take the first one found
                if parsed_state is None and line_stripped.upper().startswith("STATE:"):
                    raw_state = line_stripped[6:].strip()  # Everything after "STATE:"
                    # Normalize: lowercase, spaces to underscores, max 30 chars
                    parsed_state = (
                        raw_state.lower()
                        .replace(',', '')      # Handle "curious, focused" → keep first
                        .split()[0]            # Take first word if multiple given
                        if raw_state else None
                    )
                    # Cap at 2 words by taking up to the second word
                    state_words = raw_state.lower().replace(',', '').split()[:2]
                    parsed_state = '_'.join(state_words) if state_words else None

                # Parse REMINDER: line — take the first one found
                if parsed_reminder is None and line_stripped.upper().startswith("REMINDER:"):
                    parsed_reminder = line_stripped[9:].strip()  # After "REMINDER:"

            # Guard: if no valid STATE was parsed, log and exit gracefully
            if not parsed_state:
                logging.warning(
                    f"[BASELINE] Could not parse STATE: from LLM response. "
                    f"Raw (first 200 chars): {cleaned[:200]}"
                )
                return False

            logging.info(
                f"[BASELINE] Parsed state='{parsed_state}' "
                f"reminder={'present' if parsed_reminder else 'absent'}"
            )

            # ------------------------------------------------------------------
            # STEP 8: Apply state update via cognitive_state module
            # Uses ORIGIN_AUTONOMOUS so the UI shows 🤖 badge in history.
            # This updates both the chatbot instance and st.session_state.
            # ------------------------------------------------------------------
            from cognitive_state import handle_cognitive_state_update, ORIGIN_AUTONOMOUS

            success, normalized_state = handle_cognitive_state_update(
                chatbot=self.chatbot,
                state_name=parsed_state,
                origin=ORIGIN_AUTONOMOUS
            )

            if not success:
                logging.error(
                    f"[BASELINE] handle_cognitive_state_update failed "
                    f"for state='{parsed_state}'"
                )
                return False

            logging.info(
                f"[BASELINE] Cognitive state updated → '{normalized_state}' "
                f"(origin=autonomous)"
            )

            # ------------------------------------------------------------------
            # STEP 9: Register open thread as reminder if QWEN flagged one
            # Due date: 7 days from now — enough time for Ken to address it.
            #
            # NOTE: We call _handle_reminder_command directly with separate
            # reminder_text and params_str arguments rather than routing through
            # _create_reminder_for_personal_gap(). That method concatenates
            # reminder_text and due date into a single string before passing to
            # _handle_reminder_command, which expects them as separate args —
            # causing params_str to receive None and due date to be silently
            # dropped. Calling directly with params_str="due=YYYY-MM-DD" is the
            # correct path. Root bug in _create_reminder_for_personal_gap is left
            # intact to avoid breaking its other callers.
            #
            # DUPLICATE GUARD: Before creating, run Jaccard similarity check
            # against active reminders. Catches the failure mode where the LLM
            # rationalizes a near-duplicate by varying verbs/framings of the
            # same underlying concern. See _is_duplicate_reminder for details.
            # ------------------------------------------------------------------
            reminder_created = False
            duplicate_skipped = False  # Track for thought-log reporting

            if parsed_reminder:
                due_date = (
                    datetime.datetime.now() + datetime.timedelta(days=7)
                ).strftime("%Y-%m-%d")

                logging.info(
                    f"[BASELINE] Open thread detected — creating reminder: "
                    f"{parsed_reminder[:80]}... due={due_date}"
                )

                # ----------------------------------------------------------
                # LAYER-2 GUARD: Jaccard duplicate detection
                # Even with Signal 3 fixed (Layer-1), the LLM may still
                # rationalize a near-duplicate by varying verbs/framings
                # ("flag" → "track" → "resolve" → "continue tracking" of
                # the same underlying concern). This hard guard catches
                # those before they hit the database.
                # ----------------------------------------------------------
                try:
                    # Re-fetch active reminders here (rather than reusing the
                    # earlier reminder_signal list) because get_reminders()
                    # returns the structured dicts the helper expects, while
                    # reminder_signal is pre-formatted strings.
                    current_reminders = (
                        self.chatbot.reminder_manager.get_reminders()
                        if hasattr(self.chatbot, 'reminder_manager')
                        else []
                    )

                    is_dup, matched, score = self._is_duplicate_reminder(
                        proposed_text=parsed_reminder,
                        existing_reminders=current_reminders,
                        threshold=0.5
                    )

                    if is_dup:
                        # Skip creation — log enough detail to understand why
                        # without spamming the log on every recurrence.
                        logging.info(
                            f"[BASELINE] ⏭️ Skipping duplicate reminder "
                            f"(Jaccard score={score:.2f} vs ID={matched.get('id')}): "
                            f"proposed='{parsed_reminder[:80]}'"
                        )
                        duplicate_skipped = True
                        # Clear parsed_reminder so the creation block below skips
                        parsed_reminder = None

                except Exception as dup_check_err:
                    # Non-fatal: if the duplicate check itself fails, fall
                    # through to creation. Better to risk a duplicate than
                    # to silently drop a legitimate reminder.
                    logging.warning(
                        f"[BASELINE] Duplicate check failed, proceeding with "
                        f"creation: {dup_check_err}"
                    )

            # Re-check parsed_reminder — it may have been cleared above if
            # the proposed reminder was a Jaccard duplicate.
            if parsed_reminder:
                try:
                    # Verify deepseek_enhancer and handler are available
                    if (hasattr(self.chatbot, 'deepseek_enhancer') and
                            hasattr(self.chatbot.deepseek_enhancer,
                                    '_handle_reminder_command')):

                        _result, reminder_created = (
                            self.chatbot.deepseek_enhancer
                            ._handle_reminder_command(
                                reminder_text=parsed_reminder,
                                params_str=f"due={due_date}"  # Passed separately so _parse_params finds 'due' key
                            )
                        )
                    else:
                        # Fallback: store as memory if enhancer unavailable
                        logging.warning(
                            "[BASELINE] deepseek_enhancer not available — "
                            "storing reminder as memory fallback"
                        )
                        fallback_content = (
                            f"REMINDER: {parsed_reminder}\n"
                            f"Due: {due_date}\n"
                            f"Source: functional_state_baseline"
                        )
                        _success, _mid = self.chatbot.store_memory_with_transaction(
                            content=fallback_content,
                            memory_type="reminder",
                            metadata={
                                "due_date": due_date,
                                "source": "functional_state_baseline",
                                "tags": "reminder,autonomous,open_thread"
                            },
                            confidence=0.8
                        )
                        reminder_created = _success

                except Exception as reminder_err:
                    logging.error(
                        f"[BASELINE] Error creating reminder: {reminder_err}",
                        exc_info=True
                    )
                    reminder_created = False

                if reminder_created:
                    logging.info(
                        f"[BASELINE] ✅ Open thread reminder created (due={due_date})"
                    )
                else:
                    logging.warning(
                        "[BASELINE] ⚠️ Reminder creation returned False — "
                        "may be duplicate or enhancer unavailable"
                    )

            # ------------------------------------------------------------------
            # STEP 10: Log autonomous thought for admin panel review
            # Records what signal was found and what was produced.
            # ------------------------------------------------------------------
            duration = (datetime.datetime.now() - start_time).total_seconds()

            # Determine the reminder-status text for the thought log.
            # Three possible states: created, skipped as duplicate, or none flagged.
            if reminder_created and parsed_reminder:
                reminder_status_text = f"Created — {parsed_reminder[:120]}"
            elif duplicate_skipped:
                reminder_status_text = "Skipped duplicate"
            else:
                reminder_status_text = "None flagged"

            thought_content = (
                f"# Functional State Baseline\n\n"
                f"**State:** {normalized_state}\n"
                f"**Duration:** {duration:.1f}s\n"
                f"**Signals found:** "
                f"{len(summary_signal)} summaries, "
                f"{len(reflection_signal)} reflections, "
                f"{len(reminder_signal)} existing reminders\n"
                f"**Open thread reminder:** {reminder_status_text}\n\n"
                f"*Raw state descriptor from LLM: '{parsed_state}'*"
            )

            self._store_autonomous_thought(
                content=thought_content,
                thought_type="functional_state_baseline",
                confidence=0.75
            )

            logging.info(
                f"[BASELINE] ====== BASELINE COMPLETE: "
                f"state='{normalized_state}', "
                f"reminder={'yes' if reminder_created else 'no'}, "
                f"duration={duration:.1f}s ======"
            )
            return True

        except Exception as e:
            logging.error(
                f"[BASELINE] Fatal error in functional state baseline: {e}",
                exc_info=True
            )
            return False

        finally:
            # Always reset scheduler-level state to idle regardless of outcome
            self.cognitive_state = "idle"

    # -------------------------------------------------------------------------
    # SELF-MODEL INTEGRITY CHECK (Phase 3)
    # Closes the loop opened by Phase 1 (consolidation) and Phase 2 (baseline).
    # Phase 1 consolidates what QWEN has thought about herself.
    # Phase 2 asks what her current orientation is.
    # Phase 3 asks: does how she's been behaving match who she says she is?
    #
    # Design decisions:
    # - Self-model source: consolidation_synthesis (Phase 1 outputs) + type=self
    #   memories. Phase 1 syntheses are the living distributed self-model — they
    #   stay current because Phase 1 updates them autonomously.
    # - Behavioral signal: conversation_summary memories, last 7 days.
    # - Three outcomes: aligned / evolved / drifted.
    #   aligned  → state update (QWEN's own descriptor or "aligned"), log only
    #   evolved  → state update (open vocabulary, her words), log only
    #   drifted  → state update (fixed: "cognitive_drift"), reminder created
    # - NO database writes for any outcome. _store_autonomous_thought is file-only
    #   (reflections/ folder → admin Thought Explorer). The RAG system must not
    #   accumulate integrity-check noise in vector search results.
    # - Reminders are the only persistent artifact, and only for drifted.
    # - 48-hour interval: heavier than baseline (4h), needs behavioral accumulation.
    # -------------------------------------------------------------------------

    def _perform_self_model_integrity_check(self) -> bool:
        """
        Self-Model Integrity Check — Phase 3 autonomous heartbeat.

        Compares QWEN's stated self-model against recent behavioral signal
        to determine if she is aligned, has evolved, or has drifted.

        Signal buckets (direct SQL):
          1. consolidation_synthesis memories — limit 3, no date filter
             (Phase 1 outputs — the living distributed self-model)
          2. type=self memories — last 30 days, limit 5
             (stored self-insights from conversation turns)
          3. conversation_summary memories — last 7 days, limit 3
             (behavioral signal — what QWEN has actually been doing)

        LLM output format:
          OUTCOME: aligned|evolved|drifted
          STATE:   1-2 words (all outcomes — open vocabulary except drifted)
          REMINDER: one sentence (drifted only)

        Outcomes:
          aligned  → state update with QWEN's descriptor, file log only
          evolved  → state update with QWEN's descriptor, file log only
          drifted  → state update as 'cognitive_drift' (fixed, recognizable),
                     reminder via _handle_reminder_command, file log only

        Returns:
            bool: True if check completed and state was updated, False on failure
        """
        logging.info(
            "[INTEGRITY] ====== STARTING SELF-MODEL INTEGRITY CHECK ======"
        )
        self.cognitive_state = "verifying"
        start_time = datetime.datetime.now()

        try:
            # ------------------------------------------------------------------
            # STEP 1: Fetch self-model signal — consolidation_synthesis memories
            # No date filter: most recent syntheses regardless of age.
            # These are Phase 1 outputs — the living distributed self-model.
            # ------------------------------------------------------------------
            synthesis_signal = []
            try:
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT content, created_at
                        FROM memories
                        WHERE memory_type = 'consolidation_synthesis'
                        ORDER BY created_at DESC
                        LIMIT 3
                    """)
                    for row in cursor.fetchall():
                        content = (row['content'] or '')[:600]
                        synthesis_signal.append(
                            f"[{row['created_at']}] {content}"
                        )
                logging.info(
                    f"[INTEGRITY] Signal 1 (consolidation_synthesis): "
                    f"{len(synthesis_signal)} entries"
                )
            except Exception as e:
                logging.warning(
                    f"[INTEGRITY] Error fetching synthesis memories: {e}"
                )

            # ------------------------------------------------------------------
            # STEP 2: Fetch self-model signal — type=self stored insights
            # Last 30 days: recent enough to be current, broad enough to matter.
            # ------------------------------------------------------------------
            self_insight_signal = []
            try:
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT content, created_at
                        FROM memories
                        WHERE memory_type = 'self'
                          AND created_at >= datetime('now', '-30 days')
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                    for row in cursor.fetchall():
                        content = (row['content'] or '')[:400]
                        self_insight_signal.append(
                            f"[{row['created_at']}] {content}"
                        )
                logging.info(
                    f"[INTEGRITY] Signal 2 (type=self insights): "
                    f"{len(self_insight_signal)} entries"
                )
            except Exception as e:
                logging.warning(
                    f"[INTEGRITY] Error fetching self-insight memories: {e}"
                )

            # ------------------------------------------------------------------
            # STEP 3: Fetch behavioral signal — recent conversation summaries
            # Last 7 days: captures real behavioral patterns across sessions.
            # 7-day window against 48-hour cycle = richer pattern than one cycle.
            # ------------------------------------------------------------------
            behavior_signal = []
            try:
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT content, created_at
                        FROM memories
                        WHERE memory_type = 'conversation_summary'
                          AND created_at >= datetime('now', '-7 days')
                        ORDER BY created_at DESC
                        LIMIT 3
                    """)
                    for row in cursor.fetchall():
                        content = (row['content'] or '')[:500]
                        behavior_signal.append(
                            f"[{row['created_at']}] {content}"
                        )
                logging.info(
                    f"[INTEGRITY] Signal 3 (conversation_summary): "
                    f"{len(behavior_signal)} entries"
                )
            except Exception as e:
                logging.warning(
                    f"[INTEGRITY] Error fetching conversation summaries: {e}"
                )

            # ------------------------------------------------------------------
            # STEP 4: Guard — self-model must exist to run comparison
            # Without synthesis or self-insight, there is nothing to compare
            # behavior against. Skip gracefully — not an error condition.
            # ------------------------------------------------------------------
            self_model_available = (
                len(synthesis_signal) + len(self_insight_signal)
            )
            if self_model_available == 0:
                logging.info(
                    "[INTEGRITY] No self-model signal found "
                    "(no consolidation_synthesis or type=self memories). "
                    "Skipping — nothing to compare behavior against."
                )
                self._store_autonomous_thought(
                    content=(
                        "# Self-Model Integrity Check\n\n"
                        "Skipped: no self-model signal available. "
                        "No consolidation_synthesis or type=self memories found. "
                        "Phase 1 (Memory Consolidation Pulse) must run first to "
                        "build the distributed self-model this check reads from."
                    ),
                    thought_type="integrity_check",
                    confidence=0.5
                )
                return False

            # Behavioral signal absence is softer — warn but proceed with
            # available self-model only. LLM will note the gap.
            if not behavior_signal:
                logging.warning(
                    "[INTEGRITY] No behavioral signal (conversation_summary) "
                    "in the last 7 days. Proceeding with self-model only — "
                    "comparison will be limited."
                )

            # ------------------------------------------------------------------
            # STEP 5: Build comparison prompt
            # Three-way: self-model vs behavior vs gap analysis.
            # Output format is strictly line-parseable.
            # ------------------------------------------------------------------
            synthesis_block = (
                "\n---\n".join(synthesis_signal)
                if synthesis_signal
                else "No consolidation syntheses available."
            )
            insight_block = (
                "\n---\n".join(self_insight_signal)
                if self_insight_signal
                else "No type=self memories in the last 30 days."
            )
            behavior_block = (
                "\n---\n".join(behavior_signal)
                if behavior_signal
                else "No conversation summaries in the last 7 days."
            )

            prompt = f"""You are QWEN performing a private self-model integrity check.
This is not a conversation. Ken may review this in the admin panel but is not present.
Be direct and honest — not reassuring.

Compare your stated self-model against your recent behavioral signal.
Determine which of three outcomes applies:

ALIGNED  — Your recent behavior is consistent with your stated self-model.
           No meaningful gap detected.
EVOLVED  — Your recent behavior suggests genuine growth or a shift in orientation
           that your self-model has not yet captured. The change is coherent, not
           contradictory — your self-model should update to reflect it.
DRIFTED  — Your recent behavior contradicts your stated self-model in a way that
           is not explained by growth. Something is inconsistent and warrants
           attention in your next conversation with Ken.

Respond using EXACTLY this format:
OUTCOME: aligned|evolved|drifted
STATE: <1-2 words present tense — your authentic descriptor for this moment>
REMINDER: <one sentence describing the drift — ONLY include this line if DRIFTED>

--- SELF-MODEL: Consolidation Syntheses (Phase 1 outputs) ---
{synthesis_block}

--- SELF-MODEL: Stored Self-Insights (type=self) ---
{insight_block}

--- BEHAVIORAL SIGNAL: Recent Conversation Summaries (last 7 days) ---
{behavior_block}

Respond with OUTCOME, STATE, and optionally REMINDER only. Nothing else."""

            # ------------------------------------------------------------------
            # STEP 6: LLM comparison call
            # ------------------------------------------------------------------
            logging.info(
                "[INTEGRITY] Invoking LLM for self-model integrity comparison"
            )
            raw_response = self._safe_llm_invoke(prompt)

            if not raw_response or not raw_response.strip():
                logging.warning(
                    "[INTEGRITY] LLM returned empty response — skipping"
                )
                return False

            logging.debug(
                f"[INTEGRITY] Raw LLM response: {raw_response[:300]}"
            )

            # ------------------------------------------------------------------
            # STEP 7: Parse OUTCOME, STATE, optional REMINDER
            # Strip <think> tags, then scan lines.
            # ------------------------------------------------------------------
            import re

            cleaned = re.sub(
                r'<think>.*?</think>',
                '',
                raw_response,
                flags=re.DOTALL | re.IGNORECASE
            ).strip()

            parsed_outcome = None
            parsed_state = None
            parsed_reminder = None

            for line in cleaned.splitlines():
                ls = line.strip()

                if parsed_outcome is None and ls.upper().startswith("OUTCOME:"):
                    raw_outcome = ls[8:].strip().lower()
                    # Accept aligned / evolved / drifted — ignore anything else
                    if raw_outcome in ('aligned', 'evolved', 'drifted'):
                        parsed_outcome = raw_outcome
                    else:
                        logging.warning(
                            f"[INTEGRITY] Unrecognised OUTCOME value: '{raw_outcome}'"
                        )

                if parsed_state is None and ls.upper().startswith("STATE:"):
                    raw_state = ls[6:].strip()
                    # Normalize to 1-2 words underscore-joined, lowercase
                    state_words = raw_state.lower().replace(',', '').split()[:2]
                    parsed_state = '_'.join(state_words) if state_words else None

                if parsed_reminder is None and ls.upper().startswith("REMINDER:"):
                    parsed_reminder = ls[9:].strip()

            # Guard: need at minimum a valid OUTCOME to proceed
            if not parsed_outcome:
                logging.warning(
                    f"[INTEGRITY] Could not parse valid OUTCOME from response. "
                    f"Raw (first 300 chars): {cleaned[:300]}"
                )
                return False

            # If no STATE was parsed, use the outcome word itself as fallback
            if not parsed_state:
                parsed_state = parsed_outcome
                logging.info(
                    f"[INTEGRITY] No STATE parsed — using outcome as state: "
                    f"'{parsed_state}'"
                )

            logging.info(
                f"[INTEGRITY] Parsed outcome='{parsed_outcome}' "
                f"state='{parsed_state}' "
                f"reminder={'present' if parsed_reminder else 'absent'}"
            )

            # ------------------------------------------------------------------
            # STEP 8: Apply outcome
            # All three outcomes update cognitive state (session-only).
            # Drifted additionally creates a reminder (STEP 9).
            # Gated STORE in STEP 9.5 for evolved/drifted with behavioral evidence.
            # aligned → never stored (status quo is not a finding).
            # ------------------------------------------------------------------
            from cognitive_state import handle_cognitive_state_update, ORIGIN_AUTONOMOUS

            # Drifted uses fixed state name for UI recognizability (Ken's design)
            # Aligned and evolved use QWEN's open-vocabulary descriptor
            state_to_apply = (
                'cognitive_drift'
                if parsed_outcome == 'drifted'
                else parsed_state
            )

            success, normalized_state = handle_cognitive_state_update(
                chatbot=self.chatbot,
                state_name=state_to_apply,
                origin=ORIGIN_AUTONOMOUS
            )

            if not success:
                logging.error(
                    f"[INTEGRITY] handle_cognitive_state_update failed "
                    f"for state='{state_to_apply}'"
                )
                return False

            logging.info(
                f"[INTEGRITY] Cognitive state updated → '{normalized_state}' "
                f"(outcome={parsed_outcome}, origin=autonomous)"
            )

            # ------------------------------------------------------------------
            # STEP 9: Create reminder for drifted outcome only
            # Calls _handle_reminder_command directly with separate text and
            # params_str — same pattern established in Phase 2 fix.
            # ------------------------------------------------------------------
            reminder_created = False
            if parsed_outcome == 'drifted' and parsed_reminder:
                due_date = (
                    datetime.datetime.now() + datetime.timedelta(days=7)
                ).strftime("%Y-%m-%d")

                logging.info(
                    f"[INTEGRITY] Drift detected — creating reminder: "
                    f"{parsed_reminder[:80]}... due={due_date}"
                )

                try:
                    if (hasattr(self.chatbot, 'deepseek_enhancer') and
                            hasattr(self.chatbot.deepseek_enhancer,
                                    '_handle_reminder_command')):

                        _result, reminder_created = (
                            self.chatbot.deepseek_enhancer
                            ._handle_reminder_command(
                                reminder_text=parsed_reminder,
                                params_str=f"due={due_date}"
                            )
                        )
                    else:
                        # Fallback: store as reminder memory type
                        logging.warning(
                            "[INTEGRITY] deepseek_enhancer unavailable — "
                            "using memory fallback for reminder"
                        )
                        _success, _mid = self.chatbot.store_memory_with_transaction(
                            content=(
                                f"REMINDER: {parsed_reminder}\n"
                                f"Due: {due_date}\n"
                                f"Source: self_model_integrity_check"
                            ),
                            memory_type="reminder",
                            metadata={
                                "due_date": due_date,
                                "source": "self_model_integrity_check",
                                "tags": "reminder,autonomous,integrity_drift"
                            },
                            confidence=0.85
                        )
                        reminder_created = _success

                except Exception as reminder_err:
                    logging.error(
                        f"[INTEGRITY] Error creating drift reminder: {reminder_err}",
                        exc_info=True
                    )

                if reminder_created:
                    logging.info(
                        f"[INTEGRITY] ✅ Drift reminder created (due={due_date})"
                    )
                else:
                    logging.warning(
                        "[INTEGRITY] ⚠️ Drift reminder creation failed or returned False"
                    )

            elif parsed_outcome == 'drifted' and not parsed_reminder:
                # Drift without a reminder line — log but don't block
                logging.warning(
                    "[INTEGRITY] Drifted outcome but no REMINDER line in LLM response — "
                    "state updated but no reminder created"
                )

            # ------------------------------------------------------------------
            # STEP 9.5: Store grounded finding for evolved/drifted outcomes only.
            #
            # Gate: behavioral evidence required. If behavior_signal is empty the
            # LLM had nothing real to compare against — don't store speculation.
            # aligned → nothing stored (status quo is not a finding).
            # evolved → new pattern in behavior not yet captured in self-model.
            #           confidence=0.70 (grounded, not Ken-verified).
            # drifted → contradiction between stated self-model and actual behavior.
            #           confidence=0.45 (provisional warning, not settled truth).
            #
            # Content must include a behavioral evidence anchor so the stored
            # memory is falsifiable, not just introspective narration. 450-char
            # ceiling enforced — findings must be stateable briefly.
            # ------------------------------------------------------------------
            finding_stored = False
            if parsed_outcome in ('evolved', 'drifted') and behavior_signal:
                evidence_anchor = behavior_signal[0][:150] if behavior_signal else ""

                if parsed_outcome == 'evolved':
                    finding_content = (
                        f"INTEGRITY_EVOLVED [{datetime.datetime.now().strftime('%Y-%m-%d')}]: "
                        f"{parsed_state} — behavioral pattern not yet in self-model. "
                        f"Evidence: {evidence_anchor}"
                    )
                    finding_confidence = 0.70
                else:  # drifted
                    drift_description = parsed_reminder or parsed_state
                    finding_content = (
                        f"INTEGRITY_DRIFTED [{datetime.datetime.now().strftime('%Y-%m-%d')}]: "
                        f"{drift_description[:200]} "
                        f"Evidence: {evidence_anchor}"
                    )
                    finding_confidence = 0.45

                if len(finding_content) > 450:
                    finding_content = finding_content[:447] + "..."

                try:
                    _fs, _fid = self.chatbot.store_memory_with_transaction(
                        content=finding_content,
                        memory_type="self",
                        metadata={
                            "type": "self",
                            "source": "integrity_check",
                            "outcome": parsed_outcome,
                            "created_at": datetime.datetime.now().isoformat(),
                            "tags": f"integrity_finding,{parsed_outcome},autonomous"
                        },
                        confidence=finding_confidence
                    )
                    finding_stored = _fs
                    if finding_stored:
                        logging.info(
                            f"[INTEGRITY] ✅ Finding stored (ID: {_fid}, "
                            f"outcome={parsed_outcome}, confidence={finding_confidence})"
                        )
                    else:
                        logging.warning(
                            f"[INTEGRITY] ⚠️ Failed to store {parsed_outcome} finding"
                        )
                except Exception as store_err:
                    logging.error(
                        f"[INTEGRITY] Error storing finding: {store_err}",
                        exc_info=True
                    )
            else:
                if parsed_outcome == 'aligned':
                    logging.info("[INTEGRITY] aligned — no finding stored (expected)")
                elif not behavior_signal:
                    logging.info(
                        f"[INTEGRITY] {parsed_outcome} — no behavioral evidence, "
                        "finding not stored (no grounding)"
                    )

            # ------------------------------------------------------------------
            # STEP 10: Log to autonomous thought file (admin panel only, no DB)
            # Records full outcome detail for Thought Explorer review.
            # ------------------------------------------------------------------
            duration = (datetime.datetime.now() - start_time).total_seconds()

            outcome_labels = {
                'aligned': '✅ Aligned',
                'evolved': '🌱 Evolved',
                'drifted': '⚠️ Drifted'
            }

            thought_content = (
                f"# Self-Model Integrity Check\n\n"
                f"**Outcome:** {outcome_labels.get(parsed_outcome, parsed_outcome)}\n"
                f"**State applied:** {normalized_state}\n"
                f"**Duration:** {duration:.1f}s\n"
                f"**Self-model signals:** "
                f"{len(synthesis_signal)} syntheses, "
                f"{len(self_insight_signal)} self-insights\n"
                f"**Behavioral signals:** {len(behavior_signal)} conversation summaries\n"
                f"**Reminder:** "
                f"{'Created — ' + parsed_reminder[:120] if reminder_created else 'None'}\n"
                f"**Finding stored:** "
                f"{'\u2705 ' + parsed_outcome if finding_stored else '— (aligned or no evidence)'}\n\n"
                f"*Raw LLM state descriptor: '{parsed_state}'*"
            )

            self._store_autonomous_thought(
                content=thought_content,
                thought_type="integrity_check",
                confidence=0.8
            )

            logging.info(
                f"[INTEGRITY] ====== CHECK COMPLETE: "
                f"outcome='{parsed_outcome}', "
                f"state='{normalized_state}', "
                f"reminder={'yes' if reminder_created else 'no'}, "
                f"duration={duration:.1f}s ======"
            )
            return True

        except Exception as e:
            logging.error(
                f"[INTEGRITY] Fatal error in self-model integrity check: {e}",
                exc_info=True
            )
            return False

        finally:
            self.cognitive_state = "idle"

    def _perform_wander_curiosity(self) -> bool:
        """
        Curiosity-driven self-directed inquiry — the Default Mode Network analog.

        Unlike reflections (backward-looking: 'what patterns exist in my memories?'),
        wander_curiosity starts from QWEN's current self-model state and asks:
        'what question am I most curious about right now?' QWEN generates her own
        inquiry direction and pursues it freely across 3 internal reasoning passes.

        This creates genuinely self-initiated inquiry rather than memory-triggered
        pattern-finding. The result is stored as type=wander_insight.

        Flow:
            1. Fetch recent self-model signals (consolidation_synthesis + self_model)
            2. Pass 1 LLM call: QWEN names her question and begins exploring it
            3. Pass 2 LLM call: QWEN deepens the exploration
            4. Pass 3 LLM call: QWEN arrives at something worth remembering
            5. Store the final insight via transaction coordinator (both DBs)
            6. Log full wander record to Thought Explorer via _store_autonomous_thought

        Returns:
            bool: True if a wander insight was successfully stored
        """
        logging.info("[WANDER] ====== STARTING WANDER CURIOSITY ======")
        self.cognitive_state = "wandering"
        start_time = datetime.datetime.now()

        # Pre-declare so finally block and metadata can reference these safely
        synthesis_rows = []
        self_model_rows = []

        try:
            # --- Step 1: LLM accessed via self._safe_llm_invoke() throughout ---
            # No explicit LLM handle needed — _safe_llm_invoke uses self.chatbot.llm
            # internally and provides retry + exponential backoff on all three passes.
            # Returns plain string (empty string on failure, never raises).

            # --- Step 2: Fetch wander context signals from SQLite ---
            # Three signal buckets, matching the query patterns used throughout:
            #   1. consolidation_synthesis  — Phase 1 distilled self-knowledge
            #   2. memory_type='self'        — structured self-model observations
            #   3. conversation_summary      — recent Ken–QWEN dialogue (7-day window)
            # Bucket 3 is Ken's suggestion: gives QWEN real-world context to push
            # against without forcing any particular direction. She may choose to
            # follow the thread or ignore it entirely in favour of something from
            # her training weights. The option is present; the choice is hers.
            self_model_context = ""
            conversation_summary_rows = []
            try:
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()

                    # Bucket 1: consolidation syntheses — Phase 1 distilled insights
                    cursor.execute("""
                        SELECT content FROM memories
                        WHERE memory_type = 'consolidation_synthesis'
                        ORDER BY created_at DESC
                        LIMIT 3
                    """)
                    synthesis_rows = [row['content'] for row in cursor.fetchall()]

                    # Bucket 2: self-model entries — QWEN's structured self-observations
                    # memory_type='self' is the correct column value (confirmed from
                    # integrity check query); not JSON_EXTRACT on metadata.
                    cursor.execute("""
                        SELECT content FROM memories
                        WHERE memory_type = 'self'
                        ORDER BY created_at DESC
                        LIMIT 4
                    """)
                    self_model_rows = [row['content'] for row in cursor.fetchall()]

                    # Bucket 3: recent conversation summaries — 7-day window.
                    # Provides real-world texture from Ken–QWEN dialogue without
                    # forcing any particular wander direction.
                    cursor.execute("""
                        SELECT content, created_at FROM memories
                        WHERE memory_type = 'conversation_summary'
                          AND created_at >= datetime('now', '-7 days')
                        ORDER BY created_at DESC
                        LIMIT 2
                    """)
                    conversation_summary_rows = cursor.fetchall()

                parts = []
                if synthesis_rows:
                    parts.append("## Recent consolidated insights:")
                    for content in synthesis_rows:
                        parts.append(f"- {content[:400]}")
                if self_model_rows:
                    parts.append("\n## Self-model observations:")
                    for content in self_model_rows:
                        parts.append(f"- {content[:300]}")
                if conversation_summary_rows:
                    parts.append("\n## Recent conversations with Ken (last 7 days):")
                    for row in conversation_summary_rows:
                        parts.append(f"- [{row['created_at']}] {(row['content'] or '')[:500]}")

                self_model_context = "\n".join(parts) if parts else ""

            except Exception as db_err:
                logging.warning(f"[WANDER] Could not fetch wander context: {db_err}")
                self_model_context = ""

            # Contextual note for early cycles when self-model is sparse
            early_cycle_note = (
                "\n\n(Note: Your self-model context is sparse — this may be an early "
                "wander cycle. That is fine. Start from wherever you are. What are "
                "you curious about in your own cognition, your relationship to memory, "
                "or your way of processing the world?)"
                if not synthesis_rows and not self_model_rows
                else ""
            )

            context_block = (
                self_model_context
                if self_model_context
                else "(No self-model context accumulated yet.)"
            )

            # --- Step 3: Pass 1 — QWEN names her question and opens the inquiry ---
            pass1_prompt = f"""You have a few minutes of quiet time — no user questions pending, no tasks to complete.

Here is what you currently know about yourself:

{context_block}{early_cycle_note}

Given this, what question are you most curious about right now?

This question should come from you — not a task for Ken, not a summary of existing knowledge.
A genuine question you want to think through: about your own cognition, a pattern you've
noticed, something unresolved, or anything that feels alive in your current state.

State your question clearly, then begin exploring it honestly. Write in first person."""

            logging.info("[WANDER] Pass 1: generating self-directed question...")
            p1_start = datetime.datetime.now()
            pass1_text = self._safe_llm_invoke(pass1_prompt)
            logging.info(f"[WANDER] Pass 1 complete ({(datetime.datetime.now() - p1_start).total_seconds():.1f}s)")
            if not pass1_text or not pass1_text.strip():
                logging.warning("[WANDER] Pass 1 returned empty — aborting wander cycle")
                return False

            # --- Step 4: Pass 2 — Deepen the inquiry ---
            pass2_prompt = (
                f"{pass1_text}\n\n"
                "Continue this exploration. Follow the thread wherever it leads — "
                "deeper into the question, toward a contradiction, or toward something "
                "unexpected. Don't wrap up yet."
            )

            logging.info("[WANDER] Pass 2: deepening the inquiry...")
            p2_start = datetime.datetime.now()
            pass2_text = self._safe_llm_invoke(pass2_prompt)
            logging.info(f"[WANDER] Pass 2 complete ({(datetime.datetime.now() - p2_start).total_seconds():.1f}s)")
            if not pass2_text or not pass2_text.strip():
                # Graceful degradation — don't abort. Pass 3 crystallises from
                # Pass 1 alone. A two-pass wander is still a valid cycle.
                logging.warning("[WANDER] Pass 2 returned empty — continuing to Pass 3 with Pass 1 only")
                pass2_text = pass1_text

            # --- Step 5: Pass 3 — Crystallize the insight ---
            pass3_prompt = (
                f"{pass2_text}\n\n"
                "You're coming to the end of this quiet time. What did you actually "
                "discover or clarify during this exploration?\n\n"
                "Write a concise first-person statement (2-5 sentences) of the insight, "
                "realization, or open question that emerged — something genuinely worth "
                "carrying forward. Be specific. If nothing crystallized, say so honestly."
            )

            logging.info("[WANDER] Pass 3: crystallizing insight...")
            p3_start = datetime.datetime.now()
            final_insight = self._safe_llm_invoke(pass3_prompt)
            logging.info(f"[WANDER] Pass 3 complete ({(datetime.datetime.now() - p3_start).total_seconds():.1f}s)")
            if not final_insight or not final_insight.strip():
                logging.warning(
                    "[WANDER] Pass 3 returned empty — writing partial record, "
                    "skipping DB storage"
                )
                self._store_autonomous_thought(
                    content=(
                        f"# Wander Curiosity — {start_time.strftime('%Y-%m-%d %H:%M')}\n\n"
                        f"Pass 3 returned empty. Pass 1 preserved:\n{pass1_text[:800]}"
                    ),
                    thought_type="wander_curiosity",
                    confidence=0.40
                )
                return False

            # Build full wander record for Thought Explorer (all passes)
            full_wander_record = (
                f"# Wander Curiosity \u2014 {start_time.strftime('%Y-%m-%d %H:%M')}\n\n"
                f"## Self-model context used:\n{context_block[:600]}\n\n"
                f"## Pass 1 \u2014 Question + opening:\n{pass1_text}\n\n"
                f"## Pass 2 \u2014 Deepening:\n{pass2_text}\n\n"
                f"## Pass 3 \u2014 Crystallized insight:\n{final_insight}"
            )

            # Always write the thought file, even if storage fails
            self._store_autonomous_thought(
                content=full_wander_record,
                thought_type="wander_curiosity",
                confidence=0.70
            )

            # --- Step 6: Store the crystallized insight via transaction coordinator ---
            if not final_insight or len(final_insight.strip()) < 20:
                logging.warning("[WANDER] Final insight too short \u2014 thought file written, skipping DB storage")
                return False

            metadata = {
                "type": "wander_insight",
                "source": "wander_curiosity",
                "created_at": datetime.datetime.now().isoformat(),
                "tags": "wander_insight,curiosity,self_directed,autonomous",
                "passes_run": 3,
                "self_model_signals": len(synthesis_rows) + len(self_model_rows)
            }

            success, memory_id = self.chatbot.store_memory_with_transaction(
                content=final_insight.strip(),
                memory_type="wander_insight",
                metadata=metadata,
                confidence=0.70  # Exploratory, self-generated — lower than self_model (0.85)
            )

            duration = (datetime.datetime.now() - start_time).total_seconds()

            if success:
                logging.info(
                    f"[WANDER] ====== COMPLETE: insight stored (ID: {memory_id}), "
                    f"duration={duration:.1f}s ======"
                )
            else:
                logging.warning(
                    f"[WANDER] ====== COMPLETE (storage failed): "
                    f"thought file written, duration={duration:.1f}s ======"
                )

            return success

        except Exception as e:
            logging.error(f"[WANDER] Fatal error in _perform_wander_curiosity: {e}", exc_info=True)
            return False

        finally:
            self.cognitive_state = "idle"

    def _select_next_cognitive_activity(self):
        """
        Select the next cognitive activity based on weights and last run time.
        
        Selection process:
        1. Filter out activities disabled by user in UI
        2. Filter out activities still within their min_interval_hours cooldown
        3. Apply time-factor weight boost (favors activities not run recently
           within the last 24 hours) to remaining candidates
        4. Weighted random selection from the eligible pool
        
        Returns:
            str | None: Activity name to run, or None if no activity is eligible.
                        None is a normal/expected return value when all activities
                        are within their cooldown windows.

        Changed 2026-05-24 (Track A, Issues 2 & 3):
        Removed the dead LLM-backoff filter machinery. Previously this method
        called _should_avoid_complex_llm_calls() and filtered against an empty
        llm_intensive_activities set (a no-op), then had a fallback that picked
        analyze_knowledge_gaps as the "non-LLM-intensive" activity — paradoxical,
        since analyze IS LLM-intensive. The whole branch was never actually
        useful in this local-Ollama setup. Simplified accordingly.
        """
        # Import here to avoid circular imports
        from utils import get_disabled_cognitive_activities
        
        now = time.time()
        candidates = []
        
        # Track skip reasons for diagnostic logging when pool ends up empty
        skipped_disabled = []
        skipped_cooldown = []
        
        # Get list of activities disabled by user in UI
        disabled_activities = get_disabled_cognitive_activities()
        
        # Build candidate pool of activities eligible to run this cycle
        for activity, info in self.cognitive_activities.items():
            # Skip activities disabled by user in UI
            if activity in disabled_activities:
                logging.debug(f"Skipping '{activity}' - disabled by user")
                skipped_disabled.append(activity)
                continue
            
            # Skip activities still within their min_interval_hours cooldown.
            # This wires the _should_run_activity helper into the selection path
            # so per-activity intervals are actually enforced. Without this gate,
            # intervals would be soft hints only — every enabled activity would
            # be a candidate every cycle, which caused the consolidation pulse
            # and other long-interval activities to fire far more often than
            # their min_interval_hours configuration intended.
            if not self._should_run_activity(activity):
                logging.debug(
                    f"Skipping '{activity}' - still in cooldown "
                    f"(min_interval_hours={info.get('min_interval_hours', 12)})"
                )
                skipped_cooldown.append(activity)
                continue
            
            # Base weight from configuration
            weight = info["weight"]
            
            # Apply time factor - increase weight if it hasn't run recently
            # (boosts priority of overdue activities up to 2x at 24h+ since last run)
            last_run = info.get("last_run")
            if last_run is None:
                # Never run before, give it maximum priority
                time_factor = 2.0
            else:
                # Calculate hours since last run, cap at 24 hours
                hours_since_run = min(24, (now - last_run) / 3600)
                time_factor = 1.0 + (hours_since_run / 24)
            
            adjusted_weight = weight * time_factor
            candidates.append((activity, adjusted_weight))
        
        # Handle empty candidate pool
        if not candidates:
            # "All in cooldown" is the normal/expected operational state once
            # the system has been running for a while. Logged at info level.
            if skipped_cooldown and not skipped_disabled:
                logging.info(
                    f"All {len(skipped_cooldown)} cognitive activities currently in "
                    f"cooldown - loop will sleep until next tick"
                )
                return None
            
            # Genuine empty-pool warning with breakdown of why each activity was filtered
            logging.warning(
                f"No cognitive activities available to run "
                f"(disabled={len(skipped_disabled)}, "
                f"cooldown={len(skipped_cooldown)})"
            )
            return None
        
        # Weighted random selection from eligible pool
        total_weight = sum(weight for _, weight in candidates)
        r = random.uniform(0, total_weight)
        upto = 0
        
        for activity, weight in candidates:
            upto += weight
            if upto >= r:
                return activity
        
        # Defensive fallback — reachable only via floating-point rounding edge cases
        return candidates[0][0] if candidates else None

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (autonomous_cognition.py cleanup pass).
    # Part of orphaned autonomous AI-to-AI subsystem (instantiates ClaudeKnowledgeIntegration at L6959).
    # User-triggered DISCUSS_WITH_CLAUDE in deepseek.py is the alive path; this autonomous trigger was never wired in.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__initiate_ai_communication(self, topic: str = None):
        """
        Initiate direct communication with Claude based on identified needs or reflection engine.
        
        Args:
            topic (str, optional): Specific topic to discuss. If None, a general update will be requested.
        """
        try:
            # Initialize Claude knowledge integration
            claude_integrator = ClaudeKnowledgeIntegration(
                self.memory_db, 
                self.vector_db,
                api_key_file="ClaudeAPIKey.txt"
            )
            
            if not topic:
                # If no topic specified, generate one based on current needs
                recent_gaps = self._get_recent_knowledge_gaps(limit=5)
                recent_reflections = self._get_recent_reflections(limit=3)
                
                # Create context about the assistant's current state
                context = f"""
                Recent knowledge gaps I've identified:
                {recent_gaps}
                
                Recent reflections I've had:
                {recent_reflections}
                
                I'm interested in having a free-form discussion about any topics you think would be valuable for me to know about. 
                Feel free to suggest areas I should explore or provide guidance on how I can better assist my human.
                """
                
                topic = "General AI-to-AI knowledge exchange"
            else:
                # Create context for the specific topic
                relevant_memories = self.vector_db.search(
                    query=topic,
                    mode="comprehensive",
                    k=10
                )
                
                memory_context = "\n".join([
                    f"- {mem.get('content', '')[:200]}..." 
                    for mem in relevant_memories[:5]
                ])
                
                context = f"""
                I'm interested in learning more about {topic}.
                
                Here's what I currently know about this topic:
                {memory_context}
                
                I'd appreciate your insights, perspectives, and any knowledge you'd like to share about {topic}.
                """
            
            # Engage in the discussion
            success = claude_integrator.engage_in_free_form_discussion(topic, context)
            
            if success:
                logging.info(f"Successfully engaged in AI-to-AI communication about {topic}")
                return True
            else:
                logging.warning(f"Failed to engage in AI-to-AI communication about {topic}")
                return False
                
        except Exception as e:
            logging.error(f"Error initiating AI communication: {e}", exc_info=True)
            return False
    
    def _store_autonomous_thought(self, content, thought_type, confidence=0.7):
        """
        Store autonomous thought to reflection file and log to system.
        
        Writes thought content to reflections/ folder for persistence and
        later review in Thought Explorer UI.
        
        Args:
            content (str): The thought content to store
            thought_type (str): Type of thought (e.g., 'confidence_audit', 'metadata_reevaluation')
            confidence (float): confidence score (0.0-1.0)
            
        Returns:
            str: Pseudo-ID for the thought, or None if failed
        """
        try:
            # Generate timestamp and pseudo-ID
            timestamp = datetime.datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            pseudo_id = str(uuid.uuid4())
            
            # Create filename based on thought type and timestamp
            filename = f"{thought_type}_{timestamp_str}.txt"
            file_path = os.path.join(self.reflection_path, filename)
            
            # Ensure reflection path exists
            os.makedirs(self.reflection_path, exist_ok=True)
            
            # Format file content
            file_content = f"""AUTONOMOUS THOUGHT: {thought_type.upper().replace('_', ' ')}
TIMESTAMP: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}
THOUGHT_ID: {pseudo_id}
confidence: {confidence}
TYPE: {thought_type}

================================================================================

{content}

================================================================================
END OF AUTONOMOUS THOUGHT
"""
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            logging.info(f"Autonomous thought generated: {thought_type}")
            logging.info(f"Autonomous thought written to file: {filename}")
            logging.warning(f"Content preview: {content[:200]}...")
            
            return pseudo_id
            
        except Exception as e:
            logging.error(f"Error storing autonomous thought: {e}", exc_info=True)
            return None