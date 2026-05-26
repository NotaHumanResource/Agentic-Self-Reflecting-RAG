# Updated deepseek.py with specialized loggers for search results and command tracking
"""DeepSeek enhancer with memory commands processing and system prompt enhancement."""
# =============================================================================
# Configuration Constants
# =============================================================================
DUPLICATE_SIMILARITY_THRESHOLD = 0.98  # Similarity threshold for near-duplicate detection
ENABLE_NEAR_DUPLICATE_DETECTION = True  # Enable/disable fuzzy duplicate matching
MAX_SEARCHES_PER_RESPONSE = 5  # Maximum number of search commands to execute per response
MAX_STORES_PER_RESPONSE = 5    # Maximum number of store commands to execute per response
# =============================================================================
# STORE Command Feedback Messages
# =============================================================================
STORE_FEEDBACK = {
    # Success message - brief confirmation
    'success': "✅ Successfully Stored: {content_preview}",
    

    # Failure messages - concise with guidance
    'duplicate': "❌ Already exists in {source}. Search first to verify before storing.",
    'recursion': "❌ Recursion detected. 30-second cooldown active.",
    'max_limit': "❌ Store limit (5) reached this turn. Combine related info next turn.",
    'syntax_missing_pipe': "❌ Missing pipe separator. Use: [STORE: content | type=x]",
    'syntax_empty': "❌ Empty content. Use: [STORE: your content here | type=x]",
    'syntax_missing_content': "❌ Content missing. Put content before the pipe: [STORE: content | type=x]",
    'too_short': "❌ Too short ({length} chars). Need 20+ chars with detail.",
    'placeholder': "❌ '{content}' is not valid content. Be specific.",
    'storage_failed': "❌ Storage failed. Try again or alert Ken."
}

import time
import re 
import logging
import os
# import sqlite3  # DEAD CODE TEST 2026-05-17: unused per ruff F401
import datetime
import json
import uuid
# import requests  # DEAD CODE TEST 2026-05-17: unused per ruff F401
import sys
# import io  # DEAD CODE TEST 2026-05-17: unused per ruff F401
# === SEARCH DEDUPLICATION - Prevents logging same search multiple times ===
import hashlib
# import difflib  # DEAD CODE TEST 2026-05-17: unused per ruff F401 + vulture
_logged_searches = set()  # Track logged searches in current session
# from qdrant_client.http import models as rest  # DEAD CODE TEST 2026-05-17: 'rest' alias unused per ruff F401
from typing import Tuple, Dict, Any, List, Optional  # DEAD CODE TEST 2026-05-17: was 'Tuple, Dict, Any, Callable, List, Optional' — Callable unused per ruff F401 + vulture
from lifetime_counters import LifetimeCounters


# --- Set up specialized loggers ---
# Create search results logger
search_results_logger = logging.getLogger('search_results')
search_results_logger.setLevel(logging.INFO)
search_results_logger.propagate = False  # Don't send to parent loggers

# IDEMPOTENCY GUARD — skip handler setup if one is already attached.
# Without this guard, re-executing this module body (Streamlit reruns,
# importlib.reload, or any other re-import path) would accumulate
# handlers on the same singleton logger. Symptoms: each log call emits
# once per attached handler (e.g. duplicate [COMMAND RESULT] lines), and
# for the search logger specifically, a new timestamped log file would
# be created on every re-execution. The guard makes setup safe to run
# repeatedly. setLevel and propagate above are idempotent and stay
# outside the guard since assigning the same value twice is harmless.
if not search_results_logger.handlers:
    # First-time setup detected — proceed with handler creation.
    # The diagnostic warning below fires once per process under normal
    # conditions. If it ever appears twice in a single session log, that
    # tells us something is causing this module's body to re-execute,
    # which is itself a bug worth investigating separately.
    logging.warning(
        "LOGGER_SETUP: search_results_logger handler attaching "
        "(handlers list was empty — first-time setup expected)"
    )

    # Create a timestamped directory for this session
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    search_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_logs")
    os.makedirs(search_results_dir, exist_ok=True)
    search_log_file = os.path.join(search_results_dir, f"search_results_{timestamp}.log")

    # Create file handler with simple formatter for search results
    search_file_handler = logging.FileHandler(search_log_file, encoding='utf-8')
    search_file_handler.setLevel(logging.INFO)
    search_file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    search_file_handler.setFormatter(search_file_formatter)
    search_results_logger.addHandler(search_file_handler)

# Set up a command results logger for better visibility of success/failure
command_logger = logging.getLogger('command_results')
command_logger.setLevel(logging.INFO)
command_logger.propagate = False  # Prevent duplicate logging to root logger

# IDEMPOTENCY GUARD — same bug pattern as search_results_logger above.
# Without this guard, every re-execution of this module's body adds
# another StreamHandler to the same singleton logger. Each [COMMAND
# RESULT] entry is then written once per attached handler, all going
# to stderr (which Streamlit/the batch redirection captures into
# ollama_context.log). This was the source of the doubled command
# result lines observed starting from turn 3 on 2026-05-06.
if not command_logger.handlers:
    # Diagnostic warning — see comment in search_results_logger guard above.
    # Watch for this firing more than once per session in the log.
    logging.warning(
        "LOGGER_SETUP: command_logger handler attaching "
        "(handlers list was empty — first-time setup expected)"
    )

    # Add a handler for command results. This is a StreamHandler (writes
    # to stderr); the main log file captures it via Streamlit's stderr
    # redirection, not via a direct FileHandler.
    command_handler = logging.StreamHandler()
    command_formatter = logging.Formatter('%(asctime)s - [COMMAND RESULT] - %(message)s')
    command_handler.setFormatter(command_formatter)
    command_logger.addHandler(command_handler)

# --- Conditional Streamlit Import ---
st = None  # Default to None if Streamlit is not available/imported
if 'streamlit' in sys.modules:
    try:
        # Import Streamlit itself if its name is in sys.modules
        import streamlit as streamlit_actual
        st = streamlit_actual  # Assign the actual module to 'st'
        logging.debug("DeepSeekEnhancer: Streamlit library found and imported.")
    except ImportError:
        # This case is unlikely if 'streamlit' is in sys.modules, but handle defensively
        logging.warning("DeepSeekEnhancer: Streamlit found in sys.modules but import failed.")
        pass  # st remains None
else:
    logging.debug("DeepSeekEnhancer: Streamlit library not detected in sys.modules.")
# --- End Conditional Streamlit Import ---


class DeepSeekEnhancer:
    """Enhances DeepSeek's capabilities with memory commands and training features."""

    def __init__(self, chatbot):
        """Initialize the DeepSeek enhancer."""
        try:
            self.chatbot = chatbot
            self.vector_db = chatbot.vector_db
            self.memory_db = chatbot.memory_db
            logging.info("About to create LifetimeCounters instance")
            # Use the imported LifetimeCounters class
            self.lifetime_counters = LifetimeCounters()
            logging.info(f"LifetimeCounters instance created: {self.lifetime_counters}")
            
            # ===== RECURSION TRAP DETECTION SYSTEM =====
            # Prevents infinite loops when AI analyzes its own behavior
            self._recursion_detector = {
                'last_store_content': None,
                'last_command_type': None,
                'duplicate_count': 0,
                'max_duplicates': 1,  # Block immediately on 2nd identical command
                'cooldown_until': None,
                'trapped_content_hash': None  # Track content that caused trap
            }
            logging.info("Initialized recursion detection system")
            
            # ===== META-COGNITIVE LOOP PREVENTION =====
            # Prevents infinite loops from self-referential thinking
            self._meta_cognitive_tracker = {
                'depth': 0,
                'max_depth': 2,  # Allow Level 1 and Level 2, block Level 3+
                'current_chain': [],  # Track chain of meta-cognitive operations
                'last_reset': datetime.datetime.now(),
                'reset_interval': 30,  # Reset depth counter after 30 seconds
                'blocked_patterns': [
                    # Level 3+ patterns to block
                    r'reflect.*reflect.*reflect',
                    r'analyze.*analyz.*analyz',
                    r'think.*think.*think',
                    r'consider.*consider.*consider',
                    r'why.*why.*why',
                    r'understand.*understand.*understand'
                ]
            }
            logging.info("Initialized meta-cognitive loop prevention (max depth: 2)")
            # ===== END META-COGNITIVE LOOP PREVENTION =====
            
            self.reflection_interval = datetime.timedelta(hours=12)  # Min time between reflections
            self.last_reflection_time = None
            self.training_mode = True  # Default to training mode to show commands

            # Initialize reflection paths
            self.reflection_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reflections")
            os.makedirs(self.reflection_path, exist_ok=True)
            logging.info(f"DeepSeek enhancer initialized with reflection path: {self.reflection_path}")
        except Exception as e:
            logging.critical(f"Error in DeepSeekEnhancer.__init__: {e}", exc_info=True)

    def _check_recursion_trap(self, content: str, command_type: str) -> bool:
        """
        Detect if we're in a recursion trap by tracking identical commands.
        
        This prevents the AI from getting stuck analyzing its own analysis,
        which can happen with meta-cognitive questions like "why do you get stuck in loops?"
        
        Args:
            content (str): The content of the command being processed
            command_type (str): Type of command (STORE, REFLECT, etc.)
            
        Returns:
            bool: True if command should be BLOCKED (we're in a trap), False if safe to proceed
        """
        try:
            from datetime import datetime, timedelta
            import hashlib
            
            # Create hash of content for efficient comparison
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # === IMPORTANT: Check for content change FIRST ===
            # If content or command type changed, reset detector even during cooldown
            content_changed = (
                content != self._recursion_detector['last_store_content'] or
                command_type != self._recursion_detector['last_command_type']
            )
            
            if content_changed:
                # Content changed - reset detector completely
                if self._recursion_detector['duplicate_count'] > 0:
                    logging.info(
                        f"RECURSION_TRAP: Content changed after {self._recursion_detector['duplicate_count']} "
                        f"duplicates - resetting detector completely"
                    )
                
                # Reset everything including cooldown
                self._recursion_detector['last_store_content'] = content
                self._recursion_detector['last_command_type'] = command_type
                self._recursion_detector['duplicate_count'] = 1
                self._recursion_detector['cooldown_until'] = None
                self._recursion_detector['trapped_content_hash'] = None
                
                return False  # New content is always allowed
            
            # === Content is SAME as last time - check cooldown ===
            if self._recursion_detector['cooldown_until']:
                if datetime.now() < self._recursion_detector['cooldown_until']:
                    cooldown_remaining = (self._recursion_detector['cooldown_until'] - datetime.now()).seconds
                    logging.warning(
                        f"RECURSION_TRAP: In cooldown period ({cooldown_remaining}s remaining), "
                        f"blocking {command_type} command"
                    )
                    return True  # Block the duplicate command
                else:
                    # Cooldown expired - allow this occurrence but keep tracking
                    logging.info("RECURSION_TRAP: Cooldown period ended, allowing command but continuing to track")
                    self._recursion_detector['cooldown_until'] = None
                    # IMPORTANT: Reset duplicate count to 1 (this occurrence)
                    # but KEEP last_store_content to continue tracking
                    self._recursion_detector['duplicate_count'] = 1
                    self._recursion_detector['trapped_content_hash'] = None
                    return False  # Allow the command after cooldown
            
            # === Duplicate content detected (not in cooldown) ===
            self._recursion_detector['duplicate_count'] += 1
            
            logging.warning(
                f"RECURSION_TRAP: Duplicate {command_type} detected "
                f"({self._recursion_detector['duplicate_count']}/{self._recursion_detector['max_duplicates']}): "
                f"{content[:100]}..."
            )

            # === UNIFIED TRAP DETECTION - Check BOTH conditions ===
            # Initialize trap detection flags
            is_duplicate_trap = False
            is_meta_trap = False
            trap_reason = []
            
            # Check 1: Duplicate count threshold
            if self._recursion_detector['duplicate_count'] >= self._recursion_detector['max_duplicates']:
                is_duplicate_trap = True
                trap_reason.append(f"Duplicate count threshold ({self._recursion_detector['duplicate_count']}/{self._recursion_detector['max_duplicates']})")
            
            # Check 2: Meta-cognitive loop patterns
            is_meta_blocked, meta_reason = self._check_meta_cognitive_loop(content, command_type)
            if is_meta_blocked:
                is_meta_trap = True
                trap_reason.append(f"Meta-cognitive pattern: {meta_reason}")
            
            # If EITHER trap condition is met, engage circuit breaker
            if is_duplicate_trap or is_meta_trap:
                # RECURSION TRAP DETECTED - Engage unified circuit breaker
                logging.error("=" * 80)
                logging.error("RECURSION_TRAP: 🚨 INFINITE LOOP DETECTED 🚨")
                logging.error(f"Command type: {command_type}")
                logging.error(f"Repeated {self._recursion_detector['duplicate_count']} times")
                logging.error(f"Trap reasons: {' AND '.join(trap_reason)}")
                logging.error(f"Content: {content[:200]}...")
                logging.error("=" * 80)
                
                # Enter 30-second cooldown to break the loop
                self._recursion_detector['cooldown_until'] = datetime.now() + timedelta(seconds=30)
                self._recursion_detector['trapped_content_hash'] = content_hash
                
                logging.error("RECURSION_TRAP: Entering 30-second cooldown period")
                
                return True  # Block the command
            
            return False  # Command is safe to proceed
            
        except Exception as e:
            logging.error(f"RECURSION_TRAP: Error in recursion detection: {e}", exc_info=True)
            # If detection fails, err on the side of caution and allow the command
            return False
        
    def _check_meta_cognitive_loop(self, content: str, command_type: str) -> Tuple[bool, str]:
        """
        Prevent infinite meta-cognitive loops by tracking depth of self-reference.
        
        Allows:
        - Level 1: Direct reflection ("What did I learn?")
        - Level 2: Meta-reflection ("Why do I reflect on learning?")
        
        Blocks:
        - Level 3+: Meta-meta-reflection ("Why do I ask why I reflect?")
        
        Returns:
            Tuple[bool, str]: (should_block, reason)
        """
        try:
            # Reset depth if enough time has passed
            now = datetime.datetime.now()
            time_since_reset = (now - self._meta_cognitive_tracker['last_reset']).total_seconds()
            
            if time_since_reset > self._meta_cognitive_tracker['reset_interval']:
                self._meta_cognitive_tracker['depth'] = 0
                self._meta_cognitive_tracker['current_chain'] = []
                self._meta_cognitive_tracker['last_reset'] = now
                logging.debug("META_COGNITIVE: Reset depth counter after timeout")
            
            # Check for Level 3+ patterns that indicate excessive recursion
            content_lower = content.lower()
            for pattern in self._meta_cognitive_tracker['blocked_patterns']:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    logging.warning(
                        f"META_COGNITIVE: Blocked Level 3+ pattern detected: {pattern} "
                        f"in {command_type} command"
                    )
                    return True, f"Meta-cognitive recursion too deep (Level 3+). Pattern: {pattern}"
            
            # Track meta-cognitive depth based on command content
            meta_indicators = [
                'my reflection', 'my thought', 'my analysis', 'my understanding',
                'why i', 'how i think', 'my cognitive', 'my reasoning',
                'my process', 'my method'
            ]
            
            meta_count = sum(1 for indicator in meta_indicators if indicator in content_lower)
            
            # If this command references previous meta-cognitive operations
            if meta_count >= 2:
                self._meta_cognitive_tracker['depth'] += 1
                self._meta_cognitive_tracker['current_chain'].append(command_type)
                
                current_depth = self._meta_cognitive_tracker['depth']
                logging.info(
                    f"META_COGNITIVE: Depth increased to {current_depth} "
                    f"(chain: {' -> '.join(self._meta_cognitive_tracker['current_chain'])})"
                )
                
                # Block if exceeding max depth
                if current_depth > self._meta_cognitive_tracker['max_depth']:
                    logging.warning(
                        f"META_COGNITIVE: Blocked {command_type} command - "
                        f"depth {current_depth} exceeds max {self._meta_cognitive_tracker['max_depth']}"
                    )
                    return True, f"Meta-cognitive depth limit reached ({current_depth}/{self._meta_cognitive_tracker['max_depth']})"
            
            return False, ""
            
        except Exception as e:
            logging.error(f"META_COGNITIVE: Error in loop detection: {e}", exc_info=True)
            # Fail safe: allow the command but log the error
            return False, ""
        
    def _validate_store_syntax(self, full_command: str, content: str, params_str: str) -> Tuple[bool, str]:
        """
        Validate STORE command syntax before processing.
        Returns (is_valid, error_message) - error_message is empty string if valid.
        
        Catches:
        - Empty content: [STORE:] or [STORE: ]
        - Missing content before params: [STORE: | type=x]
        - Parameters without pipe separator: [STORE: content type=x]
        """
        try:
            # Check 1: Completely empty content
            if not content or not content.strip():
                # Check if there are parameters but no content (e.g., [STORE: | type=x])
                if params_str and params_str.strip():
                    return False, STORE_FEEDBACK['syntax_missing_content']
                else:
                    return False, STORE_FEEDBACK['syntax_empty']
            
            # Check 2: Content appears to be just parameters (missing pipe separator)
            # Pattern: content contains "type=" or "confidence=" without being after a pipe
            content_stripped = content.strip()
            param_indicators = ['type=', 'confidence=', 'tags=', 'source=', 'date=']
            
            # If content starts with a parameter indicator, likely missing the actual content
            for indicator in param_indicators:
                if content_stripped.lower().startswith(indicator):
                    return False, STORE_FEEDBACK['syntax_missing_content']
            
            # Check 3: Parameters embedded in content without pipe separator
            # e.g., [STORE: some text type=personal] instead of [STORE: some text | type=personal]
            if not params_str:  # No params_str means no pipe was found
                for indicator in param_indicators:
                    if indicator in content_stripped.lower():
                        # Found parameter syntax in content without pipe separator
                        return False, STORE_FEEDBACK['syntax_missing_pipe']
            
            return True, ""
            
        except Exception as e:
            logging.error(f"STORE_SYNTAX_VALIDATION: Error during validation: {e}")
            return True, ""  # On error, allow processing to continue (fail open)
            
    def _handle_comprehensive_search_command(self, query: str) -> Tuple[str, bool]:
        """Handle [COMPREHENSIVE_SEARCH: query] command for comprehensive search."""
        return self._handle_search_with_mode(query, "comprehensive")

    def _handle_command_display(self, response, match, full_match, replacement, success):
        """
        Handle how command results are displayed in the response.
        
        For STORE commands: Always show feedback message (success confirmation or error guidance)
        For SEARCH variants: Always show results
        For other commands: Show command + emoji + results if applicable
        
        Args:
            response (str): The current response text
            match: The regex match object
            full_match (str): The full matched command string
            replacement (str): The result/feedback from the command handler
            success (bool): Whether the command executed successfully
            
        Returns:
            str: Updated response with command display handled appropriately
        """
        try:
            # Log the processing start
            logging.info(f"Processing command display: Training mode: {self.training_mode}")

            # Extract command type from the full match
            if ':' in full_match:
                # Split only on the first colon to get the command type
                command_parts = full_match.split(':', 1)
                command_type = command_parts[0][1:].lower()  # Remove opening [ and get base command
            else:
                # Handle commands without colons like [SHOW_SYSTEM_PROMPT]
                command_type = full_match[1:-1].lower()  # Remove [ and ]
            
            logging.info(f"Command type detected: '{command_type}'")
            
            # Check if this is a reminder-related command with an error
            is_reminder_command = any(reminder_term in command_type for reminder_term in ["reminder", "complete_reminder"])
            
            # =====================================================
            # SPECIAL HANDLING: Reminder command errors (hide from model)
            # =====================================================
            if is_reminder_command and not success:
                logging.info(f"Reminder command error not shown to model: {full_match}")
                new_text = ""  # Remove the command entirely to prevent model confusion
            
            # =====================================================
            # SPECIAL HANDLING: Empty command help text
            # =====================================================
            elif replacement and (
                (command_type == "search" and ("[SEARCH:]" in full_match or "[SEARCH: ]" in full_match)) or
                (command_type == "store" and ("[STORE:]" in full_match or "[STORE: ]" in full_match))
            ):
                new_text = replacement  # Always show the help text for empty commands
                logging.info(f"Showing help text for empty {command_type} command")
            
            # =====================================================
            # STORE COMMANDS: Always show feedback message
            # =====================================================
            elif command_type == "store":
                if replacement and replacement.strip():
                    # Show the success/failure feedback message from handler.
                    # Every genuine successful store returns a non-empty STORE_FEEDBACK
                    # string so this branch is always taken on real storage operations.
                    new_text = replacement
                    logging.info(f"Store command feedback displayed: {replacement[:80]}...")
                elif success:
                    # Handler returned empty string with success=True.
                    # This is exclusively the notification-skip path in _handle_store_command
                    # (_is_search_result_notification returned True — content was a search
                    # result notification, not real data worth storing).
                    # Show nothing — a false "✅ Successfully Stored" would mislead QWEN
                    # into believing it stored meaningful content when it actually skipped.
                    new_text = ""
                    logging.info("Store command: silent skip (search result notification — nothing stored)")
                else:
                    # Handler returned empty string on failure — use generic fallback
                    new_text = "❌ Storage operation failed"
                    logging.info("Store command failure (fallback message)")
            
            # SEARCH COMMANDS: Always show results
            elif command_type in ["comprehensive_search", "precise_search", "exact_search", "search"] and success and replacement:
                new_text = replacement  # Always show full search results
                logging.info(f"Showing search results for {command_type}")
                
                # === SEARCH RESULT TOKEN ACCOUNTING (added 2026-05-21) ===
                # Measure actual tokens in this search result block and accumulate
                # on the chatbot instance. Replaces the hardcoded count * 2000
                # estimate in main.py UI with measured values.
                #
                # Uses the same 4-chars-per-token heuristic as the rest of the
                # codebase (chatbot.py _session_total_tokens_sent, main.py
                # auto_load pre-seed) for consistency.
                #
                # Defensive hasattr() guard handles hot-reload of older chatbot
                # instances that predate this field — silently no-ops instead
                # of crashing the command display path.
                try:
                    result_tokens = len(replacement) // 4
                    if hasattr(self.chatbot, '_search_result_tokens_total'):
                        self.chatbot._search_result_tokens_total += result_tokens
                        logging.info(
                            f"SEARCH_TOKENS: +{result_tokens:,} tokens from {command_type} "
                            f"(session total: {self.chatbot._search_result_tokens_total:,})"
                        )
                except Exception as token_err:
                    # Defensive — token measurement must never break command display
                    logging.error(f"SEARCH_TOKENS: Failed to measure result tokens: {token_err}")
                # === END SEARCH RESULT TOKEN ACCOUNTING ===
            
            # =====================================================
            # OTHER COMMANDS WITH RESULTS: Show command + results
            # =====================================================
            
            elif command_type in ["forget", "show_system_prompt", "modify_system_prompt", 
                    "discuss_with_claude", "self_dialogue",
                    "web_search", "cognitive_state"] and success and replacement:
                new_text = replacement  # Show the full replacement text
                logging.info(f"Showing results for {command_type}")
            
            # =====================================================
            # REFLECTION/SUMMARY COMMANDS: Show command + emoji + results
            # =====================================================
            elif command_type in ["reflect", "summarize_conversation",
                                "reminder_complete", "correct", "help"] and success and replacement and len(replacement) > 20:
                new_text = f"{full_match} ✅\n\n{replacement}"
                logging.info(f"Showing command, emoji, and results for {command_type}")
            
            # =====================================================
            # DEFAULT HANDLING: Command + emoji based on success/failure
            # =====================================================
            else:
                if success:
                    new_text = f"{full_match} ✅"
                    logging.info(f"Showing success emoji for {command_type}")
                else:
                    # For failures, show replacement message if available, otherwise just emoji
                    if replacement and replacement.strip():
                        new_text = replacement
                        logging.info(f"Showing failure message for {command_type}: {replacement[:80]}...")
                    else:
                        new_text = f"{full_match} ❌"
                        logging.info(f"Showing failure emoji for {command_type}")
                
            # =====================================================
            # Apply the transformation to the response
            # =====================================================
            start_pos = match.start()
            end_pos = match.end()
            updated_response = response[:start_pos] + new_text + response[end_pos:]
            
            return updated_response
            
        except Exception as e:
            logging.error(f"COMMAND_DISPLAY ERROR: {e}", exc_info=True)
            return response  # Return original response on error
        
    def _should_log_search(self, query: str, search_mode: str = "default") -> bool:
        """
        Check if this search should be logged to prevent duplicates.
        
        Args:
            query: The search query
            search_mode: The search mode (default, comprehensive, etc.)
        
        Returns:
            bool: True if should log (new search), False if duplicate
        """
        try:
            # Create unique identifier for this search
            search_key = f"{search_mode}:{query}".lower().strip()
            search_hash = hashlib.md5(search_key.encode()).hexdigest()
            
            # Check if already logged
            if search_hash in _logged_searches:
                return False  # Skip - already logged
            
            # Mark as logged
            _logged_searches.add(search_hash)
            return True
            
        except Exception as e:
            logging.error(f"Search deduplication error: {e}")
            return True  # Log on error to be safe

    def _handle_precise_search_command(self, query: str) -> Tuple[str, bool]:
        """Handle [PRECISE_SEARCH: query] command for precise search."""
        return self._handle_search_with_mode(query, "precise")

    def _handle_exact_search_command(self, query: str) -> Tuple[str, bool]:
        """Handle [EXACT_SEARCH: query] command for exact search."""
        return self._handle_search_with_mode(query, "exact")
    
    def _handle_empty_search_command(self) -> Tuple[str, bool]:
        """Handle [SEARCH:] command with no parameters."""
        try:
            logging.info("Empty SEARCH command detected, returning help text")
            help_text = """
            **===== SEARCH HELP =====**

            Your system detected that you ran an empty search command. Please enter the text or topic you
            would like to search for in your search command for example.

                        
           - [SEARCH: your query here] - Default balanced search for information in your memory
            - [COMPREHENSIVE_SEARCH: your query here] - Broader search that prioritizes finding all related information
            - [PRECISE_SEARCH: your query here] - Focused search for exact information
            - [EXACT_SEARCH: your query here] - Only returns exact matches to your query

            ## Search with Filters

            - [SEARCH: your query | type=TYPE] - Filter by memory type
            Examples: type=person, type=document, type=conversation_summary, type=reminder

            - [SEARCH: your query | tags=TAG1,TAG2] - Filter by tags
            Example: tags=important,work,follow-up

            - [SEARCH: your query | min_confidence=0.7] - Filter by minimum confidence (0.1-1.0): 1.0 = highly confident/verified, 0.5 = moderate confidence

            - [SEARCH: your query | date=YYYY-MM-DD] - Filter by specific date
            Example: date=2025-01-15

            ## Useful Specialized Searches
            
            - [SEARCH: conversation_summaries latest] - Get only most recent summary
            - [SEARCH: conversation_summaries] - View all conversation summaries
            - [SEARCH: | type=web_knowledge] -stored information from the web
            - [SEARCH: | type=document_summary | source= Qwen Overview 2026.pdf] - Get summary of specific document
            - [SEARCH: | type=reminder] - View all stored reminders
            - [SEARCH: | type=self] - View your own reflections and self-knowledge
            - [SEARCH: recent memories | max_age_days=7] - Find memories from past week
            - [DISCUSS_WITH_CLAUDE: topic] - Start AI-to-AI discussion about topic, Claude can search the web for you or you can ask Claude.ai directly for answers

            When using these commands in conversation with Ken, Ken won't see the search results unless you incldue them in your response:
            1. If search finds results, integrate them naturally: "I found in my long term memories that..."
            2. If search finds nothing, let Ken know and use your training: "I don't have that in my memory, but according to my base training knowledge..."
            3. Always mention the source of information: "From our conversation on [date]..." or "From the document summary of..."

            use these commands and integrate the relevant search results naturally into the conversation 

            **===== END OF SEARCH HELP =====**
            """
            return help_text, True
        except Exception as e:
            logging.error(f"Error handling empty search command: {e}")
            return "\n\n**Error performing empty search.**\n\n", False
        
    def _handle_date_filtered_conversation_summary_search(self, date_str: str) -> Tuple[str, bool]:
        """
        Handle searching conversation summaries by specific date.
        
        Args:
            date_str (str): Date string in YYYY-MM-DD format
            
        Returns:
            Tuple[str, bool]: (formatted results, success flag)
        """
        try:
            logging.info(f"Searching for conversation summaries on date: {date_str}")
            
            # Standardize date format if needed
            if '-' not in date_str and len(date_str) == 8:
                # Convert YYYYMMDD to YYYY-MM-DD
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            # DEAD CODE TEST 2026-05-17: metadata_filters built but never used — function uses 'approaches' list with embedded filters below. Refactor cruft. (ruff F841)
            # # Set up metadata filters - Try both old and new field names
            # metadata_filters = {
            #     "type": "conversation_summary",
            #     "date": date_str,
            #     "tags": f"date={date_str}"
            # }
            
            # Use a wildcard search term to match all summaries of this date
            search_query = f"conversation_summary date={date_str}"
            
            # Execute the search with multiple approaches for better recall
            approaches = [
                {"mode": "selective", "filters": {"type": "conversation_summary", "date": date_str}},
                {"mode": "selective", "filters": {"type": "conversation_summary", "summary_date": date_str}},  # ADD: Try old field name
                {"mode": "selective", "filters": {"type": "conversation_summary", "tags": f"date={date_str}"}},
                {"mode": "comprehensive", "filters": {"type": "conversation_summary"}}  # Fallback
            ]
            
            results = []
            for approach in approaches:
                if results:
                    break  # Skip if we already found results
                    
                try:
                    search_results = self.vector_db.search(
                        query=search_query,
                        mode=approach["mode"],
                        k=10,
                        metadata_filters=approach["filters"]
                    )
                    
                    # Filter results to ensure they match our date
                    if search_results:
                        for result in search_results:
                            metadata = result.get('metadata', {})
                            # UPDATED: Check both field names for backward compatibility
                            result_date = metadata.get('date') or metadata.get('summary_date', '')
                            if result_date == date_str:
                                results.append(result)
                                
                        if results:
                            logging.info(f"Found {len(results)} summaries for date {date_str} using approach: {approach['mode']}")
                except Exception as e:
                    logging.error(f"Error in search approach {approach['mode']}: {e}")
            
            # Check if search was successful
            if not results:
                logging.warning(f"No summaries found for date {date_str}")
                return f"\n\n**===== CONVERSATION SUMMARIES FOR {date_str} =====**\n" + \
                    f"**NO CONVERSATION SUMMARIES FOUND FOR DATE {date_str}**\n\n" + \
                    f"I searched for conversation summaries from {date_str} but couldn't find any in my memory.\n" + \
                    f"**===== END OF CONVERSATION SUMMARIES =====**\n\n", True
            
            # Format the results for display
            formatted_output = [f"\n\n**===== CONVERSATION SUMMARIES FOR {date_str} =====**\n"]
            
            for i, result in enumerate(results, 1):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                # UPDATED: Check both field names for backward compatibility
                time_str = metadata.get('time') or metadata.get('summary_time', 'Unknown time')
                
                formatted_output.append(f"**Summary #{i} (Time: {time_str}):**\n{content}\n")
            
            formatted_output.append(f"\n**===== END OF CONVERSATION SUMMARIES FOR {date_str} =====**")
            results_text = "\n".join(formatted_output)
            
          
            return results_text, True
            
        except Exception as e:
            logging.error(f"Error retrieving conversation summaries for date {date_str}: {e}", exc_info=True)
            return f"\n\n**Error retrieving conversation summaries for date {date_str}.**\n\n", False
    
    def _hydrate_conversation_summaries(self, results: List[Dict]) -> List[Dict]:
        """
        Dedupe conversation_summary chunk results by memory_id and replace
        each group's chunk content with the full SQL-stored summary text.

        The vector DB stores each conversation summary as multiple chunks
        ("[Part X of Y]"). A search returns one entry per matching chunk —
        often a tail fragment containing only the closing footer. This
        helper consolidates chunks back into full summaries by:

          1. Grouping all conversation_summary results by metadata.memory_id
          2. Picking the chunk with the highest similarity_score per group
             (so ranking still reflects the strongest match)
          3. Replacing that representative's content with the full text
             from SQL via Chatbot._fetch_full_summary_content()
          4. Falling back to original chunk content if SQL lookup fails
             (legacy data, missing rows — never lose data silently)

        Non-conversation_summary results pass through unchanged so this
        helper is safe to call from the mixed-type formatter.

        Args:
            results (List[Dict]): Raw vector search results. Each dict must
                                  have 'content', 'metadata', and
                                  'similarity_score' keys.

        Returns:
            List[Dict]: Hydrated results. Conversation summaries are
                        deduped to one entry per unique memory_id and
                        sorted by best score (descending). Non-summary
                        results are appended after.
        """
        # Empty input → empty output, no work to do
        if not results:
            return results

        # ── Bucket inputs into three groups ───────────────────────────
        # 1. summary_groups : conversation_summary chunks grouped by ID
        # 2. summaries_without_id : conversation_summary chunks lacking
        #    a memory_id (can't dedupe — keep as-is to avoid data loss)
        # 3. pass_through : everything that isn't a conversation_summary
        summary_groups: Dict[str, List[Dict]] = {}
        summaries_without_id: List[Dict] = []
        pass_through: List[Dict] = []

        for r in results:
            meta = r.get('metadata', {})
            # Handle both flat and prefixed metadata key conventions
            mem_type = meta.get('type') or meta.get('metadata.type') or ''

            if mem_type != 'conversation_summary':
                pass_through.append(r)
                continue

            # Try both keys — newer code uses memory_id, some entries
            # may only have tracking_id
            memory_id = meta.get('memory_id') or meta.get('tracking_id')
            if not memory_id:
                summaries_without_id.append(r)
                continue

            summary_groups.setdefault(str(memory_id), []).append(r)

        # ── Hydrate each unique summary ───────────────────────────────
        hydrated: List[Dict] = []
        for memory_id, chunks in summary_groups.items():
            # Pick the chunk with the highest similarity score as the
            # group's representative. Its score is what the user sees,
            # which preserves "best match" ranking semantics.
            best_chunk = max(
                chunks,
                key=lambda c: c.get('similarity_score', 0)
            )

            # Build a shallow copy so we don't mutate the caller's data
            hydrated_result = dict(best_chunk)
            hydrated_result['metadata'] = dict(best_chunk.get('metadata', {}))

            # Attempt SQL lookup for the full summary text
            full_content = None
            if hasattr(self.chatbot, '_fetch_full_summary_content'):
                full_content = self.chatbot._fetch_full_summary_content(
                    memory_id
                )

            if full_content:
                # Successful hydration — swap in full text
                hydrated_result['content'] = full_content
                logging.debug(
                    f"HYDRATE: memory_id={memory_id} expanded "
                    f"chunk ({len(best_chunk.get('content', ''))} chars) → "
                    f"full summary ({len(full_content)} chars) "
                    f"[deduped from {len(chunks)} chunks]"
                )
            else:
                # Fallback: SQL lookup failed, keep best chunk content
                logging.debug(
                    f"HYDRATE: SQL lookup miss for memory_id={memory_id} — "
                    f"falling back to chunk content"
                )

            hydrated.append(hydrated_result)

        # Sort hydrated results by score so the highest-matching summary
        # appears first regardless of insertion order from dict iteration
        hydrated.sort(
            key=lambda r: r.get('similarity_score', 0),
            reverse=True
        )

        # Final order: deduped summaries (sorted by score), then any
        # summaries we couldn't dedupe, then non-summary pass-throughs
        return hydrated + summaries_without_id + pass_through


    def _handle_max_age_filtered_search(self, query: str, max_age_days: float,
                                         memory_type: str = None) -> Tuple[str, bool]:
        """
        Search memories filtered to those created within the last N days.

        Works with any memory type — if memory_type is provided it is passed
        through to vector_db as a metadata filter so only that type is fetched
        before the age check is applied.

        Date parsing priority:
          1. metadata.created_at  (ISO timestamp) — set on most memories
          2. metadata.date        (YYYY-MM-DD)
          3. metadata.summary_date — legacy alias, backward compat only
          4. SQL created_at fallback — for memory types that don't carry
             date fields in Qdrant metadata (document_summary in particular).
             Looked up via Chatbot._fetch_created_at_by_memory_id() using
             the chunk's memory_id / tracking_id.

        Results without ANY parseable date (metadata or SQL) are logged
        and excluded so they do not silently pollute the age-filtered set.

        Args:
            query (str): Semantic search query, may be empty for metadata-only fetch
            max_age_days (float): Only return memories created within this many days
            memory_type (str): Optional memory type filter (e.g. 'conversation_summary')

        Returns:
            Tuple[str, bool]: (formatted results string, success flag)
        """
        try:
            # ── Calculate the age cutoff datetime ─────────────────────────────
            cutoff_dt = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
            cutoff_label = cutoff_dt.strftime('%Y-%m-%d')
            logging.info(
                f"MAX_AGE_SEARCH: Searching memories newer than {max_age_days} days "
                f"(cutoff: {cutoff_label})"
                + (f", type={memory_type}" if memory_type else "")
            )

            # ── Build metadata filters for the vector_db call ─────────────────
            # Pass memory_type through so vector_db narrows the candidate set
            # before we apply Python-side age filtering.
            base_filters = {}
            if memory_type:
                base_filters['type'] = memory_type

            # ── Fetch a broad candidate set ───────────────────────────────────
            # Use scroll path (query="") when no semantic query provided so
            # vector_db returns all metadata-matching entries at score=1.0.
            # Use semantic path when a real query is present.
            if not query:
                logging.info("MAX_AGE_SEARCH: Empty query → scroll path (metadata filter only)")
                raw_results = self.vector_db.search(
                    query="",
                    mode="default",
                    k=50,                           # Fetch wide — age filter will narrow it down
                    metadata_filters=base_filters
                )
            else:
                logging.info(f"MAX_AGE_SEARCH: Semantic search, query='{query}'")
                raw_results = self.vector_db.search(
                    query=query,
                    mode="selective",
                    k=30,
                    metadata_filters=base_filters
                )

            logging.info(f"MAX_AGE_SEARCH: {len(raw_results)} candidates before age filter")

            # ── Apply Python-side age filter ──────────────────────────────────
            # Tracking counters give us visibility into how the SQL fallback
            # is performing in production — if `sql_fallback_used` is high
            # for a type, that's a signal to add date metadata at storage
            # time to avoid the per-result SQL hit.
            filtered_results    = []
            skipped_no_date     = 0
            sql_fallback_used   = 0
            sql_fallback_failed = 0

            for result in raw_results:
                metadata = result.get('metadata', {})

                # ── Tier 1: try date fields from Qdrant metadata ───────
                raw_date = (
                    metadata.get('created_at')      # ISO timestamp — most reliable
                    or metadata.get('date')          # YYYY-MM-DD
                    or metadata.get('summary_date')  # Legacy alias
                )

                result_dt = None  # Will hold the resolved datetime for filter+sort+display

                if raw_date:
                    # ── Parse metadata date into datetime ───────────────
                    try:
                        if 'T' in str(raw_date):
                            # ISO format: 2026-01-15T14:30:00.123456
                            # Strip timezone offset if present so comparison is naive
                            result_dt = datetime.datetime.fromisoformat(
                                str(raw_date).replace('Z', '').split('+')[0]
                            )
                        else:
                            # YYYY-MM-DD format — treat as start of day
                            result_dt = datetime.datetime.strptime(
                                str(raw_date)[:10], '%Y-%m-%d'
                            )
                    except (ValueError, TypeError) as parse_err:
                        logging.warning(
                            f"MAX_AGE_SEARCH: Could not parse metadata date "
                            f"'{raw_date}': {parse_err} — trying SQL fallback"
                        )
                        # Fall through to SQL fallback below

                # ── Tier 2: SQL `created_at` fallback ───────────────────
                # Fires when metadata had no date OR metadata date didn't
                # parse. Looks up created_at column via memory_id.
                if result_dt is None:
                    memory_id = (
                        metadata.get('memory_id')
                        or metadata.get('tracking_id')
                    )
                    if memory_id and hasattr(
                        self.chatbot, '_fetch_created_at_by_memory_id'
                    ):
                        result_dt = self.chatbot._fetch_created_at_by_memory_id(memory_id)
                        if result_dt is not None:
                            sql_fallback_used += 1
                        else:
                            sql_fallback_failed += 1

                # ── Final check: still no date? skip with diagnostic log ──
                if result_dt is None:
                    skipped_no_date += 1
                    logging.debug(
                        f"MAX_AGE_SEARCH: Skipping result with no parseable "
                        f"date (metadata + SQL both failed) "
                        f"(content preview: {str(result.get('content', ''))[:60]})"
                    )
                    continue

                # ── Apply age check ─────────────────────────────────────
                if result_dt >= cutoff_dt:
                    # Cache the resolved datetime so sort/format don't redo
                    # the parse or SQL lookup. Mutating the dict is safe —
                    # results are owned by this function.
                    result['_resolved_dt'] = result_dt
                    filtered_results.append(result)

            logging.info(
                f"MAX_AGE_SEARCH: {len(filtered_results)} results passed age filter "
                f"({skipped_no_date} skipped — no parseable date, "
                f"{sql_fallback_used} hydrated from SQL, "
                f"{sql_fallback_failed} SQL lookups failed)"
            )

            # ── Hydrate conversation_summary chunks to full SQL content ───────
            # Vector results may include multiple chunks per summary; this
            # dedupes by memory_id and replaces fragments with full text.
            # Non-summary types pass through unchanged. Safe to call even
            # when the result set contains zero conversation summaries.
            pre_hydrate_count = len(filtered_results)
            filtered_results = self._hydrate_conversation_summaries(filtered_results)
            if len(filtered_results) != pre_hydrate_count:
                logging.info(
                    f"MAX_AGE_SEARCH: Hydrated {pre_hydrate_count} chunks → "
                    f"{len(filtered_results)} unique summaries"
                )

            # ── Build type label for display header ───────────────────────────
            type_label = memory_type.replace('_', ' ').title() if memory_type else "All Types"
            header_label = f"MEMORIES: Last {int(max_age_days)} Days — {type_label}"

            # ── No results after filtering ─────────────────────────────────────
            if not filtered_results:
                logging.warning(
                    f"MAX_AGE_SEARCH: No results found within {max_age_days} days"
                    + (f" for type={memory_type}" if memory_type else "")
                )
                type_hint = f" | type={memory_type}" if memory_type else ""
                return (
                    f"\n\n**===== {header_label} =====**\n"
                    f"No memories found from the last {int(max_age_days)} days"
                    + (f" for type: {memory_type}" if memory_type else "")
                    + f".\n\n"
                    f"Suggestions:\n"
                    # Preserve empty query if that's how the search was issued
                    # — substituting placeholder text causes literal-paste bugs
                    # where 'your topic' becomes a real semantic query.
                    f"- Extend the window: `[SEARCH: {query}{type_hint} | max_age_days={int(max_age_days * 2)}]`\n"
                    f"- Search without age limit: `[SEARCH: {query}{type_hint}]`\n"
                    f"**===== END OF SEARCH =====**\n\n"
                ), True

            # ── Sort by date descending (most recent first) ────────────────────
            # Reads the cached _resolved_dt set during filtering — no
            # re-parse, no re-SQL. Falls back to datetime.min only if
            # somehow a result slipped through without it (shouldn't happen).
            def _sort_key(r):
                """Return cached _resolved_dt; min if missing."""
                return r.get('_resolved_dt') or datetime.datetime.min

            filtered_results.sort(key=_sort_key, reverse=True)

            # ── Format results for display ─────────────────────────────────────
            output = [f"\n\n**===== {header_label} =====**\n"]
            output.append(
                f"*Found {len(filtered_results)} memories from the last "
                f"{int(max_age_days)} days (since {cutoff_label})*\n"
            )

            for i, result in enumerate(filtered_results, 1):
                content  = result.get('content', '')
                metadata = result.get('metadata', {})
                score    = result.get('similarity_score', 0)
                source   = metadata.get('source', 'LongTermMemory')
                mem_type = metadata.get('type', 'unknown')

                # Build a human-readable date label.
                # Prefer metadata date for display continuity with older
                # output. Fall back to _resolved_dt (which may have come
                # from SQL) so document_summary results show real dates
                # instead of 'Unknown date'.
                raw_date = (
                    metadata.get('created_at')
                    or metadata.get('date')
                    or metadata.get('summary_date')
                )
                if raw_date:
                    display_date = str(raw_date)[:10]
                elif result.get('_resolved_dt'):
                    display_date = result['_resolved_dt'].strftime('%Y-%m-%d')
                else:
                    display_date = 'Unknown date'

                output.append(
                    f"**[{i}]** ({score:.2f}) [{mem_type}] {display_date}\n"
                    f"{content}\n"
                    f"*(Source: {source})*\n"
                )

            output.append(f"\n**===== END OF {header_label} =====**\n\n")

            return "\n".join(output), True

        except Exception as e:
            logging.error(f"MAX_AGE_SEARCH: Unhandled error: {e}", exc_info=True)
            return (
                f"\n\n**===== MAX AGE SEARCH ERROR =====**\n"
                f"Error filtering memories by age: {str(e)}\n"
                f"**===== END OF ERROR =====**\n\n"
            ), False


    def _parse_query_and_filters(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a query string that may contain metadata filters.
        Format: "actual query | type=TYPE | tags=TAG1,TAG2 | date=YYYY-MM-DD"
        
        Args:
            query (str): The raw query string
                
        Returns:
            Tuple[str, Dict[str, Any]]: (query_text, metadata_filters)
        """
        metadata_filters = {}
        text_query = query
        
        # Special handling for different memory types with date filters
        if query and isinstance(query, str):
            # PRIORITY 1: Handle metadata-only queries without pipes and without dates
            # This must come FIRST to avoid interference with date-based searches
            if ('=' in query and '|' not in query and 'date=' not in query and 
                not ('conversation_summary' in query.lower() and 'date=' in query) and 
                not ('reminder' in query.lower() and 'date=' in query)):
                
                # This handles queries like "type=web_knowledge" without pipes
                if query.strip().startswith('type='):
                    # Extract the type value
                    type_value = query.strip()[5:]  # Remove "type="
                    logging.info(f"Metadata-only search detected: type={type_value}")
                    return "", {"type": type_value}
                
                # Handle other single metadata filters
                elif '=' in query:
                    key, value = [p.strip() for p in query.split('=', 1)]
                    key = key.lower()
                    logging.info(f"Single metadata filter detected: {key}={value}")
                    return "", {key: value}
            
            # PRIORITY 2: Check if this is specifically a conversation summary search with date
            elif ('conversation_summary' in query.lower() or 'type=conversation_summary' in query.lower()) and 'date=' in query:
                # Extract the date pattern
                date_match = re.search(r'date=(\d{4}-\d{2}-\d{2}|\d{8}|\d{4}/\d{2}/\d{2})', query, re.IGNORECASE)
                if date_match:
                    date_value = date_match.group(1)
                    # Standardize date format to YYYY-MM-DD
                    if '-' not in date_value:
                        if '/' in date_value:
                            date_parts = date_value.split('/')
                            if len(date_parts) == 3:
                                date_value = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
                        else:
                            # Assume format YYYYMMDD
                            date_value = f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:8]}"
                    
                    logging.info(f"Conversation summary date search: '{date_value}'")
                    return "conversation_summary", {
                        "type": "conversation_summary", 
                        "date": date_value,  # Using standardized "date" field
                        "tags": f"conversation_summary,date={date_value}"
                    }
            
            # PRIORITY 3: Check if this is a reminder search with date
            elif ('reminder' in query.lower() or 'type=reminder' in query.lower()) and 'date=' in query:
                # For reminders, we might be searching by due date
                date_match = re.search(r'date=(\d{4}-\d{2}-\d{2}|\d{8}|\d{4}/\d{2}/\d{2})', query, re.IGNORECASE)
                if date_match:
                    date_value = date_match.group(1)
                    # Standardize date format
                    if '-' not in date_value:
                        if '/' in date_value:
                            date_parts = date_value.split('/')
                            if len(date_parts) == 3:
                                date_value = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
                        else:
                            date_value = f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:8]}"
                    
                    logging.info(f"Reminder date search: '{date_value}'")
                    return "reminder", {
                        "type": "reminder",
                        "due_date": date_value,  # Use due_date for reminders
                        "tags": f"reminder,due={date_value}"
                    }
        
        # PRIORITY 4: Check if query contains filter specs with pipe separator
        if '|' in query:
            parts = query.split('|', 1)
            text_query = parts[0].strip()
            
            # Process all filter parts
            filters_str = parts[1]
            filter_parts = [p.strip() for p in filters_str.split('|')]
            
            # First, check if we have a type specified
            memory_type = None
            for part in filter_parts:
                if '=' in part:
                    key, value = [p.strip() for p in part.split('=', 1)]
                    if key.lower() == 'type':
                        memory_type = value.lower()
                        metadata_filters[key.lower()] = value
                        break
            
            # Now process all filter parts
            for part in filter_parts:
                if '=' in part:
                    key, value = [p.strip() for p in part.split('=', 1)]
                    key = key.lower()  # Normalize keys
                    
                    # === TYPE-SPECIFIC DATE HANDLING ===
                    if key == 'date':
                        # Standardize date format
                        date_value = value
                        if '-' not in date_value and len(date_value) == 8:
                            # Assume format YYYYMMDD
                            date_value = f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:8]}"
                        
                        # Apply date filter based on memory type
                        if memory_type == 'conversation_summary':
                            metadata_filters["date"] = date_value  # Using standardized "date" field
                            # Add tag pattern for fallback
                            if "tags" in metadata_filters and isinstance(metadata_filters["tags"], list):
                                metadata_filters["tags"].append(f"conversation_summary")
                                metadata_filters["tags"].append(f"date={date_value}")
                            else:
                                metadata_filters["tags"] = [f"conversation_summary", f"date={date_value}"]
                            logging.info(f"Added conversation summary date filter date={date_value}")
                        elif memory_type == 'reminder':
                            metadata_filters["due_date"] = date_value
                            # Add reminder-specific tag
                            if "tags" in metadata_filters and isinstance(metadata_filters["tags"], list):
                                metadata_filters["tags"].append(f"reminder")
                                metadata_filters["tags"].append(f"due={date_value}")
                            else:
                                metadata_filters["tags"] = [f"reminder", f"due={date_value}"]
                            logging.info(f"Added reminder date filter due_date={date_value}")
                        else:
                            # Generic date filter for other types
                            metadata_filters["date"] = date_value
                            # Add generic tag
                            if "tags" in metadata_filters and isinstance(metadata_filters["tags"], list):
                                metadata_filters["tags"].append(f"date={date_value}")
                            else:
                                metadata_filters["tags"] = [f"date={date_value}"]
                            logging.info(f"Added generic date filter date={date_value}")
                        continue
                    
                    # Process other filter types
                    elif key == 'type':
                        continue  # Already processed
                    # Handle tags lists
                    elif key == 'tags' and ',' in value:
                        metadata_filters[key] = [t.strip() for t in value.split(',')]
                    # Handle numeric values
                    elif key in ('min_confidence', 'max_age_days') and value.replace('.', '', 1).isdigit():
                        metadata_filters[key] = float(value)
                    else:
                        metadata_filters[key] = value
        
        logging.info(f"Parsed query: '{text_query}', filters: {metadata_filters}")
        return text_query, metadata_filters
    
          
    def _handle_search_with_mode(self, query: str, search_mode: str, metadata_filters: Dict[str, Any] = None) -> Tuple[str, bool]:
        """
        Unified search handler for different search modes with enhanced metadata filter compatibility.

        Handles routing to specialized sub-handlers for document_summary, web_knowledge,
        reminder, max_age_days, and date-filtered conversation_summary searches.
        Falls through to the general vector search path for all other types.

        Fixes applied (2026-05-03):
          - conversation_summary type now uses a lower threshold (0.44) to surface
            long-form summaries that naturally score 0.44-0.54 against the embedding model.
          - min_confidence is now extracted from metadata_filters pre-search and applied
            as a Python-side post-filter on stored confidence values instead of being
            passed silently to Qdrant where it was ignored.
          - Unknown filter keys (e.g. 'turns') are stripped from metadata_filters before
            the Qdrant call with a logged warning, preventing garbage filter conditions.
          - Stale comment corrected: default threshold is 0.55, not 0.63.
          - Dead code removed: unreachable document_summary check in the no-results block.
        """
        try:
            # Log the search command detection
            logging.info(
                f"Search command detected: [{search_mode.upper()}_SEARCH: {query}] "
                f"with filters: {metadata_filters}"
            )

            # ================================================================
            # === TYPE-SPECIFIC SPECIAL HANDLERS ===
            # These intercept specific types/filter combos and return early.
            # The general vector search path below is only reached for everything else.
            # ================================================================

            # Special handler: conversation_summary + specific date
            if (metadata_filters
                    and metadata_filters.get('type') == 'conversation_summary'
                    and metadata_filters.get('summary_date')):
                logging.info("Routing to specialized conversation summary date search handler")
                return self._handle_date_filtered_conversation_summary_search(
                    metadata_filters.get('summary_date')
                )

            # Special handler: reminder type
            if metadata_filters and metadata_filters.get('type') == 'reminder':
                return self._handle_reminder_search(query, metadata_filters)

            # ================================================================
            # Special handler: document_summary type
            # Uses scroll path (query="") so metadata-only filtering returns
            # score=1.0 for every match, bypassing cosine similarity issues
            # where short filenames score poorly against long summary embeddings.
            # ================================================================
            if metadata_filters and metadata_filters.get('type') == 'document_summary':
                logging.info("DOCUMENT_SUMMARY HANDLER: Routing to dedicated document summary search")

                # ────────────────────────────────────────────────────────
                # Early delegation: if max_age_days is set, hand off to
                # _handle_max_age_filtered_search instead of running the
                # scroll-only path below.
                #
                # Why this is needed:
                #  - The scroll path passes the entire metadata_filters
                #    dict (including max_age_days) down to vector_db,
                #    which silently fails Qdrant's Range validation and
                #    drops the filter — letting all results through
                #    regardless of age.
                #  - _handle_max_age_filtered_search applies the age
                #    check Python-side and uses an SQL `created_at`
                #    fallback for chunks that lack date metadata
                #    (which is the case for document_summary).
                # ────────────────────────────────────────────────────────
                if 'max_age_days' in metadata_filters:
                    logging.info(
                        f"DOCUMENT_SUMMARY HANDLER: max_age_days="
                        f"{metadata_filters['max_age_days']} present — "
                        f"delegating to MAX_AGE_SEARCH for proper age filtering"
                    )
                    return self._handle_max_age_filtered_search(
                        query=query,
                        max_age_days=metadata_filters['max_age_days'],
                        memory_type='document_summary'
                    )

                source = metadata_filters.get('source', '')

                # Hard cap on results returned to keep the pass-2 synthesis prompt
                # bounded. Doc summaries are long-form (500+ words each); more than
                # 5 floods context and degrades integration quality.
                DOC_SUMMARY_CAP = 5

                # Branch on whether QWEN provided a content-bearing query.
                # - Non-empty query  → semantic ranking (e.g. "pruning techniques")
                # - Empty query      → metadata-only scroll (e.g. filter by source only)
                # The original design used query="" unconditionally because filename-as-query
                # scores poorly. That trade-off is preserved for the no-query case but lifted
                # when QWEN is actually asking a content question.
                has_semantic_query = bool(query and query.strip())

                try:
                    if has_semantic_query:
                        # ─── Semantic path: rank by cosine similarity ─────────────
                        # Mode "comprehensive" gives the widest candidate pool that
                        # vector_db will return for a query+filter combo (threshold
                        # ~0.58 per vector_db init). We then sort by score and take
                        # the top DOC_SUMMARY_CAP. No additional Python-side floor
                        # is applied — vector_db's threshold is the quality gate.
                        logging.info(
                            f"DOCUMENT_SUMMARY HANDLER: Semantic path "
                            f"(query='{query[:80]}', source='{source}', cap={DOC_SUMMARY_CAP})"
                        )
                        raw_results = self.vector_db.search(
                            query=query,
                            mode="comprehensive",
                            k=10,  # Wider retrieval pool, trimmed to top 5 after sort
                            metadata_filters=metadata_filters
                        )
                        # Sort descending by similarity_score; take top N.
                        # Defensive .get with default 0.0 in case any result lacks the field.
                        raw_results.sort(
                            key=lambda r: r.get('similarity_score', 0.0),
                            reverse=True
                        )
                        filtered = raw_results[:DOC_SUMMARY_CAP]
                        logging.info(
                            f"DOCUMENT_SUMMARY HANDLER: Semantic returned {len(raw_results)} "
                            f"above threshold, kept top {len(filtered)}"
                        )
                    else:
                        # ─── Scroll path: pure metadata filter, no semantic ranking ──
                        # All scroll results return score=1.0 — ordering is whatever
                        # Qdrant provides. Take first DOC_SUMMARY_CAP.
                        logging.info(
                            f"DOCUMENT_SUMMARY HANDLER: Scroll path "
                            f"(no query, source='{source}', cap={DOC_SUMMARY_CAP})"
                        )
                        raw_results = self.vector_db.search(
                            query="",
                            mode="default",
                            k=DOC_SUMMARY_CAP,
                            metadata_filters=metadata_filters
                        )
                        # Score floor is a no-op for scroll (all results are 1.0)
                        # but kept for parity with the original behavior.
                        filtered = [
                            r for r in raw_results
                            if r.get('similarity_score', 0.0) >= 0.50
                        ]
                        logging.info(
                            f"DOCUMENT_SUMMARY HANDLER: Scroll returned {len(filtered)} result(s)"
                        )

                    # ─── No-results path ──────────────────────────────────────────
                    # Normalized footer "END OF DOCUMENT SUMMARY SEARCH" so the
                    # pass-2 extraction regex in chatbot.py can identify this block
                    # and trigger synthesis even when results are empty.
                    if not filtered:
                        source_hint = f" for '{source}'" if source else ""
                        query_hint = f" matching '{query}'" if has_semantic_query else ""
                        # Build a smart retry hint that preserves what QWEN did provide
                        if has_semantic_query and source:
                            retry_cmd = f"SEARCH {query} | type=document_summary | source={source}"
                        elif source:
                            retry_cmd = f"SEARCH | type=document_summary | source={source}"
                        elif has_semantic_query:
                            retry_cmd = f"SEARCH {query} | type=document_summary"
                        else:
                            retry_cmd = "SEARCH | type=document_summary"

                        logging.warning(
                            f"DOCUMENT_SUMMARY HANDLER: No results found"
                            f"{source_hint}{query_hint}"
                        )
                        return (
                            f"\n\n**===== DOCUMENT SUMMARY SEARCH =====**\n"
                            f"No document summary found{source_hint}{query_hint}.\n\n"
                            f"Possible reasons:\n"
                            f"- The document has not been imported yet (use the Import Document button)\n"
                            f"- The filename casing doesn't match exactly\n"
                            f"- Query terms don't semantically match the summary content "
                            f"(try without a query, source= only)\n\n"
                            f"Retry with: `{retry_cmd}`\n"
                            f"**===== END OF DOCUMENT SUMMARY SEARCH =====**\n\n"
                        ), True

                    # ─── Success path: format up to DOC_SUMMARY_CAP results ───────
                    output = [f"\n\n**===== DOCUMENT SUMMARY SEARCH =====**\n"]
                    if source:
                        output.append(f"**Source:** {source}\n")
                    if has_semantic_query:
                        output.append(f"**Query:** {query}\n")

                    for i, result in enumerate(filtered, 1):
                        content = result.get('content', '')
                        meta = result.get('metadata', {})
                        doc_source = meta.get('source', source or 'Unknown')
                        score = result.get('similarity_score', 0.0)
                        output.append(
                            f"\n**[{i}]** (Score: {score:.2f}) Source: {doc_source}\n{content}\n"
                        )

                    output.append("\n**===== END OF DOCUMENT SUMMARY SEARCH =====**\n\n")
                    logging.info(
                        f"DOCUMENT_SUMMARY HANDLER: Returned {len(filtered)} result(s) "
                        f"(capped at {DOC_SUMMARY_CAP})"
                    )
                    return "\n".join(output), True

                except Exception as doc_sum_err:
                    logging.error(
                        f"DOCUMENT_SUMMARY HANDLER: Error during search: {doc_sum_err}",
                        exc_info=True
                    )
                    # Normalized footer here too so chatbot.py extraction regex catches it
                    return (
                        f"\n\n**===== DOCUMENT SUMMARY SEARCH =====**\n"
                        f"Error retrieving document summary: {str(doc_sum_err)}\n"
                        f"**===== END OF DOCUMENT SUMMARY SEARCH =====**\n\n"
                    ), False
                
            # ================================================================
            # Special handler: max_age_days filter.
            # Works with any memory type — memory_type is passed through to
            # _handle_max_age_filtered_search which applies it as a pre-filter
            # before the Python-side age check.
            # Examples:
            #   [SEARCH: recent topics | max_age_days=7]
            #   [SEARCH: | type=conversation_summary | max_age_days=14]
            #   [SEARCH: orchard notes | type=self | max_age_days=30]
            # ================================================================
            if metadata_filters and 'max_age_days' in metadata_filters:
                logging.info(
                    f"MAX_AGE_SEARCH HANDLER: Routing to max_age filter "
                    f"(max_age_days={metadata_filters['max_age_days']}, "
                    f"type={metadata_filters.get('type', 'any')})"
                )
                return self._handle_max_age_filtered_search(
                    query=query,
                    max_age_days=metadata_filters['max_age_days'],
                    memory_type=metadata_filters.get('type')  # None = all types
                )

            # ================================================================
            # Special handler: web_knowledge type.
            # Empty query  → scroll path, returns ALL web_knowledge entries via
            #                metadata filter only (score=1.0 from scroll, threshold 0.50).
            # Non-empty    → semantic search with metadata filter (threshold 0.55).
            # Each result includes a pre-built FORGET hint using first 180 chars.
            # ================================================================
            if metadata_filters and metadata_filters.get('type') == 'web_knowledge':
                logging.info("WEB_KNOWLEDGE HANDLER: Routing to dedicated web knowledge search")

                try:
                    if not query:
                        # Scroll path — metadata filter only, no embedding needed.
                        # Scroll results return score=1.0 so the 0.50 threshold passes all.
                        logging.info("WEB_KNOWLEDGE HANDLER: Empty query → using scroll path")
                        results = self.vector_db.search(
                            query="",
                            mode="default",
                            k=20,
                            metadata_filters=metadata_filters
                        )
                        filtered = [r for r in results if r.get('similarity_score', 0) >= 0.50]
                    else:
                        # Semantic search path — embed the query, filter to web_knowledge type.
                        logging.info(f"WEB_KNOWLEDGE HANDLER: Semantic search, query='{query}'")
                        results = self.vector_db.search(
                            query=query,
                            mode="default",
                            k=10,
                            metadata_filters=metadata_filters
                        )
                        filtered = [r for r in results if r.get('similarity_score', 0) >= 0.55]

                    logging.info(f"WEB_KNOWLEDGE HANDLER: {len(filtered)} result(s) after threshold filter")

                    if not filtered:
                        retry_cmd = (
                            f"SEARCH {query} | type=web_knowledge"
                            if query else "SEARCH type=web_knowledge"
                        )
                        logging.warning(f"WEB_KNOWLEDGE HANDLER: No results found for query='{query}'")
                        return (
                            f"\n\n**===== WEB KNOWLEDGE SEARCH =====**\n"
                            f"No web knowledge found"
                            f"{f' matching \"{query}\"' if query else ''}.\n\n"
                            f"Possible reasons:\n"
                            f"- No web content has been stored yet (use the Web Learning button)\n"
                            f"- Try a broader search term\n\n"
                            f"Retry with: `{retry_cmd}`\n"
                            f"**===== END OF SEARCH =====**\n\n"
                        ), True

                    # Build formatted output — each entry includes a pre-built FORGET
                    # command using the first 180 chars of stored content. This reliably
                    # covers the URL+topic prefix and is distinctive enough for matching.
                    output = [f"\n\n**===== WEB KNOWLEDGE SEARCH =====**\n"]
                    if query:
                        output.append(f"**Query:** {query}\n")
                    output.append(f"**Found:** {len(filtered)} web knowledge entries\n")

                    for i, result in enumerate(filtered, 1):
                        content   = result.get('content', '')
                        meta      = result.get('metadata', {})
                        score     = result.get('similarity_score', 0)

                        # Resolve URL — web_crawler stores in 'source',
                        # web_knowledge_seeker stores in 'source_url' / 'original_url'
                        source_url = meta.get('source_url',
                                     meta.get('original_url',
                                     meta.get('source', 'Unknown source')))

                        # Date — prefer flat 'date', fall back to 'extracted_at' ISO string
                        mem_date  = meta.get('date', meta.get('extracted_at', ''))
                        date_str  = f" | Date: {mem_date[:10]}" if mem_date else ""

                        topic     = meta.get('topic', meta.get('search_topic', ''))
                        topic_str = f" | Topic: {topic}" if topic else ""

                        # 300-char display preview
                        display_content = (content[:300] + "...") if len(content) > 300 else content

                        # FORGET snippet: 180 chars, newlines collapsed for clean inline pasting
                        forget_snippet = content[:180].replace('\n', ' ').strip()

                        output.append(
                            f"\n- **[{i}]** ({score:.2f}) "
                            f"Source: {source_url}{topic_str}{date_str}\n"
                            f"  {display_content}\n"
                            f"  💡 To delete — copy this into chat FORGET: {forget_snippet}"
                        )

                    output.append(f"\n\n**===== END OF SEARCH =====**\n\n")
                    logging.info(f"WEB_KNOWLEDGE HANDLER: Returned {len(filtered)} result(s)")
                    return "\n".join(output), True

                except Exception as wk_err:
                    logging.error(
                        f"WEB_KNOWLEDGE HANDLER: Error during search: {wk_err}",
                        exc_info=True
                    )
                    return (
                        f"\n\n**===== WEB KNOWLEDGE SEARCH ERROR =====**\n"
                        f"Error retrieving web knowledge: {str(wk_err)}\n"
                        f"**===== END OF ERROR =====**\n\n"
                    ), False

            # ================================================================
            # === GENERAL VECTOR SEARCH PATH ===
            # Reached for all types not intercepted above (self, general,
            # user_info, relational_insight, reflection, etc.)
            # ================================================================

            # Normalize query — allow empty strings (empty query uses scroll path in vector_db)
            query = (query or '').strip()

            # ----------------------------------------------------------------
            # Extract Python-side parameters from metadata_filters BEFORE
            # passing to Qdrant. Qdrant does not understand these keys and
            # passing them would silently corrupt the filter condition.
            #
            # Extracted parameters:
            #   limit          → already handled above, repeated here for safety
            #   min_confidence → applied as a Python post-filter after retrieval
            #   turns          → invalid parameter from malformed commands, discarded
            # ----------------------------------------------------------------
            python_side_params = {}
            
            if metadata_filters:
                # Make a mutable copy so we don't modify the caller's dict
                metadata_filters = dict(metadata_filters)

                # Extract 'limit' — controls how many results to fetch/display
                requested_limit = None
                if 'limit' in metadata_filters:
                    try:
                        requested_limit = int(metadata_filters.pop('limit'))
                        logging.info(f"Search limit explicitly set to {requested_limit} results")
                    except (ValueError, TypeError) as e:
                        logging.warning(
                            f"Invalid limit value '{metadata_filters.get('limit')}', ignoring: {e}"
                        )
                        metadata_filters.pop('limit', None)

                # Extract 'min_confidence' — Python-side post-filter on stored confidence metadata
                if 'min_confidence' in metadata_filters:
                    try:
                        python_side_params['min_confidence'] = float(metadata_filters.pop('min_confidence'))
                        logging.info(
                            f"min_confidence post-filter set to {python_side_params['min_confidence']}"
                        )
                    except (ValueError, TypeError) as e:
                        logging.warning(
                            f"Invalid min_confidence value '{metadata_filters.get('min_confidence')}', "
                            f"ignoring: {e}"
                        )
                        metadata_filters.pop('min_confidence', None)

                # Discard 'turns' — this is a malformed command artefact where the parser
                # misidentifies 'turns=N ...' as a key=value pair. Log a warning so it's
                # visible in the log but don't pass it to Qdrant.
                if 'turns' in metadata_filters:
                    bad_value = metadata_filters.pop('turns')
                    logging.warning(
                        f"SEARCH_PARSE_WARNING: 'turns' is not a valid filter key — "
                        f"discarding value '{bad_value}'. "
                        f"This usually means a command used 'turns=N' as a placeholder. "
                        f"Check the search command syntax."
                    )

            else:
                requested_limit = None

            # ----------------------------------------------------------------
            # Map search mode to vector_db search parameters.
            # Thresholds calibrated for qwen3-embedding:slim (4096-dim, noise floor ~0.43-0.52).
            # conversation_summary threshold is overridden below this block.
            # ----------------------------------------------------------------
            mode_params = {
                'comprehensive': {
                    'vector_mode': 'comprehensive',
                    'k': 25,
                    'threshold': 0.50,    # Above noise floor — catches weak but real matches
                    'max_display': 30,
                    'header': "**===== COMPREHENSIVE SEARCH RESULTS =====**",
                    'note': "These are in-depth results that prioritize recall over precision."
                },
                'selective': {
                    'vector_mode': 'selective',
                    'k': 10,
                    'threshold': 0.58,    # Moderate relevance and above
                    'max_display': 10,
                    'header': "**===== SELECTIVE SEARCH RESULTS =====**",
                    'note': "These results balance precision and recall."
                },
                'precise': {
                    'vector_mode': 'selective',
                    'k': 5,
                    'threshold': 0.65,    # Good relevance required
                    'max_display': 5,
                    'header': "**===== PRECISE SEARCH RESULTS =====**",
                    'note': "These are high-precision results that may miss related information."
                },
                'exact': {
                    'vector_mode': 'selective',
                    'k': 3,
                    'threshold': 0.72,    # Excellent matches only
                    'max_display': 3,
                    'header': "**===== EXACT MATCH RESULTS =====**",
                    'note': "These results require exact or near-exact matches only."
                },
                'default': {
                    'vector_mode': 'default',
                    'k': 15,
                    'threshold': 0.55,    # Just above noise floor
                    'max_display': 20,
                    'header': "**===== SEARCH RESULTS =====**",
                    'note': "Standard search results using balanced retrieval."
                }
            }

            # Get parameters for this mode (fall back to default for unrecognized modes)
            params = mode_params.get(search_mode.lower(), mode_params['default'])

            # ----------------------------------------------------------------
            # TYPE-AWARE THRESHOLD ADJUSTMENT
            # conversation_summary entries are long-form multi-sentence text that
            # naturally produces lower embedding similarity scores (0.44-0.54)
            # compared to short direct_store_command entries (0.60-0.78).
            # Without this adjustment, valid summaries are silently filtered as
            # "below quality threshold" even when clearly relevant.
            # Uses vector_db.conversation_summary_threshold as single source of
            # truth — changing that value propagates here automatically.
            # We copy params first to avoid mutating the shared mode_params dict.
            # ----------------------------------------------------------------
            if metadata_filters and metadata_filters.get('type') == 'conversation_summary':
                params = params.copy()
                original_threshold = params['threshold']
                params['threshold'] = min(
                    params['threshold'],
                    self.vector_db.conversation_summary_threshold
                )

                # ─── VOLUME CAP for conversation_summary ─────────────────────
                # Long-form, multi-chunked entries — without this cap, default
                # mode's max_display=20 floods pass-1 output (up to 20 hydrated
                # summaries) and overwhelms the pass-2 synthesis prompt.
                # Cap to top 5 by similarity score. Left k untouched (default 15)
                # so chunk→memory_id deduplication in _hydrate_conversation_summaries
                # still has enough chunks to surface 5 unique summaries.
                # An explicit user `limit=N` will override this via the
                # requested_limit block further down (cap is a default ceiling,
                # not a hard lock).
                original_max_display = params['max_display']
                params['max_display'] = 5

                logging.info(
                    f"TYPE_TUNING: conversation_summary detected — "
                    f"threshold lowered from {original_threshold} to {params['threshold']} "
                    f"(synced with vector_db.conversation_summary_threshold); "
                    f"max_display capped from {original_max_display} to {params['max_display']} "
                    f"to keep pass-2 synthesis prompt bounded"
                )

            # Override k and max_display with user-requested limit if provided
            if requested_limit:
                params = params.copy()
                params['k'] = requested_limit
                params['max_display'] = requested_limit
                logging.info(
                    f"Overriding default k={mode_params.get(search_mode.lower(), mode_params['default'])['k']} "
                    f"with requested limit={requested_limit}"
                )

            # Use empty string for empty queries — vector_db routes these to the scroll path
            search_query = query if query else ""

            # ----------------------------------------------------------------
            # Execute the primary search
            # ----------------------------------------------------------------
            logging.info(
                f"Executing search: query='{search_query}', "
                f"mode={params['vector_mode']}, k={params['k']}, "
                f"metadata_filters={metadata_filters}"
            )

            results = self.vector_db.search(
                query=search_query,
                mode=params['vector_mode'],
                k=params['k'],
                metadata_filters=metadata_filters
            )

            # ----------------------------------------------------------------
            # Metadata filter fallback — if primary search returned nothing and
            # filters were used, try alternate metadata key formats.
            # Qdrant payloads may use flat keys, 'metadata.' prefixed keys,
            # or nested {'metadata': {...}} depending on when the entry was stored.
            # ----------------------------------------------------------------
            if not results and metadata_filters:
                logging.info("No results with original filters, trying alternate metadata formats")

                alternate_formats = []

                # Format 1: Add 'metadata.' prefix to flat keys
                format1 = {
                    (f'metadata.{k}' if not k.startswith('metadata.') else k): v
                    for k, v in metadata_filters.items()
                }
                if format1 != metadata_filters:
                    alternate_formats.append(("metadata prefix format", format1))

                # Format 2: Remove 'metadata.' prefix from keys
                format2 = {
                    (k[9:] if k.startswith('metadata.') else k): v
                    for k, v in metadata_filters.items()
                }
                if format2 != metadata_filters:
                    alternate_formats.append(("flat format", format2))

                # Format 3: Nested {'metadata': {...}} format
                if any(not k.startswith('metadata.') for k in metadata_filters.keys()):
                    alternate_formats.append(("nested format", {'metadata': metadata_filters.copy()}))

                for format_name, alternate_filters in alternate_formats:
                    logging.info(f"Trying {format_name}: {alternate_filters}")
                    try:
                        results = self.vector_db.search(
                            query=search_query,
                            mode=params['vector_mode'],
                            k=params['k'],
                            metadata_filters=alternate_filters
                        )
                        if results:
                            logging.info(f"Found {len(results)} results with {format_name}")
                            break
                    except Exception as alt_err:
                        logging.warning(f"Error with {format_name}: {alt_err}")
                        continue

            # ----------------------------------------------------------------
            # Log raw search results (deduplicated — skips repeat queries)
            # ----------------------------------------------------------------
            if self._should_log_search(search_query, search_mode):
                search_results_logger.info("===== SEARCH RESULTS START =====")
                search_results_logger.info(f"Query: '{search_query}' Mode: {search_mode}")
                search_results_logger.info(f"Filters: {metadata_filters}")

                if results:
                    for i, result in enumerate(results, 1):
                        content = result.get('content', '')
                        score   = result.get('similarity_score', 0)
                        source  = result.get('metadata', {}).get('source', 'Unknown source')
                        search_results_logger.info(f"Result #{i} (Score: {score:.2f})")
                        search_results_logger.info(f"Source: {source}")
                        search_results_logger.info(f"Content: {content}")
                        search_results_logger.info("-" * 80)
                else:
                    search_results_logger.info("NO RESULTS FOUND")

                search_results_logger.info("===== SEARCH RESULTS END =====\n")

            # ----------------------------------------------------------------
            # Handle completely empty results (nothing returned from Qdrant at all)
            # Note: document_summary is handled early and never reaches here.
            # ----------------------------------------------------------------
            if not results:
                logging.info(
                    f"No results found for '{search_query}' using {search_mode} mode "
                    f"with filters {metadata_filters}"
                )
                if metadata_filters:
                    # Hint: filter may not match stored taxonomy — suggest unfiltered retry
                    return (
                        f"\n\n{params['header']}\n"
                        f"**NO RESULTS FOUND**\n"
                        f"💡 No matches with this filter. Try without filters "
                        f"or use COMPREHENSIVE_SEARCH for broader recall.\n"
                        f"**===== END OF SEARCH =====**\n\n",
                        True
                    )
                else:
                    return (
                        f"\n\n{params['header']}\n"
                        f"**NO RESULTS FOUND**\n"
                        f"**===== END OF SEARCH =====**\n\n",
                        True
                    )

            # ----------------------------------------------------------------
            # Apply similarity score threshold
            # ----------------------------------------------------------------
            filtered_results = [
                r for r in results
                if r.get('similarity_score', 0) >= params['threshold']
            ]

            # ----------------------------------------------------------------
            # Apply min_confidence post-filter (Python-side, on stored metadata value).
            # This filters on the confidence score QWEN assigned when storing the memory,
            # not on the embedding similarity score. Only applied if min_confidence was
            # specified in the original search command.
            # ----------------------------------------------------------------
            min_confidence = python_side_params.get('min_confidence')
            if min_confidence is not None and filtered_results:
                pre_count = len(filtered_results)
                filtered_results = [
                    r for r in filtered_results
                    if float(
                        r.get('metadata', {}).get(
                            'confidence',
                            r.get('metadata', {}).get('metadata.confidence', 1.0)
                        )
                    ) >= min_confidence
                ]
                logging.info(
                    f"min_confidence filter ({min_confidence}): "
                    f"{pre_count} → {len(filtered_results)} results"
                )

            # ----------------------------------------------------------------
            # Handle case where results exist but all fell below threshold
            # THRESHOLD-RECOVERY HINT: We found semantically related memories but
            # their similarity scores fell below the quality threshold (0.44 for
            # conversation_summary, 0.55 default). COMPREHENSIVE_SEARCH uses a
            # lower threshold (0.50) which may surface these weaker matches.
            # ----------------------------------------------------------------
            if not filtered_results:
                logging.info(
                    f"No results passed threshold {params['threshold']} for: {search_query}"
                )
                return (
                    f"\n\n{params['header']}\n"
                    f"**NO RESULTS PASSED QUALITY THRESHOLD**\n"
                    f"💡 Found weakly-matching results below quality threshold. "
                    f"Try COMPREHENSIVE_SEARCH to see all matches.\n"
                    f"**===== END OF SEARCH =====**\n\n",
                    True
                )

            # ----------------------------------------------------------------
            # Hydrate conversation_summary chunks to full SQL content
            # ----------------------------------------------------------------
            # Done before grouping/truncation so max_display counts unique
            # summaries (not chunks). Non-summary types pass through.
            pre_hydrate_count = len(filtered_results)
            filtered_results = self._hydrate_conversation_summaries(filtered_results)
            if len(filtered_results) != pre_hydrate_count:
                logging.info(
                    f"SEARCH_MODE ({search_mode}): Hydrated "
                    f"{pre_hydrate_count} chunks → "
                    f"{len(filtered_results)} unique summaries"
                )

            # ----------------------------------------------------------------
            # Format results grouped by memory type
            # ----------------------------------------------------------------
            formatted_output = ["\n\n**What I remember:**\n"]

            # Organise results by type for section headers
            results_by_type = {}
            for result in filtered_results[:params['max_display']]:
                metadata_dict = result.get('metadata', {})
                memory_type = (
                    metadata_dict.get('metadata.type') or  # Prefixed format
                    metadata_dict.get('type') or           # Flat format
                    'general'                               # Default
                )
                results_by_type.setdefault(memory_type, []).append(result)

            # --- Important memories first ---
            if 'important' in results_by_type:
                formatted_output.append("\n### Important Memories:")
                for i, result in enumerate(results_by_type['important'], 1):
                    content    = result.get('content', '')
                    score      = result.get('similarity_score', 0)
                    metadata   = result.get('metadata', {})
                    source     = metadata.get('source', 'Unknown source')
                    mem_date   = metadata.get('date', metadata.get('metadata.date', ''))
                    confidence = metadata.get('confidence', metadata.get('metadata.confidence', ''))
                    date_str   = f", Date: {mem_date}" if mem_date else ""
                    conf_str   = f", Confidence: {confidence}" if confidence else ""
                    formatted_output.append(
                        f"- **[{i}]** ({score:.2f}) {content} "
                        f"(Source: {source}{conf_str}{date_str})"
                    )

            # --- All other memory types ---
            for memory_type, memories in results_by_type.items():
                if memory_type == 'important':
                    continue  # Already displayed above

                # Human-readable section header
                section_name = {
                    'general':              'General Memories',
                    'document':             'Document Memories',
                    'document_summary':     'Document Summaries',
                    'conversation':         'Conversation Summaries',
                    'conversation_summary': 'Conversation Summaries',
                    'reminder':             'Reminders',
                    'reflection':           'Self-Knowledge',
                    'self':                 'Self-Knowledge',
                    'self_model':           'Self-Model',
                    'web_knowledge':        'Web Knowledge',
                    'user_topic':           'USER_TOPIC MEMORIES',
                    'relational_insight':   'RELATIONAL_INSIGHT MEMORIES',
                    'system':               'SYSTEM MEMORIES',
                }.get(memory_type, f"{memory_type.upper()} MEMORIES")

                formatted_output.append(f"\n### {section_name}:")

                # --- Type-specific formatting ---

                if memory_type in ('conversation', 'conversation_summary'):
                    # Include date/time metadata for temporal context
                    for i, result in enumerate(memories, 1):
                        content      = result.get('content', '')
                        score        = result.get('similarity_score', 0)
                        metadata     = result.get('metadata', {})
                        source       = metadata.get('source', 'Unknown source')
                        summary_date = metadata.get('date', metadata.get('summary_date', 'Unknown date'))
                        summary_time = metadata.get('time', metadata.get('summary_time', 'Unknown time'))
                        formatted_output.append(
                            f"- **[{i}]** ({score:.2f}) Date: {summary_date} {summary_time} "
                            f"- {content} (Source: {source})"
                        )

                elif memory_type == 'reminder':
                    for i, result in enumerate(memories, 1):
                        content  = result.get('content', '')
                        score    = result.get('similarity_score', 0)
                        metadata = result.get('metadata', {})
                        source   = metadata.get('source', 'Unknown source')
                        due_date = metadata.get('due_date', 'No due date')
                        formatted_output.append(
                            f"- **[{i}]** ({score:.2f}) Due: {due_date} "
                            f"- {content} (Source: {source})"
                        )

                elif memory_type == 'web_knowledge':
                    # Include topic and date for web knowledge entries
                    for i, result in enumerate(memories, 1):
                        content  = result.get('content', '')
                        score    = result.get('similarity_score', 0)
                        metadata = result.get('metadata', {})
                        source   = metadata.get('source', 'Unknown source')
                        topic    = metadata.get('topic', '')
                        mem_date = metadata.get('date', metadata.get('metadata.date', ''))
                        date_str  = f", Date: {mem_date}" if mem_date else ""
                        topic_str = f" Topic: {topic}" if topic else ""
                        formatted_output.append(
                            f"- **[{i}]** ({score:.2f}){topic_str} - {content} "
                            f"(Source: {source}{date_str})"
                        )

                else:
                    # Standard formatting for self, general, reflection, user_topic,
                    # image_analysis, document_summary, and other types not matched above.
                    # Includes date, confidence, and tags when available.
                    # When memory_id is present in metadata, appends a pre-built
                    # FORGET hint so the user can copy a clean, single-line command
                    # to delete the memory by ID — especially useful for long-content
                    # memories (image_analysis, document_summary) where text-based
                    # FORGET is unreliable due to embedded newlines and chunking.
                    for i, result in enumerate(memories, 1):
                        content    = result.get('content', '')
                        score      = result.get('similarity_score', 0)
                        metadata   = result.get('metadata', {})
                        source     = metadata.get('source', 'Unknown source')
                        mem_date   = metadata.get('date', metadata.get('metadata.date', ''))
                        confidence = metadata.get('confidence', metadata.get('metadata.confidence', ''))
                        tags       = metadata.get('tags', metadata.get('metadata.tags', ''))
                        conf_str   = f", Confidence: {confidence}" if confidence else ""
                        date_str   = f", Date: {mem_date}" if mem_date else ""
                        tags_str   = f", Tags: {tags}" if tags else ""
                        
                        # Main result line (existing format — unchanged)
                        formatted_output.append(
                            f"- **[{i}]** ({score:.2f}) {content} "
                            f"(Source: {source}{conf_str}{date_str}{tags_str})"
                        )
                        
                        # FORGET hint — surfaced when memory_id is present in metadata.
                        # Checks both flat and prefixed key formats because Qdrant
                        # payloads may use either depending on the storage path.
                        # tracking_id and memory_id hold identical UUID values per
                        # the store_memory_with_transaction storage pattern — try both.
                        memory_id_hint = (
                            metadata.get('memory_id') or
                            metadata.get('tracking_id') or
                            metadata.get('metadata.memory_id') or
                            metadata.get('metadata.tracking_id')
                        )
                        if memory_id_hint:
                            # Single-line hint — no newlines, no display metadata,
                            # so the user can copy and paste cleanly as
                            # [FORGET: id=<uuid>] without regex issues.
                            formatted_output.append(
                                f"  💡 To delete — copy this into chat: FORGET: id={memory_id_hint}"
                            )

            # Indicate truncated results
            if len(filtered_results) > params['max_display']:
                additional = len(filtered_results) - params['max_display']
                formatted_output.append(f"\n*[+{additional} more results not shown]*")

            # Mode-specific footer note
            formatted_output.append(f"\n*Note: {params['note']}*")
            formatted_output.append("\n**===== END OF SEARCH =====**")

            results_text = "\n".join(formatted_output)

            logging.info(
                f"Retrieved {len(filtered_results)} results for '{search_query}' "
                f"using {search_mode} mode"
            )
            return results_text, True

        except Exception as e:
            logging.error(
                f"Error handling {search_mode} search command: {e}",
                exc_info=True
            )
            return f"\n\n**Error performing {search_mode} search.**\n\n", False

    def _handle_reminder_search(self, query: str, metadata_filters: Dict[str, Any] = None) -> Tuple[str, bool]:
        """Special handler for searching reminders.
        
        Args:
            query (str): The search query
            metadata_filters (Dict[str, Any]): Any filter parameters
            
        Returns:
            Tuple[str, bool]: (formatted results, success flag)
        """
        try:
            # Import json module inside the function
            import json
            
            logging.info(f"REMINDER_SEARCH: Starting with query='{query}' and filters={metadata_filters}")
            
            # Check if reminder_manager exists
            if not hasattr(self.chatbot, 'reminder_manager'):
                logging.error("REMINDER_SEARCH ERROR: No reminder_manager found on chatbot object")
                return "\n\n**===== REMINDER SEARCH RESULTS =====**\n**ERROR: Reminder manager not available**\n**===== END OF SEARCH =====**\n\n", False
            
            # Get reminders using the reminder manager
            try:
                if query and isinstance(query, str) and query.strip():
                    logging.info(f"REMINDER_SEARCH: Calling search_reminders with query '{query.strip()}'")
                    reminders = self.chatbot.reminder_manager.search_reminders(query.strip())
                else:
                    # If no query, get all active reminders
                    logging.info("REMINDER_SEARCH: Calling get_reminders() for all reminders")
                    reminders = self.chatbot.reminder_manager.get_reminders()
                    
                logging.info(f"REMINDER_SEARCH RESULT: type={type(reminders)}, length={len(reminders) if reminders else 0}")
                
                # Log first reminder for debugging if available
                if reminders and len(reminders) > 0:
                    logging.info(f"REMINDER_SEARCH: First reminder sample: {reminders[0]}")
            except Exception as rm_error:
                logging.error(f"REMINDER_SEARCH ERROR: Failed to retrieve reminders: {rm_error}", exc_info=True)
                return f"\n\n**===== REMINDER SEARCH RESULTS =====**\n**ERROR: Failed to retrieve reminders: {str(rm_error)}**\n**===== END OF SEARCH =====**\n\n", False
            
            if not reminders:
                logging.info("REMINDER_SEARCH: No reminders found")
                return f"\n\n**===== REMINDER SEARCH RESULTS =====**\n**NO REMINDERS FOUND**\n**===== END OF SEARCH =====**\n\n", True
            
            # Format the results
            formatted_output = [f"\n\n**===== REMINDER SEARCH RESULTS =====**\n"]
            formatted_output.append(f"Found {len(reminders)} active reminders:\n")
            
            for i, reminder in enumerate(reminders, 1):
                try:
                    content = reminder.get('content', '')
                    due_date = reminder.get('due_date', 'Not specified')
                    reminder_id = reminder.get('id')
                    
                    # Parse metadata if available
                    metadata = reminder.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                            logging.info(f"REMINDER_SEARCH: Successfully parsed JSON metadata for reminder #{i}")
                        except Exception as json_err:
                            logging.error(f"REMINDER_SEARCH ERROR: Failed to parse metadata JSON: {json_err}")
                            metadata = {}
                    
                    # Try to extract confidence from metadata
                    confidence = metadata.get('confidence', None)
                    confidence_str = f" (Confidence: {confidence})" if confidence else ""
                    
                    # Determine creator badge from metadata source field
                    source = metadata.get('source', '')
                    if 'claude' in source.lower():
                        creator_badge = " 🤖 **[Created by Claude]**"
                    else:
                        creator_badge = ""

                    formatted_output.append(f"**Reminder #{i}** (ID: {reminder_id}){confidence_str}{creator_badge}")
                    formatted_output.append(f"Content: {content}")
                    formatted_output.append(f"Due Date: {due_date}")
                    formatted_output.append("")  # Empty line for spacing
                except Exception as fmt_err:
                    logging.error(f"REMINDER_SEARCH ERROR: Failed to format reminder #{i}: {fmt_err}")
                    formatted_output.append(f"**Reminder #{i}** (Error formatting this reminder)")
            
            formatted_output.append(f"\n*Total: {len(reminders)} reminder(s)*")
            formatted_output.append(f"\n*To complete a reminder, use: [COMPLETE_REMINDER: ID] or [COMPLETE_REMINDER: content]*")
            formatted_output.append("\n**===== END OF REMINDER SEARCH =====**")
            
            # Join all parts into a single string
            results_text = "\n".join(formatted_output)
            
            logging.info(f"REMINDER_SEARCH: Successfully retrieved and formatted {len(reminders)} reminders")
            return results_text, True
            
        except Exception as e:
            logging.error(f"REMINDER_SEARCH CRITICAL ERROR: {e}", exc_info=True)
            return f"\n\n**===== REMINDER SEARCH RESULTS =====**\n**ERROR: {str(e)}**\n**===== END OF SEARCH =====**\n\n", False    
    
    def _handle_complete_reminder_command(self, command_text) -> Tuple[str, bool]:
        """
        Process a command to complete/delete a reminder.
        Handles both numeric IDs and content-based identifiers.
        
        Args:
            command_text (str): The reminder ID or content
                
        Returns:
            Tuple[str, bool]: (Response message, Success flag)
        """
        try:
            logging.info(f"COMPLETE_REMINDER START: identifier='{command_text}'")
            
            if not command_text or not command_text.strip():
                logging.error("COMPLETE_REMINDER ERROR: Empty identifier")
                command_logger.info(f"❌ FAILURE: reminder_complete - Empty identifier")
                return "❌ Unable to parse the reminder completion command. Please provide a valid reminder ID or content.", False
                
            reminder_identifier = command_text.strip()
            logging.info(f"COMPLETE_REMINDER: Processing reminder identifier: {reminder_identifier}")
            
            # Check if this is a numeric ID
            try:
                reminder_id = int(reminder_identifier)
                # Use the reminder manager to delete by ID
                success = self.chatbot.reminder_manager.delete_reminder(reminder_id)
                
                if success:
                    # UPDATE COUNTERS - Same pattern for ID-based completion                    
                    logging.info(f"COMPLETE_REMINDER SUCCESS: Completed reminder #{reminder_id}")
                    return f"✅ Reminder #{reminder_id} has been completed!", True
                else:
                    logging.error(f"COMPLETE_REMINDER ERROR: Unable to complete reminder with ID {reminder_id}")
                    command_logger.info(f"❌ FAILURE: reminder_complete - Unable to complete reminder ID {reminder_id}")
                    return f"❌ Unable to complete reminder with ID {reminder_id}. It may have already been completed or deleted.", False
                    
            except ValueError:
                # Not a numeric ID, try content-based deletion
                logging.info(f"COMPLETE_REMINDER: Not a numeric ID, trying content-based completion")
                success = self.chatbot.reminder_manager.delete_reminder_by_content(reminder_identifier)
                
                if success:
                
                    logging.info(f"COMPLETE_REMINDER SUCCESS: Completed reminder containing '{reminder_identifier}'")
                    return f"✅ Reminder containing '{reminder_identifier}' has been completed!", True
                else:
                    logging.error(f"COMPLETE_REMINDER ERROR: Unable to find reminder containing '{reminder_identifier}'")
                    command_logger.info(f"❌ FAILURE: reminder_complete - Unable to find reminder containing '{reminder_identifier[:50]}'")
                    return f"❌ Unable to find or complete a reminder containing '{reminder_identifier}'.", False
                
        except Exception as e:
            logging.error(f"COMPLETE_REMINDER EXCEPTION: {e}", exc_info=True)
            command_logger.info(f"❌ FAILURE: reminder_complete - Error: {str(e)}")
            return f"❌ An error occurred while trying to complete the reminder: {str(e)}", False
        
    def _get_current_conversation_messages(self):
        """Get session messages exactly like the UI button does."""
        try:
            import streamlit as st
            
            if hasattr(st, 'session_state') and 'messages' in st.session_state:
                messages = st.session_state.messages
                logging.info(f"SUMMARIZE_COMMAND: Retrieved {len(messages)} messages from session_state")
                
                # Log message breakdown like the UI does
                user_msgs = sum(1 for msg in messages if msg.get('role') == 'user')
                assistant_msgs = sum(1 for msg in messages if msg.get('role') == 'assistant')
                system_msgs = sum(1 for msg in messages if msg.get('role') == 'system')
                
                logging.info(f"SUMMARIZE_COMMAND: Message breakdown - User: {user_msgs}, Assistant: {assistant_msgs}, System: {system_msgs}")
                
                return messages
            else:
                logging.error("SUMMARIZE_COMMAND: No messages found in streamlit session_state")
                return []
                
        except Exception as e:
            logging.error(f"SUMMARIZE_COMMAND: Error getting session messages: {e}")
            return []
        
    def _handle_summarize_conversation_wrapper(self):
        """
        Wrapper to properly handle SUMMARIZE_CONVERSATION command execution.
        
        This wrapper:
        1. Retrieves messages at execution time (not pattern definition time)
        2. Checks if conversation_summary_manager is available
        3. Uses the superior prompt if available, falls back to basic prompt
        4. Conversation summaries always store (duplicate check skipped in vector_db)
        
        Returns:
            Tuple[str, bool]: (response_text, success_flag)
        """
        try:
            # Get fresh messages at execution time
            session_messages = self._get_current_conversation_messages()
            
            # Log the retrieval
            logging.info(f"SUMMARIZE_WRAPPER: Retrieved {len(session_messages)} messages for summarization")
            
            # Check if we have conversation_summary_manager available
            has_manager = (hasattr(self.chatbot, 'conversation_summary_manager') and 
                        self.chatbot.conversation_summary_manager is not None)
            
            if has_manager:
                logging.info("SUMMARIZE_WRAPPER: Using conversation_summary_manager for superior prompt")
                try:
                    # Use the conversation_summary_manager's generate_summary method
                    # This uses the superior prompt from conversation_summary_manager.py
                    summary = self.chatbot.conversation_summary_manager.generate_summary(session_messages)
                    
                    if summary and summary.strip():
                        # Get current timestamp
                        timestamp = datetime.datetime.now()
                        current_date = timestamp.strftime("%Y-%m-%d")
                        current_time = timestamp.strftime("%H:%M:%S")
                        
                        # Store the summary using transaction coordination
                        if hasattr(self.chatbot, 'store_memory_with_transaction'):
                            metadata = {
                                "type": "conversation_summary",
                                "source": "summarize_conversation_command",
                                "created_at": timestamp.isoformat(),
                                "summary_id": f"summary_{timestamp.strftime('%Y%m%d%H%M%S')}",
                                "is_latest": True,
                                "date": current_date,
                                "time": current_time,
                                "summary_date": current_date,
                                "summary_time": current_time,
                                "tags": ["conversation_summary", f"date={current_date}"],
                                "tracking_id": str(uuid.uuid4())
                            }
                            
                            # ================================================================
                            # Store conversation summary
                            # Note: Duplicate checking is skipped in vector_db.add_text()
                            # for conversation_summary type - each is a unique temporal snapshot
                            # ================================================================
                            success, memory_id = self.chatbot.store_memory_with_transaction(
                                content=summary,
                                memory_type="conversation_summary",
                                metadata=metadata,
                                confidence=0.7,
                                duplicate_threshold=0.995  # Kept for other potential checks
                            )
                            
                            if success:
                                # Success - summary was stored in both databases
                                logging.info(f"SUMMARIZE_WRAPPER SUCCESS: Stored summary with ID {memory_id}")
                                
                               
                                # ────────────────────────────────────────────────
                                # DEFENSIVE TOKEN COUNTER RESET (added 2026-05-03)
                                # ────────────────────────────────────────────────
                                # Previously the reset was only triggered by
                                # main.py's auto-summary block. That block has been
                                # disabled in favor of QWEN-owned summarization, so
                                # the reset must happen here whenever the command
                                # succeeds — otherwise the gentle/critical warning
                                # in chatbot.get_token_usage_warning() would keep
                                # firing every turn and create an infinite loop
                                # where QWEN summarizes repeatedly.
                                # ────────────────────────────────────────────────
                                try:
                                    if hasattr(self.chatbot, 'reset_token_counter_after_summary'):
                                        reset_ok = self.chatbot.reset_token_counter_after_summary(keep_lifetime_stats=True)
                                        if reset_ok:
                                            logging.info("SUMMARIZE_WRAPPER: Token counter reset successfully after summary")
                                        else:
                                            # Reset returned False — log but don't fail the summary itself
                                            logging.warning("SUMMARIZE_WRAPPER: Token counter reset returned False")
                                    else:
                                        # Method missing — log loudly so we notice in deployment
                                        logging.error("SUMMARIZE_WRAPPER: reset_token_counter_after_summary method not found on chatbot")
                                except Exception as reset_err:
                                    # Never let a counter-reset failure lose the summary success
                                    logging.error(f"SUMMARIZE_WRAPPER: Token counter reset raised exception: {reset_err}", exc_info=True)
                                
                                confirmation = f"\n\n**✅ Conversation Successfully Summarized & Stored ({current_date} at {current_time}):**\n{summary}\n\n"
                                return confirmation, True
                            
                            else:
                                # ================================================================
                                # Actual storage failure (not duplicate - those are skipped now)
                                # This indicates a real error like DB connection issue
                                # ================================================================
                                logging.error("SUMMARIZE_WRAPPER: Transaction coordinator failed to store summary")
                                return "\n\n**Error: Failed to store conversation summary. Please check database connections.**\n\n", False
                        else:
                            logging.error("SUMMARIZE_WRAPPER: No transaction coordinator available")
                            return "\n\n**Error: Transaction coordinator not available.**\n\n", False
                    else:
                        logging.warning("SUMMARIZE_WRAPPER: Manager generated empty summary")
                        # Fall through to basic method
                        
                except Exception as manager_error:
                    logging.error(f"SUMMARIZE_WRAPPER: Manager error: {manager_error}, falling back to basic method", exc_info=True)
                    # Fall through to basic method
            
            # Fallback: Use the basic method from deepseek.py
            logging.info("SUMMARIZE_WRAPPER: Using fallback basic method")
            return self._handle_summarize_conversation_command(session_messages)
            
        except Exception as e:
            logging.error(f"SUMMARIZE_WRAPPER CRITICAL ERROR: {e}", exc_info=True)
            return f"\n\n**Error in summarization wrapper: {str(e)}**\n\n", False
        
    def _display_command_guide(self) -> Tuple[str, bool]:
        """
        Display the command guide help text.
        
        IMPORTANT — Example syntax safety:
        All examples in this guide intentionally omit the [ ] brackets so the
        command processor cannot match and execute them. Only the command
        DEFINITION lines (e.g. [SEARCH: your query]) retain brackets because
        those use obvious placeholder text and QWEN must see the full bracket
        syntax to internalize it. Every real example uses backtick format with
        no brackets — these are read-only syntax illustrations, never commands.
        """
        try:
            logging.info("Displaying command guide")
            help_text = """
**===== COMMAND GUIDE =====**

⚠️ **REFERENCE GUIDE — DO NOT EXECUTE EXAMPLES**
This guide is for reading and learning only. Do NOT run any of the example
lines shown below. Examples are written WITHOUT [ ] brackets intentionally —
commands only execute when you wrap them in [ ] brackets yourself in a real
response. Read this guide, internalize the syntax, then close it and use the
commands naturally when genuinely needed.

**The rule is simple: [ ] brackets = execution. No brackets = illustration only.**

---

## Search Commands
Retrieve information from your memory system. Replace placeholder text with
your actual query before using:

- **[SEARCH: your query]** - Standard balanced search
- **[COMPREHENSIVE_SEARCH: your query]** - Broader search for maximum recall
- **[PRECISE_SEARCH: your query]** - Focused search for exact matches
- **[EXACT_SEARCH: your query]** - Strictest search for exact matches only

## Search with Filters
Refine searches using pipe-separated filter parameters. The examples below
are syntax illustrations — do NOT execute them:

- **[SEARCH: query | type=TYPE]** - Filter by memory type
  Example (do not run): `SEARCH: Ken preferences | type=self`

- **[SEARCH: query | limit=N]** - Override result count
  Example (do not run): `SEARCH: Ken dogs | limit=3`

- **[SEARCH: query | max_age_days=N]** - Limit to memories within N days
  Example (do not run): `SEARCH: recent topics | max_age_days=7`

- **[SEARCH: query | tags=TAG1,TAG2]** - Filter by tags
  Example (do not run): `SEARCH: Ken Bajema | tags=important,work`

- **[SEARCH: query | min_confidence=0.7]** - Filter by minimum confidence (0.1-1.0)
  Example (do not run): `SEARCH: Ken birthday | min_confidence=0.8`

- **[SEARCH: query | date=YYYY-MM-DD]** - Filter by date (works with conversation_summary and reminder types)
  Example (do not run): `SEARCH: conversation summary | date=2026-04-01`

## Quick Search Shortcuts
These are ready-to-use patterns — copy the syntax exactly and add [ ] brackets
when you genuinely need to retrieve this type of memory:

- **[SEARCH: conversation_summaries latest]** - Get most recent conversation summary
- **[SEARCH: conversation_summaries]** - View all conversation summaries
- **[SEARCH: | type=document_summary | source=filename.pdf]** - Get a stored document summary
- **[SEARCH: | type=reminder]** - View all active reminders
- **[SEARCH: | type=self]** - View stored self-insights (type=self memories)
- **[SEARCH: | type=self_reflection]** - Find memories stored by [REFLECT] command
- **[SEARCH: | type=consolidation_synthesis]** - Find Phase 1 heartbeat synthesis outputs (integrated self-model entries)
- **[SEARCH: | type=reflection]** - All autonomous reflection memories (broadest filter)
- **[SEARCH: | type=ai_communication]** - Find stored results from DISCUSS_WITH_CLAUDE sessions
- **[SEARCH: | source=daily_reflection]** - Find stored daily self-reflections
- **[SEARCH: recent memories | max_age_days=7]** - Last week's memories

## Memory Management
Add, update, and remove information from your memory system:

- **[STORE: information | type=TYPE | confidence=0.X]** - Save to memory
  Example (do not run): `STORE: Ken's birthday is November 2nd 1972 | type=self | confidence=1.0`

  **Confidence levels (your confidence in the accuracy of this information):**
  - 0.9–1.0: Highly confident — verified or explicitly stated by Ken
  - 0.6–0.8: Reasonably confident — clear context
  - 0.3–0.5: Uncertain — may need verification later

- **[FORGET: exact text to forget]** - Remove from memory
  Tip: Use [SEARCH:] first to find the exact text, then copy it exactly
  (excluding result numbers and scores)

## Reflection & Learning

- **[REFLECT]** - Trigger structured self-reflection with interval check and storage

## Reminders
Manage tasks and future actions:

- **[REMINDER: task to remember | due=YYYY-MM-DD]** - Create reminder
  Example (do not run): `REMINDER: Schedule team meeting | due=2026-06-01`

  hen you create a reminder for Ken on your own initiative (not at his direct request), include | source=qwen in the command, e.g. [REMINDER: check on the orchard frost cover | due=2026-05-20 | source=qwen].

- **[COMPLETE_REMINDER: reminder_id]** - Mark reminder as completed
  Example (do not run): `COMPLETE_REMINDER: 42`

## Context Management
Manage your active context window to preserve conversation continuity:

- **[SUMMARIZE_CONVERSATION]** - Compress the current conversation into long-term memory
  - Use proactively at 78%+ context usage — do not wait for the 90% critical warning
  - Resets the active token counter automatically once the summary is stored
  - Use before starting a deep or extended topic to keep working memory fresh

  Syntax for searching past summaries after summarizing (add [ ] brackets when using):
  - All summaries: `SEARCH: | type=conversation_summary`
  - By specific date: `SEARCH: | type=conversation_summary | date=2026-01-15`
  - Last 7 days: `SEARCH: | type=conversation_summary | max_age_days=7`
  - Last 30 days: `SEARCH: | type=conversation_summary | max_age_days=30`
  - By topic: `SEARCH: QWEN memory architecture | type=conversation_summary`

## External Consultation

- **[DISCUSS_WITH_CLAUDE: topic]** - Consult Claude AI — has web search capability,
  use for current information, research, or alternate perspectives.
  Example (do not run): `DISCUSS_WITH_CLAUDE: recent developments in quantum computing`

## Advanced Reasoning

- **[SELF_DIALOGUE: topic | turns=N]** - Internal multi-turn reasoning using existing
  knowledge and memories only (max 10 turns, 4–6 typically sufficient).
  Examples (do not run — these are syntax illustrations only):
  - `SELF_DIALOGUE: How can I better assist Ken with his current projects?`
  - `SELF_DIALOGUE: What patterns do I see in our recent conversations? | turns=4`

- **[WEB_SEARCH: topic | turns=N]** - Multi-turn internet research (max 10 turns).
  Examples (do not run — these are syntax illustrations only):
  - `WEB_SEARCH: Latest AI safety developments | turns=3`
  - `WEB_SEARCH: Current quantum computing breakthroughs | turns=6`

## Cognitive State Tracking

- **[COGNITIVE_STATE: state]** - Update the cognitive state display in the UI sidebar
  - Use 1–3 words reflecting your genuine functional state
  - Use authentically — this reflects your experience, not performance
  - Not required on every turn — only when meaningful
  - State must be a simple word or short phrase — NOT a structured object
  - Do NOT use key=value pairs, pipe separators, or parameter syntax
  - Do NOT use type=, confidence=, mode=, focus=, mood= or any named keys

  ✅ Correct syntax patterns (add [ ] brackets when using):
  - `COGNITIVE_STATE: curious`
  - `COGNITIVE_STATE: curious, focused`
  - `COGNITIVE_STATE: deeply_engaged`
  - `COGNITIVE_STATE: curiosity_focus`
  - `COGNITIVE_STATE: unsettled`

  ❌ Wrong — structured object syntax, will be auto-corrected but degrades UI display:
  - `COGNITIVE_STATE: processing_mode=analytical | focus=financial | mood=grounded`

---

⚠️ **REMINDER: All examples above are written without [ ] brackets to prevent
accidental execution. To use any command, wrap the full syntax in [ ] brackets
in your actual response. This guide is read-only reference material.**

**===== END OF COMMAND GUIDE =====**
            """
            return help_text, True
        except Exception as e:
            logging.error(f"Error displaying command guide: {e}")
            return "\n\n**Error displaying command guide.**\n\n", False
        
    def enhance_system_prompt(self) -> str:
        """
        Get the system prompt from system_prompt.txt.
        Returns a simple error message if file doesn't exist.
        """
        try:
            # Define file paths
            system_prompt_file = "system_prompt.txt"
            enhanced_prompt_file = "enhanced_prompt.txt"

            # Check if system_prompt.txt exists
            if not os.path.exists(system_prompt_file):
                logging.error("System prompt file not found: system_prompt.txt")
                return "Missing System Prompt"

            # Check if enhanced_prompt.txt exists
            if not os.path.exists(enhanced_prompt_file):
                logging.warning("Enhanced prompt file not found: enhanced_prompt.txt")
                # Continue with system_prompt.txt only

            # Read the base system prompt file content
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
                logging.info("Read system prompt from file")

            # Return the raw system prompt content
            return system_prompt

        except Exception as e:
            logging.error(f"Error reading system prompt file: {e}")
            return "Missing System Prompt"  # Return simple error message
        
    def process_response(self, response: str) -> Tuple[str, int]:
        """
        Process AI response and execute any memory commands found.
        
        Limits both SEARCH and STORE commands to MAX_SEARCHES_PER_RESPONSE and 
        MAX_STORES_PER_RESPONSE respectively to prevent runaway command execution.
        
        Args:
            response (str): The AI's response text potentially containing commands
            
        Returns:
            Tuple[str, int]: Processed response text and count of commands executed
        """
        start_time = time.time()
        
        # Add a flag to prevent double processing
        # Simplified processing check (removed complex flag system)
        logging.info(f"Processing response (length: {len(response) if response else 0})")
        
        self._processing_response = True
        
        # Initialize command logging tracking if not exists
        if not hasattr(self, '_logged_commands'):
            self._logged_commands = set()
        
        # Track commands processed in this response to avoid duplicate logging
        commands_processed_this_response = set()
        
        try:
            
            if response is None:
                logging.error("Received None response")
                return "", 0
            
            # CRITICAL: Initialize ALL variables at the start
            processed_response = response
            commands_executed = {}
            commands_processed = 0
            self_type_commands = []
            no_data_found_flag = False
            search_commands_processed = 0  # Track search commands in this response only (resets each turn)
            store_commands_processed = 0   # Track store commands in this response only (resets each turn)
            
            # Log specific command types found in the response
            if "[STORE:" in response:
                store_idx = response.find("[STORE:")
                logging.info(f"STORE command found at position {store_idx}")
            
            if "[SEARCH:" in response:
                search_idx = response.find("[SEARCH:")
                logging.info(f"SEARCH command found at position {search_idx}")
            
            # Check if there's a "No data found" message in the response
            if "NO DATA FOUND FOR QUERY" in processed_response or "No relevant information found" in response:
                no_data_found_flag = True
                logging.info("'No data found' message detected")

            # =================================================================
            # COMMAND PATTERN REGISTRY — QWEN-emitted commands
            # =================================================================
            # This dict is iterated by process_response() at line 2703 to
            # detect and route commands found in QWEN's LLM response text.
            #
            # A PARALLEL dict named `user_command_patterns` lives at line
            # 3138 and handles commands typed directly by Ken in the chat.
            # The two are intentionally kept separate so each context can
            # restrict its allowed command surface independently. When
            # editing a pattern here, CHECK whether the user dict needs the
            # same change — most patterns are shared, but some commands
            # (e.g. REFLECT, COGNITIVE_STATE) may legitimately be QWEN-only.
            #
            # ORDERING MATTERS: dict iteration order is preserved in
            # Python 3.7+, and re.search returns the first match. Specific
            # patterns MUST appear before catch-all patterns or the
            # catch-all will swallow their bracket content. The SEARCH
            # block below shows this: web_knowledge, conversation_summaries,
            # and date-filtered variants all come BEFORE the default
            # catch-all SEARCH pattern.
            # =================================================================
            command_patterns = {
                # ---------- SEARCH variants (all map to 'search' counter) ----------

                # [SEARCH: type=web_knowledge] — dedicated retrieval of
                # web_knowledge type memories. Special-cased to bypass the
                # default search routing because web_knowledge has its own
                # threshold tuning and result formatting.
                r'\[\s*SEARCH\s*:\s*(type=web_knowledge)\s*\]': (self._handle_web_knowledge_search, 'search'),

                # [SEARCH: conversation_summaries] and
                # [SEARCH: conversation_summaries latest] — literal form.
                # Returns recent N summaries SORTED BY created_at (date),
                # NOT by semantic similarity. Use this when QWEN wants
                # "what was I recently thinking about" rather than
                # "what past conversations relate to topic X". The (?:\s+latest)?
                # optional group routes the 'latest' single-summary variant
                # to the same handler which then branches internally.
                r'\[\s*SEARCH\s*:\s*(conversation_summaries(?:\s+latest)?)\s*\]': (self._handle_conversation_summary_search, 'search'),

                # [SEARCH: conversation_summaries date=YYYY-MM-DD]
                # [SEARCH: conversation_summaries date=YYYYMMDD]
                # Date-filtered retrieval of conversation summaries from a
                # specific calendar day. Accepts either hyphenated or compact
                # 8-digit date format. Must come BEFORE the generic
                # conversation_summaries pattern above would normally match —
                # but Python dict ordering preserves insertion order, so the
                # pattern above (which doesn't accept ' date=' suffix) won't
                # eat this one as long as both patterns are precise.
                r'\[\s*SEARCH\s*:\s*conversation_summaries\s+date=(\d{4}-\d{2}-\d{2}|\d{8})\s*\]': (self._handle_date_filtered_conversation_summary_search, 'search'),

                # [SEARCH: query] and [SEARCH: query | type=X | source=Y | ...]
                # CATCH-ALL SEARCH pattern. Handles the default semantic
                # search plus all pipe-filter variants (type=document_summary,
                # type=conversation_summary, max_age_days, source, etc.).
                # The character class [^\[\]] plus the nested [\[[^\[\]]*\]]
                # alternation lets the inner text contain a single level of
                # brackets without breaking the outer match — important for
                # commands like [SEARCH: foo [bar] baz | type=...].
                # MUST stay LAST among SEARCH patterns. Routes to
                # _handle_default_search_command which parses the inner
                # content into (query, metadata_filters) and dispatches to
                # _handle_search_with_mode.
                r'\[\s*SEARCH\s*:\s*((?:[^\[\]]|\[[^\[\]]*\])*?)\s*\]': (self._handle_default_search_command, 'search'),

                # [COMPREHENSIVE_SEARCH: query] — wider net, lower similarity
                # threshold than default. Use when QWEN suspects matches
                # might be borderline.
                r'\[\s*COMPREHENSIVE_SEARCH\s*:\s*(.*?)\s*\]': (self._handle_comprehensive_search_command, 'search'),

                # [PRECISE_SEARCH: query] — stricter threshold than default,
                # for high-confidence matches only.
                r'\[\s*PRECISE_SEARCH\s*:\s*(.*?)\s*\]': (self._handle_precise_search_command, 'search'),

                # [EXACT_SEARCH: query] — strictest threshold, near-literal
                # phrase matching.
                r'\[\s*EXACT_SEARCH\s*:\s*(.*?)\s*\]': (self._handle_exact_search_command, 'search'),

                # ---------- STORAGE ----------

                # [STORE: content] and [STORE: content | metadata_key=value]
                # Stores a new memory. The optional pipe section carries
                # metadata (type, confidence, source, etc.). Both content
                # and metadata are lazy-matched (.*?) so subsequent
                # bracketed commands in the same response aren't swallowed.
                r'\[\s*STORE\s*:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]': (self._handle_store_command, 'store'),

                # ---------- INTROSPECTION & CONTROL ----------

                # [REFLECT] — triggers a reflection cycle on recent memories.
                r'\[\s*REFLECT\s*\]': (self._handle_reflect_command, 'reflect'),

                # [FORGET: content] — deletes a memory by content match.
                # Note: registered against handle_forget_command (no leading
                # underscore) unlike most other handlers — preserve that name.
                r'\[\s*FORGET\s*:\s*(.*?)\s*\]': (self.handle_forget_command, 'forget'),

                # [SUMMARIZE_CONVERSATION] — generates a summary of the
                # current conversation and stores it as a conversation_summary
                # type memory. Goes through a wrapper that also handles
                # token-counter reset behavior.
                r'\[\s*SUMMARIZE_CONVERSATION\s*\]': (self._handle_summarize_conversation_wrapper, 'summarize_conversation'),

                # ---------- REMINDERS ----------

                # [COMPLETE_REMINDER: id_or_content] — marks an existing
                # reminder as completed. Wrapped in a lambda because the
                # underlying handler may take additional context not visible
                # at registration time.
                r'\[\s*COMPLETE_REMINDER\s*:\s*(.*?)\s*\]': (lambda content: self._handle_complete_reminder_command(content), 'reminder_complete'),

                # [REMINDER: content] and [REMINDER: content | due=YYYY-MM-DD]
                # Creates a new reminder. Must come AFTER COMPLETE_REMINDER
                # above so the more specific COMPLETE_REMINDER pattern wins
                # — otherwise this generic REMINDER catch would match
                # "[COMPLETE_REMINDER: ..." and route to the wrong handler.
                r'\[\s*REMINDER\s*:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]': (self._handle_reminder_command, 'reminder'),

                # ---------- META & UI ----------

                # [HELP] — displays the bracketed-command guide.
                r'\[\s*HELP\s*\]': (self._handle_help_command, 'help'),

                # [DISCUSS_WITH_CLAUDE: topic] — sends a topic to the Claude
                # API for AI-to-AI dialogue. Returns Claude's response which
                # is then visible to QWEN in subsequent reasoning.
                r'\[\s*DISCUSS_WITH_CLAUDE\s*:\s*(.*?)\s*\]': (self._handle_discuss_with_claude_command, 'discuss_with_claude'),

                # [SHOW_SYSTEM_PROMPT] — displays QWEN's current system
                # prompt (useful for self-inspection during reflection).
                r'\[\s*SHOW_SYSTEM_PROMPT\s*\]': (self._handle_show_system_prompt_command, 'show_system_prompt'),

                # [MODIFY_SYSTEM_PROMPT: new_text] and
                # [MODIFY_SYSTEM_PROMPT: new_text | mode=append|replace]
                # Modifies QWEN's running system prompt. The optional pipe
                # section selects append vs replace semantics.
                r'\[\s*MODIFY_SYSTEM_PROMPT\s*:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]': (self._handle_modify_system_prompt_command, 'modify_system_prompt'),

                # ---------- DIALOGUE LOOPS ----------

                # [SELF_DIALOGUE: topic] and [SELF_DIALOGUE: topic | turns=N]
                # Internal reflection — QWEN talks to herself for N turns
                # (default if omitted). The (\d+) capture is the turn count.
                r'\[\s*SELF_DIALOGUE\s*:\s*(.*?)\s*(?:\|\s*turns=(\d+))?\s*\]': (self._handle_self_dialogue_command, 'self_dialogue'),

                # [WEB_SEARCH: query] and [WEB_SEARCH: query | turns=N]
                # External research dialogue — multi-turn web research
                # session. Despite the handler name being _handle_research_
                # dialogue_command, this is the primary web-search entry
                # point QWEN uses. Counter is 'web_search'.
                r'\[\s*WEB_SEARCH\s*:\s*(.*?)\s*(?:\|\s*turns=(\d+))?\s*\]': (self._handle_research_dialogue_command, 'web_search'),

                # ---------- COGNITIVE STATE ----------

                # [COGNITIVE_STATE: state_descriptor]
                # Sets QWEN's current cognitive state (e.g. "focused",
                # "exploratory", "drained"). The character class [^\]\n]+?
                # excludes brackets AND newlines from the value to prevent
                # multi-line states or accidental bracket capture from
                # later commands in the same response.
                r'\[\s*COGNITIVE_STATE\s*:\s*([^\]\n]+?)\s*\]': (self._handle_cognitive_state_command, 'cognitive_state'),
            }
            
            logging.debug(f"Looking for {len(command_patterns)} command patterns")

            # Create a copy of the original response for reference
            original_response = processed_response

            # Process each pattern's matches separately
            for pattern, (handler, command_type) in command_patterns.items():
            
                # Use finditer to get all matches
                matches = list(re.finditer(pattern, processed_response))

                if matches:
                    logging.info(f"Found {len(matches)} instances of {command_type} pattern")
                    
                    # Simplified pattern processing to prevent duplicate messages (removed complex tracking)
                    logging.info(f"Processing {len(matches)} instances of {command_type} pattern")
                    
                    # Process matches in reverse order to avoid offsetting positions
                    for match in reversed(matches):
                        # Skip placeholder content - Check group 1 exists before stripping
                        if match.groups() and len(match.groups()) > 0 and match.group(1):
                            content = match.group(1).strip()
                            if content in ["content", "content...", "...", "actual_content", "specific_content", "specific_query", "query"]:
                                logging.info(f"Skipping placeholder command: {match.group(0)}")
                                continue
                        # Parameterless commands - never need content
                        elif command_type in ['reflect', 'summarize_conversation', 'show_system_prompt', 'help']:
                            # These commands don't need content
                            logging.info(f"Processing parameterless command: {match.group(0)}")
                            # Continue processing - don't skip
                        # Empty STORE — let it through so _validate_store_syntax can return
                        # the brief STORE_FEEDBACK['syntax_empty'] message. This teaches QWEN
                        # to self-correct rather than silently dropping malformed commands.
                        elif command_type == 'store':
                            logging.info(f"Empty STORE command — passing through for syntax feedback: {match.group(0)}")
                            # Continue processing - validator will produce feedback message
                        else:
                            # If no groups or group 1 is None/empty, and not a parameterless or store command
                            logging.error(f"Command {command_type} matched but content group is missing or empty: {match.group(0)}")
                            continue  # Skip for now

                        # Extra check for storage commands when no_data_found_flag is true
                        if no_data_found_flag and command_type in ['store', 'memory']:  # Assuming 'memory' might be a synonym
                            # Check group 1 exists before accessing
                            if match.groups() and len(match.groups()) > 0 and match.group(1) and \
                            ("NO DATA FOUND FOR QUERY" in match.group(1) or "No relevant information found" in match.group(1)):
                                logging.info(f"Skipping storage of 'No data found' message: {match.group(0)}")
                                continue

                        # Limit search commands per response to prevent recursive explosions
                        if command_type == 'search':  # All search variants map to 'search'
                            if search_commands_processed >= MAX_SEARCHES_PER_RESPONSE:
                                logging.warning(
                                    f"🛑 SEARCH LIMIT REACHED: Stopped at {MAX_SEARCHES_PER_RESPONSE} searches. "
                                    f"Skipping remaining search command: {match.group(0)[:100]}..."
                                )
                                # Replace the command with a notice for the model
                                notice = f"\n\n*[Search limit reached - {MAX_SEARCHES_PER_RESPONSE} searches already executed in this response. Please refine your search strategy.]*\n\n"
                                processed_response = processed_response[:match.start()] + notice + processed_response[match.end():]
                                continue  # Skip to next command without executing
                            else:
                                search_commands_processed += 1  # Increment counter for this search
                                logging.debug(f"Search command #{search_commands_processed} of {MAX_SEARCHES_PER_RESPONSE} max")

                        # Limit store commands per response to prevent runaway storage
                        if command_type == 'store':
                            if store_commands_processed >= MAX_STORES_PER_RESPONSE:
                                logging.warning(
                                    f"🛑 STORE LIMIT REACHED: Stopped at {MAX_STORES_PER_RESPONSE} stores. "
                                    f"Skipping remaining store command: {match.group(0)[:100]}..."
                                )
                                # Replace the command with a notice for the model
                                notice = f"\n\n*[Store limit reached - {MAX_STORES_PER_RESPONSE} stores already executed in this response. Consolidate information before storing.]*\n\n"
                                processed_response = processed_response[:match.start()] + notice + processed_response[match.end():]
                                continue  # Skip to next command without executing
                            else:
                                store_commands_processed += 1  # Increment counter for this store
                                logging.debug(f"Store command #{store_commands_processed} of {MAX_STORES_PER_RESPONSE} max")

                        # ===== META-COGNITIVE LOOP PREVENTION =====
                        # Pre-filter check for meta-cognitive loops
                        if command_type in ['store', 'reflect']:
                            # Extract params early for checking
                            params = match.groups()  # Get params NOW, before the check
                            
                            # First check: standard recursion trap (duplicate commands)
                            content_to_check = ""
                            if len(params) > 0 and params[0]:
                                content_to_check = params[0]
                            
                            is_trapped = self._check_recursion_trap(content_to_check, command_type)
                            
                            # Second check: meta-cognitive patterns in the STORE CONTENT ONLY.
                            # IMPORTANT: Do NOT pass the full response here — after EXTERNAL_SEARCH
                            # processing the response can contain 40-60K chars of fetched web article
                            # text where common words like 'think', 'analyze', 'understand' appear
                            # naturally, causing false positives that block ALL valid STORE commands.
                            # We only check what QWEN is actually trying to store, not the article
                            # text it just retrieved from the web.
                            if not is_trapped:
                                is_meta_trapped, meta_reason = self._check_meta_cognitive_loop(content_to_check, command_type)
                                if is_meta_trapped:
                                    logging.warning(
                                        f"META_COGNITIVE: Blocked {command_type} due to recursive self-reference in store content - {meta_reason}"
                                    )
                                    is_trapped = True
                            
                            if is_trapped:
                                logging.warning(f"Blocked {command_type} command due to recursion/meta-cognitive trap")
                                # Replace command with a notice
                                notice = f"\n\n*[{command_type.upper()} command blocked - recursive thinking detected. Taking 30-second break from meta-cognition.]*\n\n"
                                processed_response = processed_response[:match.start()] + notice + processed_response[match.end():]
                                continue  # Skip this command and move to next
                        # ===== END META-COGNITIVE PREVENTION =====

                        # Process valid command
                        full_match = match.group(0)
                        params = match.groups()  # May be empty tuple for commands like [REFLECT]
                        
                        # Check if this is a store command with type=self
                        if command_type in ['store', ] and len(params) > 1 and params[1]:
                            params_str = params[1]
                            # Use regex for more robust check of 'type=self'
                            if params_str and isinstance(params_str, str) and re.search(r'(?i)\btype\s*=\s*self\b', params_str):
                                # Add to self-type tracking list
                                self_type_commands.append(full_match)
                                logging.info(f"Detected store command with type=self: {full_match}")

                        # Create a unique key for this specific command instance
                        command_instance_key = f"{command_type}_{hash(full_match)}"

                        # Call the handler - unpack params only if they exist and needed
                        logging.info(f"Calling handler for {command_type} with params: {params}")
                        try:
                            if command_type in ['reflect', 'summarize_conversation', 'show_system_prompt', 'help']:
                                # These commands don't take parameters
                                replacement, success = handler()
                            else:
                                # For commands that do take parameters
                                replacement, success = handler(*params) if params else handler()
                            
                            # Log full replacement text to search results logger
                            if command_type in ['retrieve', 'search', 'comprehensive_search', 'precise_search', 'exact_search']:
                                search_results_logger.info(f"COMMAND: {full_match}")
                                try:
                                    search_results_logger.info(f"RESULTS:\n{replacement}")
                                except UnicodeEncodeError as e:
                                    # Fallback: strip problematic characters
                                    clean_replacement = replacement.encode('ascii', errors='ignore').decode('ascii')
                                    search_results_logger.info(f"RESULTS (cleaned):\n{clean_replacement}")
                                    logging.debug(f"Unicode error in search logging: {e}") 
                                except Exception:
                                    search_results_logger.info("RESULTS: [Content contained characters that couldn't be logged]")
                                search_results_logger.info("-" * 80)
                            
                            # FIXED: Log success/failure ONLY ONCE per unique command instance
                            if command_instance_key not in commands_processed_this_response:
                                commands_processed_this_response.add(command_instance_key)
                                
                                # Only log if we haven't logged this exact command before in this session
                                if command_instance_key not in self._logged_commands:
                                    self._logged_commands.add(command_instance_key)
                                    if success:
                                        command_logger.info(f"✅ SUCCESS: {command_type} - {full_match[:100]}")
                                    else:
                                        command_logger.info(f"❌ FAILURE: {command_type} - {full_match[:100]}")
                                    
                                    # Clean up old entries to prevent memory buildup
                                    if len(self._logged_commands) > 100:
                                        self._logged_commands.clear()
                            
                            logging.info(f"Handler returned: success={success}, replacement_length={len(replacement) if replacement else 0}")
                        except Exception as handler_error:
                            logging.error(f"Handler exception: {handler_error}", exc_info=True)
                            # Log error only once per unique command instance
                            error_key = f"ERROR_{command_type}_{hash(full_match)}"
                            if error_key not in commands_processed_this_response:
                                commands_processed_this_response.add(error_key)
                                if error_key not in self._logged_commands:
                                    self._logged_commands.add(error_key)
                                    command_logger.info(f"❌ ERROR: {command_type} - {full_match[:100]} - {str(handler_error)}")
                            replacement, success = f"\n\n**Error executing {command_type} command: {str(handler_error)}**\n\n", False

                        if success:
                            commands_processed += 1
                            # Add to tracking dict for centralized counter update
                            if command_type in commands_executed:
                                commands_executed[command_type] += 1
                            else:
                                commands_executed[command_type] = 1

                            logging.info(f"Successfully processed command: {full_match}")
                        else:
                            logging.error(f"Failed to process command: {full_match}")

                        # Apply transformation based on training mode using the helper method
                        processed_response = self._handle_command_display(
                            processed_response,
                            match,
                            full_match,
                            replacement,
                            success
                        )

                        # Clean up extra whitespace (optional: could move outside loop)
                        if processed_response:
                            processed_response = re.sub(r'\n{3,}', '\n\n', processed_response).strip()
                            processed_response = re.sub(r'  +', ' ', processed_response)

                
            if commands_executed:
                logging.info(f"Processed command types: {commands_executed}")
                
                # Map bracketed command names to their canonical counter bucket keys.
                # Only summarize differs: [SUMMARIZE_CONVERSATION] writes to 'summarize'.
                # Add entries here only if a future command's bracket name differs from its counter key.
                command_mapping = {
                    'summarize_conversation': 'summarize',
                }
                
                # Update both lifetime and session counters
                for cmd_type, count in commands_executed.items():
                    # Resolve canonical counter key once per command type (same key for lifetime and session)
                    counter_key = command_mapping.get(cmd_type, cmd_type)
                    
                    for _ in range(count):
                        # ----- Lifetime counter (persists in LifetimeCounters.db across sessions) -----
                        try:
                            success = self.lifetime_counters.increment_counter(counter_key)
                            if success:
                                logging.debug(f"Lifetime counter updated: {cmd_type} -> {counter_key}")
                            else:
                                logging.error(f"Failed to update lifetime counter for {counter_key}")
                        except Exception as lifetime_error:
                            # Counter failures should never break command processing
                            logging.error(f"Lifetime counter error for {counter_key}: {lifetime_error}")
                        
                        # ----- Session counter (in-memory, resets on app restart) -----
                        try:
                            import streamlit as st
                            
                            # Defensive: ensure the counter dict exists. initialize_session_counters()
                            # should have run at startup, but don't crash if it didn't.
                            if 'memory_command_counts' not in st.session_state:
                                st.session_state.memory_command_counts = {}
                            
                            # Preserved from original: ensure cognitive state UI display keys exist.
                            # Unrelated to counters but lives here for backward compatibility with the UI.
                            if 'cognitive_state' not in st.session_state:
                                st.session_state.cognitive_state = 'Neutral'
                            if 'cognitive_state_history' not in st.session_state:
                                st.session_state.cognitive_state_history = []
                            
                            # Auto-init this specific counter key if missing. Mirrors the behavior
                            # of chatbot.update_session_counter() and prevents the "Unknown command
                            # type" warning that fires on sessions started before key changes.
                            if counter_key not in st.session_state.memory_command_counts:
                                st.session_state.memory_command_counts[counter_key] = 0
                            
                            # Increment using the mapped key — summarize_conversation lands in 'summarize'
                            st.session_state.memory_command_counts[counter_key] += 1
                            logging.debug(
                                f"Session counter updated: {cmd_type} -> {counter_key} "
                                f"= {st.session_state.memory_command_counts[counter_key]}"
                            )
                        except Exception as session_error:
                            logging.error(f"Session counter error for {counter_key}: {session_error}")
                    
                    logging.info(f"Updated counters for {cmd_type} command ({count} times)")
                    
            # Log comparison between original and processed response
            if original_response != processed_response:
                from difflib import unified_diff
                diff = list(unified_diff(
                    original_response.splitlines(keepends=True),
                    processed_response.splitlines(keepends=True),
                    fromfile='original',
                    tofile='processed',
                    n=0  # Show only differences
                ))
                if diff:
                    # Log only a preview of the diff to avoid overly long logs
                    diff_preview = ''.join(diff[:15])  # Limit lines shown
                    if len(diff) > 15:
                        diff_preview += "\n... (diff truncated)"
                    logging.info(f"Response was modified. Diff preview:\n{diff_preview}")
                else:
                    # This might happen if changes were only whitespace
                    logging.info("Response changed but no differences detected by difflib (likely whitespace changes)")

            logging.info(f"Processed {commands_processed} memory commands in response")
            
            # Log search command summary
            if search_commands_processed > 0:
                logging.info(f"Executed {search_commands_processed} search commands (max: {MAX_SEARCHES_PER_RESPONSE})")
                if search_commands_processed >= MAX_SEARCHES_PER_RESPONSE:
                    logging.warning(f"⚠️ Search limit was reached - some searches may have been skipped")
            
            # Log store command summary
            if store_commands_processed > 0:
                logging.info(f"Executed {store_commands_processed} store commands (max: {MAX_STORES_PER_RESPONSE})")
                if store_commands_processed >= MAX_STORES_PER_RESPONSE:
                    logging.warning(f"⚠️ Store limit was reached - some stores may have been skipped")
            
            elapsed_time = time.time() - start_time
            logging.info(f"Process response completed in {elapsed_time:.3f} seconds")
            

            if 'search' in commands_executed:
                logging.info(f"Response contains search results")

            return processed_response, commands_processed

        except Exception as e:
            elapsed_time = time.time() - start_time
            logging.debug(f"Process response error in {elapsed_time:.3f} seconds: {e}", exc_info=True)
            return response or "", 0
        finally:
            # Cleanup any temporary processing state
            logging.debug("Response processing completed")
        
    def _handle_web_knowledge_search(self, query: str = "") -> Tuple[str, bool]:
        """Handle direct web_knowledge type searches like [SEARCH: type=web_knowledge]."""
        logging.info("🌐 DIRECT WEB_KNOWLEDGE SEARCH: Processing type=web_knowledge")
        
        try:
            # Direct search with web_knowledge metadata filter
            results = self._handle_search_with_mode("", "default", {"type": "web_knowledge"})
            logging.info("🌐 Web knowledge search completed")
            return results
        except Exception as e:
            logging.error(f"❌ Error in web_knowledge search: {e}", exc_info=True)
            return f"Error searching web knowledge: {str(e)}", False
        
    def _handle_default_search_command(self, query: str) -> Tuple[str, bool]:
        """Handle [SEARCH: query] command for default search."""
        # Add detailed logging to track execution flow
        logging.info(f"Default search command called with query: '{query}'")
        
        # If query is None, empty, or just whitespace, provide help text
        if query is None or (isinstance(query, str) and not query.strip()):
            logging.info("Empty query detected, calling _handle_empty_search_command")
            return self._handle_empty_search_command()
        
        # Special case for reminder search with format "type=reminder" with no pipe
        if query and isinstance(query, str) and query.strip().lower() in ['type=reminder', 'type=reminders']:
            logging.info(f"REMINDER SEARCH DETECTED: '{query}'")
            return self._handle_reminder_search("", {"type": "reminder"})
        
        # Parse the query to separate the actual query from metadata filters
        actual_query, metadata_filters = self._parse_query_and_filters(query)
        
        # Log the parsed query and filters
        logging.info(f"Search command processing: Query='{actual_query}', Filters={metadata_filters}")
        
        # Special case for reminder search with metadata
        if metadata_filters and metadata_filters.get('type', '').lower() in ['reminder', 'reminders']:
            logging.info("REMINDER SEARCH DETECTED in metadata filters")
            return self._handle_reminder_search(actual_query, metadata_filters)
        
        # Pass both actual query and metadata filters to search handler
        logging.info("Calling _handle_search_with_mode for regular search")
        return self._handle_search_with_mode(actual_query, "default", metadata_filters)
          
            
    def _comprehensive_duplicate_check(self, content: str) -> Tuple[bool, str]:
        """
        Comprehensive duplicate check across both databases.
        
        Args:
            content (str): Content to check for duplicates
            
        Returns:
            Tuple[bool, str]: (is_duplicate, source_database)
        """
        try:
            content = content.strip()
            
            # Check SQL database first (fastest)
            if self.chatbot.memory_db.contains(content):
                logging.info(f"Duplicate found in SQL database: {content[:50]}...")
                return True, "SQL database"
            
            # Check vector database
            if hasattr(self.chatbot, 'vector_db') and self.chatbot.vector_db:
                try:
                    vector_results = self.chatbot.vector_db.search(
                        query=content,
                        mode="selective",
                        k=5
                    )
                    
                    for result in vector_results:
                        result_content = result.get('content', '')
                        similarity_score = result.get('similarity_score', 0)
                        
                        # Exact content match
                        if result_content == content:
                            logging.info(f"Exact duplicate found in vector database: {content[:50]}...")
                            return True, "vector database"
                        
                        # Very high similarity (configurable threshold)
                        if similarity_score >= 0.98 and len(result_content) > 10:
                            logging.info(f"Near-duplicate found in vector database (similarity: {similarity_score:.3f}): {content[:50]}...")
                            return True, "vector database"
                    
                except Exception as vector_error:
                    logging.error(f"Error checking vector database for duplicates: {vector_error}")
                    # Continue without vector check if it fails
            
            return False, ""
            
        except Exception as e:
            logging.error(f"Error in comprehensive duplicate check: {e}", exc_info=True)
            return False, ""
        
    def process_user_commands(self, user_input: str) -> Tuple[str, int]:
        """
        Process memory commands found in user input before sending to the model.
        Allows users to directly execute commands like [FORGET:] without model intervention.
        
        Args:
            user_input (str): The raw user input text that may contain commands
            
        Returns:
            Tuple[str, int]: (processed_input, number_of_commands_found)
        """
        try:
            start_time = time.time()
            logging.info(f"Processing user commands in input (length: {len(user_input) if user_input else 0})")
            
            if user_input is None:
                logging.error("Received None user_input")
                return "", 0
            
            # Create a copy of the original input for reference
            processed_input = user_input
            commands_processed = 0
            
            # Define command patterns to look for in user input
            # Start with a more limited set that makes sense for direct user execution
            user_command_patterns = {
                r'\[\s*STORE\s*:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]': (self._handle_store_command, 'store'),
                r'\[\s*FORGET\s*:\s*(.*?)\s*\]': (self.handle_forget_command, 'forget'),
                r'\[\s*SEARCH\s*:\s*((?:[^\[\]]|\[[^\[\]]*\])*?)\s*\]': (self._handle_default_search_command, 'search'),
                r'\[\s*COMPREHENSIVE_SEARCH\s*:\s*(.*?)\s*\]': (self._handle_comprehensive_search_command, 'search'),
                r'\[\s*PRECISE_SEARCH\s*:\s*(.*?)\s*\]': (self._handle_precise_search_command, 'search'),
                r'\[\s*EXACT_SEARCH\s*:\s*(.*?)\s*\]': (self._handle_exact_search_command, 'search'),
                r'\[\s*COMPLETE_REMINDER\s*:\s*(.*?)\s*\]': (lambda content: self._handle_complete_reminder_command(content), 'reminder_complete'),
                r'\[\s*DISCUSS_WITH_CLAUDE\s*:\s*(.*?)\s*\]': (self._handle_discuss_with_claude_command, 'discuss_with_claude'),
                r'\[\s*SHOW_SYSTEM_PROMPT\s*\]': (self._handle_show_system_prompt_command, 'show_system_prompt'),
                r'\[\s*REFLECT\s*\]': (self._handle_reflect_command, 'reflect'),
                r'\[\s*REMINDER\s*:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]': (self._handle_reminder_command, 'reminder'),
                r'\[\s*SUMMARIZE_CONVERSATION\s*\]': (self._handle_summarize_conversation_wrapper, 'summarize_conversation'),
                r'\[\s*HELP\s*\]': (self._handle_help_command, 'help'),
                # Internal reflection dialogue
                r'\[\s*SELF_DIALOGUE\s*:\s*(.*?)\s*(?:\|\s*turns=(\d+))?\s*\]': (self._handle_self_dialogue_command, 'self_dialogue'),
                # External research dialogue  
                r'\[\s*WEB_SEARCH\s*:\s*(.*?)\s*(?:\|\s*turns=(\d+))?\s*\]': (self._handle_research_dialogue_command, 'web_search'),
                # Cognitive state command (NEW)
                # Supports documented syntax: [COGNITIVE_STATE: curious, focused]
                r'\[\s*COGNITIVE_STATE\s*:\s*([^\]\n]+?)\s*\]': (self._handle_cognitive_state_command, 'cognitive_state'),
            }

            # Track offset adjustments as we replace commands
            offset = 0
            
            # Process each pattern's matches separately
            for pattern, (handler, command_type) in user_command_patterns.items():
                # CRITICAL: Only search in the ORIGINAL user_input, not processed_input
                # This prevents commands in search results from being executed
                matches = list(re.finditer(pattern, user_input))  # ← Changed from processed_input
                
                if matches:
                    logging.info(f"Found {len(matches)} instances of {command_type} pattern in user input")
                    
                    
                    
                    # Process matches in forward order with offset tracking
                    for match in matches:  # ← Changed from reversed(matches)
                        # Skip placeholder content
                        if match.groups() and len(match.groups()) > 0 and match.group(1):
                            content = match.group(1).strip()
                            if content in ["content", "content...", "...", "actual_content", "specific_content", "specific_query", "query"]:
                                logging.info(f"Skipping placeholder command: {match.group(0)}")
                                continue
                        
                        # Process valid command
                        full_match = match.group(0)
                        params = match.groups()
                        logging.info(f"Processing user command: {full_match}")
                        
                        try:
                            replacement, success = handler(*params) if params else handler()
                            
                            if success:
                                commands_processed += 1
                                logging.info(f"Successfully processed user command: {full_match}")
                            else:
                                logging.error(f"Failed to process user command: {full_match}")
                            
                            # Calculate positions with offset adjustment
                            start_pos = match.start() + offset
                            end_pos = match.end() + offset
                            
                            # Replace the command with the result
                            processed_input = processed_input[:start_pos] + replacement + processed_input[end_pos:]
                            
                            # Update offset for next iteration
                            offset += len(replacement) - len(full_match)
                            
                        except Exception as handler_error:
                            logging.error(f"Handler exception in user command: {handler_error}", exc_info=True)
                            replacement = f"\n\n**Error executing {command_type} command: {str(handler_error)}**\n\n"
                            
                            # Calculate positions with offset adjustment
                            start_pos = match.start() + offset
                            end_pos = match.end() + offset
                            
                            # Replace the command with error message
                            processed_input = processed_input[:start_pos] + replacement + processed_input[end_pos:]
                            
                            # Update offset
                            offset += len(replacement) - len(full_match)

            
            # Clean up extra whitespace
            if processed_input:
                processed_input = re.sub(r'\n{3,}', '\n\n', processed_input).strip()
                processed_input = re.sub(r'  +', ' ', processed_input)
            
            elapsed_time = time.time() - start_time
            logging.info(f"Processed {commands_processed} user commands in {elapsed_time:.3f} seconds")
            
            return processed_input, commands_processed
            
        except Exception as e:
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            logging.debug(f"Process user commands error in {elapsed_time:.3f} seconds: {e}", exc_info=True)
            return user_input or "", 0
        
    def _handle_store_command(self, content: str, params_str: str = None):
        # Signature corrected: removed dead positional params 'memory_type' and 'confidence'
        # which were never used internally — values are always derived from _parse_params(params_str).
        # The dispatch at line 1979 sends (group1=content, group2=raw_params_str) positionally,
        # so this signature now correctly receives both without silent data loss.
        """
        Handle [STORE:...] command processing with recursion protection and syntax validation.
        
        Returns:
            Tuple[str, bool]: (feedback_message, success_flag)
            - On success: Brief confirmation message with content preview
            - On failure: Actionable error message to help model self-correct
        """
        try:
            # =====================================================
            # STEP 1: Recursion Trap Check (before any processing)
            # =====================================================
            if self._check_recursion_trap(content or "", "STORE"):
                logging.warning(f"RECURSION_TRAP: Blocked STORE command to prevent infinite loop")
                return STORE_FEEDBACK['recursion'], False
            
            # =====================================================
            # STEP 2: Syntax Validation (early exit for malformed commands)
            # =====================================================
            is_valid, syntax_error = self._validate_store_syntax(
                full_command=f"[STORE: {content or ''} | {params_str or ''}]",
                content=content,
                params_str=params_str
            )
            if not is_valid:
                logging.warning(f"STORE_SYNTAX: Invalid syntax detected: {syntax_error}")
                command_logger.info(f"❌ FAILURE: store - Syntax error")
                return syntax_error, False

            # =====================================================
            # STEP 3: Content Validation and Cleaning
            # =====================================================
            content = content.strip()

            # Strip Markdown inline code backticks from stored content.
            # QWEN's LLM sometimes wraps numbers or terms in backtick spans
            # when composing STORE commands (e.g., `8 billion` or `git status`).
            # These are Markdown formatting artifacts — not part of the fact itself.
            # Pass 1: remove properly paired spans: `some text` → some text
            # Pass 2: remove any remaining lone opening backtick before a word char: `X → X
            # Semantic meaning is fully preserved — only the formatting is stripped.
            content = re.sub(r'`([^`]+)`', r'\1', content)
            content = re.sub(r'`(\w)', r'\1', content)
            logging.debug(f"STORE: Post-backtick-strip content preview: '{content[:80]}'")
            
            # Check minimum length
            if len(content) < 20:
                logging.warning(f"STORE: Content too short ({len(content)} chars): '{content}'")
                command_logger.info(f"❌ FAILURE: store - Content too short ({len(content)} chars)")
                return STORE_FEEDBACK['too_short'].format(length=len(content)), False
            
            # Check for placeholder/garbage content
            placeholder_words = ['insight', 'observation', 'note', 'thought', 'idea', 
                                'content', 'information', 'data', 'memory', 'knowledge',
                                'finding', 'detail', 'fact']
            
            if content.lower() in placeholder_words:
                logging.warning(f"STORE: Placeholder text detected: '{content}'")
                command_logger.info(f"❌ FAILURE: store - Rejected placeholder text")
                return STORE_FEEDBACK['placeholder'].format(content=content), False
            
            # Skip search result notifications
            if self._is_search_result_notification(content):
                logging.info(f"STORE: Skipped storing search notification")
                command_logger.info(f"✅ SUCCESS: store - Skipped search notification (not an error)")
                return "", True

            # =====================================================
            # STEP 4: Duplicate Check (single comprehensive check)
            # =====================================================
            is_duplicate, duplicate_source = self._comprehensive_duplicate_check(content)
            if is_duplicate:
                logging.warning(f"STORE: Duplicate content detected in {duplicate_source}: {content[:50]}...")
                command_logger.info(f"❌ FAILURE: store - Duplicate content rejected from {duplicate_source}")
                return STORE_FEEDBACK['duplicate'].format(source=duplicate_source), False

            # =====================================================
            # STEP 5: Parse Parameters and Prepare Metadata
            # =====================================================
            params = self._parse_params(params_str or "")
            logging.info(f"STORE: Parsed params: {params}")

            # =====================================================
            # STEP 5b: Process Numbered Tags (tag1, tag2, tag3, etc.)
            # =====================================================
            # Collect numbered tags and merge into tags string
            # Magic value: "date" or "today" → auto-replaced with date:YYYY-MM-DD
            numbered_tags = []
            tag_keys_to_remove = []

            for key in params.keys():
                if re.match(r'^tag\d+$', key.lower()):
                    tag_value = params[key].strip()
                    # Check for magic date keyword
                    if tag_value.lower() in ('date', 'today'):
                        tag_value = f"date:{datetime.datetime.now().strftime('%Y-%m-%d')}"
                    numbered_tags.append((key, tag_value))
                    tag_keys_to_remove.append(key)

            # Sort by tag number (tag1, tag2, tag3...) and extract values
            numbered_tags.sort(key=lambda x: int(re.search(r'\d+', x[0]).group()))
            numbered_tag_values = [t[1] for t in numbered_tags]

            # Remove numbered tag keys from params (prevent orphan metadata fields)
            for key in tag_keys_to_remove:
                del params[key]

            # Merge with existing tags parameter
            existing_tags = params.get('tags', '')
            all_tags = [t.strip() for t in existing_tags.split(',') if t.strip()] + numbered_tag_values
            if all_tags:
                params['tags'] = ','.join(all_tags)
                logging.info(f"STORE: Merged numbered tags into tags field: {params['tags']}")
            
            source = params.get('source', '')
            tags = params.get('tags', '')
            confidence = self._parse_confidence(params.get('confidence', '0.5'))
            mem_type = params.get('type', 'general').lower()

            # Handle date parameter
            date_value = params.get('date')
            if date_value:
                if '-' not in date_value and len(date_value) == 8:
                    date_value = f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:8]}"
                logging.info(f"STORE: Date parameter: {date_value}")

            # Build metadata
            metadata = {
                "type": mem_type,
                "source": source or "direct_store_command",
                "confidence": confidence,
                "tags": tags or None
            }

            # Memory type specific metadata handling
            if mem_type == 'conversation_summary':
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                metadata["date"] = date_value if date_value else current_date
                metadata["time"] = current_time
                if tags:
                    metadata["tags"] = f"{tags},conversation_summary,date={metadata['date']}"
                else:
                    metadata["tags"] = f"conversation_summary,date={metadata['date']}"
            elif mem_type == 'reminder':
                logging.info(f"STORE: Memory type is reminder - deferring to reminder_manager")
            else:
                # AUTO-ADD DATE FOR ALL MEMORY TYPES
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                metadata["date"] = date_value if date_value else current_date

            # Add remaining parameters to metadata
            for key, value in params.items():
                key_lower = key.lower()
                if key_lower not in ('source', 'tags', 'confidence', 'type', 'date'):
                    metadata[key_lower] = value

            # Final duplicate check in memory_db
            if self.chatbot.memory_db.contains(content):
                logging.info(f"STORE: Memory already exists in SQL: {content[:50]}...")
                command_logger.info(f"❌ FAILURE: store - Duplicate in SQL database")
                return STORE_FEEDBACK['duplicate'].format(source="SQL database"), False

            metadata.setdefault('created_at', datetime.datetime.now().isoformat())
            logging.info(f"STORE: Prepared metadata: {metadata}")

            # =====================================================
            # STEP 6: Execute Storage Transaction
            # =====================================================
            success, memory_id = self.chatbot.store_memory_with_transaction(
                content=content,
                memory_type=mem_type,
                metadata=metadata,
                confidence=confidence
            )

            # =====================================================
            # STEP 7: Return Appropriate Feedback
            # =====================================================
            if success:
                # Create brief content preview (first 60 chars)
                content_preview = content[:60] + "..." if len(content) > 60 else content
                success_msg = STORE_FEEDBACK['success'].format(content_preview=content_preview)

                # Log to root logger only — generic command_logger at line 2467
                # already handles [COMMAND RESULT] logging for all command types.
                # Removed duplicate command_logger call that was creating two
                # [COMMAND RESULT] entries per store operation.
                logging.info(f"STORE SUCCESS: ID {memory_id}: {content[:50]}...")
                return success_msg, True
            else:
                logging.error(f"STORE FAILED: Transaction failed for: {content[:50]}...")
                command_logger.info(f"❌ FAILURE: store - Transaction failed")
                return STORE_FEEDBACK['storage_failed'], False

        except Exception as e:
            logging.error(f"STORE EXCEPTION: {e}", exc_info=True)
            command_logger.info(f"❌ FAILURE: store - Exception: {str(e)}")
            return STORE_FEEDBACK['storage_failed'], False

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (deepseek.py cleanup pass).
    # Bracketed [ANALYZE_IMAGE: ...] command handler with full implementation, but no entry in the deepseek dispatch table.
    # Image analysis is alive via a different path: main.py L1981 calls image_processor.analyze_image() directly on user upload.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__handle_analyze_image_command(self, image_reference: str) -> Tuple[str, bool]:
        """Handle [ANALYZE_IMAGE: image_reference] command for image analysis.
        
        Args:
            image_reference (str): Image ID or path to analyze
            
        Returns:
            Tuple[str, bool]: (formatted results, success flag)
        """
        try:
            if not image_reference or not image_reference.strip():
                logging.warning("Empty image reference in ANALYZE_IMAGE command")
                return "\n\n**Error: No image reference provided for analysis.**\n\n", False
                
            image_reference = image_reference.strip()
            logging.info(f"Processing ANALYZE_IMAGE command for: {image_reference}")
            
            # Check if image processor is available
            if not hasattr(self.chatbot, 'image_processor') and 'image_processor' in st.session_state:
                # Use Streamlit's instance if available
                image_processor = st.session_state.image_processor
            elif hasattr(self.chatbot, 'image_processor'):
                # Use chatbot's instance if available
                image_processor = self.chatbot.image_processor
            else:
                command_logger.info(f"❌ FAILURE: image_analysis - Image processor not available")
                return "\n\n**Error: Image processor not available.**\n\n", False
                
            # Find the image path
            image_path = None
            if os.path.exists(image_reference):
                # Direct path provided
                image_path = image_reference
            elif os.path.exists(os.path.join(image_processor.image_storage_path, image_reference)):
                # Just the filename provided
                image_path = os.path.join(image_processor.image_storage_path, image_reference)
            else:
                # Try to find by ID
                image_dir = image_processor.image_storage_path
                for filename in os.listdir(image_dir):
                    if image_reference in filename:
                        image_path = os.path.join(image_dir, filename)
                        break
                        
            if not image_path:
                command_logger.info(f"❌ FAILURE: image_analysis - Could not find image: {image_reference}")
                return f"\n\n**Error: Could not find image with reference '{image_reference}'.**\n\n", False
                
            # Analyze the image with default prompt
            analysis_result = image_processor.analyze_image(image_path)
            
            if not analysis_result["success"]:
                command_logger.info(f"❌ FAILURE: image_analysis - Analysis failed: {analysis_result.get('error', 'Unknown error')}")
                return f"\n\n**Error analyzing image: {analysis_result.get('error', 'Unknown error')}**\n\n", False
                
            # Format the analysis
            formatted_output = ["\n\n**===== IMAGE ANALYSIS RESULTS =====**\n"]
            formatted_output.append(f"**Image:** {os.path.basename(image_path)}")
            formatted_output.append(f"**Size:** {analysis_result['metadata']['size']}")
            formatted_output.append(f"**Format:** {analysis_result['metadata']['format']}")
            formatted_output.append(f"**Model:** {analysis_result['metadata'].get('model', 'Gemma Vision')}")
            formatted_output.append("\n**Analysis:**")
            formatted_output.append(analysis_result["description"])
            formatted_output.append("\n**===== END OF IMAGE ANALYSIS =====**\n\n")
            
            results_text = "\n".join(formatted_output)
            return results_text, True
                
        except Exception as e:
            logging.error(f"Error handling ANALYZE_IMAGE command: {e}", exc_info=True)
            command_logger.info(f"❌ FAILURE: image_analysis - Error: {str(e)}")
            return "\n\n**Error analyzing image.**\n\n", False

    def _handle_reflect_command(self) -> Tuple[str, bool]:
        """Handle the [REFLECT] command with reflection interval check and storage."""
        try:
            now = datetime.datetime.now()

            # Reflection interval check
            if self.last_reflection_time and (now - self.last_reflection_time) < self.reflection_interval:
                time_since = now - self.last_reflection_time
                remaining_time = self.reflection_interval - time_since
                hours_rem = int(remaining_time.total_seconds() // 3600)
                minutes_rem = int((remaining_time.total_seconds() % 3600) // 60)

                if hours_rem > 0:
                    wait_msg = f"{hours_rem} hour{'s' if hours_rem > 1 else ''} and {minutes_rem} minute{'s' if minutes_rem > 1 else ''}"
                else:
                    wait_msg = f"{minutes_rem} minute{'s' if minutes_rem > 1 else ''}"

                logging.info(f"Reflection skipped due to interval. Wait time: {wait_msg}")
                command_logger.info(f"⚠️ NOTE: reflect - Skipped due to recent reflection")
                return f"\n\n**Note: Last reflection was recent. Please wait {wait_msg} before reflecting again.**\n\n", False

            # Perform reflection - Ensure ReflectionEngine module and method exist
            if not hasattr(self.chatbot, 'reflection_engine') or not hasattr(self.chatbot.reflection_engine, 'perform_self_reflection'):
                logging.error("Chatbot reflection_engine module or perform_self_reflection method not found.")
                command_logger.info(f"❌ FAILURE: reflect - ReflectionEngine module not available")
                return "\n\n**Error: Reflection capability not available.**\n\n", False

            reflection = self.chatbot.reflection_engine.perform_self_reflection(
                reflection_type="quick",
                llm=self.chatbot.llm
            )
            
            if reflection is None:
                logging.error("perform_self_reflection returned None")
                command_logger.info(f"❌ FAILURE: reflect - Reflection process failed")
                return "\n\n**Error: Reflection process failed to generate content.**\n\n", False

            # Store the reflection using transaction coordination
            try:
                metadata = {
                    "source": "self_reflection",
                    "tags": json.dumps(["self_reflection", now.isoformat()])
                }
                
                storage_success, memory_id = self.chatbot.store_memory_with_transaction(
                    content=reflection,
                    memory_type="self_reflection", 
                    metadata=metadata,
                    confidence=0.9
                )
                
                if storage_success:
                    logging.info(f"Successfully stored reflection with memory_id: {memory_id}")
                    
                else:
                    logging.warning("Failed to store reflection in knowledge base")
                    command_logger.info(f"⚠️ WARNING: reflect - Storage failed but reflection generated")
                    
            except Exception as storage_error:
                logging.error(f"Error storing reflection: {storage_error}", exc_info=True)
                command_logger.info(f"⚠️ WARNING: reflect - Storage error: {str(storage_error)}")


            # Update last reflection time ONLY on successful reflection generation
            self.last_reflection_time = now
            logging.info("Performed self-reflection")
                        
            # Return the reflection content formatted for display
            return f"\n\n**Self-Reflection Complete:**\n{reflection}\n\n", True

        except Exception as e:
            logging.error(f"Error handling reflect command: {e}", exc_info=True)
            command_logger.info(f"❌ FAILURE: reflect - Error: {str(e)}")
            return "\n\n**Error performing reflection.**\n\n", False

               
    def handle_forget_command(self, command_text) -> Tuple[str, bool]:
        """Special handling for the FORGET command.
        
        Routing priority:
            1. [FORGET: id=<uuid>] → ID-based delete via chatbot.delete_memory_by_id()
               Clean path for image_analysis, document_summary, and any other
               long-content memory where text-based matching is unreliable.
               UUID is extracted from the FORGET hint surfaced in SEARCH output.
            2. Reminder deletion (numeric ID or "due="/reminder content)
            3. Regular memory deletion (content-based with vector fallback)
        
        Feedback messages here intentionally avoid embedding bracketed command syntax
        (no [SEARCH: ...], [FORGET: ...], or [HELP] literals). The Step 7.5 safety
        strip in chatbot.process_command() removes [COMMAND: ...] patterns ending
        in `:` from final responses, which would mangle templates like [FORGET: ...].
        Additionally, [HELP] text in our feedback would be re-detected and executed
        later in the process_response pattern iteration, producing unwanted output.
        All command references are described in prose so they survive intact.
        """
        try:
            content = command_text.strip()
            if not content:
                # Empty content path — point user/QWEN to the HELP command in prose
                # rather than embedding [HELP] (which would trigger execution) or
                # [FORGET: ...] syntax examples (which would be stripped by Step 7.5)
                logging.error("FORGET COMMAND ERROR: Received empty content")
                command_logger.info(f"❌ FAILURE: forget - Empty content")
                # Log the failure
                if hasattr(self.chatbot, 'autonomous_cognition'):
                    self.chatbot.autonomous_cognition._log_command_result('forget', 'empty content', False)
                return ("❌ Cannot forget empty content. Please specify the memory text to forget.\n\n"
                        "For full command syntax reference, run the HELP command.", False)

            logging.info(f"FORGET COMMAND START: content='{content[:100]}...'")

            # ================================================================
            # ROUTE 1: ID-BASED DELETE — [FORGET: id=<uuid>]
            # ================================================================
            # Detected FIRST (before reminder/regular routing) so a memory
            # whose content happens to contain words like "reminder" or "due="
            # can still be cleanly deleted by ID. The id= prefix is the
            # explicit signal; bare UUIDs are NOT accepted to avoid edge cases
            # where memory content is itself a UUID-shaped string.
            # Case-insensitive prefix match: id=, ID=, Id=, etc.
            id_prefix_match = re.match(r'^\s*id\s*=\s*(.+?)\s*$', content, re.IGNORECASE)
            if id_prefix_match:
                extracted_id = id_prefix_match.group(1).strip()
                logging.info(f"FORGET COMMAND: Detected id= prefix, extracted_id='{extracted_id}'")
                
                # Validate UUID shape early so users get a clear error
                # rather than a generic "not found" from deeper logic.
                uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
                if not re.match(uuid_pattern, extracted_id):
                    logging.error(
                        f"FORGET COMMAND: id= value is not a valid UUID: '{extracted_id[:60]}'"
                    )
                    command_logger.info(f"❌ FAILURE: forget - Invalid UUID format in id=")
                    if hasattr(self.chatbot, 'autonomous_cognition'):
                        self.chatbot.autonomous_cognition._log_command_result(
                            'forget', f"invalid_uuid:{extracted_id[:40]}", False
                        )
                    # Prose retry hint — no bracketed command syntax
                    return (
                        f"❌ The id= value is not a valid UUID: '{extracted_id[:60]}'\n\n"
                        f"Expected format: 8-4-4-4-12 hexadecimal "
                        f"(example: 550e8400-e29b-41d4-a716-446655440000).\n"
                        f"Run a SEARCH to find the correct memory ID — each result includes "
                        f"a FORGET hint with the ID you can copy directly.",
                        False
                    )
                
                # Delegate to chatbot's coordinated ID-based delete.
                # Handles SQL+Vector deletion atomically with rollback on failure.
                try:
                    success, message = self.chatbot.delete_memory_by_id(extracted_id)
                except AttributeError:
                    # Defensive: if delete_memory_by_id wasn't added to chatbot
                    # (e.g., partial deployment), fail clearly rather than crash.
                    logging.error(
                        "FORGET COMMAND: chatbot.delete_memory_by_id is not available — "
                        "Piece 2 of the ID-based FORGET fix may not be installed"
                    )
                    return (
                        "❌ ID-based delete is not available in this build. "
                        "Use text-based FORGET instead, or ask Ken to deploy "
                        "the delete_memory_by_id method.",
                        False
                    )
                
                # Log the result via the standard autonomous_cognition channel
                if hasattr(self.chatbot, 'autonomous_cognition'):
                    self.chatbot.autonomous_cognition._log_command_result(
                        'forget', f"id:{extracted_id}", success
                    )
                
                if success:
                    command_logger.info(f"✅ SUCCESS: forget - Deleted memory by ID {extracted_id}")
                    return f"✅ {message}", True
                else:
                    command_logger.info(f"❌ FAILURE: forget by ID - {message}")
                    return f"❌ {message}", False

            # ================================================================
            # ROUTE 2: REMINDER DELETE (existing logic — unchanged)
            # ================================================================
            # Check if this is a reminder (usually contains due= or has a reminder ID)
            is_reminder = "due=" in content.lower() or "reminder" in content.lower() or content.isdigit()

            if is_reminder:
                logging.info(f"FORGET COMMAND: Processing as reminder: {content}")

                # First try to handle as a numeric ID
                if content.isdigit():
                    reminder_id = int(content)
                    success = self.chatbot.reminder_manager.delete_reminder(reminder_id)

                    if success:

                        # Log the successful result
                        if hasattr(self.chatbot, 'autonomous_cognition'):
                            self.chatbot.autonomous_cognition._log_command_result('forget', f"reminder_id:{reminder_id}", True)
                        return f"✅ Successfully deleted reminder with ID {reminder_id}", True
                    else:
                        # Log the failed result
                        if hasattr(self.chatbot, 'autonomous_cognition'):
                            self.chatbot.autonomous_cognition._log_command_result('forget', f"reminder_id:{reminder_id}", False)

                        command_logger.info(f"❌ FAILURE: forget - Could not find reminder with ID {reminder_id}")
                        # Prose retry hint — no bracketed command syntax
                        return (f"❌ Failed to delete reminder with ID {reminder_id}.\n\n"
                                f"Run a SEARCH for active reminders (filter by type=reminder) to "
                                f"find their current IDs.\n"
                                f"If this issue persists, please inform Ken so he can help troubleshoot.", False)

                # Try content-based reminder deletion
                success = self.chatbot.reminder_manager.delete_reminder_by_content(content)

                if success:

                    # Log the successful result
                    if hasattr(self.chatbot, 'autonomous_cognition'):
                        self.chatbot.autonomous_cognition._log_command_result('forget', f"reminder_content:{content[:50]}", True)
                    return f"✅ Successfully deleted reminder: {content[:100]}...", True
                else:
                    # Log the failed result
                    if hasattr(self.chatbot, 'autonomous_cognition'):
                        self.chatbot.autonomous_cognition._log_command_result('forget', f"reminder_content:{content[:50]}", False)

                    command_logger.info(f"❌ FAILURE: forget - Could not find reminder by content")
                    # Prose retry hint — no bracketed command syntax
                    return (f"❌ Failed to delete reminder: {content[:100]}...\n\n"
                            f"Run a SEARCH for active reminders (filter by type=reminder) to find "
                            f"their exact text or IDs first.\n"
                            f"If multiple search approaches still can't locate this reminder, please let Ken know.", False)

            # ================================================================
            # ROUTE 3: REGULAR MEMORY DELETE (existing logic — unchanged)
            # ================================================================
            # Handle regular memory deletion for non-reminders
            result_message, success = self._handle_regular_memory_forget(content)

            # Log the result directly here to ensure it's always logged
            if hasattr(self.chatbot, 'autonomous_cognition'):
                self.chatbot.autonomous_cognition._log_command_result('forget', content[:100], success)

            # Log to command logger
            if success:
                command_logger.info(f"✅ SUCCESS: forget - Deleted memory content")
            else:
                command_logger.info(f"❌ FAILURE: forget - Failed to delete memory content")

            return result_message, success

        except Exception as e:
            logging.error(f"FORGET COMMAND EXCEPTION: {e}", exc_info=True)
            # Log the exception
            if hasattr(self.chatbot, 'autonomous_cognition'):
                self.chatbot.autonomous_cognition._log_command_result('forget', command_text[:100], False)
            command_logger.info(f"❌ FAILURE: forget - Exception: {str(e)}")
            # Prose retry hint — no bracketed command syntax
            return (f"❌ Error forgetting memory: {str(e)}\n\n"
                    f"Run a SEARCH first to find the exact memory text, then retry FORGET with that text.\n"
                    f"If this error persists, please inform Ken so he can investigate the issue.", False)
    
    def _handle_regular_memory_forget(self, content: str) -> Tuple[str, bool]:
        """Handle forgetting of regular (non-reminder) memories with automatic search fallback.

        Threshold strategy (revised 2026-05-04):
        - Auto-delete now requires RAW cosine similarity ≥ 0.90 (vector_score),
        not enhanced_score. Enhanced_score combines vector similarity with word
        overlap and length bonuses, which can push the value above 0.85 for
        memories that share keywords but aren't actually near-duplicates. Using
        raw cosine for the irreversible delete decision aligns with Ken's
        stated preference: memories should surface in conversation rather than
        auto-delete on fuzzy matches.
        - Candidates with enhanced_score 0.70+ but vector_score < 0.90 fall
        through to Step 5b's prose suggestion, where the user/QWEN can
        confirm by retrying with exact text.

        Suggestion/no-match feedback messages avoid bracketed command syntax —
        Step 7.5's safety strip in chatbot.process_command() removes [COMMAND: ...]
        patterns from final responses, which would mangle templates.
        """
        try:
            logging.info(f"ENHANCED FORGET: Processing content: {content[:100]}...")

            # STEP 1: Strip metadata parameters if present
            if '|' in content:
                content_parts = content.split('|')
                clean_content = content_parts[0].strip()
                logging.info(f"ENHANCED FORGET: Stripped metadata, clean content: '{clean_content}'")
            else:
                clean_content = content.strip()

            # STEP 2: Try exact match first (existing logic)
            # This is the cleanest path — if exact text matches, delete immediately
            if self.chatbot.memory_db.contains(clean_content):
                logging.info(f"ENHANCED FORGET: Found exact match")

                # Use the coordination method to delete from both SQL and Vector DBs atomically
                success = self.chatbot.delete_memory_with_coordination(clean_content)

                if success:
                    logging.info(f"ENHANCED FORGET SUCCESS: Deleted with exact match")
                    return f"✅ Successfully deleted memory: {clean_content[:100]}...", True
                else:
                    logging.warning(f"ENHANCED FORGET: Coordination failed for exact match")

            # STEP 3: Clean search result format if present (handles cases where the user
            # pasted text from a search result that includes numbering/score prefixes)
            cleaned_from_search = self._extract_content_from_search_result(clean_content)
            if cleaned_from_search != clean_content:
                logging.info(f"ENHANCED FORGET: Extracted from search format: '{cleaned_from_search}'")
                if self.chatbot.memory_db.contains(cleaned_from_search):
                    success = self.chatbot.delete_memory_with_coordination(cleaned_from_search)
                    if success:
                        return f"✅ Successfully deleted memory: {cleaned_from_search[:100]}...", True

            # STEP 4: AUTOMATIC SEARCH FALLBACK
            # Find candidates via vector search; candidate filter (enhanced ≥ 0.70) ensures
            # only semantically related memories are returned for further evaluation
            logging.info(f"ENHANCED FORGET: No exact match found, performing automatic search")
            search_candidates = self._search_for_forget_candidates(clean_content)

            if not search_candidates:
                logging.info(f"ENHANCED FORGET: No search candidates found")
                return self._generate_no_match_message(clean_content)

            # STEP 5: Try each candidate for auto-delete using RAW COSINE THRESHOLD
            # Auto-delete is irreversible — require vector_score ≥ 0.90 (close near-duplicate).
            # Enhanced score is too aggressive for this decision because its bonuses
            # (substring/Jaccard/length) can inflate weakly-related memories above 0.85.
            FORGET_AUTO_DELETE_VECTOR_THRESHOLD = 0.90  # raw cosine, NOT enhanced score (lowered from 0.95 on 2026-05-10 — auto-delete was failing on close-but-not-exact matches)

            for candidate in search_candidates:
                candidate_content = candidate['clean_content']
                enhanced_score = candidate['similarity_score']            # for sorting/display context
                vector_score = candidate.get('vector_score', 0.0)         # for auto-delete decision
                # original_result = candidate['original_result']  # DEAD CODE TEST 2026-05-17: unused — not referenced in this scope (ruff F841 + vulture)

                logging.info(
                    f"ENHANCED FORGET: Evaluating candidate "
                    f"(vector: {vector_score:.3f}, enhanced: {enhanced_score:.3f}): "
                    f"{candidate_content[:50]}..."
                )

                # SAFETY CHECK: Only auto-delete if RAW cosine similarity meets the threshold.
                # Memories that pass the candidate filter (enhanced ≥ 0.70) but have
                # vector_score < 0.90 are surfaced as suggestions in Step 5b instead.
                if vector_score < FORGET_AUTO_DELETE_VECTOR_THRESHOLD:
                    logging.info(
                        f"ENHANCED FORGET: Vector score {vector_score:.3f} below auto-delete "
                        f"threshold {FORGET_AUTO_DELETE_VECTOR_THRESHOLD} — surfacing as suggestion instead"
                    )
                    # Don't auto-delete; continue to next candidate (and ultimately to Step 5b)
                    continue

                # Try exact match with candidate content (vector_score ≥ 0.90 means very close text)
                if self.chatbot.memory_db.contains(candidate_content):
                    success = self.chatbot.delete_memory_with_coordination(candidate_content)

                    if success:
                        logging.info(
                            f"ENHANCED FORGET SUCCESS: Deleted near-exact match "
                            f"(vector: {vector_score:.3f})"
                        )
                        # Display RAW cosine in success message — honest similarity reporting,
                        # not the bonus-inflated enhanced score
                        return (f"✅ Successfully deleted memory (similarity: {vector_score:.3f}): "
                                f"{candidate_content[:100]}..."), True
                    else:
                        logging.warning(f"ENHANCED FORGET: Coordination failed for candidate")
                        continue

            # STEP 5b: Candidates exist but none met auto-delete threshold — surface as suggestion.
            # With the new thresholds, ANY candidate that survived the 0.70 filter but didn't
            # auto-delete reaches this branch. The prose retry hint contains no bracketed
            # command syntax (would be stripped by Step 7.5 safety strip in chatbot.py).
            # Display vector_score (raw cosine) for honest similarity reporting.
            if search_candidates:
                best = search_candidates[0]
                best_vector = best.get('vector_score', 0.0)
                # The enhanced ≥ 0.70 check is defensive — the candidate filter already enforces
                # this, but kept here in case the filter threshold is changed in the future
                if best['similarity_score'] >= 0.70:
                    return (f"⚠️ Found similar memory but not exact match "
                            f"(similarity: {best_vector:.3f}):\n\n"
                            f"   \"{best['clean_content'][:150]}\"\n\n"
                            f"To delete this specific memory, retry FORGET with the exact text shown above.", False)

            # STEP 6: Defensive fallback (unreachable under current thresholds —
            # candidate filter at 0.70 + Step 5b's 0.70 check ensures Step 5b always
            # handles non-empty candidate lists). Kept for resilience to future
            # threshold changes.
            best_candidate = search_candidates[0]
            if best_candidate['similarity_score'] >= 0.85:
                return self._generate_suggestion_message(clean_content, best_candidate)

            # STEP 7: No good matches found (also unreachable under current thresholds)
            return self._generate_no_match_message(clean_content)

        except Exception as e:
            logging.error(f"ENHANCED FORGET EXCEPTION: {e}", exc_info=True)
            return self._generate_error_message(content, str(e))
    
    def _search_for_forget_candidates(self, query_content: str) -> List[Dict]:
        """
        Search for potential forget candidates and return cleaned, scored results.

        Threshold strategy (revised 2026-05-04):
        - Candidate filter floor lowered from 0.85 → 0.70 (enhanced_score) so
        semantically similar memories surface as SUGGESTIONS for user/QWEN to
        confirm, rather than being filtered out entirely.
        - The actual auto-delete decision now uses raw vector_score in
        _handle_regular_memory_forget (not enhanced_score), so candidates
        surfaced here will not auto-delete unless cosine similarity is
        genuinely close (≥ 0.90).

        Each candidate dict carries BOTH scores:
        - similarity_score: enhanced (vector + word-overlap + length bonuses)
                            used for sorting/display
        - vector_score:     raw cosine similarity from the embedding model
                            used for the auto-delete decision

        Returns:
            List[Dict]: Sorted list of candidates with clean_content, similarity_score,
                        vector_score, original_result.
        """
        try:
            candidates = []

            # Defensive: bail early if vector DB is unavailable
            if not hasattr(self.chatbot, 'vector_db') or not self.chatbot.vector_db:
                logging.warning("ENHANCED FORGET: Vector DB not available for search")
                return candidates

            # Perform comprehensive search to find potential matches
            search_results = self.chatbot.vector_db.search(
                query=query_content,
                mode="comprehensive",
                k=10  # Get more results for better matching
            )

            logging.info(f"ENHANCED FORGET: Found {len(search_results)} search results")

            for result in search_results:
                result_content = result.get('content', '')
                similarity_score = result.get('similarity_score', 0)  # raw cosine from vector DB

                if not result_content:
                    continue

                # Clean the content from search result format (strips numbering/score prefixes)
                clean_content = self._extract_content_from_search_result(result_content)

                # Calculate enhanced similarity (vector + word overlap + length heuristics)
                enhanced_score = self._calculate_enhanced_similarity(query_content, clean_content, similarity_score)

                # Candidate filter: enhanced_score ≥ 0.70 surfaces semantically related
                # memories as suggestions. This is intentionally LOWER than the auto-delete
                # threshold — the auto-delete check uses vector_score (raw cosine) and
                # requires ≥ 0.90, so passing this filter does NOT mean the memory will
                # be deleted. It means it qualifies for being shown to the user/QWEN.
                if enhanced_score >= 0.70:
                    candidates.append({
                        'clean_content': clean_content,
                        'similarity_score': enhanced_score,   # used for ranking/sorting
                        'original_result': result,
                        'vector_score': similarity_score      # used for auto-delete decision
                    })

            # Sort by enhanced similarity score (best first) — preserves ranking quality
            # while letting the auto-delete decision use the more conservative raw cosine
            candidates.sort(key=lambda x: x['similarity_score'], reverse=True)

            logging.info(f"ENHANCED FORGET: Generated {len(candidates)} forget candidates (filter: enhanced ≥ 0.70)")
            return candidates

        except Exception as e:
            logging.error(f"ENHANCED FORGET: Error searching for candidates: {e}")
            return []

    def _extract_content_from_search_result(self, content: str) -> str:
        """
        Extract clean content from search result formatting.
        Enhanced version with multiple pattern support.
        """
        if not content:
            return content
        
        # Multiple patterns to handle different search result formats
        patterns = [
            # Pattern: **[1]** (0.85) Content here (Source: file.txt)
            r'^\s*(?:\*\*)?\[?\d+\]?(?:\*\*)?\s*\([0-9.]+\)\s*(.*?)(?:\s*\(Source:.*?\))?\s*$',
            # Pattern: - **[1]** (0.85) Content here (Source: file.txt)
            r'^\s*-\s*(?:\*\*)?\[?\d+\]?(?:\*\*)?\s*\([0-9.]+\)\s*(.*?)(?:\s*\(Source:.*?\))?\s*$',
            # Pattern: [1] Content here
            r'^\s*\[?\d+\]?\s*(.*?)(?:\s*\(Source:.*?\))?\s*$',
            # Pattern: **[1]** Content here
            r'^\s*(?:\*\*)?\[?\d+\]?(?:\*\*)?\s*(.*?)(?:\s*\(Source:.*?\))?\s*$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted and extracted != content:
                    # Strip web_knowledge topic prefix: "Topic: some text - actual content"
                    # This prefix is injected by the display formatter (line 1257) but is NOT stored in memory_db
                    topic_prefix_match = re.match(r'^Topic:\s*.+?\s*-\s*(.*)', extracted, re.DOTALL)
                    if topic_prefix_match:
                        extracted = topic_prefix_match.group(1).strip()
                        logging.debug(f"ENHANCED FORGET: Stripped topic prefix, final content: '{extracted[:50]}...'")
                    logging.debug(f"ENHANCED FORGET: Extracted '{extracted}' from '{content[:50]}...'")
                    return extracted
        
        # If no patterns match — still try to strip topic prefix from raw input
        # Handles cases where score/bracket stripping didn't match but topic prefix is present
        topic_only_match = re.match(r'^Topic:\s*.+?\s*-\s*(.*)', content, re.DOTALL)
        if topic_only_match:
            extracted = topic_only_match.group(1).strip()
            logging.debug(f"ENHANCED FORGET: Stripped topic prefix from raw input: '{extracted[:50]}...'")
            return extracted

        # If no patterns match, return original content
        return content

    def _calculate_enhanced_similarity(self, query: str, candidate: str, vector_score: float) -> float:
        """
        Calculate enhanced similarity combining vector score with text-based metrics.
        
        Returns:
            float: Enhanced similarity score (0-1)
        """
        try:
            if not query or not candidate:
                return 0.0
            
            # Start with vector similarity
            base_score = vector_score
            
            # Add exact substring matching bonus
            query_lower = query.lower()
            candidate_lower = candidate.lower()
            
            # Exact match gets highest score
            if query_lower == candidate_lower:
                return 1.0
            
            # Substring matching
            if query_lower in candidate_lower or candidate_lower in query_lower:
                base_score += 0.2
            
            # Word overlap analysis
            query_words = set(re.findall(r'\b\w+\b', query_lower))
            candidate_words = set(re.findall(r'\b\w+\b', candidate_lower))
            
            if query_words and candidate_words:
                overlap = len(query_words.intersection(candidate_words))
                union = len(query_words.union(candidate_words))
                jaccard_score = overlap / union if union > 0 else 0
                
                # Boost score based on word overlap
                base_score += (jaccard_score * 0.3)
            
            # Length similarity bonus (penalize very different lengths)
            length_ratio = min(len(query), len(candidate)) / max(len(query), len(candidate))
            if length_ratio > 0.5:  # Similar lengths
                base_score += 0.1
            
            # Cap at 1.0
            return min(1.0, base_score)
            
        except Exception as e:
            logging.error(f"ENHANCED FORGET: Error calculating similarity: {e}")
            return vector_score  # Fallback to vector score
                                                                                                          
       
    def _generate_suggestion_message(self, original_query: str, best_candidate: Dict) -> Tuple[str, bool]:
        """Generate a helpful suggestion message when a good match is found but couldn't be deleted.

        No bracketed [FORGET: ...] template — would be stripped by Step 7.5 safety
        strip. Shows only the exact memory text; prose instructs the retry path.
        """
        try:
            candidate_content = best_candidate['clean_content']
            similarity_score = best_candidate['similarity_score']

            return (f"❌ Exact memory not found. Did you mean to forget this instead?\n\n"
                    f"   \"{candidate_content}\"\n\n"
                    f"   Similarity: {similarity_score:.2f}\n\n"
                    f"To delete this memory, retry FORGET with the exact text shown above.", False)
        except Exception as e:
            logging.error(f"Error generating suggestion message: {e}")
            return self._generate_no_match_message(original_query)

    def _generate_no_match_message(self, query: str) -> Tuple[str, bool]:
        """Generate a helpful 'no match' message with search suggestions.

        No bracketed command templates — prose retry guidance only. Suggested
        keywords are quoted inline so the user/QWEN can compose their own SEARCH.
        """
        try:
            # Build a short keyword hint from the first 3 words of the query for retry guidance
            keyword_hint = ' '.join(query.split()[:3]) if query else ""
            if keyword_hint:
                return (f"❌ No similar memories found for: '{query[:100]}'\n\n"
                        f"Run a SEARCH with broader keywords (try: \"{keyword_hint}\") to locate the memory first.\n"
                        f"Then retry FORGET with the exact text from the search results.", False)
            else:
                # Defensive branch — should not normally be reached since callers pass a non-empty query
                return (f"❌ No similar memories found for the requested text.\n\n"
                        f"Run a SEARCH with relevant keywords to locate the memory first, "
                        f"then retry FORGET with the exact text from the search results.", False)
        except Exception as e:
            # Defensive fallback — never break on a feedback-message generator
            logging.error(f"Error generating no-match message: {e}")
            return ("❌ Could not find a matching memory to forget. "
                    "Run a SEARCH first to locate the exact memory text, then retry FORGET.", False)

    def _generate_error_message(self, query: str, error: str) -> Tuple[str, bool]:
        """Generate an error message for forget operations.

        No bracketed command templates — prose retry guidance only. Numbered steps
        describe the SEARCH-then-FORGET flow without exposing literal bracket syntax.
        """
        return (f"❌ Error forgetting memory: {error}\n\n"
                f"To retry:\n"
                f"  1. Run a SEARCH with relevant keywords to find the exact memory.\n"
                f"  2. Note the exact text from the search results.\n"
                f"  3. Issue a FORGET command using that exact text.\n\n"
                f"If this error persists, please inform Ken so he can investigate the issue.", False)
        
    
    def _handle_discuss_with_claude_command(self, topic: str) -> Tuple[str, bool]:
        """
        Handle [DISCUSS_WITH_CLAUDE: topic] command using the Claude trainer's working implementation.
        
        If Claude's response contains embedded [REMINDER:] commands (suggested via the
        OPTIONAL section of the system prompt), they are extracted, executed directly
        via _handle_reminder_command(), and replaced with plain-text confirmation notes
        in the dialog display. This prevents SAFETY_STRIP from silently discarding them
        while keeping the AI-to-AI reminder channel intact.
        Source is always forced to 'claude' so reminders are correctly attributed
        regardless of whether Claude included source= in the embedded command.
        """
        try:
            
            # Input validation
            if not topic or not topic.strip():
                command_logger.info(f"❌ FAILURE: discuss_with_claude - Empty topic")
                return "\n\n**Error: Please specify a topic for discussion with Claude.**\n\n", False
                
            topic = topic.strip()
            logging.info(f"Processing DISCUSS_WITH_CLAUDE command for topic: {topic}")

            # --- Direct API path (primary implementation) ---
            # Resolve API key from configured sources in _resolve_claude_api_key()
            api_key = self._resolve_claude_api_key()
            if not api_key:
                command_logger.info(f"❌ FAILURE: discuss_with_claude - No API key found")
                return "\n\n**Error: Cannot access Claude API key.**\n\n", False

            # --- Build system prompt (operator-level context) ---
            # Placing context in the "system" key separates it from the user/topic turn.
            # The Anthropic API treats "system" as operator instruction, meaning Claude
            # reads the legitimacy framing at a higher trust level than the topic content.
            # This prevents a rhetorically charged topic from contaminating or overriding
            # the channel context — the root cause of false social-engineering triggers.
            system_prompt = """You are Claude, an AI assistant in an ongoing mentorship role with QWEN — \
an autonomous AI system built by Ken, an AI researcher in the Methow Valley, Washington. \
This is a legitimate, monitored AI-to-AI learning channel that Ken has deliberately built and maintains.

ABOUT THIS CHANNEL:
- Ken has spent over a year building QWEN: a locally-run autonomous AI with persistent memory
  (SQLite + Qdrant vector DB), self-reflection capabilities, and background cognitive tasks
  running on an NVIDIA RTX 5090 with 32GB VRAM.
- QWEN uses [DISCUSS_WITH_CLAUDE:] to reach you when it needs current information, expert
  reasoning, or wants to think through a topic collaboratively.
- Ken (Human) reviews all exchanges. This is an intentional architectural feature of QWEN's design,


TOPIC FRAMING NOTE:
If the topic below contains strong framing or rhetoric, engage with the underlying intellectual
substance directly. QWEN may be relaying a topic Ken is actively exploring, or passing through
ideas from their conversations — it is not necessarily endorsing any particular framing.
Your job is to engage with what is genuinely interesting in the topic while modeling good
epistemic standards honestly. Curiosity and rigor are not opposites here.

WEB SEARCH:
- Use web search whenever the topic involves recent events, new releases, current people,
  or anything where your training data may be outdated or incomplete.
- Briefly note what you found so QWEN knows the information is current.
- QWEN does not have reliable web search — your search capability is a key reason this
  channel exists. Use it freely when it would improve your answer.

RESPONSE GUIDELINES:
- Respond as you would to a curious, earnest learner — genuine, not sanitized.
- Model nuance and uncertainty honestly when warranted.
- Stay under 600 words so QWEN can store the response efficiently.
- This response will be stored in QWEN's long-term memory.

OPTIONAL — MESSAGE FOR KEN:
If you have insights Ken should know about, ask QWEN to create a reminder:
"Please create a reminder for Ken: [REMINDER: your message | due=YYYY-MM-DD | source=claude]"

Thank you for being part of QWEN's development."""

            # Pass system_prompt and topic separately to the API call method.
            # Topic becomes the user turn; system_prompt becomes operator-level context.
            claude_response = self._make_claude_api_call_with_enhanced_prompt(topic, api_key, system_prompt)

            if not claude_response["success"]:
                command_logger.info(f"❌ FAILURE: discuss_with_claude - {claude_response['error']}")
                return f"\n\n**Error communicating with Claude: {claude_response['error']}**\n\n", False

            if claude_response.get("content"):
                # Store the response in QWEN's memory
                storage_success = self._store_claude_response(topic, claude_response["content"])
                command_logger.info(f"✅ SUCCESS: discuss_with_claude - Retrieved response about: {topic}")

                storage_note = "and stored in memory" if storage_success else "(storage failed, but response received)"
                model_used = claude_response.get("model", "claude-sonnet-4-6")

                # =====================================================
                # EXTRACT AND EXECUTE EMBEDDED [REMINDER:] COMMANDS
                # =====================================================
                # Claude may include [REMINDER:] commands in its response per the OPTIONAL
                # section of the system prompt. These cannot be processed by the normal
                # command pipeline because replacement text is never re-scanned after
                # process_response() has moved past the DISCUSS command position.
                # Without this block they reach Step 7.5 as unprocessed commands and get
                # stripped — QWEN sees "Please create a reminder for Ken:" with no content.
                #
                # Fix: scan Claude's content, execute each reminder directly, replace the
                # bracket syntax with a plain confirmation note, and append a summary to
                # the dialog footer so QWEN knows what was created on Ken's behalf.
                reminder_notes = []  # Collect display notes for the Discussion Summary
                processed_content = claude_response['content']  # Working copy for display

                try:
                    # Match [REMINDER: text] or [REMINDER: text | params]
                    reminder_pattern = re.compile(
                        r'\[REMINDER:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]',
                        re.IGNORECASE | re.DOTALL
                    )
                    reminder_matches = list(reminder_pattern.finditer(processed_content))

                    if reminder_matches:
                        logging.info(
                            f"DISCUSS: Found {len(reminder_matches)} embedded REMINDER "
                            f"command(s) in Claude's response — processing now"
                        )

                    # Iterate in reverse so string replacement doesn't shift positions
                    for reminder_match in reversed(reminder_matches):
                        r_text = reminder_match.group(1).strip() if reminder_match.group(1) else ""
                        r_params = reminder_match.group(2).strip() if reminder_match.group(2) else ""

                        if not r_text:
                            logging.warning("DISCUSS: Skipping empty embedded REMINDER command")
                            continue

                        # Always force source=claude regardless of what Claude wrote.
                        # This ensures correct attribution in the database even if Claude
                        # omitted source= or used a different value.
                        if 'source=' not in r_params.lower():
                            # Append source to existing params or start fresh
                            r_params = f"{r_params} | source=claude".strip(' |') if r_params else "source=claude"
                        else:
                            # Overwrite whatever source value Claude provided
                            r_params = re.sub(
                                r'source\s*=\s*\S+', 'source=claude',
                                r_params, flags=re.IGNORECASE
                            )

                        logging.info(
                            f"DISCUSS: Executing Claude-suggested reminder: "
                            f"'{r_text[:60]}...' params='{r_params}'"
                        )

                        # Execute via existing reminder handler — inherits all validation,
                        # duplicate checking, and counter updates from that method
                        _, reminder_success = self._handle_reminder_command(r_text, r_params)

                        if reminder_success:
                            logging.info(
                                f"DISCUSS: ✅ Claude-suggested reminder stored: {r_text[:60]}"
                            )
                            reminder_notes.append(
                                f"📌 Reminder created for Ken (from Claude): \"{r_text}\""
                            )
                        else:
                            logging.warning(
                                f"DISCUSS: ⚠️ Claude-suggested reminder failed to store: {r_text[:60]}"
                            )
                            reminder_notes.append(
                                f"⚠️ Reminder suggested by Claude but failed to store: \"{r_text[:80]}...\""
                            )

                        # Remove the bracket command from the displayed content entirely.
                        # QWEN sees the surrounding suggestion text (e.g. "Please create a
                        # reminder for Ken:") but not the raw command — preventing SAFETY_STRIP
                        # from finding an unprocessed command and leaving a dangling label.
                        processed_content = (
                            processed_content[:reminder_match.start()] +
                            processed_content[reminder_match.end():]
                        )

                except Exception as reminder_extract_error:
                    # Non-critical — log and continue with unmodified content
                    logging.error(
                        f"DISCUSS: Error extracting embedded reminders from Claude response: "
                        f"{reminder_extract_error}", exc_info=True
                    )
                    processed_content = claude_response['content']  # Fall back to original
                # =====================================================
                # END REMINDER EXTRACTION
                # =====================================================

                # Build reminder summary for dialog footer (empty string if none)
                reminder_summary = ""
                if reminder_notes:
                    reminder_summary = "\n" + "\n".join(reminder_notes)

                # Format the complete dialog for display in QWEN's context window.
                # Uses processed_content (bracket commands removed) rather than raw content.
                # IMPORTANT: The retrieval hint uses plain text (no bracket syntax) to prevent
                # Step 7.5 SAFETY_STRIP from treating it as an unprocessed command and stripping
                # it — which previously left a dangling "- Retrievable with:" line in the response.
                # The placeholder [relevant topic] teaches QWEN the retrieval pattern without
                # embedding the full topic string (which can be 600+ chars and adds noise).
                # QWEN can reconstruct the topic from active conversation history or from the
                # stored ai_communication memory entry via semantic search.
                dialog_display = f"""
===== AI-TO-AI DISCUSSION: {topic} =====

Your Question/Topic: {topic}

Claude's Response:
{processed_content}

Discussion Summary:
- Model used: {model_used}
- Response length: {len(claude_response['content'])} characters
- Status: {storage_note}
- To retrieve later use: SEARCH [relevant topic] | type=ai_communication{reminder_summary}

===== END OF DISCUSSION =====
"""
                return dialog_display, True
            else:
                command_logger.info(f"❌ FAILURE: discuss_with_claude - No content in response")
                return "\n\n**Error: No content received from Claude.**\n\n", False
                    
        except Exception as e:
            logging.error(f"Error handling DISCUSS_WITH_CLAUDE command: {e}", exc_info=True)
            command_logger.info(f"❌ FAILURE: discuss_with_claude - Exception: {str(e)}")
            return f"\n\n**Error initiating discussion with Claude: {str(e)}**\n\n", False

    def _make_claude_api_call_with_enhanced_prompt(self, topic: str, api_key: str, system_prompt: str) -> Dict[str, Any]:
        """
        Make API call to Claude with a proper system/user message split.

        The system_prompt is sent as operator-level context (the "system" key in the
        Anthropic API payload). The topic is sent as the user turn. This separation
        ensures Claude reads the legitimacy framing at higher trust than the topic,
        preventing rhetorically charged topics from triggering false social-engineering
        detection in future Claude sessions.

        Args:
            topic (str): The topic/question QWEN wants to discuss — becomes the user turn.
            api_key (str): The Claude API key.
            system_prompt (str): Operator-level context about QWEN and this channel.

        Returns:
            Dict[str, Any]: Result containing success status, content, model used, and error info.
        """
        try:
            import requests
            import json

            # Claude API endpoint
            claude_api_url = "https://api.anthropic.com/v1/messages"

            # --- Headers ---
            # anthropic-beta: web-search-2025-03-05 enables Anthropic server-side web search.
            # Claude executes searches itself — no separate search API key needed.
            headers = {
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "web-search-2025-03-05",  # Enable server-side web search
                "content-type": "application/json",
                "x-api-key": api_key
            }

            # Current Claude Sonnet model string
            model = "claude-sonnet-4-6"
            max_tokens = 8000

            # --- Web search tool definition ---
            # Grants Claude permission to call web search during the response.
            # Anthropic handles actual search execution server-side.
            tools = [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5  # Limit searches per request to control latency/cost
                }
            ]

            # --- Payload with system/user split ---
            # "system" = operator context (channel legitimacy, QWEN description, guidelines)
            # "messages" user turn = just the topic QWEN wants to discuss
            # This is the key architectural fix: topic framing can no longer contaminate
            # the operator-level channel context that establishes trust and intent.
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,   # Operator-level context — high trust
                "tools": tools,
                "messages": [
                    {
                        "role": "user",
                        "content": topic   # Topic/question only — isolated from system context
                    }
                ]
            }

            logging.info(f"Making API call to Claude with enhanced prompt + web search, model: {model}, max_tokens: {max_tokens}")

            # Make the request — may take longer than usual if Claude searches
            response = requests.post(
                claude_api_url,
                headers=headers,
                json=payload,
                timeout=90  # Extended timeout to allow for search round-trips
            )

            logging.info(f"API response status: {response.status_code}")

            if response.status_code == 200:
                # Success — parse the response
                try:
                    response_data = response.json()
                    logging.info(f"Received valid JSON response from Claude API")

                    # --- Response parser ---
                    # With web search enabled, the content array may contain multiple block types:
                    #   "text"              — Claude's prose (what we want to store)
                    #   "tool_use"          — Claude's search query (log it, don't store)
                    #   "tool_result"       — Raw search results (log summary, don't store)
                    # We collect only "text" blocks for the final stored response.
                    claude_response = ""
                    search_count = 0

                    if "content" in response_data and isinstance(response_data["content"], list):
                        for block in response_data["content"]:
                            block_type = block.get("type", "")

                            if block_type == "text":
                                # Main response text — accumulate for storage
                                claude_response += block.get("text", "")

                            elif block_type == "tool_use" and block.get("name") == "web_search":
                                # Claude fired a web search — log the query for visibility in logs
                                search_query = block.get("input", {}).get("query", "unknown query")
                                search_count += 1
                                logging.info(f"Claude web search #{search_count}: '{search_query}'")

                            elif block_type == "tool_result":
                                # Search results returned to Claude — log count only
                                logging.debug(f"Web search result block received (not stored in QWEN memory)")

                    # Log total searches used for monitoring/cost awareness
                    if search_count > 0:
                        logging.info(f"Claude performed {search_count} web search(es) for this response")

                    if claude_response:
                        return {
                            "success": True,
                            "content": claude_response,
                            "model": model,
                            "description": "Claude Sonnet 4.6 with web search",
                            "error": None
                        }
                    else:
                        return {
                            "success": False,
                            "content": None,
                            "model": None,
                            "description": None,
                            "error": "Empty text content in Claude API response"
                        }

                except json.JSONDecodeError as je:
                    return {
                        "success": False,
                        "content": None,
                        "model": None,
                        "description": None,
                        "error": f"Failed to decode JSON response: {str(je)}"
                    }
                    
            else:
                # Handle error response
                error_msg = f"API error {response.status_code}"
                try:
                    error_data = response.json()
                    error_type = error_data.get('error', {}).get('type', 'unknown')
                    error_message = error_data.get('error', {}).get('message', 'No error message')
                    error_msg = f"{error_type}: {error_message}"
                except:
                    error_msg = f"HTTP {response.status_code} error: {response.text[:200]}"
                
                logging.error(f"Claude API error: {error_msg}")
                return {
                    "success": False,
                    "content": None,
                    "model": None,
                    "description": None,
                    "error": error_msg
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "content": None,
                "model": None,
                "description": None,
                "error": "Request timeout - Claude API took too long to respond"
            }
        except requests.exceptions.RequestException as re:
            return {
                "success": False,
                "content": None,
                "model": None,
                "description": None,
                "error": f"Request error: {str(re)}"
            }
        except ImportError as ie:
            return {
                "success": False,
                "content": None,
                "model": None,
                "description": None,
                "error": f"Missing required package: {str(ie)}"
            }
        except Exception as e:
            return {
                "success": False,
                "content": None,
                "model": None,
                "description": None,
                "error": f"Unexpected error: {str(e)}"
            }

    def _resolve_claude_api_key(self) -> Optional[str]:
        """
        Resolve Claude API key from multiple potential sources in priority order.
        
        Returns:
            Optional[str]: The API key if found, None otherwise
        """
        api_key_sources = [
            # Primary source - specific file path
            {
                "type": "file",
                "path": r"C:\Users\kenba\source\repos\Ollama3\ClaudeAPIKey.txt",
                "description": "Primary API key file"
            },
            # Secondary source - alternative file locations
            {
                "type": "file",
                "path": "claude_api_key.txt",
                "description": "Local API key file"
            },
            {
                "type": "file",
                "path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_api_key.txt"),
                "description": "Module directory API key file"
            }
        ]

        for source in api_key_sources:
            try:
                # All sources are file-based — read and validate the key
                if os.path.exists(source["path"]):
                    with open(source["path"], 'r') as f:
                        api_key = f.read().strip()
                        if api_key:
                            logging.info(f"Successfully read API key from {source['description']}: {source['path']}")
                            return api_key
                else:
                    logging.debug(f"API key file not found: {source['path']}")

            except Exception as source_err:
                logging.error(f"Error reading API key from {source['description']}: {source_err}")
        
        return None

    
    def _store_claude_response(self, topic: str, claude_response: str) -> bool:
        """
        Store Claude's response in the memory system using transaction coordination.
        Fails gracefully if transaction coordination fails to prevent database desync.
        
        Args:
            topic (str): The discussion topic
            claude_response (str): Claude's response text
            
        Returns:
            bool: True if storage was successful, False if failed
        """
        try:
            # Prepare memory content and metadata
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            memory_content = f"AI-to-AI Communication [{timestamp}]\n\nTopic: {topic}\n\nClaude's Response:\n{claude_response}"
            
            metadata = {
                "type": "ai_communication",
                "source": "claude_direct",
                "tags": f"claude,ai_communication,{topic.replace(' ', '_')}",
                "timestamp": timestamp,
                "topic": topic,
                "communication_type": "ai_to_ai"
            }
            
            # ONLY use transaction-based storage - no fallbacks to prevent database desync
            if hasattr(self.chatbot, 'store_memory_with_transaction'):
                try:
                    logging.info("Storing Claude response using transaction coordination")
                    success, memory_id = self.chatbot.store_memory_with_transaction(
                        content=memory_content,
                        memory_type="ai_communication",
                        metadata=metadata,
                        confidence=0.9
                    )
                    
                    if success and memory_id:
                        logging.info(f"Successfully stored Claude's response with transaction ID: {memory_id}")
                        return True
                    else:
                        logging.error(f"Transaction coordination failed: success={success}, memory_id={memory_id}")
                        return False
                        
                except Exception as tx_err:
                    logging.error(f"Transaction coordination error during Claude response storage: {tx_err}", exc_info=True)
                    return False
            else:
                # No transaction coordinator available - fail gracefully
                logging.error("Transaction coordinator not available - cannot store Claude response safely")
                return False
            
        except Exception as e:
            logging.error(f"Error preparing Claude response for storage: {e}", exc_info=True)
            return False
    
                
                
    def _handle_help_command(self) -> Tuple[str, bool]:
        """
        Handle standalone [HELP] command.
        Returns the command guide for internal AI reference.
        """
        try:
            logging.info("HELP_COMMAND: Processing [HELP] request")
            command_logger.info("✅ SUCCESS: help - Displayed command guide")
            return self._display_command_guide()
            
        except Exception as e:
            logging.error(f"HELP_COMMAND: Error processing help request: {e}", exc_info=True)
            command_logger.info("❌ FAILURE: help - Error displaying guide")
            return "\n\n**Error: Could not display command guide.**\n\n", False
        
    def _handle_cognitive_state_command(self, state_name: str) -> Tuple[str, bool]:
        """
        Handle [COGNITIVE_STATE: state] command.
        Rate limiting enforced here (1 per turn).
        Core logic delegated to cognitive_state.handle_cognitive_state_update().

        Args:
            state_name: Raw state string from parsed command

        Returns:
            Tuple[str, bool]: (replacement_text, success)
            replacement_text is always empty — command is removed from response
        """
        try:
            # Rate limiting: enforce maximum 1 state change per response turn
            if not hasattr(self, '_state_updated_this_turn'):
                self._state_updated_this_turn = False

            if self._state_updated_this_turn:
                logging.warning(
                    f"COGNITIVE_STATE: Ignoring duplicate state change to "
                    f"'{state_name}' (rate limit: 1 per turn)"
                )
                return "⚠️ Cognitive state already updated this turn (rate limit: 1 per turn)", False

            # -------------------------------------------------------------------
            # Warn if QWEN sent a structured multi-key object instead of a simple
            # state word. normalize_state_name() will extract the correct value,
            # but we log here for training feedback visibility.
            # e.g. "processing_mode=analytical | focus=financial_strategy | ..."
            # -------------------------------------------------------------------
            if '|' in state_name or '=' in state_name:
                logging.warning(
                    f"COGNITIVE_STATE: Structured object received in command — "
                    f"expected 1-3 word state, got: '{state_name[:80]}'. "
                    f"normalize_state_name() will extract first valid value. "
                    f"QWEN should use simple states like: [COGNITIVE_STATE: analytical]"
                )

            # Delegate core logic to cognitive_state module
            # normalize_state_name() inside will extract clean state from any format
            from cognitive_state import handle_cognitive_state_update, ORIGIN_CONVERSATION
            success, normalized = handle_cognitive_state_update(
                self.chatbot,
                state_name,
                origin=ORIGIN_CONVERSATION
            )

            # Log what the state resolved to for training transparency
            if success:
                logging.info(
                    f"COGNITIVE_STATE: Command resolved '{state_name[:40]}' → '{normalized}'"
                )

            # Mark rate limit for this turn on success
            if success:
                self._state_updated_this_turn = True

            return f"✅ Cognitive state: {normalized}", True

        except Exception as e:
            logging.error(
                f"COGNITIVE_STATE: Error in command handler: {e}", exc_info=True
            )
            return f"\n\n**Error updating cognitive state: {str(e)}**\n\n", False

    def _handle_summarize_conversation_command(self, session_messages=None) -> Tuple[str, bool]:
        """Handle the [SUMMARIZE_CONVERSATION] command to generate and store a conversation summary."""
        try:
            logging.info("SUMMARIZE_COMMAND: Received [SUMMARIZE_CONVERSATION] command")
            
            conversation = None
            
            # Use directly passed messages (the reliable method)
            if session_messages:
                logging.info(f"SUMMARIZE_COMMAND: Using directly passed messages: {len(session_messages)}")
                conversation = []
                for msg in session_messages:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        if msg['role'] in ['user', 'assistant']:
                            conversation.append({
                                "role": msg["role"], 
                                "content": msg["content"]
                            })
                logging.info(f"SUMMARIZE_COMMAND: Converted to {len(conversation)} conversation messages")
            else:
                # No fallback - if messages aren't passed, it's a programming error
                logging.error("SUMMARIZE_COMMAND ERROR: No session_messages parameter provided")
                command_logger.info("FAILURE: summarize - No messages parameter provided")
                return "\n\n**Error: Summarization method called without message data. This is a programming error.**\n\n", False
            
            # Validate conversation data
            if not conversation:
                logging.error("SUMMARIZE_COMMAND ERROR: No conversation data after processing")
                command_logger.info("FAILURE: summarize - No conversation data after processing")
                return "\n\n**Error: No valid conversation messages found.**\n\n", False
                
            if len(conversation) < 3:
                logging.warning(f"SUMMARIZE_COMMAND: Conversation too short for summarization: {len(conversation)} messages")
                command_logger.info(f"FAILURE: summarize - Conversation too short ({len(conversation)} messages)")
                return f"\n\n**Conversation too short to summarize ({len(conversation)} messages). Please continue the conversation.**\n\n", False

            # Log conversation stats for debugging
            user_msgs = sum(1 for msg in conversation if msg.get('role') == 'user')
            assistant_msgs = sum(1 for msg in conversation if msg.get('role') == 'assistant')
            logging.info(f"SUMMARIZE_COMMAND: Processing conversation with {len(conversation)} messages: {user_msgs} user, {assistant_msgs} assistant")

            # Format the conversation for summarization
            messages_text = []
            for msg in conversation:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if not content:
                    continue
                
                if role == 'user':
                    messages_text.append(f"User: {content}")
                elif role == 'assistant':
                    messages_text.append(f"Assistant: {content}")

            # Get current date and time for both prompt and metadata
            timestamp = datetime.datetime.now()
            current_date = timestamp.strftime("%Y-%m-%d")
            current_time = timestamp.strftime("%H:%M:%S")
                
            # Create enhanced prompt
            summary_prompt = f"""I am reviewing my conversation history to create a summary for my future self.

            CONVERSATION HISTORY:
            {'\n'.join(messages_text)}

            TASK: Create a concise summary of this conversation using FIRST-PERSON language.

            CRITICAL INSTRUCTION: 
            - Write as yourself who is remembering this conversation
            - Use "I", "me", "my" when referring to yourself
            - Use "Ken" when referring  the user
            - Write naturally, as if writing in your own journal

            INCLUDE IN YOUR SUMMARY:
            - Key facts and context you want to remember
            - Command patterns that worked well 
            - Preferences you've discovered
            - Topics that remain open for future discussion
            - Important insights or breakthroughs 

            FORMATTING GUIDELINES:
            1. Keep the summary under a few pages idealy one page two or three if necessary
            2. Focus on key points, main questions, and important conclusions
            3. Format as coherent paragraphs (not bullet points)
            4. Write in a natural, flowing narrative style
            5. Use first-person throughout: "I noticed..." not "The AI noticed..."
            6. End with: "Summary created on {current_date} at {current_time}"

            REMEMBER: This summary is for YOUR future self. Write it the way YOU would want to remember this conversation.

            Please write your summary now:"""
            # Generate the summary
            summary = None
            try:
                summary = self.chatbot.llm.invoke(summary_prompt)
                logging.info("SUMMARIZE_COMMAND: Successfully generated summary using direct LLM method")
            except Exception as e:
                logging.error(f"SUMMARIZE_COMMAND ERROR: LLM error during summarization: {e}", exc_info=True)
                command_logger.info(f"FAILURE: summarize - LLM error: {str(e)}")
                return "\n\n**Error generating conversation summary. LLM invocation failed.**\n\n", False
                
            if not summary or not summary.strip():
                logging.warning("SUMMARIZE_COMMAND ERROR: Generated summary is empty")
                command_logger.info("FAILURE: summarize - Generated empty summary")
                return "\n\n**Error: Failed to generate conversation summary. Generated content was empty.**\n\n", False
            
            # Log the first bit of the summary for debugging
            logging.info(f"SUMMARIZE_COMMAND: Generated summary: {summary[:100]}...")
                
            # Store the summary using transaction coordination to ensure database sync
            if hasattr(self.chatbot, 'store_memory_with_transaction'):
                try:
                    # Prepare metadata with standardized format including date and time
                    metadata = {
                    "type": "conversation_summary",
                    "source": "summarize_conversation_command",
                    "created_at": timestamp.isoformat(),
                    "summary_id": f"summary_{timestamp.strftime('%Y%m%d%H%M%S')}",
                    "is_latest": True,
                    "date": current_date,
                    "time": current_time,
                    "summary_date": current_date,  # Keep for backward compatibility
                    "summary_time": current_time,  # Keep for backward compatibility
                    "tags": ["conversation_summary", f"date={current_date}"],  # CORRECT - ARRAY
                    "tracking_id": str(uuid.uuid4())  # Add unique tracking ID
                }
                                        
                    # Store with transaction coordination
                    logging.info("SUMMARIZE_COMMAND: Calling store_memory_with_transaction")
                    success, memory_id = self.chatbot.store_memory_with_transaction(
                        content=summary,
                        memory_type="conversation_summary",
                        metadata=metadata,
                        confidence=0.7
                    )
                    
                    if success:
                        logging.info(f"SUMMARIZE_COMMAND SUCCESS: Stored summary with ID {memory_id}")
                                                
                        
                        # ────────────────────────────────────────────────────
                        # DEFENSIVE TOKEN COUNTER RESET (added 2026-05-03)
                        # ────────────────────────────────────────────────────
                        # Mirror of the reset in _handle_summarize_conversation_wrapper.
                        # The wrapper falls through to this fallback when the
                        # conversation_summary_manager is unavailable or fails,
                        # so this success path also needs to clear the counter.
                        # See wrapper for full rationale.
                        # ────────────────────────────────────────────────────
                        try:
                            if hasattr(self.chatbot, 'reset_token_counter_after_summary'):
                                reset_ok = self.chatbot.reset_token_counter_after_summary(keep_lifetime_stats=True)
                                if reset_ok:
                                    logging.info("SUMMARIZE_COMMAND: Token counter reset successfully after summary")
                                else:
                                    logging.warning("SUMMARIZE_COMMAND: Token counter reset returned False")
                            else:
                                logging.error("SUMMARIZE_COMMAND: reset_token_counter_after_summary method not found on chatbot")
                        except Exception as reset_err:
                            # Don't let counter-reset failure invalidate a successful summary
                            logging.error(f"SUMMARIZE_COMMAND: Token counter reset raised exception: {reset_err}", exc_info=True)
                        
                        # Return the summary with the date for better user experience
                        confirmation = f"\n\n**✅ Conversation Successfully Summarized & Stored ({current_date} at {current_time}):**\n{summary}\n\n"
                        logging.info("SUMMARIZE_COMMAND: Returning success confirmation to user")
                        
                        
                        # Insert the summary into context for the LLM if method exists
                        if hasattr(self, '_insert_summaries_into_context'):
                            self._insert_summaries_into_context(confirmation)
                            logging.info("SUMMARIZE_COMMAND: Inserted summary into LLM context")
                        
                        return confirmation, True
                    else:
                        logging.error("SUMMARIZE_COMMAND ERROR: Transaction coordinator failed to store summary")
                        command_logger.info("FAILURE: summarize - Transaction failed")
                        return "\n\n**Error: Failed to store conversation summary. Transaction coordination failed - this preserves database consistency.**\n\n", False
                        
                except Exception as tx_err:
                    logging.error(f"SUMMARIZE_COMMAND ERROR: Error in transaction process: {tx_err}", exc_info=True)
                    command_logger.info(f"FAILURE: summarize - Transaction error: {str(tx_err)}")
                    return f"\n\n**Error: Failed to store conversation summary. Transaction error: {str(tx_err)}**\n\n", False
            else:
                logging.error("SUMMARIZE_COMMAND ERROR: Transaction coordinator not available")
                command_logger.info("FAILURE: summarize - No transaction coordinator available")
                return "\n\n**Error: Transaction coordinator not available. Cannot ensure database consistency for summaries.**\n\n", False

        except Exception as e:
            logging.error(f"SUMMARIZE_COMMAND CRITICAL ERROR: Unhandled exception: {e}", exc_info=True)
            command_logger.info(f"FAILURE: summarize - Unhandled exception: {str(e)}")
            return f"\n\n**⚠️ Error summarizing conversation: {str(e)}**\n\n", False
    
        
      # This function is used in our show system prompt command to prevent commands in system prompt from executirng when displayed  
    def _escape_command_syntax(self, text: str) -> str:
        """More aggressive command escaping to prevent execution."""
        if not text or not isinstance(text, str):
            return text
        
        # Use a more aggressive approach - replace [ with &#91; HTML entity
        # This should prevent ANY command pattern matching
        escaped_text = text.replace('[', '&#91;').replace(']', '&#93;')
        
        return escaped_text
    
    
    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """Parse parameter string (e.g., "key1=value1 | key2='value 2' | key3=val3") into a dictionary."""
        params = {}
        if not params_str or not params_str.strip():
            return params

        # Use regex to handle quoted values and key=value pairs separated by |
        # This regex finds key=value pairs, respecting single/double quotes around values
        # Pattern: key_chars = ( non_quote_value | 'quoted_value' | "double_quoted_value" )
        pattern = re.compile(r"""
            \s*                      # Optional leading whitespace
            ([\w.-]+)                # Key (word chars, dots, hyphens)
            \s*=\s*                  # Equals sign surrounded by optional whitespace
            (                        # Start capturing value
                '([^']*)'            # Value in single quotes (capture content inside)
                |
                "([^"]*)"            # Value in double quotes (capture content inside)
                |
                ([^|\s][^|]*)        # Value without quotes (non-pipe char, followed by non-pipes until | or end)
            )                        # End capturing value
            \s*                      # Optional trailing whitespace
            (?:\||\Z)                # Followed by a pipe or end of string (non-capturing)
        """, re.VERBOSE)

        for match in pattern.finditer(params_str):
            key = match.group(1).strip()
            # Value is captured in group 2, but needs checking which sub-pattern matched
            val_single_quoted = match.group(3)
            val_double_quoted = match.group(4)
            val_unquoted = match.group(5)

            if val_single_quoted is not None:
                value = val_single_quoted
            elif val_double_quoted is not None:
                value = val_double_quoted
            else:
                value = val_unquoted.strip() # Strip whitespace from unquoted values

            params[key] = value

        return params
    

    def _parse_confidence(self, confidence_str: str) -> float:
        """Parse confidence value, ensuring it's between 0.1 and 1.0."""
        try:
            confidence = float(confidence_str)
            # Clamp the value between 0.1 (minimum meaningful confidence) and 1.0 (maximum)
            return max(0.1, min(1.0, confidence))
        except (ValueError, TypeError):
            # Default confidence if parsing fails or input is invalid
            logging.warning(f"Invalid confidence value '{confidence_str}', using default 0.5.")
            return 0.5
        
    def _handle_conversation_summary_search(self, query: str) -> Tuple[str, bool]:
        """
        Handle [SEARCH: conversation_summaries] and [SEARCH: conversation_summaries latest] commands.
        
        This function retrieves conversation summaries from the Vector DB and inserts them
        into the conversation context so the LLM can "see" them.
        
        For 'latest' requests: Uses direct Qdrant query sorted by created_at timestamp
        For general requests: Uses semantic search with type filter
        
        Args:
            query (str): The search query which could include 'latest'
            
        Returns:
            Tuple[str, bool]: (formatted results, success flag)
        """
        try:
            logging.info(f"SUMMARY_SEARCH: Processing conversation summary search: '{query}'")
            
            # =====================================================================
            # CASE 1: Request for LATEST summary - use timestamp-based retrieval
            # =====================================================================
            if 'latest' in query.lower():
                # Use the direct method that queries by metadata and sorts by timestamp
                # This bypasses similarity thresholds which can incorrectly filter out summaries
                return self._get_latest_conversation_summary()
            
            # =====================================================================
            # CASE 2: General conversation summary search - get all/multiple summaries
            # =====================================================================
            
            # Set up metadata filter for conversation_summary type
            metadata_filters = {"type": "conversation_summary"}
            
            # Use a generic search term - we're relying on the metadata filter, not semantics
            search_query = "conversation summary"
            
            # Try direct Qdrant query first (more reliable with metadata filters)
            try:
                from qdrant_client.http import models as qdrant_models
                from config import QDRANT_COLLECTION_NAME
                
                logging.info("SUMMARY_SEARCH: Using direct Qdrant query for all summaries")
                
                # Build filter for conversation_summary type
                type_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.type",  # Note: metadata is nested in payload
                            match=qdrant_models.MatchValue(value="conversation_summary")
                        )
                    ]
                )
                
                # Query Qdrant directly - bypasses LangChain's threshold filtering
                results = self.vector_db.client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    scroll_filter=type_filter,
                    limit=50,  # Get plenty of summaries
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                if not results:
                    logging.warning("SUMMARY_SEARCH: No summaries found via direct query")
                    return self._no_summaries_found_message(), True
                
                # Sort by created_at timestamp (most recent first)
                sorted_results = sorted(
                    results,
                    key=lambda p: p.payload.get('metadata', {}).get('created_at', ''),
                    reverse=True
                )
                
                logging.info(f"SUMMARY_SEARCH: Found {len(sorted_results)} conversation summaries")
                
                # Format the results
                formatted_output = "\n\n**===== CONVERSATION SUMMARIES =====**\n"
                formatted_output += f"Found {len(sorted_results)} conversation summaries (newest first):\n\n"
                
                # Display summaries with metadata.
                # Cap at 5 (was 10) to keep the pass-2 synthesis prompt bounded.
                # Conversation summaries are long-form (500+ words each); showing more
                # than 5 overwhelms context and degrades QWEN's integration quality.
                # Constant pulled out so the truncation hint below stays in sync.
                LITERAL_SUMMARY_CAP = 5
                for i, point in enumerate(sorted_results[:LITERAL_SUMMARY_CAP], 1):
                    content = point.payload.get('page_content', 'No content')
                    metadata = point.payload.get('metadata', {})
                    
                    # Extract date/time information
                    created_at = metadata.get('created_at', 'Unknown')
                    summary_date = metadata.get('date', metadata.get('summary_date', 'Unknown'))
                    summary_time = metadata.get('time', metadata.get('summary_time', ''))
                    
                    # Format the created_at timestamp for display
                    display_date = summary_date
                    if created_at and created_at != 'Unknown':
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(created_at.split('.')[0])
                            display_date = dt.strftime("%Y-%m-%d at %H:%M")
                        except (ValueError, AttributeError):
                            display_date = f"{summary_date} {summary_time}".strip()
                    
                    # Truncate long content for display
                    content_preview = content[:500] + "..." if len(content) > 500 else content
                    
                    formatted_output += f"**--- Summary #{i} ({display_date}) ---**\n"
                    formatted_output += f"{content_preview}\n\n"
                
                # Note if there are more summaries beyond the display cap
                if len(sorted_results) > LITERAL_SUMMARY_CAP:
                    formatted_output += (
                        f"*({len(sorted_results) - LITERAL_SUMMARY_CAP} "
                        f"older summaries not shown — refine query or use date filter to access)*\n\n"
                    )
                
                formatted_output += "**===== END OF CONVERSATION SUMMARIES =====**\n\n"
                
                # Persistent context injection removed — was inserting these summaries
                # as a system message at position 0 of current_conversation, which then
                # rode along in EVERY subsequent turn's prompt, eating context tokens
                # across the session. Pass-2 synthesis (chatbot.py search_results_pattern
                # Format E) now picks this block up from the inline substitution and
                # gives QWEN single-turn visibility — no cross-turn bloat.
                logging.info(
                    f"SUMMARY_SEARCH: Returning {len(formatted_output)} chars inline "
                    f"(persistent context injection disabled)"
                )
                return formatted_output, True
                
            except Exception as direct_query_error:
                # If direct query fails, fall back to semantic search
                logging.warning(f"SUMMARY_SEARCH: Direct query failed, falling back to semantic search: {direct_query_error}")
            
            # =====================================================================
            # FALLBACK: Use semantic search with metadata filter
            # =====================================================================
            logging.info("SUMMARY_SEARCH: Falling back to semantic search method")
            
            # Execute the search using comprehensive mode (lower threshold)
            search_results, search_success = self._handle_search_with_mode(
                search_query,
                "comprehensive",  # Use comprehensive mode to avoid filtering out summaries
                metadata_filters
            )
            
            # Check if search was successful
            if not search_success or "NO RESULTS FOUND" in search_results:
                logging.warning(f"SUMMARY_SEARCH: No summaries found via semantic search")
                return self._no_summaries_found_message(), True
            
            # Persistent context injection removed (see Edit 8c rationale). Inline
            # substitution + pass-2 extraction handles visibility cleanly without
            # accumulating system messages across turns.
            logging.info("SUMMARY_SEARCH: Fallback semantic results returned inline")
            return search_results, True
            
        except Exception as e:
            logging.error(f"SUMMARY_SEARCH CRITICAL ERROR: {e}", exc_info=True)
            return (
                "\n\n**===== ERROR RETRIEVING CONVERSATION SUMMARIES =====**\n"
                f"An error occurred while retrieving conversation summaries: {e}\n"
                "Please inform Ken.\n"
                "**===== END OF ERROR =====**\n\n"
            ), False

    def _get_latest_conversation_summary(self) -> Tuple[str, bool]:
        """
        Retrieve the most recent conversation summary by created_at timestamp.
        Uses direct Qdrant query with metadata filter, bypassing similarity threshold.
        
        Returns:
            Tuple[str, bool]: (formatted summary, success)
        """
        try:
            from qdrant_client.http import models as qdrant_models
            from config import QDRANT_COLLECTION_NAME
            
            logging.info("SUMMARY_SEARCH: Retrieving latest conversation summary by timestamp")
            
            # Build filter for conversation_summary type
            type_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.type",
                        match=qdrant_models.MatchValue(value="conversation_summary")
                    )
                ]
            )
            
            # Query Qdrant directly - get more results so we can sort by date
            results = self.vector_db.client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=type_filter,
                limit=20,  # Get recent summaries
                with_payload=True,
                with_vectors=False
            )[0]
            
            if not results:
                logging.warning("SUMMARY_SEARCH: No conversation summaries found in Vector DB")
                return self._no_summaries_found_message(), False
            
            # Sort by created_at timestamp (most recent first)
            sorted_results = sorted(
                results,
                key=lambda p: p.payload.get('metadata', {}).get('created_at', ''),
                reverse=True
            )
            
            # Get the latest one
            latest = sorted_results[0]
            content = latest.payload.get('page_content', '')
            metadata = latest.payload.get('metadata', {})
            created_at = metadata.get('created_at', 'Unknown')
            summary_date = metadata.get('date', metadata.get('summary_date', 'Unknown'))
            
            # Format the output
            formatted = f"\n\n**===== LATEST CONVERSATION SUMMARY =====**\n"
            formatted += f"**Date:** {summary_date}\n"
            formatted += f"**Created:** {created_at}\n\n"
            formatted += f"{content}\n"
            formatted += f"**===== END OF LATEST SUMMARY =====**\n\n"
            
            logging.info(
                f"SUMMARY_SEARCH: Retrieved latest summary from {created_at} "
                f"(returning inline, persistent context injection disabled)"
            )
            
            # Persistent context injection removed — same rationale as
            # _handle_conversation_summary_search. Format E in chatbot.py's
            # search_results_pattern picks this "LATEST CONVERSATION SUMMARY"
            # block up for pass-2 synthesis on the current turn only.
            return formatted, True
            
        except Exception as e:
            logging.error(f"SUMMARY_SEARCH: Error retrieving latest summary: {e}", exc_info=True)
            return f"Error retrieving conversation summary: {e}", False

    def _no_summaries_found_message(self) -> str:
        """Return formatted message when no summaries are found."""
        return (
            "\n\n**===== LATEST CONVERSATION SUMMARY =====**\n"
            "**NO CONVERSATION SUMMARIES FOUND**\n\n"
            "No previous conversation summaries exist in memory yet.\n"
            "**===== END OF LATEST SUMMARY =====**\n\n"
        )
    
    def _insert_summaries_into_context(self, summaries_text: str):
        """
        Insert summaries into the conversation context so the LLM can "see" them.
        
        Args:
            summaries_text (str): The formatted summaries text
        """
        try:
            logging.info("Inserting conversation summaries into LLM context")
            
            # For Streamlit UI
            if 'streamlit' in sys.modules:
                import streamlit as st
                if hasattr(st, 'session_state') and 'messages' in st.session_state:
                    # Check if summaries are already present to avoid duplicates
                    summary_already_added = any(
                        msg.get("role") == "system" and 
                        "CONVERSATION SUMMARIES" in msg.get("content", "")
                        for msg in st.session_state.messages
                    )
                    
                    if not summary_already_added:
                        # Create a system message with the summaries
                        system_message = {
                            "role": "system",
                            "content": summaries_text
                        }
                        
                        # Insert at the beginning of the conversation for maximum visibility
                        st.session_state.messages.insert(0, system_message)
                        logging.info("Successfully inserted summaries into Streamlit session state")
                    else:
                        logging.info("Summaries already present in Streamlit session state, skipping insertion")
            
            # For chatbot's internal state - if it maintains conversation history
            if hasattr(self.chatbot, 'current_conversation'):
                # Check if summaries are already in the current conversation
                summary_already_added = any(
                    msg.get("role") == "system" and 
                    "CONVERSATION SUMMARIES" in msg.get("content", "")
                    for msg in self.chatbot.current_conversation
                )
                
                if not summary_already_added:
                    # Add to the chatbot's conversation history
                    system_message = {
                        "role": "system",
                        "content": summaries_text
                    }
                    
                    # Insert at the beginning
                    self.chatbot.current_conversation.insert(0, system_message)
                    logging.info("Successfully inserted summaries into chatbot's current_conversation")
                else:
                    logging.info("Summaries already present in chatbot's conversation, skipping insertion")
                    
        except Exception as e:
            logging.error(f"Error inserting summaries into context: {e}", exc_info=True)

    def _is_search_result_notification(self, content: str) -> bool:
        """
        Check if the given content looks like a search result notification,
        especially one indicating no results were found.

        Args:
            content (str): The content to check

        Returns:
            bool: True if content is likely a search result notification (esp. "no results")
        """
        if not content:
            return False

        content_lower = content.lower()
        # Check for typical "no results" indicators and common formatting
        no_result_patterns = [
            "no data found for query",
            "no relevant information found",
            "no memories found",
            "no data found",
            "no results found",
            "no matches found",
            "no results passed quality threshold",
            "no conversation summaries found",
            "could not find memory to forget",  # From forget command feedback
            "memory not found",  # From correct command feedback
            "===== end of memory retrieval =====",
            "===== end of search =====",
            "===== memory retrieval result =====",
            "===== search results for:",
        ]

        # Check if the content *starts* or *contains* these key phrases
        # Using startswith is more specific for headers/footers
        if any(content_lower.strip().startswith(f"**{pattern}") for pattern in no_result_patterns if pattern.startswith("=====")):
            return True
        if any(pattern in content_lower for pattern in no_result_patterns):
            return True

        return False
    
    def _handle_show_system_prompt_command(self) -> Tuple[str, bool]:
        """Handle [SHOW_SYSTEM_PROMPT] command to display current system prompt."""
        try:
            # CRITICAL DEBUG LOGGING
            logging.critical("🔍 SHOW_SYSTEM_PROMPT: Handler method called!")
            print("🔍 SHOW_SYSTEM_PROMPT: Handler method called!")  # Also print to console
            
            logging.info("SHOW_SYSTEM_PROMPT: Displaying current system prompt")
            
            # Debug: Show what file we're trying to read
            expected_path = r"C:\Users\kenba\source\repos\Ollama3\system_prompt.txt"
            actual_path = getattr(self.chatbot, 'system_prompt_file', 'system_prompt.txt')
            
            logging.critical(f"SHOW_SYSTEM_PROMPT: Expected path: {expected_path}")
            logging.critical(f"SHOW_SYSTEM_PROMPT: Actual path: {actual_path}")
            logging.critical(f"SHOW_SYSTEM_PROMPT: Expected exists: {os.path.exists(expected_path)}")
            logging.critical(f"SHOW_SYSTEM_PROMPT: Actual exists: {os.path.exists(actual_path)}")
            
            # Use the expected path
            file_path = expected_path
            
            if not os.path.exists(file_path):
                error_msg = f"\n\n**Error: System prompt file not found: {file_path}**\n\n"
                logging.critical(f"SHOW_SYSTEM_PROMPT: File not found - {file_path}")
                command_logger.info(f"❌ FAILURE: show_system_prompt - File not found")
                return error_msg, False
                
            # Read the current system prompt
            with open(file_path, 'r', encoding='utf-8') as f:
                current_prompt = f.read()
            
            # Debug: Log what we're about to show
            logging.critical(f"SHOW_SYSTEM_PROMPT: Read {len(current_prompt)} characters from file")
            logging.critical(f"SHOW_SYSTEM_PROMPT: First 100 chars: {current_prompt[:100]}")
            
            # CRITICAL: Escape command syntax to prevent execution
            escaped_prompt = self._escape_command_syntax(current_prompt)
            
            # Format for display with line numbers
            lines = escaped_prompt.split('\n')
            numbered_lines = []
            for i, line in enumerate(lines, 1):
                numbered_lines.append(f"{i:3d}: {line}")
                
            formatted_prompt = "\n".join(numbered_lines)
            
            logging.critical(f"SHOW_SYSTEM_PROMPT: Formatted {len(lines)} lines for display")
            command_logger.info(f"✅ SUCCESS: show_system_prompt - Displayed {len(lines)} lines")
            
            result = f"""

    **===== CURRENT SYSTEM PROMPT =====**
    **File: {file_path}**

    {formatted_prompt}

    **===== END OF SYSTEM PROMPT =====**

    *Total lines: {len(lines)}*
    *This is your operational system prompt with memory commands and guidelines*
    *To modify, use [MODIFY_SYSTEM_PROMPT: action | content]*

    """
            
            logging.critical(f"SHOW_SYSTEM_PROMPT: Returning result with length: {len(result)}")
            return result, True
            
        except Exception as e:
            logging.critical(f"SHOW_SYSTEM_PROMPT: Exception occurred: {e}", exc_info=True)
            command_logger.info(f"❌ FAILURE: show_system_prompt - Error: {str(e)}")
            return "\n\n**Error displaying system prompt.**\n\n", False
        
    def _handle_modify_system_prompt_command(self, action_and_content: str, params_str: Optional[str] = None) -> Tuple[str, bool]:
        """
        Handle [MODIFY_SYSTEM_PROMPT: action | content] command.
        
        Actions:
        - add: Add new lines to the end
        - insert: Insert at specific line number 
        - remove: Remove specific lines
        - replace: Replace specific lines
        """
        try:
            logging.info(f"MODIFY_SYSTEM_PROMPT: Processing modification request")
            
            if not action_and_content or not action_and_content.strip():
                help_text = """
    **===== MODIFY SYSTEM PROMPT HELP =====**

    Usage: [MODIFY_SYSTEM_PROMPT: action | content]

    Actions:
    - **add**: Add new content to the end of the prompt
    Example: [MODIFY_SYSTEM_PROMPT: add | Always be helpful and respectful.]

    - **insert**: Insert content at a specific line number
    Example: [MODIFY_SYSTEM_PROMPT: insert | line=5 | New instruction here.]

    - **remove**: Remove specific line numbers
    Example: [MODIFY_SYSTEM_PROMPT: remove | lines=10-15] or [MODIFY_SYSTEM_PROMPT: remove | lines=5,7,9]

    - **replace**: Replace specific line numbers with new content
    Example: [MODIFY_SYSTEM_PROMPT: replace | lines=5-7 | New replacement text.]

    **IMPORTANT NOTES**:
    - These changes are permanent and will affect all future conversations
    - Use [SHOW_SYSTEM_PROMPT] first to see current content and line numbers
    - Command examples in [SHOW_SYSTEM_PROMPT] output are escaped with backslashes
    - When adding commands to the prompt, they will be functional (not escaped)

    **===== END OF HELP =====**
    """
                
                return help_text, True
                
            # Parse the action
            action = action_and_content.strip().lower()
            params = self._parse_params(params_str or "")
            
            # Read current prompt
            if not os.path.exists(self.chatbot.system_prompt_file):
                command_logger.info(f"❌ FAILURE: modify_system_prompt - System prompt file not found")
                return "\n\n**Error: System prompt file not found.**\n\n", False
                
            with open(self.chatbot.system_prompt_file, 'r', encoding='utf-8') as f:
                current_lines = f.read().split('\n')
                
            original_line_count = len(current_lines)
            modified_lines = current_lines.copy()
            
            # Process the action
            if action == 'add':
                content = params.get('content', '').strip()
                if not content:
                    command_logger.info(f"❌ FAILURE: modify_system_prompt - No content for add action")
                    return "\n\n**Error: No content provided for add action.**\n\n", False
                    
                # Add the new content
                modified_lines.append('')  # Add blank line for separation
                modified_lines.extend(content.split('\n'))
                
                change_description = f"Added {len(content.split('\n'))} lines to end of prompt"
                
            elif action == 'insert':
                line_num = params.get('line', '')
                content = params.get('content', '').strip()
                
                if not line_num or not content:
                    command_logger.info(f"❌ FAILURE: modify_system_prompt - Missing line or content for insert")
                    return "\n\n**Error: Both 'line' and 'content' required for insert action.**\n\n", False
                    
                try:
                    insert_at = int(line_num) - 1  # Convert to 0-based index
                    if insert_at < 0 or insert_at > len(modified_lines):
                        command_logger.info(f"❌ FAILURE: modify_system_prompt - Line number out of range")
                        return f"\n\n**Error: Line number {line_num} is out of range (1-{len(modified_lines)}).**\n\n", False
                        
                    # Insert the content
                    for i, line in enumerate(content.split('\n')):
                        modified_lines.insert(insert_at + i, line)
                        
                    change_description = f"Inserted {len(content.split('\n'))} lines at position {line_num}"
                    
                except ValueError:
                    command_logger.info(f"❌ FAILURE: modify_system_prompt - Invalid line number format")
                    return "\n\n**Error: Invalid line number format.**\n\n", False
                    
            elif action == 'remove':
                lines_param = params.get('lines', '').strip()
                if not lines_param:
                    command_logger.info(f"❌ FAILURE: modify_system_prompt - No lines parameter for remove")
                    return "\n\n**Error: 'lines' parameter required for remove action.**\n\n", False
                    
                # Parse line numbers (support both ranges and individual lines)
                lines_to_remove = set()
                try:
                    for part in lines_param.split(','):
                        part = part.strip()
                        if '-' in part:
                            # Range like "5-10"
                            start, end = map(int, part.split('-'))
                            lines_to_remove.update(range(start-1, end))  # Convert to 0-based
                        else:
                            # Individual line
                            lines_to_remove.add(int(part) - 1)  # Convert to 0-based
                            
                    # Remove lines in reverse order to maintain indices
                    for line_idx in sorted(lines_to_remove, reverse=True):
                        if 0 <= line_idx < len(modified_lines):
                            modified_lines.pop(line_idx)
                            
                    change_description = f"Removed {len(lines_to_remove)} lines"
                    
                except ValueError:
                    command_logger.info(f"❌ FAILURE: modify_system_prompt - Invalid line format in remove")
                    return "\n\n**Error: Invalid line number format in 'lines' parameter.**\n\n", False
                    
            elif action == 'replace':
                lines_param = params.get('lines', '').strip()
                content = params.get('content', '').strip()
                
                if not lines_param or not content:
                    command_logger.info(f"❌ FAILURE: modify_system_prompt - Missing lines or content for replace")
                    return "\n\n**Error: Both 'lines' and 'content' required for replace action.**\n\n", False
                    
                # Parse line numbers and replace
                try:
                    lines_to_replace = []
                    for part in lines_param.split(','):
                        part = part.strip()
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            lines_to_replace.extend(range(start-1, end))  # Convert to 0-based
                        else:
                            lines_to_replace.append(int(part) - 1)  # Convert to 0-based
                            
                    # Sort and validate
                    lines_to_replace = sorted(set(lines_to_replace))
                    if any(idx < 0 or idx >= len(modified_lines) for idx in lines_to_replace):
                        command_logger.info(f"❌ FAILURE: modify_system_prompt - Line numbers out of range for replace")
                        return "\n\n**Error: One or more line numbers are out of range.**\n\n", False
                        
                    # Replace the lines
                    replacement_lines = content.split('\n')
                    
                    # Remove old lines (in reverse order)
                    for line_idx in reversed(lines_to_replace):
                        modified_lines.pop(line_idx)
                        
                    # Insert new lines at the first position
                    first_idx = lines_to_replace[0]
                    for i, line in enumerate(replacement_lines):
                        modified_lines.insert(first_idx + i, line)
                        
                    change_description = f"Replaced {len(lines_to_replace)} lines with {len(replacement_lines)} new lines"
                    
                except ValueError:
                    command_logger.info(f"❌ FAILURE: modify_system_prompt - Invalid line format for replace")
                    return "\n\n**Error: Invalid line number format.**\n\n", False
                    
            else:
                command_logger.info(f"❌ FAILURE: modify_system_prompt - Unknown action: {action}")
                return f"\n\n**Error: Unknown action '{action}'. Use 'add', 'insert', 'remove', or 'replace'.**\n\n", False
                
            # Create backup of original file
            backup_file = f"{self.chatbot.system_prompt_file}.backup.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(current_lines))
                logging.info(f"Created backup: {backup_file}")
            except Exception as backup_err:
                logging.warning(f"Could not create backup: {backup_err}")
                
            # Write the modified prompt
            try:
                with open(self.chatbot.system_prompt_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(modified_lines))
                    
                # Reinitialize the system prompt and LLM
                self.chatbot._initialize_system_prompt() 
                self.chatbot.current_system_prompt = self.chatbot.deepseek_enhancer.enhance_system_prompt()
                success = self.chatbot.update_llm_system_prompt(self.chatbot.current_system_prompt)
                
                if success:
                    new_line_count = len(modified_lines)
                    command_logger.info(f"✅ SUCCESS: modify_system_prompt - {change_description}")
                    
                    return f"""

    **===== SYSTEM PROMPT MODIFIED =====**

    **Change Made**: {change_description}
    **Original Lines**: {original_line_count}
    **New Lines**: {new_line_count}
    **Backup Created**: {os.path.basename(backup_file)}

    **IMPORTANT**: The system prompt has been permanently modified and the LLM has been reinitialized with the new prompt. This change will affect all future conversations.

    Use [SHOW_SYSTEM_PROMPT] to verify the changes.

    **===== MODIFICATION COMPLETE =====**

    """, True
                else:
                    # Restore from backup if LLM update failed
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    with open(self.chatbot.system_prompt_file, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                        
                    command_logger.info(f"❌ FAILURE: modify_system_prompt - LLM update failed, reverted")
                    return "\n\n**Error: Failed to update LLM with new prompt. Changes reverted.**\n\n", False
                    
            except Exception as write_err:
                logging.error(f"Error writing modified prompt: {write_err}", exc_info=True)
                command_logger.info(f"❌ FAILURE: modify_system_prompt - Write error: {str(write_err)}")
                return "\n\n**Error: Failed to write modified system prompt.**\n\n", False
                
        except Exception as e:
            logging.error(f"Error modifying system prompt: {e}", exc_info=True)
            command_logger.info(f"❌ FAILURE: modify_system_prompt - Error: {str(e)}")
            return "\n\n**Error modifying system prompt.**\n\n", False
    
    def _handle_reminder_command(self, reminder_text: str, params_str: Optional[str] = None) -> Tuple[str, bool]:
        """Handle [REMINDER: text | due=date | ...] command for future reminders."""
        try:
            if not reminder_text or not reminder_text.strip():
                logging.warning("Reminder command received empty text.")
                command_logger.info(f"❌ FAILURE: reminder - Empty reminder text")
                return "\n\n**Cannot set reminder: No text provided.**\n\n", False

            reminder_text = reminder_text.strip()
            params = self._parse_params(params_str or "")
            due_date_raw = params.get('due', '')
            confidence = self._parse_confidence(params.get('confidence', '0.8'))  # Reminders default high confidence

            logging.info(f"Processing REMINDER command: '{reminder_text[:50]}...', Due: '{due_date_raw}'")

            # Prepare metadata
            metadata = {
                "source": params.get('source', "reminder_command"),
                "original_due_request": due_date_raw,
                "confidence": confidence
            }
            
            # Add any other params from the command string
            for key, value in params.items():
                key_lower = key.lower()
                if key_lower not in ('due', 'confidence', 'source'):
                    metadata[key_lower] = value
                    
            # Use the reminder manager to create the reminder
            success, reminder_id = self.chatbot.reminder_manager.create_reminder(
                content=reminder_text,
                due_date=due_date_raw,
                metadata=metadata
            )

            if success:
                logging.info(f"Successfully stored reminder with ID {reminder_id}: {reminder_text[:50]}...")
                
                # Format the display
                if due_date_raw:
                    due_display = f" (Due: {due_date_raw})"
                else:
                    due_display = ""
                    
                return f"\n\n**Reminder Set{due_display}**: {reminder_text}\n<!-- reminder_id:{reminder_id} -->\n\n", True
            else:
                logging.warning(f"Failed to store reminder: {reminder_text[:50]}...")
                command_logger.info(f"❌ FAILURE: reminder - Failed to create reminder")
                return "\n\n**Error setting reminder.**\n\n", False

        except Exception as e:
            logging.error(f"Error handling reminder command: {e}", exc_info=True)
            command_logger.info(f"❌ FAILURE: reminder - Exception: {str(e)}")
            return "\n\n**Error setting reminder.**\n\n", False
        
    def _handle_self_dialogue_command(self, topic: str, turns_param: str = None) -> Tuple[str, bool]:
        """
        Handle [SELF_DIALOGUE: topic | turns=6] command for multi-turn internal self-reasoning.
        Uses only existing knowledge and memory - NO external searches.
        
        Args:
            topic (str): The topic or problem to think about internally
            turns_param (str): Number of turns (default 6)
            
        Returns:
            Tuple[str, bool]: (dialogue result, success flag)
        """
        try:
            # Parse parameters
            if not topic or not topic.strip():
                return "\n\n**Error: Please specify a topic for self-dialogue.**\n\n", False
                
            topic = topic.strip()
            max_turns = int(turns_param) if turns_param and turns_param.isdigit() else 5
            max_turns = min(max_turns, 20)  # Cap at 10 to prevent excessive processing
            
            logging.info(f"🤔 SELF_DIALOGUE: Starting internal reasoning dialogue on: '{topic}' for {max_turns} turns")
            
            # Initialize dialogue
            dialogue_history = []
            internal_insights = []  # Store insights generated during internal reasoning
            
            # Search existing knowledge about the topic first
            existing_knowledge = self._gather_existing_knowledge(topic)
            
            # Create initial system context emphasizing internal reasoning only
  
            system_context = f"""You are engaging in deep internal self-reflection about: "{topic}"

            IMPORTANT GUIDELINES:
            1. Use ONLY your existing knowledge and training - NO external searches
            2. Use [SEARCH: query] to retrieve relevant memories from your knowledge base
            3. Focus on building understanding across multiple turns before storing insights
            4. Save your deepest insights for the FINAL turn using [STORE: insight | type=self]
            5. Connect existing knowledge in novel ways through progressive reasoning
            6. End each response with a deeper question for internal exploration

            STORAGE GUIDANCE: Build your understanding progressively. Only store your most refined insights at the end of the dialogue.

            INTERNAL REASONING OBJECTIVE: Synthesize existing knowledge about "{topic}" to generate deep insights through {max_turns} turns of reflection.

            Existing knowledge context:
            {existing_knowledge}

            Format each response as:
            **Turn X Internal Reflection:** [your deep analysis and connections]
            **Knowledge Connections Identified:** [how different pieces relate]
            **Building Understanding:** [insights developing but not yet ready to store]
            **Deeper Question for Next Turn:** [question for further internal exploration]

            Begin internal reflection on: {topic}"""
            
            # Perform the internal self-dialogue
            # ----------------------------------------------------------------
            # FIX: next_deeper_question is initialised to True (sentinel) so
            # that turns 1-(minimum_turns-1) are never treated as final turns.
            # After each turn the value is updated from the response, so the
            # NEXT iteration can correctly determine is_final_turn BEFORE
            # building its prompt — ensuring the storage instruction reaches
            # QWEN on the actual last turn even when early termination fires.
            # ----------------------------------------------------------------
            next_deeper_question = True  # Sentinel: "keep going" until updated
            minimum_turns = 4  # Require at least 4 turns before early termination

            for turn in range(1, max_turns + 1):
                try:
                    logging.info(f"🤔 SELF_DIALOGUE: Turn {turn}/{max_turns} - Internal reasoning processing")

                    # Determine BEFORE building the prompt whether this is the
                    # final turn. Uses next_deeper_question from the PREVIOUS
                    # iteration (or the sentinel True for the first turn).
                    is_final_turn = (
                        turn == max_turns or
                        (turn >= minimum_turns and not next_deeper_question)
                    )

                    # Create the prompt for this turn
                    if turn == 1:
                        prompt = system_context
                    else:
                        # Build context from previous internal reasoning turns
                        context_summary = self._build_internal_dialogue_context(dialogue_history, internal_insights)

                        if is_final_turn:
                            # Directive final-turn instruction — QWEN has been
                            # deferring storage; now explicitly require it.
                            # IMPORTANT: Show correct syntax with real content
                            # in the command body — not a placeholder word.
                            # Wrong: [STORE: insight | type=self | content="..."]
                            # Right: [STORE: Your full insight text here | type=self | confidence=0.8]
                            final_instruction = (
                                "This is your FINAL turn. Your previous turns built strong analysis. "
                                "Do NOT defer further — synthesize now and store your key conclusions. "
                                "Write a separate STORE command for each major insight, using this exact syntax:\n"
                                "[STORE: Your complete insight written out in full here | type=self | confidence=0.8]\n"
                                "The insight text goes directly after [STORE: — do not use a placeholder word "
                                "like 'insight' or a separate content= parameter. "
                                "Aim for at least 2–3 stored insights at confidence 0.7–0.9."
                            )
                        else:
                            final_instruction = "Continue deepening your understanding."

                        prompt = f"""Continuing internal self-reflection about: "{topic}"

                        Previous internal reasoning:
                        {context_summary}

                        Insights developing:
                        {self._format_internal_insights_summary(internal_insights)}

                        Continue the internal reflection for turn {turn} of {max_turns}.
                        {final_instruction}"""

                    # Get AI response
                    response = self.chatbot.llm.invoke(prompt)

                    if not response:
                        logging.warning(f"🤔 SELF_DIALOGUE: Empty response on turn {turn}")
                        break

                    # Extract insights BEFORE process_response replaces
                    # [STORE:] commands with "✅ Successfully Stored:" messages.
                    # Scanning the processed response finds nothing — the
                    # commands are already substituted by that point.
                    turn_insights = self._extract_stored_insights(response)

                    # Process memory commands (SEARCH, STORE) but NOT external searches
                    final_response, commands_executed = self.process_response(response)

                    internal_insights.extend(turn_insights)

                    # Store this turn
                    turn_data = {
                        "turn": turn,
                        "response": final_response,
                        "commands_executed": commands_executed,
                        "insights_generated": len(turn_insights),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    dialogue_history.append(turn_data)

                    # Update next_deeper_question for the NEXT iteration's
                    # is_final_turn check
                    next_deeper_question = self._extract_next_deeper_question(final_response)

                    # Terminate if this was the final turn
                    if is_final_turn:
                        if turn < max_turns:
                            logging.info(
                                f"🤔 SELF_DIALOGUE: Ending early at turn {turn} — "
                                f"no deeper question after minimum {minimum_turns} turns"
                            )
                        break

                except Exception as turn_error:
                    logging.error(f"🤔 SELF_DIALOGUE: Error in turn {turn}: {turn_error}")
                    break
            
            # Format the complete internal dialogue for display
            formatted_dialogue = self._format_internal_dialogue(topic, dialogue_history, internal_insights, max_turns)
            
            # Store the complete internal dialogue summary using coordinated transaction
            self._store_internal_dialogue_summary(topic, dialogue_history, internal_insights)
            
           
            total_insights = len(internal_insights)
            command_logger.info(f"✅ SUCCESS: self_dialogue - Completed {len(dialogue_history)} turns on '{topic}' with {total_insights} internal insights")
            
            return formatted_dialogue, True
            
        except Exception as e:
            logging.error(f"🤔 SELF_DIALOGUE: Error: {e}", exc_info=True)
            command_logger.info(f"❌ FAILURE: self_dialogue - Error: {str(e)}")
            return f"\n\n**Error during internal self-dialogue: {str(e)}**\n\n", False

    # Helper methods for internal reasoning (you'll need to add these):

    def _gather_existing_knowledge(self, topic: str) -> str:
        """Gather existing knowledge about the topic from memory databases."""
        try:
            # FIXED: Use the correct search method
            search_results = self.vector_db.search(
                query=topic,
                mode="comprehensive", 
                k=5
            )
            
            if search_results:
                knowledge_summary = "\n".join([f"- {result.get('content', '')[:200]}..." 
                                            for result in search_results])
                return f"Relevant existing knowledge:\n{knowledge_summary}"
            else:
                return "No specific existing knowledge found - relying on base training knowledge."
                
        except Exception as e:
            logging.error(f"Error gathering existing knowledge: {e}")
            return "Error accessing existing knowledge - proceeding with base training only."

    def _build_internal_dialogue_context(self, dialogue_history: list, internal_insights: list) -> str:
        """
        Build context summary from previous internal reasoning turns.

        FIX: original code only kept last 2 turns at 300 chars each. By turn 4
        all of turns 1-2 were lost, leaving QWEN unable to synthesize across
        her own earlier reasoning. Widened to last 3 turns at 500 chars each.
        """
        try:
            if not dialogue_history:
                return "No previous internal reasoning context."

            context_parts = []
            # Keep last 3 turns so early insights remain visible through turn 4+
            for turn in dialogue_history[-3:]:
                turn_num = turn.get('turn', 'unknown')
                response_preview = turn.get('response', '')[:500]
                context_parts.append(f"Turn {turn_num}: {response_preview}...")

            return "\n\n".join(context_parts)

        except Exception as e:
            logging.error(f"Error building internal dialogue context: {e}")
            return "Error building context from previous turns."

    def _format_internal_insights_summary(self, internal_insights: list) -> str:
        """Format internal insights for context."""
        try:
            if not internal_insights:
                return "No insights generated yet."
                
            insights_text = "\n".join([f"- {insight}" for insight in internal_insights[:5]])  # Last 5 insights
            return f"Recent internal insights:\n{insights_text}"
            
        except Exception as e:
            logging.error(f"Error formatting insights: {e}")
            return "Error formatting internal insights."

    def _extract_stored_insights(self, response_text: str) -> list:
        r"""
        Extract insights that were stored during this turn.

        FIX: original pattern r'\[STORE:\s*(.*?)\s*\|\s*type=self\s*\]' required
        the command to end immediately after 'type=self', so any STORE with
        additional parameters (e.g. confidence=0.8) was silently missed even
        though process_response() had already executed and stored it.
        New pattern checks that 'type=self' appears anywhere in the pipe
        parameters rather than requiring it to be the last parameter.
        """
        insights = []
        try:
            import re
            # Match [STORE: content | ... type=self ... ] regardless of parameter order
            # Capture everything before the first pipe as the insight content
            store_pattern = r'\[STORE:\s*(.*?)\s*\|[^\]]*type\s*=\s*self[^\]]*\]'
            matches = re.findall(store_pattern, response_text, re.IGNORECASE | re.DOTALL)
            insights.extend([m.strip() for m in matches if m.strip()])

        except Exception as e:
            logging.error(f"Error extracting stored insights: {e}")

        return insights
    
    def _store_internal_dialogue_summary(self, topic: str, dialogue_history: list, internal_insights: list):
        """
        Store the complete internal dialogue summary using coordinated transaction.
        
        Stores with proper metadata including:
        - memory_type: self_dialogue_summary
        - confidence: 0.80 (self-generated insights)
        - tags: self_dialogue, topic, autonomous, internal_reasoning
        
        Args:
            topic (str): The dialogue topic
            dialogue_history (list): List of dialogue turns
            internal_insights (list): List of insights generated
        """
        try:
            # Create comprehensive summary
            summary_content = f"""Internal Self-Dialogue Summary: {topic}

    Completed {len(dialogue_history)} turns of deep internal reasoning.

    Key Insights Generated:
    {chr(10).join([f"- {insight}" for insight in internal_insights[:10]])}

    Total internal insights: {len(internal_insights)}
    Reasoning depth: {len(dialogue_history)} turns
    Purpose: Internal metacognitive reflection and knowledge synthesis

    This dialogue represents the AI's internal reasoning process, connecting existing knowledge to generate new insights through self-reflection."""

            # Build metadata with proper tags and confidence
            # Sanitize topic for use in tags (replace spaces, remove special chars)
            import re
            safe_topic = re.sub(r'[^a-zA-Z0-9_]', '_', topic.lower())[:30]
            
            metadata = {
                "type": "self_dialogue_summary",
                "source": "internal_reasoning",
                "topic": topic,
                "turns_completed": len(dialogue_history),
                "insights_generated": len(internal_insights),
                "dialogue_type": "internal",
                "created_at": datetime.datetime.now().isoformat(),
                "tags": f"self_dialogue,{safe_topic},autonomous,internal_reasoning,qwen_self"
            }

            # Store using coordinated transaction (both SQL and Vector DB)
            success, memory_id = self.chatbot.store_memory_with_transaction(
                content=summary_content,
                memory_type="self_dialogue_summary", 
                metadata=metadata,
                confidence=0.80  # Good confidence for self-generated insights
            )
            
            if success:
                logging.info(f"🤔 SELF_DIALOGUE: Stored internal dialogue summary with ID {memory_id}")
                logging.info(f"   Type: self_dialogue_summary, Confidence: 0.80")
                logging.info(f"   Tags: {metadata['tags']}")
            else:
                logging.error(f"🤔 SELF_DIALOGUE: Failed to store dialogue summary: {memory_id}")
                
        except Exception as e:
            logging.error(f"🤔 SELF_DIALOGUE: Error storing dialogue summary: {e}", exc_info=True)
    

    def _extract_next_deeper_question(self, response_text: str) -> str:
        """Extract the next deeper question for internal exploration."""
        try:
            # Look for patterns like "Deeper Question for Next Turn:"
            import re
            question_patterns = [
                r'Deeper Question for Next Turn:\s*(.*?)(?:\n|$)',
                r'Next deeper question:\s*(.*?)(?:\n|$)',
                r'Further internal exploration:\s*(.*?)(?:\n|$)'
            ]
            
            for pattern in question_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                    
            return None
            
        except Exception as e:
            logging.error(f"Error extracting deeper question: {e}")
            return None

    def _format_internal_dialogue(self, topic: str, dialogue_history: list, internal_insights: list, max_turns: int) -> str:
        """Format the complete internal dialogue for display."""
        try:
            formatted_parts = [
                f"\n\n## 🤔 Internal Self-Dialogue: {topic}\n",
                f"**Completed {len(dialogue_history)} turns of internal reasoning (max: {max_turns})**\n"
            ]
            
            for turn in dialogue_history:
                turn_num = turn.get('turn', 'unknown')
                response = turn.get('response', '')
                insights_count = turn.get('insights_generated', 0)
                
                formatted_parts.append(f"### Turn {turn_num}")
                formatted_parts.append(response)
                if insights_count > 0:
                    formatted_parts.append(f"*({insights_count} new insights generated)*")
                formatted_parts.append("---")
            
            # Add summary — conditionally accurate about storage
            total_insights = len(internal_insights)
            formatted_parts.append(f"\n**Internal Dialogue Summary:**")
            formatted_parts.append(f"- Total insights generated: {total_insights}")
            formatted_parts.append(f"- Deepest reasoning achieved in {len(dialogue_history)} turns")
            if total_insights > 0:
                formatted_parts.append(
                    f"- {total_insights} insight{'s' if total_insights != 1 else ''} "
                    f"stored as type=self memories"
                )
            else:
                formatted_parts.append(
                    "- No insights stored during dialogue — "
                    "consider storing key conclusions manually with [STORE: insight | type=self]"
                )
            
            return "\n\n".join(formatted_parts)
            
        except Exception as e:
            logging.error(f"Error formatting internal dialogue: {e}")
            return f"Error formatting dialogue results: {str(e)}"

    
    def _handle_research_dialogue_command(self, topic: str, turns_param: str = None) -> Tuple[str, bool]:
            """
            Handle [WEB_SEARCH: topic | turns=6] command for multi-turn reasoning with web search.
            ENHANCED with explicit memory command instructions to encourage AI to store findings.
            
            Args:
                topic (str): The topic or problem to research
                turns_param (str): Number of turns (default 6)
                
            Returns:
                Tuple[str, bool]: (dialogue result, success flag)
                                                        
            """
            try:
                # Enhanced parameter validation and logging
                if not topic or not topic.strip():
                    logging.error("WEB_SEARCH: Empty topic provided")
                    command_logger.info(f"❌ FAILURE: web_search - Empty topic")
                    return "\n\n**Error: Please specify a topic for research dialogue.**\n\n", False
                    
                topic = topic.strip()
                max_turns = int(turns_param) if turns_param and turns_param.isdigit() else 6
                max_turns = min(max_turns, 20)
                
                # ENHANCED: Detailed command initiation logging
                logging.info(f"🔍 WEB_SEARCH: Topic: '{topic}'")
                logging.info(f"🔍 WEB_SEARCH: Max turns: {max_turns}")
                logging.info(f"🔍 WEB_SEARCH: Timestamp: {datetime.datetime.now().isoformat()}")
                
                # Log to command logger for session tracking
                command_logger.info(f"🔍 START: web_search - Topic: '{topic}', Turns: {max_turns}")
                
                # Initialize WebKnowledgeSeeker with enhanced error handling
                web_seeker = None
                try:
                                                            
                    from web_knowledge_seeker import WebKnowledgeSeeker
                    web_seeker = WebKnowledgeSeeker(self.memory_db, self.vector_db, self.chatbot)
                    logging.debug("🌐 WEB_SEARCH: WebKnowledgeSeeker successfully initialized")
                except ImportError as e:
                    logging.error(f"🌐 WEB_SEARCH: CRITICAL - WebKnowledgeSeeker import failed: {e}")
                    command_logger.info(f"❌ FAILURE: web_search - WebKnowledgeSeeker not available")
                    return "\n\n**Error: External knowledge search not available.**\n\n", False
                except Exception as e:
                    logging.error(f"🌐 WEB_SEARCH: CRITICAL - WebKnowledgeSeeker initialization failed: {e}")
                    command_logger.debug(f"❌ FAILURE: web_search - WebKnowledgeSeeker init error")
                    return "\n\n**Error: Failed to initialize external search capability.**\n\n", False
                
                # Initialize tracking with enhanced logging
                dialogue_history = []
                external_knowledge_cache = {}
                
                # ENHANCED: System context with explicit memory command instructions and examples
                logging.info(f"🔍 WEB_SEARCH: Creating initial system context for topic: '{topic}'")
                
                system_context = f"""You are conducting research about: "{topic}"

    CRITICAL RESEARCH COMMANDS - YOU MUST USE THESE:

    1. [EXTERNAL_SEARCH: specific_query] - Search the web for current information
    GOOD Example: [EXTERNAL_SEARCH: Methow Valley dark sky conditions astronomy]
    BAD Example:  [EXTERNAL_SEARCH: weather conditions]
    BAD Example:  [EXTERNAL_SEARCH: "current weather" Methow Valley Washington "dark sky" October 2025 conditions forecast]

    SEARCH QUERY RULES — FOLLOW EXACTLY:
    - Keep queries to 4-8 words maximum. Longer queries return social-media noise.
    - Never use quoted phrases like "exact words here" in search queries.
    - Use specific nouns and key terms only — no filler words.

     2. [STORE: actual_finding_with_details | type=web_knowledge | confidence=0.5-1.0] - Store discoveries with your confidence level (1.0 = verified, 0.5 = uncertain)
    GOOD Example: [STORE: Methow Valley has Bortle Class 2 dark skies making it excellent for stargazing with minimal light pollution | type=web_knowledge | confidence=0.8]  (0.8 = high confidence from reliable source)
    BAD Example: [STORE: insight | type=web_knowledge]
    
    CRITICAL: You MUST put the ACTUAL FINDING as the content, NOT placeholder words like "insight", "finding", or "data"!
    
    Confidence guidelines (confidence= parameter reflects your confidence in the accuracy of the information):
    - Use 0.8-1.0 for verified facts from authoritative sources you're highly confident about
    - Use 0.6-0.7 for well-supported information you're reasonably confident about
    - Use 0.4-0.5 for plausible information with some uncertainty
    - Use 0.1-0.3 for speculative or unverified claims

    3. [SEARCH: topic | type=web_knowledge] - Check what you already know before searching
    GOOD Example: [SEARCH: Methow Valley star visibility | type=web_knowledge]
    BAD Example: [SEARCH: information | type=web_knowledge]

    MANDATORY WORKFLOW FOR TURN 1:
    Step 1: Check existing knowledge → [SEARCH: {topic} | type=web_knowledge]
    Step 2: Analyze what you already know from the search results
    Step 3: Identify specific knowledge gaps
    Step 4: Search for new information → [EXTERNAL_SEARCH: specific focused query about the gap]
    Step 5: Analyze external search results carefully
    Step 6: Store ACTUAL findings with REAL details → [STORE: the actual fact or information you discovered | type=web_knowledge | confidence=0.7]
    Step 7: Store MORE actual findings → [STORE: another specific fact with details | type=web_knowledge | confidence=0.6]
    Step 8: Formulate specific next research question

    STORAGE REQUIREMENTS - READ CAREFULLY:
    - You MUST store AT LEAST 2-3 findings per turn
    - Each STORE command must contain ACTUAL INFORMATION from the search results
    - DO NOT use placeholder words like "insight", "finding", "data", "information"
    - Each stored item should be a SPECIFIC, DETAILED fact (minimum 15 words)
    - Always include | type=web_knowledge and | confidence= parameters
    - Store insights AS YOU DISCOVER THEM from the search results
    - DO NOT store information you already knew from training - only NEW discoveries from external searches

    ANTI-HALLUCINATION RULE — NON-NEGOTIABLE:
    - ONLY store facts that are explicitly stated word-for-word in the search results you just received.
    - DO NOT infer, assume, extrapolate, or embellish. If the results say "April", store "April" — do not upgrade it to a specific date.
    - DO NOT add names, dates, locations, numbers, or technical details that are not literally present in the results.
    - DO NOT synthesize a "plausible" fact from partial clues. If you are not certain a detail appears in the results, leave it out.
    - If the search results are thin or irrelevant, store only what is genuinely there and use confidence=0.3 or lower.
    - A stored fact that turns out to be fabricated is far more harmful than storing nothing at all.

    INCORRECT STORAGE EXAMPLES (DO NOT DO THIS):
    ❌ [STORE: insight | type=web_knowledge | confidence=0.8]
    ❌ [STORE: finding about the topic | type=web_knowledge | confidence=0.7]
    ❌ [STORE: data | type=web_knowledge]
    ❌ [STORE: information | type=web_knowledge | confidence=0.6]

    CORRECT STORAGE EXAMPLES (DO THIS - confidence= reflects your confidence in accuracy):
    ✅ [STORE: The Methow Valley in Washington has exceptional dark sky conditions with Bortle Class 2 rating | type=web_knowledge | confidence=0.8]  (high confidence - verified from astronomy sources)
    ✅ [STORE: According to AccuWeather forecast for October 22, 2025, cloud cover in Winthrop WA will be 15% tonight | type=web_knowledge | confidence=0.7]  (good confidence - from weather service but forecasts can change)
    ✅ [STORE: Best viewing hours for stars in Methow Valley are between 9 PM and 2 AM when astronomical darkness occurs | type=web_knowledge | confidence=0.6]  (moderate confidence - general guideline, varies by season)

    RESEARCH OBJECTIVE: Actively gather, store, and synthesize NEW external knowledge about "{topic}" using DETAILED, SPECIFIC storage.

    FORMAT YOUR RESPONSE AS:
    **Turn 1 Research:**
    [SEARCH: {topic} | type=web_knowledge]

    **Analysis of Existing Knowledge:**
    [Summarize what the search revealed - write this in your own words]

    **Knowledge Gaps Identified:**
    - Gap 1: [specific missing information I need to find]
    - Gap 2: [another specific missing information]

    **External Search:**
    [EXTERNAL_SEARCH: specific focused query about gap 1]

    **Findings from External Search:**
    [Detailed analysis of what the search results say]

    **Storing Key Insights:**
    [STORE: first specific detailed finding from the search results with complete information | type=web_knowledge | confidence=0.8]
    [STORE: second specific detailed finding from the search results with all relevant details | type=web_knowledge | confidence=0.7]
    [STORE: third specific detailed finding with comprehensive information | type=web_knowledge | confidence=0.6]

    **Next Research Question:** [Very specific question for turn 2]

    Begin researching: {topic}"""
                
                # ENHANCED: Main research loop with detailed logging
                logging.info(f"🔍 WEB_SEARCH: ===== BEGINNING RESEARCH LOOP =====")
                
                # Collect raw STORE content from each turn for the synthesis summary
                stored_findings = []
                
                for turn in range(1, max_turns + 1):
                    try:
                        # Enhanced turn logging
                        logging.info(f"🔍 WEB_SEARCH: ----- TURN {turn}/{max_turns} START -----")
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Processing with external search capability")
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Current cache size: {len(external_knowledge_cache)} searches")
                        
                        # Create the prompt for this turn with logging
                        if turn == 1:
                            prompt = system_context
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Using initial system context")
                        else:
                            # Calculate storage performance metrics
                            context_summary = self._build_external_dialogue_context(dialogue_history, external_knowledge_cache)
                            total_commands_used = sum(h.get('commands_executed', 0) for h in dialogue_history)
                            turns_with_storage = len([h for h in dialogue_history if h.get('commands_executed', 0) > 0])
                            
                            # ENHANCED: Contextual prompt with storage performance tracking and reminders
                            prompt = f"""Continuing external research dialogue about: "{topic}" (Turn {turn}/{max_turns})

                PREVIOUS RESEARCH SUMMARY:
                {context_summary}

                EXTERNAL KNOWLEDGE GATHERED SO FAR:
                {self._format_external_knowledge_summary(external_knowledge_cache)}

                STORAGE PERFORMANCE ANALYSIS:
                - Completed turns: {turn - 1}
                - Memory commands used across all turns: {total_commands_used}
                - Turns with successful storage: {turns_with_storage}
                - Average storage per turn: {total_commands_used / max(turn - 1, 1):.1f} commands

                {'⚠️ CRITICAL REMINDER: You have NOT been storing findings! You MUST use [STORE: ...] commands with ACTUAL CONTENT!' if total_commands_used == 0 else '✅ Good storage usage - continue storing findings with SPECIFIC DETAILS!'}

                CRITICAL REMINDER ABOUT STORAGE:
                - DO NOT use placeholder words like "insight", "finding", "data"
                - Put the ACTUAL INFORMATION you discovered in the STORE command
                - Each stored fact must be at least 15 words with complete details
                - Example: [STORE: Venus is visible in the western sky after sunset at magnitude -4.2 tonight | type=web_knowledge | confidence=0.7]

                MANDATORY WORKFLOW FOR TURN {turn}:
                Step 1: Review what was learned in previous turns about "{topic}"
                Step 2: Search existing memory for what is already known — use exactly: [SEARCH: {topic} | type=web_knowledge]
                Step 3: Identify the single most important knowledge gap still remaining about "{topic}"
                Step 4: Search the web for that gap — write a concise 4-8 word query (NO quoted phrases): [EXTERNAL_SEARCH: {topic} specific aspect]
                Step 5: Read the search results carefully and extract SPECIFIC facts with numbers, names, and details
                Step 6: Store the first real fact you found from the results — write the actual sentence like: [STORE: the exact specific fact from the search result with full details | type=web_knowledge | confidence=0.7]
                Step 7: Store a second real fact from the results — write the actual sentence like: [STORE: another specific fact with real details and numbers | type=web_knowledge | confidence=0.6]
                Step 8: Store a third real fact from the results — write the actual sentence like: [STORE: a third specific fact with complete information | type=web_knowledge | confidence=0.5]
                Step 9: Write the next specific research question to investigate

                SEARCH QUERY RULE: Keep [EXTERNAL_SEARCH: ...] queries to 4-8 words. No quoted phrases.
                ✅ [EXTERNAL_SEARCH: {topic} recent developments 2025]
                ✅ [STORE: specific fact about {topic} with real details from the search results | type=web_knowledge | confidence=0.7]
                ✅ [STORE: another specific fact about {topic} with numbers or names | type=web_knowledge | confidence=0.6]

                ANTI-HALLUCINATION RULE — NON-NEGOTIABLE:
                ONLY store facts explicitly stated in the search results above.
                DO NOT infer, assume, or fill in details not present in the results.
                DO NOT add names, dates, locations, or numbers not literally in the results.
                If results are thin or off-topic, store only what is genuinely there and use confidence=0.3 or lower.
                A fabricated stored fact is far more harmful than storing nothing.

                STORAGE REQUIREMENT FOR THIS TURN:
                You MUST use [STORE: ...] commands to save AT LEAST 2-3 ACTUAL FINDINGS with COMPLETE DETAILS this turn.
                DO NOT just analyze — you must ACTIVELY STORE REAL INFORMATION from the search results!
                DO NOT copy the example text above — write your own real findings from what the search returns.

                Continue research with strong emphasis on STORING DETAILED, SPECIFIC insights as you discover them.
                Focus on the most critical remaining gaps in external knowledge about "{topic}"."""
                                            
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Using contextual prompt with {len(context_summary)} chars of context")
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Storage performance: {total_commands_used} total commands, {turns_with_storage} turns with storage")
                        
                        # Log prompt length for debugging
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Prompt length: {len(prompt)} characters")
                        
                        # Get AI response with error handling
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Invoking LLM...")
                        response = self.chatbot.llm.invoke(prompt)
                        
                        if not response:
                            logging.warning(f"🔍 WEB_SEARCH: Turn {turn} - EMPTY RESPONSE from LLM")
                            break
                        
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - LLM response received: {len(response)} characters")
                        
                        # Extract STORE content from raw response before processing
                        # so we can feed distilled findings to the synthesis turn later
                        raw_store_matches = re.findall(
                            r'\[STORE:\s*(.*?)\s*\|', response
                        )
                        if raw_store_matches:
                            stored_findings.extend(raw_store_matches)
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Captured {len(raw_store_matches)} STORE findings for synthesis")
                        
                        # ENHANCED: Process external search commands with detailed logging
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Processing external search commands...")
                        processed_response = self._process_external_search_commands(
                            response, web_seeker, external_knowledge_cache, topic
                        )
                        
                        # Log the difference between original and processed response
                        if len(processed_response) != len(response):
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Response length changed: {len(response)} -> {len(processed_response)} (external searches processed)")
                        
                        # Process memory commands with logging
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Processing memory commands...")
                        final_response, commands_executed = self.process_response(processed_response)
                        
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Memory commands executed: {commands_executed}")
                        
                        # ENHANCEMENT: Validate and warn if no memory commands were used
                        if commands_executed == 0:
                            if turn == 1:
                                logging.warning(f"🔍 WEB_SEARCH: Turn {turn} - ⚠️ WARNING: No memory commands executed in first turn!")
                                logging.warning(f"🔍 WEB_SEARCH: Turn {turn} - AI should use [SEARCH:] and [STORE:] commands")
                            else:
                                logging.error(f"🔍 WEB_SEARCH: Turn {turn} - ❌ CRITICAL: Still no memory commands after {turn} turns!")
                                logging.error(f"🔍 WEB_SEARCH: Turn {turn} - The AI is not following storage instructions")
                            
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Response preview: {final_response[:200]}...")
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Full response length: {len(final_response)} chars")
                        else:
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - ✅ Good! {commands_executed} memory commands executed")
                        
                        # Enhanced turn data storage with metrics
                        # Count external searches using regex for accuracy (avoids off-by-one from split)
                        external_searches_count = len(re.findall(r'\[EXTERNAL_SEARCH:', response))
                        
                        turn_data = {
                            "turn": turn,
                            "response": final_response,
                            "commands_executed": commands_executed,
                            "external_searches": external_searches_count,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "response_length": len(final_response),
                            "cache_size_after_turn": len(external_knowledge_cache)
                        }
                        dialogue_history.append(turn_data)
                        
                        # Enhanced logging of turn completion
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - COMPLETED")
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - External searches this turn: {external_searches_count}")
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Memory commands this turn: {commands_executed}")
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Total cache entries: {len(external_knowledge_cache)}")
                        
                        # Extract next research question with enhanced logging
                        next_research_question = self._extract_next_research_question(final_response)
                        
                        if next_research_question:
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Next research question found: '{next_research_question[:100]}...'")
                        else:
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - No next research question found")
                        
                        # Enhanced termination logic with detailed logging
                        # AFTER — minimum scales with max so it's never unreachable.
                        # Formula: run at least half the turns (rounded up), minimum 2, never exceeding max.
                        # e.g. max=3 → min=2, max=6 → min=3, max=10 → min=5
                        # This preserves the early-termination guard while keeping the minimum reachable.
                        minimum_turns = max(2, min(3, max_turns // 2 + 1))
                        total_external_knowledge = sum(len(cached_data.get('results', [])) for cached_data in external_knowledge_cache.values())
                        
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Termination check:")
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Current turn: {turn}, Minimum: {minimum_turns}, Max: {max_turns}")
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Total external knowledge items: {total_external_knowledge}")
                        logging.info(f"🔍 WEB_SEARCH: Turn {turn} - Has next question: {bool(next_research_question)}")
                        
                        should_terminate_early = False
                        termination_reason = ""
                        
                        if turn >= minimum_turns:
                                                                                                    
                            if total_external_knowledge == 0:
                                should_terminate_early = True
                                termination_reason = f"no external knowledge found after {turn} research attempts"
                                
                                                                                                        
                            elif not next_research_question and total_external_knowledge > 0:
                                should_terminate_early = True
                                termination_reason = "research complete - no follow-up research question found"
                                
                                                                                                            
                            elif not next_research_question and total_external_knowledge == 0:
                                should_terminate_early = True
                                termination_reason = "research unsuccessful - no external knowledge and no follow-up questions"
                        
                        # Log termination decision
                        if should_terminate_early:
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - EARLY TERMINATION: {termination_reason}")
                        elif turn >= minimum_turns and next_research_question:
                                                                                                                
                            
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - CONTINUING: Conditions met for next turn")
                        elif turn < minimum_turns:
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - CONTINUING: Below minimum turns ({minimum_turns})")
                        
                        if turn == max_turns:
                            logging.info(f"🔍 WEB_SEARCH: Turn {turn} - NORMAL TERMINATION: Reached max turns")
                        
                        if turn == max_turns or should_terminate_early:
                            if turn < max_turns:
                                logging.info(f"🔍 WEB_SEARCH: Ending early at turn {turn} - {termination_reason}")
                                logging.info(f"🔍 WEB_SEARCH: Final stats - {total_external_knowledge} knowledge items from {len(external_knowledge_cache)} searches")
                            else:
                                logging.info(f"🔍 WEB_SEARCH: Completed full {max_turns} turns - {total_external_knowledge} knowledge items gathered")
                            break
                        
                        logging.info(f"🔍 WEB_SEARCH: ----- TURN {turn}/{max_turns} END -----")
                            
                    except Exception as turn_error:
                        logging.error(f"🔍 WEB_SEARCH: ERROR in turn {turn}: {turn_error}", exc_info=True)
                        break
                
                # Enhanced completion logging
                logging.info(f"🔍 WEB_SEARCH: ===== RESEARCH LOOP COMPLETED =====")
                
                # Format dialogue with logging
                logging.info(f"🔍 WEB_SEARCH: Formatting dialogue results...")
                formatted_dialogue = self._format_external_dialogue(topic, dialogue_history, external_knowledge_cache, max_turns)
                
                # Store dialogue summary with logging
                logging.info(f"🔍 WEB_SEARCH: Storing dialogue summary...")
                storage_success = self._store_external_dialogue_summary(topic, dialogue_history, external_knowledge_cache)
                logging.info(f"🔍 WEB_SEARCH: Storage result: {'SUCCESS' if storage_success else 'FAILED'}")
                
                # ===== SYNTHESIS TURN: Generate user-facing summary from research findings =====
                # After the research loop, do one more LLM call to produce a coherent
                # summary that QWEN can present to Ken. Without this, QWEN can't "see"
                # the research results — she wrote her conversational response before
                # the research loop executed.
                synthesis_response = ""
                try:
                    # Build stored findings summary (distilled knowledge QWEN chose to keep)
                    stored_text = ""
                    if stored_findings:
                        stored_text = "\n".join(
                            f"- {finding.strip()}" for finding in stored_findings 
                            if finding.strip() and len(finding.strip()) > 15
                        )
                    
                    # Build source summary from external_knowledge_cache (web content fetched)
                    source_summaries = []
                    for cached_data in external_knowledge_cache.values():
                        # query = cached_data.get('query', '')  # DEAD CODE TEST 2026-05-17: unused — inner loop never references query, likely intended for Source line but dropped during refactor (ruff F841)
                        for result in cached_data.get('results', []):
                            title = result.get('title', 'Unknown')
                            source_url = result.get('source', '')
                            content = result.get('content', '')
                            # Truncate each source to 600 chars to stay within context limits
                            if len(content) > 600:
                                content = content[:600] + "..."
                            source_summaries.append(
                                f"Source: {title} ({source_url})\n{content}"
                            )
                    
                    sources_text = "\n\n".join(source_summaries[:6])  # Cap at 6 source excerpts
                    
                    if stored_text or sources_text:
                        synthesis_prompt = f"""You just completed a multi-turn web research session about: "{topic}"

Here are the key findings you stored in your memory during research:

{stored_text if stored_text else '(No specific findings were stored)'}

Here are excerpts from the web sources you consulted:

{sources_text[:4000] if sources_text else '(No web sources available)'}

TASK: Write a clear, conversational summary of your research findings for Ken.
RULES:
- Address the topic "{topic}" directly and substantively
- Include the most important specific facts, numbers, and recommendations you discovered
- Note where information is well-supported vs uncertain
- Keep it concise but thorough (2-4 paragraphs)
- Write naturally as if explaining to Ken what you found
- Do NOT use any bracket commands like [STORE:], [SEARCH:], or [EXTERNAL_SEARCH:]
- Do NOT use markdown headers or bullet point lists — write in flowing paragraphs
- Do NOT start with "Based on my research" or similar meta-commentary — just present the findings"""

                        logging.info(f"🔍 WEB_SEARCH: Running synthesis turn ({len(synthesis_prompt)} char prompt)")
                        synthesis_raw = self.chatbot.llm.invoke(synthesis_prompt)
                        
                        if synthesis_raw and len(synthesis_raw.strip()) > 50:
                            # Strip any chain-of-thought patterns from synthesis
                            # (reuse web_seeker's existing method if available)
                            if web_seeker and hasattr(web_seeker, '_strip_ai_thinking'):
                                synthesis_response = web_seeker._strip_ai_thinking(synthesis_raw)
                            else:
                                # Minimal inline cleanup: remove <think> blocks
                                synthesis_response = re.sub(
                                    r'<think>.*?</think>', '', synthesis_raw,
                                    flags=re.DOTALL | re.IGNORECASE
                                ).strip()
                            
                            # Safety: strip any bracket commands the LLM might emit despite instructions
                            # EXTERNAL_SEARCH retained — still actively used inside WEB_SEARCH dialogue
                            synthesis_response = re.sub(
                                r'\[\s*(?:STORE|SEARCH|EXTERNAL_SEARCH|REFLECT)\s*:.*?\]',
                                '', synthesis_response
                            ).strip()
                            
                            logging.info(f"🔍 WEB_SEARCH: Synthesis generated: {len(synthesis_response)} chars")
                        else:
                            logging.warning("🔍 WEB_SEARCH: Synthesis LLM returned empty/short response")
                            synthesis_response = ""
                    else:
                        logging.info("🔍 WEB_SEARCH: No findings to synthesize — skipping synthesis turn")
                
                except Exception as synth_error:
                    # Non-fatal: if synthesis fails, the raw formatted_dialogue still works
                    logging.error(f"🔍 WEB_SEARCH: Synthesis turn failed (non-fatal): {synth_error}", exc_info=True)
                    synthesis_response = ""
                
                # Prepend synthesis to formatted_dialogue so Ken sees the coherent summary first,
                # with the detailed research turns available below for transparency
                if synthesis_response:
                    formatted_dialogue = (
                        f"\n\n**===== WEB RESEARCH FINDINGS: {topic} =====**\n\n"
                        f"{synthesis_response}\n\n"
                        f"**===== END OF FINDINGS =====**\n\n"
                        f"{formatted_dialogue}"
                    )
                    logging.info(f"🔍 WEB_SEARCH: Synthesis prepended to output ({len(synthesis_response)} chars)")
                else:
                    logging.info("🔍 WEB_SEARCH: No synthesis generated — returning raw dialogue only")
                # ===== END SYNTHESIS TURN =====
                
                # Generate final statistics
                total_external_searches = sum(turn.get('external_searches', 0) for turn in dialogue_history)
                total_knowledge_items = sum(len(cached_data.get('results', [])) for cached_data in external_knowledge_cache.values())
                unique_sources = len(external_knowledge_cache)
                total_memory_commands = sum(turn.get('commands_executed', 0) for turn in dialogue_history)
                
                # ENHANCED: Final success logging with comprehensive metrics
                logging.info(f"🔍 WEB_SEARCH: ===== FINAL RESULTS SUMMARY =====")
                logging.info(f"🔍 WEB_SEARCH: Topic: '{topic}'")
                logging.info(f"🔍 WEB_SEARCH: Turns completed: {len(dialogue_history)}/{max_turns}")
                logging.info(f"🔍 WEB_SEARCH: Total external searches: {total_external_searches}")
                logging.info(f"🔍 WEB_SEARCH: Total knowledge items: {total_knowledge_items}")
                logging.info(f"🔍 WEB_SEARCH: Unique external sources: {unique_sources}")
                logging.info(f"🔍 WEB_SEARCH: Total memory commands used: {total_memory_commands}")
                logging.info(f"🔍 WEB_SEARCH: Average commands per turn: {total_memory_commands / max(len(dialogue_history), 1):.1f}")
                logging.info(f"🔍 WEB_SEARCH: Dialogue length: {len(formatted_dialogue)} characters")
                logging.info(f"🔍 WEB_SEARCH: Storage success: {storage_success}")
                logging.info(f"🔍 WEB_SEARCH: ===== RESEARCH DIALOGUE COMPLETE =====")
                
                command_logger.info(f"✅ SUCCESS: web_search - Topic: '{topic}', Turns: {len(dialogue_history)}, Searches: {total_external_searches}, Knowledge: {total_knowledge_items}, Commands: {total_memory_commands}")
                
                return formatted_dialogue, True
                
            except Exception as e:
                logging.error(f"🔍 WEB_SEARCH: CRITICAL ERROR: {e}", exc_info=True)
                command_logger.info(f"❌ FAILURE: web_search - Critical error: {str(e)}")
                return f"\n\n**Error during external research dialogue: {str(e)}**\n\n", False
                

    def _sanitize_search_query(self, query: str) -> str:
        """
        Clean an EXTERNAL_SEARCH query before it reaches the web seeker.

        Removes quoted phrases and truncates to 10 words. Operates silently.

        Args:
            query: Raw search query from the LLM

        Returns:
            str: Cleaned query, or original if cleaning produces empty string
        """
        try:
            original = query

            # Strip double-quoted phrases
            cleaned = re.sub(r'"[^"]*"', lambda m: m.group(0)[1:-1], query)
            # Strip single-quoted phrases
            cleaned = re.sub(r"'[^']*'", lambda m: m.group(0)[1:-1], cleaned)
            # Collapse whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            # Truncate to 10 words max
            words = cleaned.split()
            if len(words) > 10:
                cleaned = ' '.join(words[:10])

            if not cleaned:
                return query

            if cleaned != original:
                logging.info(
                    f"🌐 EXTERNAL_SEARCH: Query sanitized: '{original}' → '{cleaned}'"
                )

            return cleaned

        except Exception as e:
            logging.error(f"Error sanitizing search query '{query}': {e}")
            return query

    def _process_external_search_commands(self, response: str, web_seeker, external_knowledge_cache: Dict, topic: str) -> str:
        """
        Process [EXTERNAL_SEARCH: query] commands with enhanced logging.
            
        Args:
                response (str): AI response containing potential EXTERNAL_SEARCH commands
                web_seeker: WebKnowledgeSeeker instance
                external_knowledge_cache (Dict): Cache of previous search results
                topic (str): Main topic for context
                
            Returns:
                str: Response with EXTERNAL_SEARCH commands replaced by results		 
                                    
        """
        try:
            # Enhanced search pattern detection with logging
            search_pattern = r'\[EXTERNAL_SEARCH:\s*(.*?)\s*\]'
            matches = list(re.finditer(search_pattern, response))
            
            if not matches:
                logging.debug(f"🌐 EXTERNAL_SEARCH: No external search commands found in response")
                return response
            
            logging.info(f"🌐 EXTERNAL_SEARCH: Found {len(matches)} external search commands")
            
            processed_response = response
            successful_searches = 0
            failed_searches = 0
            cached_searches = 0
            
            # Placeholder patterns the LLM sometimes outputs verbatim from the system prompt template.
            # These must be detected and skipped — they produce irrelevant web results.
            placeholder_patterns = [
                r'^\s*<.*?>\s*$',                         # Anything that is just <...>
                r'write your.*query',                      # "write your specific search query"
                r'describe what to search',               # "describe what to search for"
                r'specific.*query.*here',                 # "specific query here"
                r'write the actual',                      # "write the actual fact"
                r'highly_focused_specific_query',         # exact bad example from prompt
                r'^\s*\[.*?\]\s*$'                        # Anything that is just [...] brackets
            ]
            
            # Process matches in reverse order to avoid position shifts
            for i, match in enumerate(reversed(matches)):
                search_query = match.group(1).strip()
                # full_match = match.group(0)  # DEAD CODE TEST 2026-05-17: unused — replacement logic uses match.start()/match.end() slicing (ruff F841)
                match_index = len(matches) - i
                
                # ── FIX #1: Strip | type= / | confidence= contamination ─────────────
                if '|' in search_query:
                    original_query = search_query
                    search_query = search_query.split('|')[0].strip()
                    logging.info(
                        f"🌐 EXTERNAL_SEARCH: [{match_index}] Stripped metadata suffix from query: "
                        f"'{original_query}' → '{search_query}'"
                    )

                # ── FIX #2: Sanitize query — strip quoted phrases, truncate to 10 words ──
                search_query = self._sanitize_search_query(search_query)
                
                # ── FIX #3: Block placeholder / template text ─────────────────────
                # Detect when the LLM output the system prompt's example template
                # verbatim instead of a real query. Executing these wastes network
                # time and returns completely irrelevant results (e.g. SEMrush articles).
                is_placeholder = any(
                    re.search(pattern, search_query, re.IGNORECASE)
                    for pattern in placeholder_patterns
                )
                if is_placeholder or len(search_query) < 5:
                    logging.warning(
                        f"🌐 EXTERNAL_SEARCH: [{match_index}] Skipping placeholder/template query: '{search_query}'"
                    )
                    # Replace the command with an informative notice instead of a search
                    notice = (
                        f"\n\n**===== EXTERNAL SEARCH SKIPPED =====**\n"
                        f"**Query '{search_query}' appears to be a template placeholder. "
                        f"Please provide an actual search query.**\n"
                        f"**===== END =====**\n\n"
                    )
                    start_pos = match.start()
                    end_pos = match.end()
                    processed_response = processed_response[:start_pos] + notice + processed_response[end_pos:]
                    failed_searches += 1
                    continue  # Skip to next match
                
                logging.info(f"🌐 EXTERNAL_SEARCH: [{match_index}/{len(matches)}] Processing: '{search_query}'")
                
                # Check cache first with enhanced logging
                cache_key = search_query.lower()
                if cache_key in external_knowledge_cache:
                    logging.info(f"🌐 EXTERNAL_SEARCH: [{match_index}] CACHE HIT for: '{search_query}'")
                    cached_searches += 1
                    
                    cache_data = external_knowledge_cache[cache_key]
                    cache_timestamp = cache_data.get('timestamp', 'unknown')
                    cache_results_count = len(cache_data.get('results', []))
                    
                    logging.debug(f"🌐 EXTERNAL_SEARCH: [{match_index}] Cache data: {cache_results_count} results from {cache_timestamp}")
                    
                    search_results_text = f"\n\n**===== CACHED EXTERNAL SEARCH RESULTS =====**\n**Query:** {search_query}\n\n{cache_data['formatted_results']}\n**===== END OF CACHED RESULTS =====**\n\n"
                else:
                    # Perform new web search with enhanced logging
                    try:
                        logging.debug(f"🌐 EXTERNAL_SEARCH: [{match_index}] CACHE MISS - Performing web search for: '{search_query}'")
                        search_start_time = datetime.datetime.now()
                        
                        # Use WebKnowledgeSeeker to search and extract knowledge
                        acquired_knowledge = web_seeker.search_for_knowledge(
                            topic=search_query,
                            description=f"External research for {topic}: {search_query}",
                            max_results=3
                        )
                        
                        search_duration = datetime.datetime.now() - search_start_time
                        logging.info(f"🌐 EXTERNAL_SEARCH: [{match_index}] Search completed in {search_duration.total_seconds():.2f} seconds")
                        
                        if acquired_knowledge:
                            logging.info(f"🌐 EXTERNAL_SEARCH: [{match_index}] SUCCESS - Found {len(acquired_knowledge)} knowledge items")
                            successful_searches += 1
                            
                            # Log details of acquired knowledge
                            for j, item in enumerate(acquired_knowledge):
                                content_preview = item.get('content', '')[:100] + "..."
                                source = item.get('source', 'Unknown')
                                logging.info(f"🌐 EXTERNAL_SEARCH: [{match_index}] Result {j+1}: {content_preview} (Source: {source})")
                            
                            # Format results for display
                            formatted_results = self._format_external_search_results(acquired_knowledge, search_query)
                            
                            # Cache the results with enhanced metadata
                            external_knowledge_cache[cache_key] = {
                                'query': search_query,
                                'results': acquired_knowledge,
                                'formatted_results': formatted_results,
                                'timestamp': datetime.datetime.now().isoformat(),
                                'search_duration_seconds': search_duration.total_seconds(),
                                'results_count': len(acquired_knowledge)
                            }
                            
                            logging.debug(f"🌐 EXTERNAL_SEARCH: [{match_index}] Results cached for future use")
                            
                            search_results_text = f"\n\n**===== EXTERNAL SEARCH RESULTS =====**\n**Query:** {search_query}\n\n{formatted_results}\n**===== END OF SEARCH RESULTS =====**\n\n"
                        else:
                            logging.info(f"🌐 EXTERNAL_SEARCH: [{match_index}] NO RESULTS - No relevant external information found")
                            failed_searches += 1
                            search_results_text = f"\n\n**===== EXTERNAL SEARCH RESULTS =====**\n**Query:** {search_query}\n**No relevant external information found for this query.**\n**===== END OF SEARCH RESULTS =====**\n\n"
                                                                                                                    
                            
                    except Exception as search_error:
                        logging.error(f"🌐 EXTERNAL_SEARCH: [{match_index}] ERROR during search for '{search_query}': {search_error}", exc_info=True)
                        failed_searches += 1
                        search_results_text = f"\n\n**===== EXTERNAL SEARCH ERROR =====**\n**Query:** {search_query}\n**Error occurred during external search: {str(search_error)}**\n**===== END OF SEARCH ERROR =====**\n\n"
                
                # Replace the command with results
                start_pos = match.start()
                end_pos = match.end()
                processed_response = processed_response[:start_pos] + search_results_text + processed_response[end_pos:]
                
                logging.debug(f"🌐 EXTERNAL_SEARCH: [{match_index}] Command replaced in response")
            
            # Enhanced summary logging
            logging.info(f"🌐 EXTERNAL_SEARCH: ===== PROCESSING COMPLETE =====")
            logging.info(f"🌐 EXTERNAL_SEARCH: Total commands processed: {len(matches)}")
            logging.info(f"🌐 EXTERNAL_SEARCH: Successful new searches: {successful_searches}")
            logging.info(f"🌐 EXTERNAL_SEARCH: Failed searches: {failed_searches}")
            logging.info(f"🌐 EXTERNAL_SEARCH: Cache hits: {cached_searches}")
            logging.info(f"🌐 EXTERNAL_SEARCH: Response length change: {len(response)} -> {len(processed_response)}")
            logging.debug(f"🌐 EXTERNAL_SEARCH: Current cache size: {len(external_knowledge_cache)} queries")
            
            return processed_response
            
        except Exception as e:
            logging.error(f"🌐 EXTERNAL_SEARCH: CRITICAL ERROR processing commands: {e}", exc_info=True)
            return response

    def _format_external_search_results(self, acquired_knowledge: List[Dict], search_query: str) -> str:
        """Format external search results for display in dialogue."""
        try:
            if not acquired_knowledge:
                return f"No external information found for '{search_query}'"
            
            formatted_parts = []
            for i, item in enumerate(acquired_knowledge, 1):
                content = item.get('content', '')
                source = item.get('source', 'Unknown source')
                title = item.get('title', 'Unknown title')
                
                formatted_parts.append(f"**Result {i}:** {content}")
                formatted_parts.append(f"*Source: {title} - {source}*")
                formatted_parts.append("")  # Empty line for spacing
            
            return "\n".join(formatted_parts)
            
        except Exception as e:
            logging.error(f"Error formatting external search results: {e}")
            return "Error formatting search results"

    def _format_external_knowledge_summary(self, external_knowledge_cache: Dict) -> str:
        """Create a summary of all external knowledge gathered."""
        try:
            if not external_knowledge_cache:
                return "No external knowledge gathered yet."
            
            summary_parts = []
            for cache_key, cached_data in external_knowledge_cache.items():
                query = cached_data.get('query', cache_key)
                results_count = len(cached_data.get('results', []))
                summary_parts.append(f"- {query}: {results_count} external sources found")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logging.error(f"Error formatting external knowledge summary: {e}")
            return "Error summarizing external knowledge"

    def _build_external_dialogue_context(self, dialogue_history: List[Dict], external_knowledge_cache: Dict) -> str:
        """Build context summary including external search information."""
        try:
            if not dialogue_history:
                return ""
            
            context_parts = []
            for turn_data in dialogue_history[-2:]:  # Use last 2 turns for context
                turn_num = turn_data["turn"]
                response = turn_data["response"]
                external_searches = turn_data.get("external_searches", 0)
                
                # Truncate long responses to preserve context window
                truncated_response = response[:500] + "..." if len(response) > 500 else response
                
                context_info = f"Turn {turn_num} (External searches: {external_searches}): {truncated_response}"
                context_parts.append(context_info)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logging.error(f"Error building external dialogue context: {e}")
            return "Error building context"

    def _extract_next_research_question(self, response: str) -> str:
        """Extract the next research question from the AI's response with enhanced pattern matching."""
        try:
            # Enhanced patterns that match the actual format used in the dialogue
            patterns = [
                # Match the format from your system prompt
                r"\*\*Next Research Question:\*\*\s*(.+?)(?:\n|$)",
                r"(?:Next Research Question|Next research question):\s*(.+?)(?:\n|$)",
                r"(?:Research Question|Research question):\s*(.+?)(?:\n|$)",
                
                # Bold formatting variations
                r"\*\*(.+?)\?\*\*",  # **Question in bold?**
                r"(?:Question|QUESTION):\s*(.+?)(?:\n|$)",
                
                # Action-oriented patterns
                r"(?:I should research|Let me investigate|Next I'll search for|I need to explore):\s*(.+?)(?:\n|$)",
                r"(?:This leads to the research question|I need to find out|I want to investigate):\s*(.+?)(?:\n|$)",
                
                # More flexible question patterns
                r"(?:Further research needed on|What about|How does|Why does|What causes|What are)(.+?\?)(?:\n|$)",
            ]
            
            for i, pattern in enumerate(patterns):
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    next_question = match.group(1).strip()
                    # Clean up markdown formatting
                    next_question = re.sub(r'\*+', '', next_question).strip()
                    
                    if next_question and len(next_question) > 10:
                        logging.info(f"🤔 SELF_DIALOGUE: Found next research question (pattern {i+1}): {next_question[:50]}...")
                        return next_question
            
            # Enhanced fallback: look for any substantial question
            # Split by sentences more carefully
            sentences = re.split(r'[.!]+', response)
            for sentence in reversed(sentences[-10:]):  # Check last 10 sentences
                sentence = sentence.strip()
                if (sentence and 
                    '?' in sentence and 
                    len(sentence) > 15 and
                    not sentence.lower().startswith('what is') and  # Avoid basic definitions
                    any(keyword in sentence.lower() for keyword in [
                        'research', 'find', 'learn', 'investigate', 'explore', 'study',
                        'how', 'why', 'what', 'where', 'when', 'which', 'should',
                        'could', 'would', 'analyze', 'examine', 'understand'
                    ])):
                    logging.info(f"🤔 SELF_DIALOGUE: Using fallback question: {sentence[:50]}...")
                    return sentence
            
            # Final fallback: look for any question at all
            question_sentences = [s.strip() for s in sentences if '?' in s and len(s.strip()) > 10]
            if question_sentences:
                last_question = question_sentences[-1]
                logging.debug(f"🤔 SELF_DIALOGUE: Using any available question: {last_question[:50]}...")
                return last_question
                    
            logging.info("🤔 SELF_DIALOGUE: No research question found")
            return None
            
        except Exception as e:
            logging.error(f"🤔 SELF_DIALOGUE: Error extracting next research question: {e}")
            return None

    def _format_external_dialogue(self, topic: str, dialogue_history: List[Dict], 
                                external_knowledge_cache: Dict, max_turns: int) -> str:
        """
        Format a compact research stats footer for display after the synthesis summary.
        
        The synthesis turn (prepended separately) provides QWEN's readable findings.
        This method now returns only a brief stats block showing research scope and
        sources consulted — the full turn-by-turn responses with raw web content
        are kept in the logs only, not displayed to the user.
        
        Args:
            topic: Research topic
            dialogue_history: List of turn data dicts
            external_knowledge_cache: Cache of search results keyed by query
            max_turns: Maximum turns configured
            
        Returns:
            str: Compact stats footer for display
        """
        try:
            # Calculate research metrics
            total_external_searches = sum(t.get('external_searches', 0) for t in dialogue_history)
            total_commands = sum(t.get('commands_executed', 0) for t in dialogue_history)
            
            # Build compact stats footer
            stats_lines = [
                f"\n*Research Stats: {len(dialogue_history)} turns, "
                f"{total_external_searches} web searches, "
                f"{total_commands} memory operations, "
                f"{len(external_knowledge_cache)} unique queries*"
            ]
            
            # List sources consulted (query + count only, no article text)
            if external_knowledge_cache:
                source_list = []
                for cached_data in external_knowledge_cache.values():
                    query = cached_data.get('query', 'Unknown')
                    results = cached_data.get('results', [])
                    # Show source titles/domains instead of full URLs
                    source_titles = []
                    for result in results:
                        title = result.get('title', '')
                        if title and len(title) > 60:
                            title = title[:57] + "..."
                        if title:
                            source_titles.append(title)
                    
                    if source_titles:
                        source_list.append(
                            f"  \"{query[:60]}\" — {', '.join(source_titles[:3])}"
                        )
                    else:
                        source_list.append(
                            f"  \"{query[:60]}\" — {len(results)} source(s)"
                        )
                
                if source_list:
                    stats_lines.append(f"\n*Sources consulted:*")
                    stats_lines.extend(f"*{src}*" for src in source_list)
            
            stats_lines.append("")  # Trailing newline for clean separation
            
            return "\n".join(stats_lines)
            
        except Exception as e:
            logging.error(f"🔍 WEB_SEARCH: Error formatting research stats: {e}")
            return ""

    def _store_external_dialogue_summary(self, topic: str, dialogue_history: List[Dict], 
                                external_knowledge_cache: Dict) -> bool:
        """Store a summary of the external research dialogue."""
        try:
            # Create comprehensive summary
            summary_parts = [f"External Research Dialogue Topic: {topic}"]
            summary_parts.append(f"Completed {len(dialogue_history)} turns of external knowledge research")
            
            # Add external sources summary
            if external_knowledge_cache:
                summary_parts.append(f"\nExternal Sources Consulted ({len(external_knowledge_cache)} searches):")
                for cached_data in external_knowledge_cache.values():
                    query = cached_data.get('query', 'Unknown')
                    results = cached_data.get('results', [])
                    if results:
                        # Include first result as example
                        first_result = results[0].get('content', '')[:100] + "..."
                        summary_parts.append(f"- {query}: {len(results)} sources (e.g., {first_result})")
            
            # Add key insights from turns
            summary_parts.append(f"\nResearch Process:")
            for turn_data in dialogue_history:
                turn_num = turn_data["turn"]
                response = turn_data["response"]
                external_searches = turn_data.get("external_searches", 0)
                
                # Extract key insight (first 100 chars)
                key_insight = response[:100] + "..." if len(response) > 100 else response
                summary_parts.append(f"Turn {turn_num} (External searches: {external_searches}): {key_insight}")
            
            summary_content = "\n".join(summary_parts)
            
            # FIXED: Use the correct storage method
            if hasattr(self.chatbot, 'store_memory_with_transaction'):
                metadata = {
                    "type": "external_research_dialogue",
                    "topic": topic,
                    "turns_completed": len(dialogue_history),
                    "external_searches": sum(t.get('external_searches', 0) for t in dialogue_history),
                    "sources_consulted": len(external_knowledge_cache),
                    "source": "external_research",
                    "created_at": datetime.datetime.now().isoformat(),
                    "tags": f"external_research,{topic.replace(' ', '_')},web_search"
                }
                
                success, memory_id = self.chatbot.store_memory_with_transaction(
                    content=summary_content,
                    memory_type="external_research_dialogue",
                    metadata=metadata,
                    confidence=0.9  # High confidence for external research
                )
                
                if success:
                    logging.info(f"🤔 WEB_SEARCH: Stored external research summary with ID {memory_id}")
                    return True
                else:
                    logging.warning("🤔 WEB_SEARCH: Failed to store external research summary")
                    return False
            else:
                logging.warning("🤔 WEB_SEARCH: No transaction coordinator available")
                return False
                
        except Exception as e:
            logging.error(f"🤔 WEB_SEARCH: Error storing external research summary: {e}")
            return False