"""Main chatbot logic with enhanced search, conversation summary storage."""
import datetime
import sqlite3
import logging
import os
import re
import time
import json
import uuid
# import sys  # DEAD CODE TEST 2026-05-17: unused at module level, re-imported inside functions at L644 and L679 (ruff F401/F811)
from utils import calculate_tokens
from conversation_summary_manager import ConversationSummaryManager
from conversation_state import ConversationStateManager
from typing import Dict, List, Optional
from langchain_ollama import OllamaLLM
from document_reader import DocumentReader
from reflection_engine import ReflectionEngine
from vector_db import VectorDB
from memory_db import MemoryDB
from deepseek import DeepSeekEnhancer
from datetime import datetime as dt 
# DEAD CODE TEST 2026-05-17: removed QDRANT_LOCAL_PATH, QDRANT_USE_LOCAL, QDRANT_URL — all unused per ruff F401 + vulture. To revert, restore the original 3-line import below.
from config import (OLLAMA_MODEL, MODEL_PARAMS, DOCS_PATH, DB_PATH,
                   QDRANT_COLLECTION_NAME)




class Chatbot:
    """Main chatbot class with conversation summary management."""
    
    def __init__(self):
        """Initialize the chatbot and its components, including reminder manager."""
        logging.info("Initializing Chatbot")
        try:
            # Initialize databases and readers
            self.vector_db = VectorDB()
            self.memory_db = MemoryDB()

            # Initialize flags to track LLM generation and conversation state
            # These flags prevent autonomous background tasks from interfering with active responses
            self._llm_generating = False  # True when LLM is actively generating a response
            self.conversation_in_progress = False  # True when processing user input
            logging.info("Initialized LLM generation and conversation state flags")
            
            # Set docs_path earlier so it can be used in DocumentReader initialization
            self.docs_path = DOCS_PATH
            
            # Initialize DocumentReader with chatbot reference and docs_path
            from document_reader import DocumentReader
            self.doc_reader = DocumentReader(docs_path=self.docs_path, chatbot=self)
                       
            # Initialize system prompt from file
            self._initialize_system_prompt()
            
            # Conversation history is managed via st.session_state.messages (Streamlit session store)
            
            # Initialize Reflection module with self reference
            self.reflection_engine = ReflectionEngine(memory_db=self.memory_db, chatbot=self)
            
            # Initialize DeepSeekEnhancer and enhance the system prompt
            from deepseek import DeepSeekEnhancer
            self.deepseek_enhancer = DeepSeekEnhancer(self)
            
            # Get enhanced system prompt
            self.current_system_prompt = self.deepseek_enhancer.enhance_system_prompt()
            
            # Initialize LLM only once with the enhanced system prompt
            self.llm = self._initialize_llm()

            # Initialize prompt tracking
            self._last_prompt_sent = None
            self._last_prompt_tokens = 0
            
            # Token tracking — three-field design (2026-05-14):
            # 
            # _last_sent_prompt_tokens: Current context window pressure — size of
            #   the most recently sent prompt. Updated by ASSIGNMENT in 
            #   accumulate_prompt_tokens(). Reset to 0 by reset_token_counter_after_summary().
            #   Drives the UI color tiers and the get_token_usage_warning() advisories
            #   that tell QWEN when to run [SUMMARIZE_CONVERSATION].
            #
            # _session_total_tokens_sent: True cumulative session counter. Updated 
            #   by INCREMENT (+=) in accumulate_prompt_tokens(). NOT reset on 
            #   summarization — survives the entire session. Drives the UI "Session 
            #   Total" display.
            #
            # _last_counted_prompt_text: Dedup key for two-pass retrieval mode — 
            #   prevents double-count when pass-1 and pass-2 both trigger accumulation.
            #
            # _prompt_was_sent: Gate flag — only update counters when the LLM was 
            #   actually invoked (prevents counting build-then-fail prompts).
            self._last_sent_prompt_tokens = 0
            self._session_total_tokens_sent = 0     # NEW: true cumulative session counter
            # Cumulative tokens fetched via search result blocks this session.
            # Used by main.py UI to display real search overhead instead of the
            # old count * 2000 estimate. Incremented in deepseek.py's 
            # _handle_command_display when a search returns formatted results.
            # NOT reset by reset_token_counter_after_summary — parallels
            # _session_total_tokens_sent semantics (session-work counter that
            # survives summarization for honest cumulative reporting).
            self._search_result_tokens_total = 0
            self._last_counted_prompt_text = None
            self._prompt_was_sent = False

            # Auto-summarization trigger flag — prevents repeated triggers at threshold
            # Bug #6 fix: was lazily created inside get_token_usage_warning()
            self._auto_summary_triggered = False

            # Last stored memory ID — used for tracking and rollback references  
            # Bug #5 fix: was only set on successful store, causing AttributeError if accessed before first store
            self._last_memory_id = None

            logging.info("Initialized prompt tracking variables with cumulative tracking")
                
            # Initialize conversation state manager
            self.conversation_manager = ConversationStateManager(DB_PATH)
            # Initialize session and auto-retrieve latest summary
            self.conversation_manager.initialize_session(auto_retrieve_summary=True)
            
            # Initialize the conversation summary manager
            self.conversation_summary_manager = ConversationSummaryManager(self)
            logging.info(f"✅ conversation_summary_manager initialized: {self.conversation_summary_manager is not None}")
            
            # Initialize reminder manager
            from reminders import ReminderManager
            self.reminder_manager = ReminderManager(DB_PATH)
            logging.info("Initialized ReminderManager")

            # Initialize cognitive state tracking
            self.current_cognitive_state = 'Neutral'  # Default state
            self.cognitive_state_history = []
            logging.info("Initialized cognitive state tracking with default state: Neutral")
            
            # ===== MEMORY COMMAND COUNTERS AND SESSION MANAGEMENT =====
            # Initialize session management for command tracking
            try:
                import uuid
                from session_manager import SessionManager
                
                # Generate a unique session ID for this chatbot instance
                self.session_id = str(uuid.uuid4())
                logging.info(f"Generated session ID for chatbot: {self.session_id}")
                
                # Create session manager with lifetime counters reference
                # Note: deepseek_enhancer already has lifetime_counters initialized
                self.session_manager = SessionManager(self.deepseek_enhancer.lifetime_counters)
                
                # Start the session and sync the session ID
                session_id = self.session_manager.start_new_session()
                self.session_id = session_id
                self.deepseek_enhancer.session_id = session_id
                
                # Verify database setup
                db_path = self.deepseek_enhancer.lifetime_counters.get_database_path()
                logging.info(f"LifetimeCounters database initialized at: {db_path}")
                
                # Test database functionality
                test_counters = self.deepseek_enhancer.lifetime_counters.get_counters()
                logging.info(f"LifetimeCounters test read successful. Sample counters: {dict(list(test_counters.items())[:5])}")
                
            except ImportError as import_err:
                logging.error(f"Failed to import session management modules: {import_err}")
                # Fallback: create a simple session ID without full session management
                import uuid
                self.session_id = str(uuid.uuid4())
                self.session_manager = None
                logging.warning("Session management disabled due to import error")
                
            except Exception as session_err:
                logging.error(f"Error initializing session management: {session_err}")
                # Fallback: create a simple session ID
                import uuid
                self.session_id = str(uuid.uuid4())
                self.session_manager = None
                logging.warning("Session management partially disabled due to initialization error")
            # ===== END MEMORY COMMAND COUNTERS SECTION =====
            
            # Initialize image processor for multimodal capabilities
            try:
                from image_processor import ImageProcessor
                self.image_processor = ImageProcessor()
                logging.info("Image Processor initialized for multimodal capabilities")
            except ImportError:
                logging.warning("Image Processor module not available. Install requirements for multimodal features.")
            except Exception as e:
                logging.error(f"Error initializing Image Processor: {e}")
            
            # ===== FINAL VERIFICATION AND LOGGING =====
            # Log successful initialization with session info
            if hasattr(self, 'session_manager') and self.session_manager:
                session_summary = self.session_manager.get_session_summary()
                logging.info(f"Chatbot initialization completed with session tracking: {session_summary}")
            else:
                logging.info(f"Chatbot initialization completed with basic session ID: {self.session_id}")

            # Initialize OODA Deep Research Loop engine
            try:
                from ooda_loop import OODALoop
                self.ooda_loop = OODALoop(self)
                logging.info("✅ OODALoop initialized for Autonomous Agent mode")
            except Exception as e:
                self.ooda_loop = None
                logging.error(f"❌ OODALoop failed to initialize: {e}", exc_info=True)
            
           
            # Verify all critical components are initialized
            critical_components = [
                'vector_db', 'memory_db', 'doc_reader', 'reflection_engine', 
                'deepseek_enhancer', 'llm', 'conversation_manager', 
                'conversation_summary_manager', 'reminder_manager'  
            ]

            missing_components = [comp for comp in critical_components if not hasattr(self, comp)]
            if missing_components:
                logging.error(f"Critical components missing after initialization: {missing_components}")
                raise Exception(f"Failed to initialize critical components: {missing_components}")

            logging.info("✅ Chatbot initialization completed successfully with enhanced system prompt and command tracking")
            
        except Exception as e:
            logging.error(f"Chatbot initialization error: {e}", exc_info=True)
            raise

    def get_session_memory_stats(self) -> dict:
        """
        Get current session memory command counts for display in enhanced system prompt.
        
        This provides the AI model with self-awareness of its memory usage patterns
        by tracking all command executions during the current session.
        
        Uses Streamlit session state as source of truth (same data shown in UI sidebar).
        
        Command Categories:
        - Core Memory: SEARCH, STORE, FORGET, REFLECT
        - Auxiliary: SUMMARIZE, REMINDER, DISCUSS_WITH_CLAUDE
        - Meta/Utility: WEB_SEARCH, RESEARCH_DIALOGUE, HELP
        - Self-Awareness: COGNITIVE_STATE
        
        Returns:
            dict: Complete command counts with keys matching enhanced_prompt.txt placeholders
        """
        
        # Initialize default return dictionary with all required keys
        # This ensures .format() calls never get KeyError
        default_stats = {
            # Core Memory
            'search': 0,
            'store': 0,
            'forget': 0,
            'reflect': 0,
            # Auxiliary
            'summarize': 0,
            'reminder': 0,
            'discuss': 0,
            # Meta/Utility
            'web_search': 0,
            'research_dialogue': 0,
            'help': 0,
            # Self-Awareness
            'cognitive_state': 0,
            # Total
            'total_count': 0
        }
        
        try:
            # Try to import streamlit and access session state
            # This is the same source of truth used by the UI sidebar
            import streamlit as st
            
            if not hasattr(st, 'session_state'):
                logging.warning("Streamlit session_state not available")
                return default_stats
                
            if 'memory_command_counts' not in st.session_state:
                logging.warning("memory_command_counts not in session_state")
                return default_stats
            
            # Get the session counters dictionary from Streamlit session state
            # This is the SAME data displayed in the UI sidebar
            session_counts = st.session_state.memory_command_counts
            
            # ===================================================================
            # CORE MEMORY OPERATIONS
            # ===================================================================
            # SEARCH: Get the 'search' counter (already combined in session state)
            search_count = session_counts.get('search', 0)
            
            # STORE: Memory storage operations
            store_count = session_counts.get('store', 0)
            
            # FORGET: Memory deletion operations
            forget_count = session_counts.get('forget', 0)
            
            # REFLECT: Combines regular reflect and concept reflection
            reflect_count = (
                session_counts.get('reflect', 0) + 
                session_counts.get('reflect_concept', 0)
            )
            
            # ===================================================================
            # AUXILIARY OPERATIONS
            # ===================================================================
            # SUMMARIZE: Conversation summarization
            # Note: Session state uses 'summarize_conversation' key
            summarize_count = session_counts.get('summarize_conversation', 0)
            
            # REMINDER: Reminder management operations
            reminder_count = (
                session_counts.get('reminder', 0) +
                session_counts.get('reminder_complete', 0)
            )
            
            # DISCUSS_WITH_CLAUDE: AI-to-AI communication
            discuss_count = session_counts.get('discuss_with_claude', 0)
            
            # ===================================================================
            # META/UTILITY OPERATIONS
            # ===================================================================
            # WEB_SEARCH: External web searches
            web_search_count = session_counts.get('web_search', 0)
            
            # RESEARCH_DIALOGUE: Multi-turn autonomous research
            # Note: Session state might use 'self_dialogue' key
            research_dialogue_count = (
                session_counts.get('research_dialogue', 0) +
                session_counts.get('self_dialogue', 0)
            )
            
            # HELP: Command help requests
            help_count = session_counts.get('help', 0)
            
            # ===================================================================
            # SELF-AWARENESS OPERATIONS
            # ===================================================================
            # COGNITIVE_STATE: Tracking of processing/emotional states
            cognitive_state_count = session_counts.get('cognitive_state', 0)
            
            # ===================================================================
            # CALCULATE TOTAL
            # ===================================================================
            total_count = (
                search_count + store_count + forget_count + reflect_count +
                summarize_count + reminder_count + discuss_count +
                web_search_count + research_dialogue_count + help_count +
                cognitive_state_count
            )
            
            # Log the stats for debugging
            logging.info(
                f"Session memory stats (from st.session_state) - "
                f"SEARCH: {search_count}, STORE: {store_count}, FORGET: {forget_count}, "
                f"REFLECT: {reflect_count}, SUMMARIZE: {summarize_count}, "
                f"REMINDER: {reminder_count}, DISCUSS: {discuss_count}, "
                f"WEB_SEARCH: {web_search_count}, RESEARCH_DIALOGUE: {research_dialogue_count}, "
                f"HELP: {help_count}, COGNITIVE_STATE: {cognitive_state_count}, "
                f"TOTAL: {total_count}"
            )
            
            # Return comprehensive stats dictionary
            return {
                # Core Memory
                'search': search_count,
                'store': store_count,
                'forget': forget_count,
                'reflect': reflect_count,
                # Auxiliary
                'summarize': summarize_count,
                'reminder': reminder_count,
                'discuss': discuss_count,
                # Meta/Utility
                'web_search': web_search_count,
                'research_dialogue': research_dialogue_count,
                'help': help_count,
                # Self-Awareness
                'cognitive_state': cognitive_state_count,
                # Total
                'total_count': total_count
            }
            
        except ImportError:
            # Streamlit not available
            logging.error("Could not import Streamlit for session stats")
            return default_stats
        except Exception as e:
            # Error handling: Log and return default zeros
            logging.error(f"Error getting session memory stats: {e}", exc_info=True)
            return default_stats
        
    def accumulate_prompt_tokens(self):
        """
        Update token counters after a prompt is sent to the LLM.
        Called ONCE immediately after sending a prompt.

        BEHAVIOR (updated 2026-05-14):
        Maintains two counters with different semantics:
        
        1. _last_sent_prompt_tokens — assigned (=) to current prompt size.
           Represents current context window pressure.
           
        2. _session_total_tokens_sent — incremented (+=) by current prompt size.
           Represents true cumulative tokens sent across the session.
           NOT reset by summarization — survives the entire session.

        The _prompt_was_sent gate prevents counting prompts that were built but
        never reached the model (e.g., LLM call failed). The _last_counted_prompt_text 
        dedup prevents double-counting in two-pass retrieval mode where both
        passes share the same accumulation call site.
        """
        try:
            # Initialize tracking variables if needed (defensive — survives hot reload)
            if not hasattr(self, '_last_sent_prompt_tokens'):
                self._last_sent_prompt_tokens = 0
            if not hasattr(self, '_session_total_tokens_sent'):
                # Defensive init for the cumulative counter — covers hot reloads
                # and any legacy chatbot instances that predate the 2026-05-14 redesign
                self._session_total_tokens_sent = 0
            if not hasattr(self, '_last_counted_prompt_text'):
                self._last_counted_prompt_text = None
            
            # Resolve the current prompt's token count — prefer the cached value
            # set during prompt build, fall back to recomputing from prompt text.
            if hasattr(self, '_last_prompt_tokens') and self._last_prompt_tokens > 0:
                current_prompt_tokens = self._last_prompt_tokens
            elif hasattr(self, '_last_prompt_sent') and self._last_prompt_sent:
                current_prompt_tokens = calculate_tokens(self._last_prompt_sent)
                self._last_prompt_tokens = current_prompt_tokens
            else:
                logging.warning("ACCUMULATE: No prompt tokens to update")
                return
            
            # Only update if the LLM was actually invoked with this prompt
            if hasattr(self, '_prompt_was_sent') and self._prompt_was_sent:
                current_prompt_text = getattr(self, '_last_prompt_sent', '')
                
                # Dedup: only update if this is a NEW prompt (prevents double-count
                # when both pass-1 and pass-2 of two-pass retrieval call this method)
                if self._last_counted_prompt_text != current_prompt_text:
                    previous_pressure = self._last_sent_prompt_tokens
                    previous_session_total = self._session_total_tokens_sent
                    
                    # Pressure counter: ASSIGN — always reflects most recent prompt size
                    self._last_sent_prompt_tokens = current_prompt_tokens
                    
                    # Session counter: INCREMENT — true cumulative running total
                    self._session_total_tokens_sent += current_prompt_tokens
                    
                    # Mark this prompt as counted to prevent re-accumulation
                    self._last_counted_prompt_text = current_prompt_text
                    
                    logging.debug(f"✅ TOKEN COUNTERS UPDATED:")
                    logging.debug(f"   Pressure: {previous_pressure:,} → {self._last_sent_prompt_tokens:,} tokens")
                    logging.debug(f"   Session total: {previous_session_total:,} → {self._session_total_tokens_sent:,} tokens (+{current_prompt_tokens:,})")
                    
                    # Reset the gate flag — next LLM invoke will set it again
                    self._prompt_was_sent = False
                else:
                    logging.debug(f"⏭️ SKIPPED: Prompt already counted (dedup)")
            else:
                logging.debug(f"⏸️ NO UPDATE: _prompt_was_sent = {getattr(self, '_prompt_was_sent', 'not set')}")
                
        except Exception as e:
            # Defensive: never let a token-tracking failure break the chat flow
            logging.error(f"ACCUMULATE: Error - {e}", exc_info=True)
    
    def update_llm_system_prompt(self, new_system_prompt):
        """
        Update the LLM's system prompt without reinitializing the whole model.
        
        OllamaLLM is a lightweight LangChain wrapper — reinitializing it does NOT
        reload the model from disk or affect GPU memory. The Ollama server keeps
        the model resident independently. So both paths here are fast.
        
        Args:
            new_system_prompt (str): The new system prompt text to use
            
        Returns:
            bool: True if update succeeded, False otherwise
        """
        try:
            # Guard: LLM must be initialized
            if self.llm is None:
                logging.error("Cannot update system prompt: LLM not initialized")
                return False
            
            # Guard: validate the new prompt
            if not new_system_prompt or not isinstance(new_system_prompt, str):
                logging.error("Cannot update system prompt: new_system_prompt is empty or not a string")
                return False
            
            logging.info(f"Updating LLM system prompt ({len(new_system_prompt)} chars)")
            
            # Step 1: Always update the chatbot's own reference first.
            # This ensures _initialize_llm() picks up the new prompt if the
            # fallback reinit path is taken below.
            self.current_system_prompt = new_system_prompt
            
            # Step 2: Try direct attribute update on the OllamaLLM wrapper.
            # OllamaLLM stores `system` as a top-level field (NOT inside extra_body).
            # extra_body only holds generation options (temperature, num_ctx, etc.).
            if hasattr(self.llm, 'system'):
                try:
                    self.llm.system = new_system_prompt
                    logging.info("✅ Updated LLM system prompt via direct attribute assignment")
                    return True
                except (AttributeError, TypeError, Exception) as attr_err:
                    # Pydantic may prevent direct mutation on some LangChain builds
                    logging.warning(
                        f"Direct attribute set failed ({attr_err}), "
                        "falling back to LLM reinitialization"
                    )
            else:
                logging.warning(
                    "OllamaLLM has no .system attribute — "
                    "falling back to LLM reinitialization"
                )
            
            # Step 3: Fallback — reinitialize the wrapper object.
            # self.current_system_prompt was already updated in Step 1, so
            # _initialize_llm() will build the new LLM with the correct prompt.
            # NOTE: This recreates the Python wrapper only — the Ollama model
            # stays resident in GPU memory, so this is not expensive.
            logging.info("Reinitializing LLM wrapper with updated system prompt")
            self.llm = self._initialize_llm()
            
            if self.llm is not None:
                logging.info("✅ LLM wrapper reinitialized with new system prompt")
                return True
            else:
                logging.error("❌ LLM reinitialization returned None")
                return False
                
        except Exception as e:
            logging.error(f"Error updating LLM system prompt: {e}", exc_info=True)
            return False

    def _initialize_system_prompt(self):
        """Initialize and manage the system prompt file."""
        try:
            self.system_prompt_file = "system_prompt.txt"
            if not os.path.exists(self.system_prompt_file):
                logging.error("System prompt file not found: system_prompt.txt")
                self.current_system_prompt = "Missing System Prompt"
                return

            # Read the content from the file
            with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
                self.current_system_prompt = f.read()
            logging.info("System prompt loaded successfully")
        except Exception as e:
            logging.error(f"Error initializing system prompt: {e}")
            self.current_system_prompt = "Missing System Prompt"  

    def _get_enhanced_prompt_template(self):
        """Get the enhanced prompt template from file."""
        try:
            enhanced_prompt_file = "enhanced_prompt.txt"
            if not os.path.exists(enhanced_prompt_file):
                logging.error(f"Enhanced prompt file not found: {enhanced_prompt_file}")
                # Use a simple fallback prompt instead of creating a new file
                return "User Query: {user_input}\n\nPlease respond to Ken with the search results or let him know if there are no results."
            
            with open(enhanced_prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                logging.info(f"Successfully loaded enhanced prompt file, length: {len(content)}")
                return content
        except Exception as e:
            logging.error(f"Error reading enhanced prompt file: {e}")
            return "Error loading prompt. User Query: {user_input}"
    
    def _initialize_llm(self):
        """Initialize the LLM and let Ollama handle conversation context."""
        try:
            logging.info("🚀 LLM INITIALIZATION STARTING")
            
            # Pull num_ctx from config — no hardcoded fallback. If MODEL_PARAMS
            # is missing num_ctx the resulting KeyError surfaces a real config bug
            # rather than silently using a different window size than the read methods.
            num_ctx = MODEL_PARAMS["num_ctx"]
            
            # Simple initialization - let Ollama manage conversation
            llm = OllamaLLM(
                model=OLLAMA_MODEL,
                system=self.current_system_prompt,  # System prompt only
                extra_body={
                    "options": {
                        "temperature": MODEL_PARAMS.get("temperature", 0.7),
                        "num_ctx": num_ctx,
                        "num_gpu": 99,
                        "top_k": MODEL_PARAMS.get("top_k", 40),
                        "top_p": MODEL_PARAMS.get("top_p", 0.9),
                        "num_predict": 4096,  # Limits output to 4096 tokens max
                    }
                }
            )
            
            logging.info(f"✅ LLM INITIALIZED - Context: {num_ctx} tokens, Max output: 4096 tokens")
            return llm
            
        except KeyError as ke:
            # Specific catch for the new no-fallback config access
            logging.error(f"❌ LLM INITIALIZATION FAILED: MODEL_PARAMS missing required key: {ke}")
            raise
        except Exception as e:
            logging.error(f"❌ LLM INITIALIZATION FAILED: {e}")
            raise
    
    def log_reminder_operation(self, operation_type, reminder_id, content=None, status="completed", error=None):
        """
        Log reminder operations for tracking and debugging.
        
        Args:
            operation_type (str): Type of operation ("create", "complete", "delete")
            reminder_id: The ID of the reminder
            content (str, optional): Content of the reminder
            status (str): Status of the operation ("starting", "completed", "failed")
            error (str, optional): Error message if operation failed
        """
        try:
            # Format the log prefix based on operation type
            prefix = f"[REMINDER {operation_type.upper()}]"
            
            # Create the log message
            if content:
                content_preview = content[:50] + "..." if len(content) > 50 else content
                message = f"{prefix} ID={reminder_id}, Content='{content_preview}', Status={status}"
            else:
                message = f"{prefix} ID={reminder_id}, Status={status}"
                
            # Add error details if present
            if error:
                message += f", Error: {error}"
                
            # Log the message
            logging.info(message)
            
        except Exception as e:
            # Fallback logging to ensure errors in logging don't cause additional problems
            logging.error(f"Error in log_reminder_operation: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Use unified token estimation from utils."""
        return calculate_tokens(text)
        
    def update_session_counter(self, command_type: str):
        """
        Update session counter for a specific command type.
        
        Args:
            command_type (str): The type of command ('store', 'retrieve', etc.)
        """
        import sys
        if 'streamlit' in sys.modules:
            try:
                import streamlit as st_local
                if hasattr(st_local, 'session_state') and 'memory_command_counts' in st_local.session_state:
                    # Initialize counter if it doesn't exist
                    if command_type not in st_local.session_state.memory_command_counts:
                        st_local.session_state.memory_command_counts[command_type] = 0
                    
                    # Increment the counter
                    st_local.session_state.memory_command_counts[command_type] += 1
                    logging.info(f"Updated {command_type} counter: {st_local.session_state.memory_command_counts[command_type]}")
                    return True
                else:
                    logging.debug(f"Could not update {command_type} counter - session_state not available")
                    return False
            except (ImportError, ModuleNotFoundError, Exception) as e:
                logging.debug(f"Could not update {command_type} counter: {e}")
                return False
        else:
            logging.debug("Streamlit not available for counter update")
            return False

    def initialize_session_counters(self):
        """
        Initialize all session counters to 0.
        
        Session counters track command usage within the current Streamlit session only.
        They reset on app restart. Lifetime counters in LifetimeCounters.db persist across sessions.
        
        The canonical list below should match:
        - The valid_command_types list in lifetime_counters.py
        - The session counter increments in deepseek.py's process_response centralized loop
        Note: 'total' is computed by the UI as a sum, not stored as its own session key.
        """
        import sys
        
        # Only run when Streamlit is loaded — counters live in st.session_state
        if 'streamlit' in sys.modules:
            try:
                import streamlit as st_local
                
                if hasattr(st_local, 'session_state'):
                    # Create the counter dict on the session if missing
                    if 'memory_command_counts' not in st_local.session_state:
                        st_local.session_state.memory_command_counts = {}
                    
                    # Canonical list of memory commands tracked in session counters.
                    # Removed (dead): 'retrieve' (replaced by 'search'), 'correct' (no longer used)
                    # Excluded by design: 'image_analysis' (internal to image import, not user-facing)
                    counter_types = [
                        'store',
                        'search',
                        'forget',
                        'reflect',
                        'summarize',
                        'reminder',
                        'reminder_complete',
                        'discuss_with_claude',
                        'self_dialogue',
                        'web_search',
                        'cognitive_state',
                        'show_system_prompt',
                        'modify_system_prompt',
                        'help'
                    ]
                    
                    # Seed each counter at 0 if not already present.
                    # Existing nonzero values in this session are preserved.
                    for counter_type in counter_types:
                        if counter_type not in st_local.session_state.memory_command_counts:
                            st_local.session_state.memory_command_counts[counter_type] = 0
                    
                    logging.info(f"Initialized session counters ({len(counter_types)} types)")
                    return True
                else:
                    # Streamlit imported but no session_state — likely a non-UI context
                    logging.debug("initialize_session_counters: session_state not available")
                    return False
                    
            except Exception as e:
                # Don't fail — counters are observability, not core functionality
                logging.error(f"Error initializing session counters: {e}")
                return False
        
        # Streamlit not loaded — running in CLI/headless context
        logging.debug("initialize_session_counters: Streamlit not loaded, skipping")
        return False
      
    @staticmethod
    def is_command_only_response(response_text: str) -> bool:
        """
        Detect if a response contains only memory commands without conversational text.
        
        Args:
            response_text: The model's response to check
            
        Returns:
            True if response needs guidance to include conversation, False otherwise
        """
        import re
        
        if not response_text:
            return False
        
        # Remove all memory command patterns
        stripped = response_text
        command_patterns = [
            r'\[\s*SEARCH\s*:[^\]]*\]',
            r'\[\s*STORE\s*:[^\]]*\]', 
            r'\[\s*FORGET\s*:[^\]]*\]',
            r'\[\s*REFLECT\s*\]',
            r'\[\s*REMINDER\s*:[^\]]*\]',
            r'\[\s*COGNITIVE_STATE\s*:[^\]]*\]',
            r'\[\s*WEB_SEARCH\s*:[^\]]*\]',
            r'\[\s*SELF_DIALOGUE\s*:[^\]]*\]',
            r'\[\s*DISCUSS_WITH_CLAUDE\s*:[^\]]*\]',
            r'\[\s*HELP\s*\]',
            r'\[\s*COMPLETE_REMINDER\s*:[^\]]*\]',
        ]
        
        for pattern in command_patterns:
            stripped = re.sub(pattern, '', stripped, flags=re.IGNORECASE)
        
        # Remove whitespace and common filler
        stripped = stripped.strip()
        stripped = re.sub(r'^(let me|i\'ll|checking|searching)[\s\.]*$', '', stripped, flags=re.IGNORECASE)
        stripped = stripped.strip()
        
        # If less than 20 chars of non-command text, it's command-only
        # This threshold catches things like "..." or "Let me check" but allows real responses
        return len(stripped) < 20

                
    def build_conversation_context(self, current_user_input: str = "",
                                    max_token_pct: float = 0.7) -> str:
        """
        Build formatted conversation context from Streamlit session state messages.

        Extracted from process_command() so both the normal chat flow and the
        OODA research loop share identical conversation context visibility.
        Without this shared method, OODA runs the LLM in complete isolation
        from the conversation it lives inside — causing hallucinated task context
        for referential phrases like "these findings" or "this document".

        Args:
            current_user_input (str): The current user message to exclude from
                history. Prevents duplication since it is added separately via
                {user_input} in the prompt template. Pass "" to include all messages.
            max_token_pct (float): Fraction of the model context window to allow
                for conversation history. Default 0.7 (70%) matches process_command
                behavior. OODA callers may pass a smaller value (e.g. 0.4) to leave
                headroom for accumulated research context in the same prompt.

        Returns:
            str: Formatted conversation context string with ChatML role tags,
                ready for injection into any LLM prompt. Returns "" on failure
                so callers never need to guard against None.
        """
        try:
            import streamlit as st

            # --- Guard: Streamlit session must be available ---
            if not hasattr(st, 'session_state') or 'messages' not in st.session_state:
                logging.warning("CONVO_CONTEXT: Streamlit session messages not available")
                return ""

            messages = st.session_state.messages

            # --- Duplicate prevention ---
            # Exclude the current user message when provided, because it will be
            # injected separately via {user_input} in the prompt template.
            # Failure to exclude it causes the message to appear twice in the prompt.
            if (current_user_input and messages and
                    messages[-1].get('role') == 'user' and
                    messages[-1].get('content', '').strip() ==
                    current_user_input.strip()):
                messages_to_format = messages[:-1]
                logging.debug(
                    "CONVO_CONTEXT: Excluded current user message to prevent duplication"
                )
            else:
                messages_to_format = messages

            # --- Token-budgeted context assembly ---
            formatted = []
            total_tokens = 0
            # Pull num_ctx from config directly — no hardcoded fallback so
            # any config bug surfaces immediately rather than producing a
            # silently-wrong budget.
            max_context_tokens = int(
                MODEL_PARAMS["num_ctx"] * max_token_pct
            )

            # Work backwards from most recent so newest messages are always included.
            # If history exceeds the budget, older messages are dropped first.
            for i in range(len(messages_to_format) - 1, -1, -1):
                msg = messages_to_format[i]
                role = msg.get('role', '').lower()
                content = msg.get('content', '')

                if not content:
                    continue

                # ChatML tags ensure Qwen3 correctly identifies speaker roles
                formatted_msg = f"<|im_start|>{role}\n{content.strip()}<|im_end|>"
                msg_tokens = self._estimate_tokens(formatted_msg)

                # Stop adding messages once we hit the token ceiling
                if total_tokens + msg_tokens > max_context_tokens:
                    logging.info(
                        f"CONVO_CONTEXT: Token budget reached at message index {i} "
                        f"({total_tokens:,}/{max_context_tokens:,} tokens) — "
                        f"older messages excluded"
                    )
                    break

                # Prepend so final order is oldest → newest
                formatted.insert(0, formatted_msg)
                total_tokens += msg_tokens

            convo_context = "\n".join(formatted)
            logging.info(
                f"CONVO_CONTEXT: Built context with {len(formatted)} messages, "
                f"~{total_tokens:,} tokens "
                f"(budget: {max_context_tokens:,}, pct: {max_token_pct:.0%})"
            )
            return convo_context

        except Exception as e:
            logging.error(
                f"CONVO_CONTEXT: Error building conversation context: {e}",
                exc_info=True
            )
            return ""
        
    def _invoke_llm_with_timeout(self, prompt: str, timeout_seconds: int, label: str = "LLM") -> str:
        """
        Invoke self.llm.invoke() with a hard wall-clock timeout.

        Wraps the underlying LangChain invoke in a ThreadPoolExecutor so the
        application returns control to the user even if Ollama hangs. Designed
        as a defense against streaming stalls, VRAM pressure, GPU thermal
        throttling, or any other condition where Ollama receives the request
        but takes far longer than expected to respond.

        IMPORTANT — Thread cancellation behavior on Windows:
            Python's Future.cancel() does NOT kill a thread that has already
            started executing. When the timeout fires, this method raises
            TimeoutError back to the caller on schedule, but the worker thread
            running self.llm.invoke() will continue in the background until
            Ollama either returns or the underlying socket closes. This is an
            accepted tradeoff — a 10-minute user-visible response with one
            orphaned background thread is dramatically better than a silent
            23-minute hang. ThreadPoolExecutor worker threads are daemon
            threads in Python 3.9+, so an orphaned worker will NOT prevent
            process shutdown — it gets killed when the process exits.

        IMPORTANT — Executor lifecycle (do NOT use `with` syntax):
            This method does NOT use the `with ThreadPoolExecutor(...) as
            executor:` context-manager form because the context manager's
            __exit__ calls shutdown(wait=True), which blocks the main thread
            until all worker threads finish. On the timeout path that defeats
            the entire purpose of the timeout — observed empirically on
            2026-05-14 where a 600s timeout fired correctly but the
            user-facing message was delayed by another 674s while __exit__
            waited for the orphaned Ollama call to eventually return. We
            construct the executor explicitly and call shutdown(wait=False)
            on the timeout path so the orphaned worker is truly fire-and-
            forget and the user gets relief on schedule. The success path
            uses shutdown(wait=True) since the worker has already finished
            and wait=True returns immediately in that case.

        Args:
            prompt (str): The fully-formatted prompt string to send to the LLM.
            timeout_seconds (int): Hard timeout in seconds. If the underlying
                llm.invoke() has not returned by this point, TimeoutError is
                raised to the caller.
            label (str): Short descriptor used in log messages, e.g.
                "FIRST_PASS", "SECOND_PASS", "EMPTY_RESPONSE_GUARD". Helps
                disambiguate which call hung when reading logs.

        Returns:
            str: The LLM's response text. May be an empty string if the model
                produced no content — the caller is responsible for handling
                that case (the existing EMPTY_RESPONSE_GUARD logic does so).

        Raises:
            TimeoutError: The LLM call did not return within timeout_seconds.
                Callers should catch this and return a user-friendly message
                rather than re-raising.
            Exception: Any other exception from self.llm.invoke() is
                propagated unchanged for the caller's existing exception
                handler to deal with.
        """
        # Inline import — concurrent.futures is only needed inside this method,
        # so there's no need to add it as a module-level import. Keeps the
        # change to chatbot.py minimal (no new top-level imports).
        import concurrent.futures

        invoke_start_time = time.time()

        # Construct the executor EXPLICITLY rather than using `with` syntax.
        # See class docstring "Executor lifecycle" section for full rationale.
        # Short version: __exit__ would call shutdown(wait=True) on every
        # exit path, blocking on the orphaned worker thread when the timeout
        # fires. By managing shutdown ourselves we can pass wait=False at the
        # right moment and let the user get relief on schedule.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"llm_invoke_{label.lower()}"
        )

        try:
            # Submit the invoke to the worker thread. The worker begins
            # executing immediately; the main thread will block on the future
            # below until either result is ready or the timeout fires.
            future = executor.submit(self.llm.invoke, prompt)

            try:
                # Block until either the invoke returns or the timeout fires.
                # future.result() raises concurrent.futures.TimeoutError on
                # timeout, and re-raises any exception the worker raised
                # internally (which the outer except Exception handles).
                result = future.result(timeout=timeout_seconds)

                invoke_duration = time.time() - invoke_start_time
                # Log at DEBUG so this doesn't add noise to normal operation.
                # The existing CRITICAL "LLM INVOKE COMPLETED" log in
                # process_command still fires on the success path and gives
                # the operator-visible timing info.
                logging.debug(
                    f"INVOKE_TIMEOUT[{label}]: completed normally in "
                    f"{invoke_duration:.2f}s (limit was {timeout_seconds}s)"
                )

                # Success path — worker has already finished (we just got its
                # result), so shutdown(wait=True) returns immediately. This
                # releases executor resources cleanly without leaving it
                # lingering for the atexit handler.
                executor.shutdown(wait=True)
                return result

            except concurrent.futures.TimeoutError:
                # Timeout fired — log critically with full diagnostic context
                # so future hangs leave a clear breadcrumb trail.
                # future.cancel() will return False here because the task is
                # already running; we call it anyway as a best-effort intent
                # signal (does not actually stop the worker thread).
                cancelled = future.cancel()
                invoke_duration = time.time() - invoke_start_time

                logging.critical(
                    f"⏰ INVOKE_TIMEOUT[{label}]: LLM did not return within "
                    f"{timeout_seconds}s (waited {invoke_duration:.2f}s). "
                    f"future.cancel() returned {cancelled} — on Windows this "
                    f"does NOT kill the worker thread; the orphaned worker "
                    f"may continue executing until Ollama returns. Likely "
                    f"causes: streaming stall, VRAM pressure, GPU thermal "
                    f"throttling, or Ollama internal stall. Check Ollama "
                    f"server.log for the same time window."
                )

                # CRITICAL — wait=False is the whole point of this rewrite.
                # The worker thread is still running an llm.invoke() call
                # that has hung. If we waited for it (wait=True), we'd block
                # the main thread until Ollama eventually returns, which can
                # take many more minutes (observed 674s on 2026-05-14). Using
                # wait=False marks the executor as shutting down and returns
                # control immediately. The orphaned worker keeps running in
                # the background (daemon thread, harmless) and exits cleanly
                # whenever Ollama returns.
                executor.shutdown(wait=False)

                # Re-raise as the standard library TimeoutError rather than
                # concurrent.futures.TimeoutError. Standard TimeoutError is
                # the idiomatic exception for callers to catch and is what
                # the integration sites in process_command look for.
                raise TimeoutError(
                    f"LLM invocation '{label}' exceeded {timeout_seconds}s timeout"
                )

        except TimeoutError:
            # Re-raise the TimeoutError unchanged so the caller's specific
            # timeout handler can catch it. The executor has already been
            # shut down above on the timeout path; no additional cleanup
            # needed here.
            raise

        except Exception as e:
            # Any other exception (network error, malformed response, parse
            # error inside langchain, etc.) — log and re-raise so the caller's
            # existing generic exception handler in process_command catches
            # it. This preserves backward compatibility: callers that
            # previously called self.llm.invoke() directly will see the same
            # exceptions they always did, just with extra logging.
            invoke_duration = time.time() - invoke_start_time
            logging.error(
                f"INVOKE_TIMEOUT[{label}]: LLM invocation failed after "
                f"{invoke_duration:.2f}s with {type(e).__name__}: {e}",
                exc_info=True
            )

            # Defensive shutdown: the worker may have finished with an
            # exception (in which case shutdown is instant) or may be in
            # a partially-completed state (in which case we don't want to
            # block waiting for it). wait=False is safe for both cases.
            # Wrap in try/except to avoid masking the original exception
            # if shutdown itself fails for any reason.
            try:
                executor.shutdown(wait=False)
            except Exception:
                # Already shut down or other shutdown error — don't mask
                # the original exception by raising a secondary one.
                pass

            raise
    
    def process_command(self, user_input: str, indicators: Dict) -> str:
        """
        Process user command with enhanced prompt template, memory commands, and auto-summarization.
        
        This method handles:
        1. Token tracking and tiered usage warnings (gentle 75%, critical 95%, overflow 100%+)
        2. Enhanced prompt formatting with context
        3. Two-pass retrieval system (initial + search results)
        4. Memory command processing (FIXED: Now processes BOTH passes)
        5. Response cleanup
        6. Recursion trap prevention for meta-cognitive questions
        7. Cognitive state tracking (rate-limited to 1 update per turn)
        8. Command-only response detection and guidance injection
        9. Safety validation to detect unprocessed commands
        
        Args:
            user_input (str): The user's input text
            indicators (Dict): UI indicators for status display
            
        Returns:
            str: The model's response text
        """
        start_time = time.time()
        logging.info(f"Starting process_command at {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        try:
            # =====================================================
            # STEP 0: Reset per-turn flags for new user turn
            # =====================================================
            
            # --- Reset recursion detector ---
            # This allows ONE meta-cognitive storage per response cycle
            # while preventing infinite loops from repeated identical stores
            #
            # IMPORTANT: This does NOT limit the total number of stores per turn.
            # It only prevents the SAME content from being stored repeatedly.
            # 
            # Examples of what's ALLOWED:
            # - Multiple different [STORE:] commands in one response
            # - Storing search results from multiple searches
            # - Storing various insights about different topics
            #
            # Examples of what's BLOCKED:
            # - Storing "insight about loops" 4+ times in one response
            # - Any identical content repeated 4+ times
            
            if hasattr(self, 'deepseek_enhancer') and hasattr(self.deepseek_enhancer, '_recursion_detector'):
                # Only reset if we're not in cooldown (cooldown must persist across turns)
                if not self.deepseek_enhancer._recursion_detector.get('cooldown_until'):
                    self.deepseek_enhancer._recursion_detector['duplicate_count'] = 0
                    self.deepseek_enhancer._recursion_detector['last_store_content'] = None
                    self.deepseek_enhancer._recursion_detector['last_command_type'] = None
                    logging.debug("RECURSION_DETECTOR: Reset for new user turn (allows 1 meta-storage)")
                else:
                    # Still in cooldown from previous recursion trap
                    cooldown_remaining = (
                        self.deepseek_enhancer._recursion_detector['cooldown_until'] - 
                        datetime.datetime.now()
                    ).seconds
                    logging.warning(
                        f"RECURSION_DETECTOR: Still in cooldown ({cooldown_remaining}s remaining) "
                        "- meta-cognitive storage blocked"
                    )
            
            # --- Reset cognitive state rate limiter ---
            # Allows the model to update its cognitive state ONCE per conversation turn
            # Prevents spam like changing from Frustrated → Happy → Frustrated → Curious in one response
            if hasattr(self, 'deepseek_enhancer') and hasattr(self.deepseek_enhancer, '_state_updated_this_turn'):
                self.deepseek_enhancer._state_updated_this_turn = False
                logging.debug("COGNITIVE_STATE: Reset rate limiter for new turn (allows 1 state update)")
                
            # =====================================================
            # STEP 1: Get token count for monitoring
            # =====================================================
            # Note: Auto-summarization is handled by main.py after response generation
            # Get unified token count for monitoring and dashboard
            current_tokens, max_tokens, percentage = self.get_unified_token_count()
            logging.info(f"ENHANCED: Token usage: {current_tokens:,}/{max_tokens:,} ({percentage:.1f}%)")
            
            # =====================================================
            # STEP 2: Build conversation context and check for meta-questions
            # =====================================================
            
            # Get enhanced prompt template (loads from enhanced_prompt.txt)
            enhanced_prompt_template = self._get_enhanced_prompt_template()
            
            # Build conversation context using the shared method.
            # OODA loop calls the same method — both see identical history.
            convo_context = self.build_conversation_context(user_input)
            
            # Get current time and date for temporal awareness
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_date = now.strftime("%A, %B %d, %Y")
            
            # Check for meta-questions that could trigger recursion
            meta_safety_instruction = ""
            if self._detect_meta_question(user_input):
                logging.warning("META_QUESTION: Detected potentially recursive question - adding safety instructions")
                
                # Allow limited meta-cognitive storage with safeguards
                meta_safety_instruction = """
                ⚠️ META-COGNITIVE QUESTION DETECTED: This question is about your own processing.
                BALANCED APPROACH TO SELF-REFLECTION:
                1. You MAY use ONE [STORE:] command to capture a genuine meta-cognitive insight
                2. You MAY use ONE [REFLECT:] command if it provides developmental value
                3. DO NOT store insights about "storing insights" (that creates infinite loops)
                4. DO NOT reflect on "the act of reflecting" (that triggers recursion)
                5. Focus your storage on WHAT you learned, not HOW you're learning it right now
                SAFE META-STORAGE EXAMPLE:
                ✅ [STORE: When asked about loop behavior, I recognized that self-referential questions require careful boundary setting | type=insight | confidence=0.8]
                UNSAFE META-STORAGE EXAMPLE (WILL LOOP):
                ❌ [STORE: I'm storing an insight about recognizing recursive patterns while recognizing recursive patterns...]
                Remember: Store the CONCLUSION of your meta-analysis, not the PROCESS of analyzing.
                """
            
            # =====================================================
            # STEP 3: Format the enhanced prompt using template system
            # =====================================================
            try:
                # Get token usage warning if needed (shows at 75%+; tiers: gentle 75%, critical 95%, overflow 100%+)
                token_warning = self.get_token_usage_warning()
                
                # Get session memory command stats for display
                session_stats = self.get_session_memory_stats()
                
                # For models internal memory command counter. Format the enhanced prompt with memory command stats
                enhanced_prompt = enhanced_prompt_template.format(
                    memory_context="",  # Empty for first pass, filled in second pass
                    # PASS INDICATOR: Explicit signal to QWEN about which response phase this is.
                    # Pass 1 allows commands freely. Pass 2 (second_pass_prompt below) bans them.
                    # This gives QWEN an unambiguous architectural signal rather than relying on
                    # it to infer from context whether commands are appropriate.
                    pass_indicator=(
                        "## RESPONSE MODE: FIRST PASS\n"
                        "Search your memory and store any relevant information. "
                        "You MAY use [SEARCH:], [STORE:], and other memory commands in this response.\n"
                        "IMPORTANT: Always write your conversational reply AFTER your commands, "
                        "never before them and never instead of them. "
                        "Do NOT narrate commands in prose (e.g. do not write 'I stored: [STORE:...]'). "
                        # FIX: Prevents hallucinated command results — QWEN was writing prose describing
                        # what a command 'found' or 'produced' BEFORE the command executed, anticipating
                        # results that the system hadn't injected yet. Commands must run first; only then
                        # can results be referenced in prose.
                        "Do NOT describe or anticipate the results of a command before it executes — "
                        "results are injected by the system after the command runs. "
                        # FIX: Prevents duplicate command entries — QWEN was issuing the same command
                        # in both the executable position and again embedded in prose, causing the
                        # safety strip to fire and leaving stray command fragments in the response.
                        "Do NOT issue the same command more than once per response."
                    ),
                    convo_context=convo_context,
                    user_input=user_input,
                    token_usage=f"{current_tokens}/{max_tokens} ({percentage:.1f}%)",
                    token_warning=token_warning,
                    current_time=current_time,
                    current_date=current_date,
                    # Core Memory Command Counts
                    search_count=session_stats['search'],
                    store_count=session_stats['store'],
                    forget_count=session_stats['forget'],
                    reflect_count=session_stats['reflect'],
                    # Auxiliary Command Counts
                    summarize_count=session_stats['summarize'],
                    reminder_count=session_stats['reminder'],
                    discuss_count=session_stats['discuss'],
                    # Meta/Utility Command Counts
                    web_search_count=session_stats['web_search'],
                    research_dialogue_count=session_stats['research_dialogue'],
                    help_count=session_stats['help'],
                    cognitive_state_count=session_stats['cognitive_state'],
                    # Total Operations
                    total_count=session_stats['total_count']
                )
                # Prepend safety instructions if this is a meta-question
                if meta_safety_instruction:
                    enhanced_prompt = meta_safety_instruction + "\n\n" + enhanced_prompt
                logging.info(f"ENHANCED: Formatted prompt length: {len(enhanced_prompt)}")
                            
            except Exception as format_error:
                logging.error(f"Error formatting enhanced prompt: {format_error}")
                # Fallback to simple format if template formatting fails
                enhanced_prompt = f"<|im_start|>system\n{self.current_system_prompt}<|im_end|>\n{convo_context}\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant"
            
            # =====================================================
            # STEP 4: Get initial response from LLM
            # =====================================================
            logging.info("ENHANCED: Getting initial LLM response with conversation context")
            
            # CRITICAL FIX: Store the prompt AND calculate tokens BEFORE the LLM call
            # This ensures our token tracking is accurate
            self._last_prompt_sent = enhanced_prompt
            self._last_prompt_tokens = calculate_tokens(enhanced_prompt)
            logging.info(f"TOKEN_TRACKING: Stored prompt - {len(enhanced_prompt):,} chars, {self._last_prompt_tokens:,} tokens")
            
            # Initialize command_only_detected flag before try block
            command_only_detected = False
            
            try:
                # === ENHANCED DIAGNOSTIC LOGGING START ===
                logging.debug(f"🔥 LLM INVOKE STARTING - Timestamp: {datetime.datetime.now().isoformat()}")
                
                # Log prompt characteristics
                prompt_length = len(enhanced_prompt)
                prompt_tokens = calculate_tokens(enhanced_prompt)
                logging.debug(f"📝 Prompt length: {prompt_length:,} characters")
                logging.debug(f"🔢 Estimated prompt tokens: {prompt_tokens:,}")
                
                # Log context window state
                current_tokens, max_tokens, percentage = self.get_unified_token_count()
                logging.debug(f"📊 CONTEXT STATE BEFORE LLM:")
                logging.critical(f"   Current tokens: {current_tokens:,}/{max_tokens:,} ({percentage:.1f}%)")
                # Count actual conversation turns from the live Streamlit message store
                import streamlit as st  # inline import — st is not a module-level import in chatbot.py
                turn_count = len([m for m in st.session_state.messages if m.get('role') == 'user']) if hasattr(st, 'session_state') and 'messages' in st.session_state else 0
                logging.critical(f"   Conversation turns: {turn_count}")
                # Log prompt beginning and end for pattern detection
                logging.critical(f"📄 Prompt START (first 200 chars):")
                logging.critical(f"   {enhanced_prompt[:200]}")
                logging.critical(f"📄 Prompt END (last 200 chars):")
                logging.critical(f"   {enhanced_prompt[-200:]}")
                
                # === PROMPT-SIDE PATTERN COUNT (INPUT to LLM) ===
                # Counts literal command syntax occurrences in the prompt being SENT to QWEN.
                # This includes documentation, command-guide examples, and replayed
                # conversation history — NOT commands QWEN executes. High counts here
                # indicate prompt bloat (lots of past commands in history or repeated
                # documentation), not runaway command behavior.
                # For QWEN's actual command emissions, see the response-side count below.
                # DEAD CODE TEST 2026-05-21: Prompt-side literal counts hidden per user request.
                # Diagnostic noise — only response-side "QWEN COMMANDS EMITTED" is useful day-to-day.
                # Restore if prompt bloat diagnosis is needed (e.g. suspected history-replay overflow).
                # prompt_search_count = enhanced_prompt.count('[SEARCH:')
                # prompt_store_count = enhanced_prompt.count('[STORE:')
                #
                # if prompt_search_count > 0:
                #     logging.critical(f"📥 PROMPT-SIDE: [SEARCH:] literal appears {prompt_search_count} times (docs/history/examples)")
                # if prompt_store_count > 0:
                #     logging.critical(f"📥 PROMPT-SIDE: [STORE:] literal appears {prompt_store_count} times (docs/history/examples)")
                
                # Start timing the LLM call
                llm_start_time = time.time()
                logging.critical(f"⏱️ Starting LLM invoke at {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                
                # === ACTUAL LLM CALL (wrapped with 600s / 10-minute timeout) ===
                # Wraps self.llm.invoke() in a thread executor with a hard wall-clock
                # timeout. Protects against Ollama-side hangs caused by model swap
                # thrash or VRAM pressure (the cause of the 23-minute hangs observed
                # on 2026-05-05 Run #5 and 2026-05-06 Run #13).
                #
                # IMPORTANT: 10 minutes is the generous ceiling for normal chat.
                # OODA loops, scheduled reflections, and autonomous cognition use
                # different code paths and are NOT affected by this timeout — they
                # are allowed to take longer when needed.
                #
                # See _invoke_llm_with_timeout for the thread cancellation caveat:
                # on Windows, an orphaned worker thread may continue executing in
                # the background after timeout, but the user gets control back on
                # schedule. Acceptable tradeoff vs. silent 23-minute hangs.
                response = self._invoke_llm_with_timeout(
                    enhanced_prompt,
                    timeout_seconds=600,
                    label="FIRST_PASS"
                )
                self._prompt_was_sent = True  # Set flag immediately after LLM call — accumulate_prompt_tokens() reads this
                logging.critical(f"🚀 LLM INVOKE #1 COMPLETE - Set _prompt_was_sent=True")
                
                # === LOG COMPLETION ===
                llm_duration = time.time() - llm_start_time
                logging.critical(f"✅ LLM INVOKE COMPLETED")
                logging.critical(f"⏱️ Duration: {llm_duration:.2f} seconds")
                logging.critical(f"📏 Response length: {len(response) if response else 0} characters")
                # === RESPONSE-SIDE COMMAND PATTERN COUNT (QWEN's 1st-pass output) ===
                # Counts memory commands QWEN actually emitted in her first-pass response.
                # Per architecture: commands appear in pass 1 only; pass 2 is the clean
                # natural response with results already incorporated. This log line shows
                # turn-by-turn what QWEN is doing memory-wise.
                # NOTE: This counts what QWEN EMITTED. The MAX_SEARCHES_PER_RESPONSE=5 and
                # MAX_STORES_PER_RESPONSE=5 limits in process_response() may block some
                # of these from actually executing — watch for the deepseek.py log line
                # "⚠️ Search limit was reached" to see when the cap fires.
                try:
                    # Guard against None response (already handled below at line ~1238,
                    # but this block runs before that check, so we guard here too)
                    if response:
                        # Search variants — all four map to the 'search' counter in process_response
                        response_search_count = (
                            response.count('[SEARCH:') +
                            response.count('[COMPREHENSIVE_SEARCH:') +
                            response.count('[PRECISE_SEARCH:') +
                            response.count('[EXACT_SEARCH:')
                        )
                        # Other tracked command types
                        response_store_count = response.count('[STORE:')
                        response_reflect_count = response.count('[REFLECT]')
                        response_forget_count = response.count('[FORGET:')
                        response_cognitive_state_count = response.count('[COGNITIVE_STATE:')
                        response_discuss_claude_count = response.count('[DISCUSS_WITH_CLAUDE:')
                        response_web_search_count = response.count('[WEB_SEARCH:')
                        response_self_dialogue_count = response.count('[SELF_DIALOGUE:')
                        
                        # Build a compact summary of non-zero counts for clean log output.
                        # This dict keeps log noise low when QWEN uses only a couple commands.
                        command_counts = {
                            'search': response_search_count,
                            'store': response_store_count,
                            'reflect': response_reflect_count,
                            'forget': response_forget_count,
                            'cognitive_state': response_cognitive_state_count,
                            'discuss_claude': response_discuss_claude_count,
                            'web_search': response_web_search_count,
                            'self_dialogue': response_self_dialogue_count,
                        }
                        
                        # Filter to only commands actually emitted (count > 0)
                        emitted = {k: v for k, v in command_counts.items() if v > 0}
                        
                        if emitted:
                            # Format as "search=3, store=1" — sorted by count desc for readability
                            summary = ", ".join(
                                f"{k}={v}" for k, v in 
                                sorted(emitted.items(), key=lambda x: -x[1])
                            )
                            total = sum(emitted.values())
                            logging.critical(f"🧠 QWEN COMMANDS EMITTED (1st pass, total={total}): {summary}")
                        else:
                            logging.critical(f"🧠 QWEN COMMANDS EMITTED (1st pass): none")
                    else:
                        # Response was None or empty — process_response will be skipped anyway
                        logging.critical(f"🧠 QWEN COMMANDS EMITTED (1st pass): n/a (empty response)")
                except Exception as count_err:
                    # Defensive: never let diagnostic counting break the main response flow.
                    # If counting fails for any reason, log and continue — the LLM response
                    # itself is what matters, not our visibility into it.
                    logging.warning(f"Failed to count emitted commands in response: {count_err}")
                # === END RESPONSE-SIDE COMMAND PATTERN COUNT ===
                                
                if response is None:
                    return "I encountered an unexpected error. Please try again."
                
                logging.info(f"ENHANCED: Got response length: {len(response)}")
                original_response = response
                
                # =====================================================
                # STEP 4.5: Detect command-only response
                # =====================================================
                # Check if response is essentially just commands with no conversational content
                # This flag will be used to add guidance in the second pass if needed
                command_only_detected = self.is_command_only_response(response)
                if command_only_detected:
                    logging.warning(f"COMMAND_ONLY_RESPONSE: Model responded with commands but no conversation. Response: {response[:100]}...")

            except TimeoutError as timeout_err:
                # =====================================================
                # FIRST_PASS TIMEOUT HANDLER
                # =====================================================
                # _invoke_llm_with_timeout has already logged the detailed diagnostic
                # CRITICAL message including timing, label, and likely causes. Here
                # we just need to return a user-facing message that:
                #   1. Is specific enough to be recognizable as a timeout (not a
                #      generic crash) so Ken can debug pattern occurrences
                #   2. Doesn't alarm the user — most timeouts will be transient
                #      and resolved by simply retrying
                #   3. Gives a clear retry path
                #
                # NOTE: This block must come BEFORE the generic `except Exception`
                # below. Python checks except handlers in order; if Exception came
                # first, TimeoutError would be caught by it (since TimeoutError
                # inherits from Exception) and this specific handler would never run.
                llm_duration = time.time() - llm_start_time if 'llm_start_time' in locals() else 0
                logging.critical(
                    f"⏰ FIRST_PASS TIMEOUT: User-facing timeout after {llm_duration:.2f}s. "
                    f"Returning friendly message. Detailed diagnostic logged above by "
                    f"_invoke_llm_with_timeout. Original exception: {timeout_err}"
                )
                return (
                "I took longer than expected to respond. This sometimes happens "
                "with complex queries — could you try again? If it keeps "
                "happening, let me know and we'll look into it."
            )
                
            except Exception as e:
                llm_duration = time.time() - llm_start_time if 'llm_start_time' in locals() else 0
                logging.critical(f"❌ LLM INVOKE FAILED")
                logging.critical(f"⏱️ Time before failure: {llm_duration:.2f} seconds")
                logging.critical(f"❌ Error type: {type(e).__name__}")
                logging.critical(f"❌ Error message: {str(e)}")
                logging.critical(f"📊 Context state at failure: {current_tokens:,}/{max_tokens:,} tokens")
                return f"I encountered an issue processing your request: {str(e)}"
            
            # =====================================================
            # STEP 5: Process memory commands in the response
            # =====================================================
            logging.info("ENHANCED: Processing memory commands")
            processed_response = response
            commands_processed = 0
            
            try:
                if hasattr(self, 'deepseek_enhancer'):
                    # Process any [STORE:], [RETRIEVE:], [SEARCH:] commands in the response
                    # This is where recursion detection will activate if duplicate commands are found
                    processed_response, commands_processed = self.deepseek_enhancer.process_response(response)
                    logging.info(f"ENHANCED: Processed {commands_processed} memory commands")
                    
                    if processed_response is None:
                        processed_response = response
                        
            except Exception as process_error:
                logging.error(f"Error processing memory commands: {process_error}")
                processed_response = response

            # =====================================================
            # STEP 5.5: Re-fetch session stats after pass 1 command execution
            # =====================================================
            # BUG FIX: session_stats was captured before pass 1 LLM call (Step 3).
            # process_response() in Step 5 has now executed STORE/SEARCH/FORGET commands
            # and updated the Streamlit session counters. Without this re-fetch, pass 2
            # prompt shows stale counts (e.g. STORE: 0 when QWEN just stored 2 memories).
            # Re-fetching here ensures QWEN's self-awareness display is accurate in pass 2.
            try:
                session_stats = self.get_session_memory_stats()
                logging.info(
                    f"SESSION_STATS_REFRESH: Re-fetched after pass 1 command execution - "
                    f"STORE:{session_stats['store']}, SEARCH:{session_stats['search']}, "
                    f"FORGET:{session_stats['forget']}, REFLECT:{session_stats['reflect']}, "
                    f"TOTAL:{session_stats['total_count']}"
                )
            except Exception as stats_refresh_error:
                # Non-critical: if re-fetch fails, pass 2 will use the pre-pass-1 stats
                # This is the same behavior as before the fix, so safe to continue
                logging.warning(f"SESSION_STATS_REFRESH: Failed to re-fetch stats, using pre-pass-1 values: {stats_refresh_error}")
            
            # =====================================================
            # STEP 6: Handle retrieval commands with second pass
            # =====================================================
            
            # Check if response contains retrieval commands that would have generated search results
            retrieval_pattern = re.compile(r'\[(PRECISE_SEARCH|SEARCH|COMPREHENSIVE_SEARCH|EXACT_SEARCH):\s*(.*?)\s*\]', re.IGNORECASE)
            retrieval_commands = retrieval_pattern.findall(response)
            
            # Only do second pass if:
            # 1. We found retrieval commands
            # 2. Commands were actually processed (commands_processed > 0)
            # 3. The processed response is different from original (meaning results were added)
            if retrieval_commands and commands_processed > 0 and processed_response != response:
                logging.critical(f"🔄 SECOND PASS: Found {len(retrieval_commands)} retrieval commands, preparing second LLM call")
                
                # Extract search results for second pass.
                # FIX 2: Expanded regex to match ALL search result formats:
                #   Format A (===== header): **===== SEARCH RESULTS =====** ... **===== END OF SEARCH =====**
                #   Format B (What I remember): **What I remember:** ... **===== END OF SEARCH =====**
                # Previously only Format A was matched, causing Format B results (the default
                # search handler output) to be missed, leaving search_results_sections empty
                # and skipping the second LLM pass entirely.
                search_results_pattern = re.compile(
                    r'(?:'
                    # Format A: ===== delimited headers (SEARCH RESULTS, MEMORY RETRIEVAL, etc.)
                    r'\*\*=====\s*(?:COMPREHENSIVE\s+)?(?:SELECTIVE\s+)?(?:PRECISE\s+)?'
                    r'(?:EXACT\s+)?(?:MATCH\s+)?(?:SEARCH|MEMORY RETRIEVAL|EXACT SEARCH)'
                    r'.*?=====\*\*.*?\*\*=====\s*END\s+OF\s+(?:SEARCH|MEMORY RETRIEVAL|EXACT SEARCH)\s*=====\*\*'
                    r'|'
                    # Format B: "What I remember:" header used by the default search handler.
                    # Catches [SEARCH: |type=conversation_summary] (pipe form) which routes
                    # through the general path and produces this format.
                    r'\*\*What I remember:\*\*.*?\*\*=====\s*END\s+OF\s+SEARCH\s*=====\*\*'
                    r'|'
                    # Format C: DOCUMENT SUMMARY SEARCH dedicated handler output.
                    # Catches [SEARCH: |type=document_summary] which uses its own header/footer
                    # format. Without this arm, pass-2 synthesis was skipped for doc-summary
                    # searches and QWEN never got to integrate retrieved summaries.
                    r'\*\*=====\s*DOCUMENT\s+SUMMAR(?:Y|IES)\s+SEARCH\s*=====\*\*'
                    r'.*?\*\*=====\s*END\s+OF\s+'
                    r'(?:DOCUMENT\s+SUMMAR(?:Y|IES)\s+SEARCH|SEARCH)\s*=====\*\*'
                    r'|'
                    # Format D: ERROR blocks. Surfaces failed searches to pass 2 so QWEN can
                    # acknowledge the error in her reply rather than producing a synthesis
                    # that ignores a failed retrieval. Matches any opening with the word
                    # ERROR (delimited by word boundaries) paired with "END OF ERROR" footer.
                    r'\*\*=====[^*]*?\bERROR\b[^*]*?=====\*\*'
                    r'.*?\*\*=====\s*END\s+OF\s+ERROR\s*=====\*\*'
                    r'|'
                    # Format E: CONVERSATION SUMMARIES blocks emitted by the literal
                    # [SEARCH: conversation_summaries] handler and its date-filtered variant.
                    # Previously these were not extracted, so pass 2 was skipped for those
                    # commands. .*? in both header and footer absorbs optional "FOR {date}"
                    # suffixes and any future modifiers without regex churn.
                    r'\*\*=====\s*(?:LATEST\s+)?CONVERSATION\s+SUMMAR(?:Y|IES).*?=====\*\*'
                    r'.*?\*\*=====\s*END\s+OF\s+(?:LATEST\s+SUMMARY|CONVERSATION\s+SUMMAR(?:Y|IES)).*?=====\*\*'
                    r'|'
                    # Format F: MEMORIES blocks emitted by the MAX_AGE_SEARCH handler
                    # (e.g. [SEARCH: | type=X | max_age_days=N]). The handler emits
                    # headers like **===== MEMORIES: Last 14 Days — Conversation Summary =====**.
                    # End marker echoes the same header in the with-results case OR falls
                    # back to **===== END OF SEARCH =====** in the no-results case. Both
                    # forms matched here so pass-2 synthesis fires consistently for filter
                    # searches. Without this arm, the raw MEMORIES block was being dumped
                    # into the chat window instead of being hidden behind second-pass
                    # synthesis. (Added 2026-05-18.)
                    r'\*\*=====\s*MEMORIES:.*?=====\*\*'
                    r'.*?\*\*=====\s*END\s+OF\s+(?:MEMORIES:.*?|SEARCH)\s*=====\*\*'
                    r')',
                    re.DOTALL | re.IGNORECASE
                )
                search_results_sections = search_results_pattern.findall(processed_response)
                
                # =====================================================
                # DISCUSS_WITH_CLAUDE DIALOG EXTRACTION
                # =====================================================
                # The DISCUSS dialog uses plain ===== delimiters (no ** wrapping),
                # so the search_results_pattern above never matches it.
                # If QWEN issued both [SEARCH:] and [DISCUSS_WITH_CLAUDE:] in the
                # same response, the dialog block lives in processed_response but
                # would be invisible to the second pass LLM — Claude's response
                # would be stored in memory but never shown to QWEN in context.
                # This extraction appends any dialog blocks to the sections list
                # so the second pass receives both search results AND Claude's response.
                try:
                    discuss_pattern = re.compile(
                        r'={5}\s*AI-TO-AI DISCUSSION:.*?={5}\s*END\s+OF\s+DISCUSSION\s*={5}',
                        re.DOTALL | re.IGNORECASE
                    )
                    discuss_sections = discuss_pattern.findall(processed_response)
                    if discuss_sections:
                        logging.info(
                            f"SECOND_PASS: Found {len(discuss_sections)} DISCUSS_WITH_CLAUDE "
                            f"dialog block(s) — appending to second pass context"
                        )
                        # Extend the list so the existing if/else logic below handles
                        # both cases (search results only, discuss only, or both) uniformly
                        search_results_sections.extend(discuss_sections)
                except Exception as discuss_extract_error:
                    logging.error(
                        f"SECOND_PASS: Error extracting DISCUSS dialog blocks: "
                        f"{discuss_extract_error}"
                    )
                    # Non-critical — fall through with whatever search_results_sections has

                if search_results_sections:
                    search_results_text = "\n\n".join(search_results_sections)
                    logging.critical(f"🔄 SECOND PASS: Extracted {len(search_results_sections)} search result sections ({len(search_results_text)} chars)")
                    
                    # =====================================================
                    # ADD COMMAND-ONLY GUIDANCE (if needed)
                    # =====================================================
                    # If first pass was command-only, add integration instructions
                    if command_only_detected:
                        guidance = """## ⚠️ IMPORTANT INSTRUCTIONS FOR THIS RESPONSE

    Your previous response contained ONLY commands without any conversational text.
    This is not helpful. 

    **REQUIRED FOR THIS RESPONSE:**
    1. Write a natural, conversational response that INTEGRATES the search results below
    2. DO NOT just list what you found. Integrate any relevant information naturally in your reply.
    3. DO NOT include any more commands in this response
    
    
    **SEARCH RESULTS TO INTEGRATE:**

    """
                        search_results_text = guidance + search_results_text
                        logging.warning("COMMAND_ONLY_FIX: Added integration guidance for second pass")
                    
                    # =====================================================
                    # ALWAYS ADD INTEGRATION REMINDER (even for non-command-only)
                    # =====================================================
                    # This helps ensure natural integration even when first pass had some text
                    else:
                        # FIX A: Explicit command ban added to integration_reminder.
                        # Previously the reminder only said "integrate naturally" but QWEN
                        # still generated new [SEARCH:] and [STORE:] commands in the second
                        # pass response. Those commands got re-executed, filling the response
                        # with more search result blocks (up to 28K chars) instead of a clean
                        # conversational reply. The explicit ban below prevents this loop.
                        integration_reminder = """## MEMORY CONTEXT - SECOND PASS RESPONSE INSTRUCTIONS

CRITICAL: This is your FINAL response pass. You MUST follow ALL of these rules:
1. Write a natural, conversational reply that integrates the memory context below
2. DO NOT include ANY commands - no [SEARCH:], [STORE:], [REFLECT:] or any [COMMAND:] syntax
3. DO NOT display the search results to the user - use them silently to inform your reply
4. Keep your response concise and focused on what Ken said
5. Speak naturally as QWEN, not as a system reporting data

Memory context for your reference only - integrate relevant details naturally:

"""
                        search_results_text = integration_reminder + search_results_text
                        logging.info("MEMORY_INTEGRATION: Added strengthened integration reminder with command ban for second pass")
                    
                    # Get specialized second-pass template with search results injected
                    # and updated memory command statistics
                    second_pass_prompt = enhanced_prompt_template.format(
                        memory_context=search_results_text,  # NOW POPULATED with search results
                        # PASS INDICATOR: Second pass explicitly bans all commands.
                        # QWEN now receives an unambiguous architectural signal that this is
                        # the final response pass and commands are not permitted. Previously
                        # QWEN had to infer this from the integration_reminder content alone,
                        # which was unreliable and caused command loops in the second pass.
                        pass_indicator=(
                            "## RESPONSE MODE: SYNTHESIS PASS\n"
                            "Your role has shifted. You are no longer the researcher — you are the communicator.\n"
                            "The research phase is complete. All commands have executed and their results "
                            "are in the memory context below.\n"
                            "Your only task now is to write a natural conversational reply to Ken that "
                            "integrates what you found. Speak as QWEN, not as a system reporting data.\n"
                            "There are no commands available in this mode — the command pipeline is closed "
                            "for this pass. Any command syntax will be stripped before Ken sees your response.\n"
                            "Do not display or narrate search results — weave relevant findings naturally "
                            "into your reply as if recalling from memory."
                        ),
                        convo_context=convo_context,
                        user_input=user_input,
                        token_usage=f"{current_tokens}/{max_tokens} ({percentage:.1f}%)",  # FRESH VALUES
                        token_warning=token_warning,
                        current_time=current_time,
                        current_date=current_date,
                        # Core Memory Command Counts
                        search_count=session_stats['search'],
                        store_count=session_stats['store'],
                        forget_count=session_stats['forget'],
                        reflect_count=session_stats['reflect'],
                        # Auxiliary Command Counts
                        summarize_count=session_stats['summarize'],
                        reminder_count=session_stats['reminder'],
                        discuss_count=session_stats['discuss'],
                        # Meta/Utility Command Counts
                        web_search_count=session_stats['web_search'],
                        research_dialogue_count=session_stats['research_dialogue'],
                        help_count=session_stats['help'],
                        cognitive_state_count=session_stats['cognitive_state'],
                        # Total Operations
                        total_count=session_stats['total_count']
                    )
                    
                    # CRITICAL: Update stored prompt and tokens for accurate tracking
                    self._last_prompt_sent = second_pass_prompt
                    self._last_prompt_tokens = calculate_tokens(second_pass_prompt)
                    logging.critical(f"🔄 SECOND PASS: Updated tracking - {len(second_pass_prompt):,} chars, {self._last_prompt_tokens:,} tokens")
                    
                    # Log second pass prompt details
                    logging.critical(f"🔄 SECOND PASS: Second pass prompt length: {len(second_pass_prompt):,} chars")
                    logging.critical(f"🔄 SECOND PASS: Command-only guidance added: {command_only_detected}")
                    
                    # Safety check: Only proceed if we have room in context window
                    # Use 98% threshold to leave a small safety margin
                    if self._last_prompt_tokens < MODEL_PARAMS["num_ctx"] * 0.98:
                        try:
                            start_second = time.time()
                            logging.critical(f"🔄 SECOND PASS: Starting LLM invoke at {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                            
                            # === SECOND-PASS LLM CALL (wrapped with 300s / 5-minute timeout) ===
                            # Same wrapper pattern as first pass, but with a tighter timeout.
                            # Rationale: at this point the model is already loaded, the prompt
                            # template is already constructed, and the model just needs to
                            # synthesize a reply from search results that have been injected
                            # into the prompt. This pass should be fast — typical durations
                            # observed in production logs are 30-90 seconds. A 5-minute ceiling
                            # catches stalls without cutting off legitimate slow synthesis.
                            #
                            # Graceful fallback: if this times out, the existing handler
                            # below falls back to processed_response — the first-pass output
                            # with commands already executed and search results inline. The
                            # user gets a coherent reply derived from the actual findings;
                            # they just don't get the explicit synthesis QWEN would have done.
                            final_response = self._invoke_llm_with_timeout(
                                second_pass_prompt,
                                timeout_seconds=300,
                                label="SECOND_PASS"
                            )
                            self._prompt_was_sent = True  # Set flag immediately after LLM call
                            logging.critical(f"🚀 LLM INVOKE #2 COMPLETE - Set _prompt_was_sent=True")
                            
                            elapsed_second = time.time() - start_second
                            logging.critical(f"🔄 SECOND PASS: COMPLETED in {elapsed_second:.2f}s, response length: {len(final_response) if final_response else 0}")
                            
                            if final_response:
                                # =====================================================
                                # ✅ FIX: Process second pass response for any new commands
                                # =====================================================
                                logging.info("ENHANCED: Processing second-pass response for commands")
                                try:
                                    final_response_processed, second_commands_count = self.deepseek_enhancer.process_response(final_response)
                                    
                                    # FIX: Always use final_response_processed regardless of count.
                                    # process_response applies substitutions for handlers that return
                                    # success=False with a non-empty replacement (e.g. rate-limited
                                    # cognitive_state, soft-fail warnings, error messages). The previous
                                    # else-branch discarded those substitutions because the count only
                                    # increments on success=True, leaving the original bracketed
                                    # commands in the response for safety_strip to catch downstream.
                                    # None-guard mirrors the pattern used at Step 5 in case
                                    # process_response hits an internal exception and returns None.
                                    response = (
                                        final_response_processed
                                        if final_response_processed is not None
                                        else final_response
                                    )

                                    # Logging reflects whether commands were *successfully* processed,
                                    # not whether commands were found. Rate-limited or soft-failed
                                    # commands count as 0 here even though their substitutions were applied.
                                    if second_commands_count > 0:
                                        logging.warning(
                                            f"SECOND_PASS_COMMANDS: Model included {second_commands_count} "
                                            f"commands in second pass response - these have been processed"
                                        )
                                    else:
                                        logging.info(
                                            "SECOND_PASS_COMMANDS: No commands successfully processed "
                                            "in second pass (expected behavior)"
                                        )
                                        
                                except Exception as second_process_error:
                                    logging.error(f"SECOND_PASS_COMMANDS: Error processing second pass commands: {second_process_error}")
                                    response = final_response  # Fallback to unprocessed response
                                # =====================================================
                                # END OF FIX
                                # =====================================================
                                
                                logging.info("ENHANCED: Completed two-pass retrieval process successfully")

                        except TimeoutError as timeout_err:
                            # =====================================================
                            # SECOND_PASS TIMEOUT HANDLER
                            # =====================================================
                            # _invoke_llm_with_timeout has already logged the detailed
                            # diagnostic CRITICAL message. Here we fall back to
                            # processed_response — the pass-1 content with commands
                            # already executed and search results visible inline. This
                            # gives the user a coherent reply derived from the actual
                            # search findings, just without the explicit synthesis step
                            # that pass 2 would have performed.
                            #
                            # IMPORTANT: Unlike the first-pass timeout handler, no
                            # user-facing "took too long" message is shown here. The
                            # user gets a real reply (with the search results inline)
                            # rather than an apology. The synthesis failure is graceful
                            # and largely invisible to the user.
                            #
                            # NOTE: This block must come BEFORE the generic except
                            # Exception below. TimeoutError inherits from Exception,
                            # so order matters — Python checks except handlers top to
                            # bottom and uses the first match. If Exception came first,
                            # this specific handler would never run.
                            elapsed_second = time.time() - start_second if 'start_second' in locals() else 0
                            logging.critical(
                                f"⏰ SECOND_PASS TIMEOUT: Synthesis timed out after "
                                f"{elapsed_second:.2f}s. Falling back to pass-1 "
                                f"processed_response (which has search results inline). "
                                f"User will see a coherent reply but not the explicit "
                                f"synthesized summary. Original exception: {timeout_err}"
                            )
                            response = processed_response
                            

                        except Exception as e:
                            elapsed_second = time.time() - start_second if 'start_second' in locals() else 0
                            logging.error(f"🔄 SECOND PASS: FAILED after {elapsed_second:.2f}s: {e}")
                            import traceback
                            traceback.print_exc()
                            # Fall back to processed response from first pass
                            response = processed_response
                    else:
                        # Context window too full for second pass
                        logging.warning(f"🔄 SECOND PASS: SKIPPED - prompt tokens ({self._last_prompt_tokens:,}) exceed 98% of context ({MODEL_PARAMS['num_ctx']:,})")
                        response = processed_response
                
                else:
                    # =====================================================
                    # FIX 1: No extractable search result sections found.
                    # This happens when search results use the "What I remember:"
                    # format rather than the "===== SEARCH RESULTS =====" header
                    # format that the regex at line 1036 expects. Without this
                    # else clause, `response` was left as the raw unprocessed
                    # LLM output (with visible commands) because it was never
                    # updated to processed_response.
                    # Fall back to processed_response from pass 1 which has
                    # all commands replaced with their success/result messages.
                    # =====================================================
                    logging.warning(
                        "SECOND_PASS: No extractable search result sections found in expected "
                        "===== format - falling back to pass 1 processed_response. "
                        "Commands were executed successfully but second LLM pass skipped."
                    )
                    response = processed_response
            
            # =====================================================
            # STEP 6 ELSE: No retrieval second pass triggered
            # =====================================================
            # DISCUSS_WITH_CLAUDE, STORE, REFLECT, and other non-retrieval commands are
            # fully handled in Step 5 (process_response). Their output lives in
            # `processed_response`. If no SEARCH/RETRIEVE commands were present,
            # the if-block above was never entered and `response` was never updated.
            # This else-clause ensures `processed_response` (with command output
            # already formatted) is used rather than the raw unmodified LLM output.
            else:
                if processed_response != response:
                    logging.info(
                        "ENHANCED: Step 6 skipped (no retrieval commands) — "
                        "applying processed_response from Step 5. "
                        f"response={len(response)} chars → processed={len(processed_response)} chars"
                    )
                    response = processed_response

            # =====================================================
            # STEP 7: Clean search results from final response
            # =====================================================
            
            def clean_search_results_from_response(response_text):
                """
                Remove internal result blocks from response to prevent context accumulation.
                
                These blocks are useful for the model during processing but should not
                be displayed to the user in the final response. The user has access to
                a complete HTML command guide in the UI.
                """
                # Pattern to match all internal result blocks using ===== delimiters
                # Uses flexible matching for variations in block names
                search_patterns = [
                    # FIX B1: "What I remember:" format used by the default search handler.
                    # This is the most common format and was previously missing, causing
                    # large search result blocks (up to 28K chars) to pass through to
                    # the final response unchanged. Must come before the ===== patterns
                    # since it anchors on the END OF SEARCH footer shared by all formats.
                    r'\*\*What I remember:\*\*.*?\*\*=====\s*END\s+OF\s+SEARCH\s*=====\*\*',
                    
                    # FIX B2: "Successfully Stored" feedback line emitted by store handler.
                    # Format: ✅ Successfully Stored: 'truncated memory text...'
                    # These one-line confirmations were leaking into the final response.
                    r'✅\s*Successfully\s+Stored:[^\n]*',
                    
                    # Search result variants (===== header format)
                    r'\*\*=====\s*(?:COMPREHENSIVE\s+)?(?:SELECTIVE\s+)?(?:PRECISE\s+)?(?:EXACT\s+)?(?:MATCH\s+)?SEARCH\s+RESULTS?\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+SEARCH\s*=====\*\*',
                    r'\*\*=====\s*SEARCH\s+RESULTS?:.*?=====\*\*.*?\*\*=====\s*END\s+OF\s+SEARCH\s*=====\*\*',
                    
                    # Memory retrieval
                    r'\*\*=====\s*MEMORY\s+RETRIEVAL\s+RESULTS?\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+MEMORY\s+RETRIEVAL\s*=====\*\*',
                    
                    # Document summaries — uses SUMMAR(?:Y|IES) to match both singular
                    # "DOCUMENT SUMMARY SEARCH" (the actual emitted form) and the plural
                    # variant. The legacy "SUMMARIES?" pattern silently failed to match
                    # the singular form, leaking raw dumps into the user-visible response.
                    # END clause covers the normalized footer plus the legacy "END OF SEARCH"
                    # form for backward compatibility with any in-flight error paths.
                    r'\*\*=====\s*DOCUMENT\s+SUMMAR(?:Y|IES)\s+SEARCH\s*=====\*\*'
                    r'.*?\*\*=====\s*END\s+OF\s+'
                    r'(?:DOCUMENT\s+SUMMAR(?:Y|IES)\s+SEARCH|SEARCH|DOCUMENT\s+SUMMAR(?:Y|IES))\s*=====\*\*',
                    
                    # Conversation summaries — opening already used SUMMAR(?:Y|IES) correctly.
                    # END clause had two bugs: (1) buggy SUMMARIES? quantifier, (2) required
                    # =====**  immediately after SUMMARIES so the date-suffixed footer
                    # "END OF CONVERSATION SUMMARIES FOR YYYY-MM-DD =====**" never matched.
                    # Both fixed: SUMMAR(?:Y|IES) for singular/plural, plus .*? to absorb
                    # any trailing modifier ("FOR {date}", etc.) before the closing delimiter.
                    r'\*\*=====\s*(?:LATEST\s+)?CONVERSATION\s+SUMMAR(?:Y|IES).*?=====\*\*.*?\*\*=====\s*END\s+OF\s+(?:LATEST\s+SUMMARY|CONVERSATION\s+SUMMAR(?:Y|IES)).*?=====\*\*',
                    
                    # Format F mirror (added 2026-05-21): MEMORIES blocks emitted by
                    # the MAX_AGE_SEARCH handler in deepseek.py:_handle_max_age_filtered_search
                    # (e.g. [SEARCH: | type=conversation_summary | max_age_days=14]).
                    # Pairs with the Format F arm of the extraction regex at L~1556.
                    # Without this pattern, the raw MEMORIES block leaks to the
                    # user-visible response whenever the second pass falls back to
                    # processed_response (timeout / exception / context >98% / extraction
                    # miss) OR when the second-pass LLM echoes the block in its synthesis.
                    # END clause covers BOTH emission paths in the handler:
                    #   - With-results footer:  '**===== END OF MEMORIES: Last N Days ... =====**'
                    #   - No-results footer:    '**===== END OF SEARCH =====**' (deepseek.py L1042)
                    # NOTE: user-typed filter searches travel a separate diagnostic path
                    # in deepseek.handle_user_commands and are not affected by this cleanup.
                    r'\*\*=====\s*MEMORIES:.*?=====\*\*'
                    r'.*?\*\*=====\s*END\s+OF\s+(?:MEMORIES:.*?|SEARCH)\s*=====\*\*',
                    
                    # Reminder search results
                    r'\*\*=====\s*REMINDER\s+SEARCH\s+RESULTS?\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+(?:REMINDER\s+SEARCH|SEARCH)\s*=====\*\*',
                    
                    # AI-to-AI discussion (DISCUSS_WITH_CLAUDE)
                    r'\*\*=====\s*AI-TO-AI\s+DISCUSSION:.*?=====\*\*.*?\*\*=====\s*END\s+OF\s+DISCUSSION\s*=====\*\*',
                    
                    # External research dialogue (WEB_SEARCH)
                    r'\*\*=====\s*EXTERNAL\s+RESEARCH\s+DIALOGUE:.*?=====\*\*.*?\*\*=====\s*END\s+OF\s+EXTERNAL\s+RESEARCH\s+DIALOGUE\s*=====\*\*',
                    
                    # Image analysis
                    r'\*\*=====\s*IMAGE\s+ANALYSIS\s+RESULTS?\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+IMAGE\s+ANALYSIS\s*=====\*\*',
                    
                    # Command guide / help
                    r'\*\*=====\s*(?:INTERNAL\s+)?COMMAND\s+(?:GUIDE|REFERENCE)\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+COMMAND\s+(?:GUIDE|REFERENCE)\s*=====\*\*',
                    
                    # System prompt display
                    r'\*\*=====\s*SYSTEM\s+PROMPT\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+SYSTEM\s+PROMPT\s*=====\*\*',
                    
                    # Help blocks (search help, store help, modify prompt help)
                    r'\*\*=====\s*(?:SEARCH|STORE\s+COMMAND|MODIFY\s+SYSTEM\s+PROMPT)\s+HELP\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+(?:SEARCH\s+)?HELP\s*=====\*\*',
                    
                    # Status display
                    r'\*\*=====\s*STATUS\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+STATUS\s*=====\*\*',
                    
                    # Error blocks — flexible match for any opening that contains the word
                    # ERROR delimited by word boundaries (catches "===== ERROR =====",
                    # "===== ERROR RETRIEVING CONVERSATION SUMMARIES =====", "===== WEB
                    # KNOWLEDGE ERROR =====", and similar). The previous pattern required
                    # ERROR to be immediately followed by =====, which never matched the
                    # actual emission at deepseek.py:5079 — error blocks have been silently
                    # leaking to the user any time a search errored out.
                    # [^*]*? stops at the first * to prevent overrun into adjacent blocks.
                    r'\*\*=====[^*]*?\bERROR\b[^*]*?=====\*\*.*?\*\*=====\s*END\s+OF\s+ERROR\s*=====\*\*',
                    
                    # Internal self-dialogue (uses markdown headers, not ===== pattern)
                    r'##\s*🤔\s*Internal\s+Self-Dialogue:.*?(?=\n##\s+[^🤔]|\n\*\*[A-Z]|\Z)',
                ]
                
                cleaned_response = response_text
                for pattern in search_patterns:
                    cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
                
                # Remove multiple newlines and clean up formatting
                cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)
                cleaned_response = cleaned_response.strip()
                
                return cleaned_response
            
            # Capture pre-cleanup content so EMPTY_RESPONSE_GUARD can use
            # found search results if cleanup strips everything (e.g. pass 2
            # returns only search result blocks with no conversational wrapper)
            pre_cleanup_response = response

            # Apply cleanup to final response
            original_response_length = len(response)
            response = clean_search_results_from_response(response)
            cleaned_response_length = len(response)
            
            if original_response_length != cleaned_response_length:
                logging.info(f"CLEANUP: Removed {original_response_length - cleaned_response_length} characters of search results from response")
                logging.info(f"CLEANUP: Response length: {original_response_length} → {cleaned_response_length}")
            
            # =====================================================
            # STEP 7.5: SAFETY CHECK - Detect any remaining unprocessed commands
            # =====================================================
            # This catches commands that slipped through processing
            try:
                # Pattern to match any known command syntax
                command_pattern = r'\[\s*(?:SEARCH|STORE|FORGET|REFLECT|COMPREHENSIVE_SEARCH|PRECISE_SEARCH|EXACT_SEARCH|SUMMARIZE_CONVERSATION|REMINDER|HELP|DISCUSS_WITH_CLAUDE|SHOW_SYSTEM_PROMPT|MODIFY_SYSTEM_PROMPT|SELF_DIALOGUE|WEB_SEARCH|COGNITIVE_STATE|COMPLETE_REMINDER)\s*:'
                remaining_commands = re.findall(command_pattern, response, re.IGNORECASE)
                
                if remaining_commands:
                    logging.critical(f"⚠️ UNPROCESSED COMMANDS IN FINAL RESPONSE: Found {len(remaining_commands)} unprocessed commands")
                    # Log the commands found
                    for i, cmd in enumerate(remaining_commands[:5], 1):  # Log first 5
                        logging.critical(f"   {i}. {cmd}")
                    
                    # Log context around first occurrence for debugging
                    first_match = re.search(command_pattern, response, re.IGNORECASE)
                    if first_match:
                        start_pos = max(0, first_match.start() - 50)
                        end_pos = min(len(response), first_match.end() + 50)
                        context = response[start_pos:end_pos]
                        logging.critical(f"   Context around first command: ...{context}...")
                    
                    # FIX 3: Strip unprocessed commands from the final response so they
                    # are never visible to the user. This is the last-resort safety net.
                    # Commands that reach here were either:
                    #   - Included twice by the LLM (once executable, once as prose)
                    #   - Missed by the pattern matcher in process_response()
                    #   - Left over due to a format mismatch in an earlier step
                    # We replace the full [...] block with an empty string rather than
                    # a placeholder so QWEN's response reads cleanly.
                    # The full command pattern including closing bracket:
                    full_command_pattern = r'\[\s*(?:SEARCH|STORE|FORGET|REFLECT|COMPREHENSIVE_SEARCH|PRECISE_SEARCH|EXACT_SEARCH|SUMMARIZE_CONVERSATION|REMINDER|HELP|DISCUSS_WITH_CLAUDE|SHOW_SYSTEM_PROMPT|MODIFY_SYSTEM_PROMPT|SELF_DIALOGUE|WEB_SEARCH|COGNITIVE_STATE|COMPLETE_REMINDER)\s*:[^\]]*\]'
                    response_before_strip = len(response)
                    response = re.sub(full_command_pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
                    # Clean up any double newlines left behind by the removal
                    response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response).strip()
                    stripped_chars = response_before_strip - len(response)
                    logging.critical(f"   ✅ SAFETY_STRIP: Removed {len(remaining_commands)} unprocessed command(s), {stripped_chars} chars stripped from response")
                    
            except Exception as safety_check_error:
                logging.error(f"SAFETY_CHECK: Error checking for unprocessed commands: {safety_check_error}")
            
            # =====================================================
            # STEP 7.75: Empty response guard — LLM acknowledgment pass
            # =====================================================
            # If all content was stripped (e.g. STORE-only response with no
            # conversational text), the response will be empty or whitespace.
            # Rendering an empty string in Streamlit shows only the assistant
            # avatar (yellow robot icon) with no text — confusing to the user.
            #
            # Fix: fire a lightweight LLM call asking QWEN to acknowledge what
            # it just did. If that also fails, fall back to a static message so
            # the UI always receives a non-empty string.
            try:
                if not response or not response.strip():
                    logging.warning(
                        "EMPTY_RESPONSE_GUARD: Response is empty after all cleanup steps. "
                        "This was likely a STORE-only response. Firing acknowledgment pass."
                    )
                    
                    # Build acknowledgment prompt — inject any search results found
                    # before cleanup stripped them, so QWEN can answer based on
                    # what it actually retrieved rather than flying blind.
                    # Truncate to 4000 chars to prevent context overflow.
                    memory_context_for_ack = pre_cleanup_response[:4000] if pre_cleanup_response else ""

                    ack_prompt = (
                        f"<|im_start|>system\n"
                        f"{self.current_system_prompt}\n"
                        f"<|im_end|>\n"
                        f"<|im_start|>user\n"
                        f"[INTERNAL INSTRUCTION — do NOT repeat this to the user]\n"
                        f"You just searched your memory in response to: \"{user_input}\"\n"
                        + (
                            f"Here is what you found in your memory:\n\n{memory_context_for_ack}\n\n"
                            if memory_context_for_ack else
                            f"Your response contained only memory operations with no conversational text.\n"
                        )
                        + f"Please provide a natural response to Ken that integrates what you found.\n"
                        f"If nothing relevant was found, say so naturally.\n"
                        f"Do NOT include any memory commands (no [STORE:], [SEARCH:], etc.) in this response.\n"
                        f"<|im_end|>\n"
                        f"<|im_start|>assistant"
                    )
                    
                    try:
                        # === ACKNOWLEDGMENT LLM CALL (wrapped with 120s / 2-minute timeout) ===
                        # Tightest of the three timeouts. Rationale: this is a recovery
                        # path that fires only when previous response generation produced
                        # zero conversational content (typically a STORE-only response).
                        # The acknowledgment prompt is small (~4K chars memory context +
                        # short system prompt + brief user instruction) and asks for a
                        # short conversational acknowledgment. Typical durations should be
                        # under 30 seconds. A 2-minute ceiling means we don't compound a
                        # bad situation: if the recovery itself stalls, we bail to the
                        # static fallback ("Memory updated.") rather than make the user
                        # wait further.
                        #
                        # Graceful fallback: if this times out, the new TimeoutError
                        # handler below uses the same static fallback as the existing
                        # exception handler. UI always receives a non-empty string.
                        ack_response = self._invoke_llm_with_timeout(
                            ack_prompt,
                            timeout_seconds=120,
                            label="EMPTY_RESPONSE_GUARD"
                        )
                        
                        if ack_response and ack_response.strip():
                            # Run the acknowledgment through safety strip too,
                            # in case the model still sneaks a command in
                            # RETRIEVE deprecated — removed from acknowledgment-pass strip
                            ack_response = re.sub(
                                r'\[\s*(?:SEARCH|STORE|FORGET|REFLECT|COMPREHENSIVE_SEARCH|'
                                r'PRECISE_SEARCH|EXACT_SEARCH|SUMMARIZE_CONVERSATION|REMINDER|HELP|'
                                r'DISCUSS_WITH_CLAUDE|SHOW_SYSTEM_PROMPT|MODIFY_SYSTEM_PROMPT|'
                                r'SELF_DIALOGUE|WEB_SEARCH|COGNITIVE_STATE|COMPLETE_REMINDER)\s*:[^\]]*\]',
                                '', ack_response, flags=re.IGNORECASE | re.DOTALL
                            ).strip()
                            
                            if ack_response.strip():
                                response = ack_response
                                logging.info(
                                    f"EMPTY_RESPONSE_GUARD: Acknowledgment pass succeeded. "
                                    f"Response length: {len(response)} chars."
                                )
                            else:
                                # Acknowledgment was also command-only — use static fallback
                                response = "Memory updated."
                                logging.warning(
                                    "EMPTY_RESPONSE_GUARD: Acknowledgment pass returned commands only. "
                                    "Using static fallback."
                                )
                        else:
                            # LLM returned None or empty
                            response = "Memory updated."
                            logging.warning(
                                "EMPTY_RESPONSE_GUARD: Acknowledgment pass returned empty. "
                                "Using static fallback."
                            )

                    except TimeoutError as ack_timeout_err:
                        # =====================================================
                        # EMPTY_RESPONSE_GUARD TIMEOUT HANDLER
                        # =====================================================
                        # _invoke_llm_with_timeout has already logged the detailed
                        # diagnostic CRITICAL message. Here we use the same static
                        # fallback as the generic exception handler below — the
                        # UI must never receive an empty string from this method,
                        # and "Memory updated." is honest about what just happened
                        # (the first pass DID succeed and store memories; only the
                        # acknowledgment synthesis failed).
                        #
                        # Logging at CRITICAL because a timeout in the recovery
                        # path indicates a deeper problem worth investigating —
                        # this path is supposed to be lightweight and fast.
                        #
                        # NOTE: Must come BEFORE except Exception below for the
                        # same inheritance reason as Stages 3 and 4.
                        logging.critical(
                            f"⏰ EMPTY_RESPONSE_GUARD TIMEOUT: Recovery acknowledgment "
                            f"pass timed out. This indicates a deeper issue — the "
                            f"recovery path is lightweight and should never take this "
                            f"long. Using static fallback. Original exception: "
                            f"{ack_timeout_err}"
                        )
                        response = "Memory updated."
                            
                    except Exception as ack_error:
                        # LLM call itself failed — use static fallback so UI never
                        # receives an empty string
                        response = "Memory updated."
                        logging.error(
                            f"EMPTY_RESPONSE_GUARD: Acknowledgment LLM call failed: {ack_error}. "
                            "Using static fallback."
                        )
                        
            except Exception as empty_guard_error:
                # Outer guard — should never happen, but ensures we never crash here
                logging.error(
                    f"EMPTY_RESPONSE_GUARD: Unexpected error in empty response guard: {empty_guard_error}",
                    exc_info=True
                )
                # Last resort: if response is still empty, set a static fallback
                if not response or not response.strip():
                    response = "Memory updated."
            
            # =====================================================
            # STEP 8: Reset UI indicators and return response
            # =====================================================
            
            # Reset indicators safely
            try:
                if hasattr(indicators['Model'], 'empty'):
                    indicators['Model'].empty()
                if hasattr(indicators['FAISS'], 'empty'):
                    indicators['FAISS'].empty()
                if hasattr(indicators['DB'], 'empty'):
                    indicators['DB'].empty()
            except Exception:
                pass
            
            # Log processing time
            elapsed_time = time.time() - start_time
            logging.info(f"process_command completed in {elapsed_time:.3f} seconds")
            
            return response
            
        except Exception as e:
            logging.error(f"Error in enhanced process_command: {e}", exc_info=True)
            elapsed_time = time.time() - start_time
            logging.info(f"process_command completed with error in {elapsed_time:.3f} seconds")
            return f"An error occurred: {str(e)}"
                

    def _detect_meta_question(self, user_input: str) -> bool:
        """
        Detect if user is asking about the AI's own processing/behavior.
        These meta-cognitive questions can trigger recursion traps.
        
        Args:
            user_input (str): The user's question/input
            
        Returns:
            bool: True if this appears to be a meta-question about AI's own behavior
        """
        try:
            # Keywords that indicate meta-cognitive questions
            meta_keywords = [
                'stuck in a loop', 'stuck in loop', 'why do you loop', 
                'analyze yourself', 'your own processing', 'your behavior', 
                'why did you', 'recursive', 'meta-cognitive', 'self-referential',
                'analyze your', 'your thinking', 'your response pattern',
                'why do you keep', 'infinite loop', 'repeated behavior',
                'same thing over', 'consciousness', 'self-awareness'
            ]
            
            input_lower = user_input.lower()
            
            # Check if any meta-keywords are present
            for keyword in meta_keywords:
                if keyword in input_lower:
                    logging.warning(f"META_QUESTION: Detected keyword '{keyword}' in user input")
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error in meta-question detection: {e}")
            return False  # If detection fails, treat as normal question
                   
        #used by nightly learning and web knowlegde seeker
    # QUARANTINED 2026-05-19: No callers found AND would crash if invoked (batch 4 web cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Calls web_crawler.process_knowledge_gaps_with_web_search() at L2249 — a method that does NOT exist
    # on the WebCrawler class. Would throw AttributeError immediately if reached. The stale comment above
    # claims "used by nightly learning and web knowledge seeker" but the actual fill_knowledge_gaps
    # cognitive activity dispatches to autonomous_cognition._fill_knowledge_gaps instead (see main.py L1350).
    # Leftover from a refactor where the dispatch was moved but this wrapper wasn't removed.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_fill_knowledge_gaps(self, max_gaps: int = 3) -> str:
        """
        Trigger knowledge gap filling using web search.
        
        Args:
            max_gaps: Maximum number of gaps to process
            
        Returns:
            str: User-friendly status report
        """
        try:
            from knowledge_gap import KnowledgeGapQueue
            from web_crawler import WebCrawler
            
            logging.info(f"🚀 Starting knowledge gap filling process")
            
            # Initialize components
            gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
            web_crawler = WebCrawler(chatbot=self)
            
            # Process knowledge gaps
            results = web_crawler.process_knowledge_gaps_with_web_search(
                knowledge_gap_queue=gap_queue,
                max_gaps=max_gaps
            )
            
            if not results["success"]:
                return f"❌ Knowledge gap filling failed: {results.get('error', 'Unknown error')}"
            
            # Create user-friendly report
            if results["gaps_processed"] == 0:
                return "ℹ️ No pending knowledge gaps found to process."
            
            report = f"🧠 **Knowledge Gap Filling Complete**\n\n"
            report += f"📊 **Summary:**\n"
            report += f"• Gaps processed: {results['gaps_processed']}\n"
            report += f"• Successfully filled: {results['gaps_filled']}\n"
            report += f"• Failed to fill: {results['gaps_failed']}\n\n"
            
            if results["processed_topics"]:
                report += f"📋 **Processed Topics:**\n"
                for topic_info in results["processed_topics"]:
                    status_emoji = "✅" if topic_info["status"] == "filled" else "❌"
                    report += f"{status_emoji} {topic_info['topic']} - {topic_info['status']}\n"
            
            return report
            
        except Exception as e:
            logging.error(f"❌ Error in fill_knowledge_gaps: {e}", exc_info=True)
            return f"❌ Error filling knowledge gaps: {str(e)}"

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (chatbot.py cleanup pass).
    # Memory formatting helper, unwired.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__format_memory_for_context(self, memory: Dict) -> str:
        """Format a memory entry for context inclusion."""
        try:
            content = memory.get('content', '')
            confidence = memory.get('confidence', {}).get('level', 'Unknown')
            metadata = memory.get('metadata', {})
            memory_type = metadata.get('type', 'general')
            source = metadata.get('source', 'Unknown')
            if memory_type == 'document':
                return f"[Document Memory] {content} (Source: {source}, Confidence: {confidence})"
            elif memory_type == 'important':
                return f"[Important Memory] {content} (Confidence: {confidence})"
            elif memory_type == 'conversation':
                return f"[Conversation Summary] {content} (Confidence: {confidence})"
            else:
                return f"[Memory] {content} (Confidence: {confidence})"
        except Exception as e:
            logging.error(f"Error formatting memory: {e}")
            return str(memory.get('content', 'Error formatting memory'))
 
    def store_memory_with_transaction(self, content, memory_type, metadata=None, confidence=None, max_retries=1, duplicate_threshold=0.98):
        """
        Store memory with true two-phase commit across SQL and Vector databases.
        Ensures both databases stay in sync - both succeed or both fail together.
        
        Args:
            content: The text content to store
            memory_type: Type of memory (e.g., "conversation_summary", "user_info")
            metadata: Optional metadata dictionary
            confidence: Optional confidence weight (0.0 to 1.0)
            max_retries: Number of retry attempts for vector DB storage
            duplicate_threshold: Similarity threshold for duplicate detection (default 0.98)
                                Use higher values (e.g., 0.995) for conversation summaries
                                to allow similar but different content to be stored
        
        Returns:
            tuple[bool, str]: (success, memory_id or None)
        """
        memory_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}
        metadata['tracking_id'] = memory_id

        # Validate vector_db
        if not hasattr(self, 'vector_db') or self.vector_db is None:
            logging.error("VectorDB is not initialized")
            return False, None

        # Log the threshold being used
        logging.info(f"TRANSACTION: Storing {memory_type} with duplicate_threshold={duplicate_threshold}")

        # Convert tags array to comma-separated string for SQL storage
        tags_value = metadata.get("tags", None)
        if isinstance(tags_value, list):
            tags_str = ",".join(tags_value)  # Convert array to string for SQL
        else:
            tags_str = tags_value  # Already a string or None

        # ✅ PHASE 1: Prepare SQL transaction (but don't commit yet)
        conn = None
        cursor = None
        try:
            import sqlite3
            import json
            
            conn = sqlite3.connect(self.memory_db.db_path)
            conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            conn.execute("BEGIN IMMEDIATE TRANSACTION")
            cursor = conn.cursor()
            
            # Calculate confidence weight if not provided
            if confidence is not None:
                initial_weight = confidence
            else:
                initial_weight = self.memory_db.calculate_memory_weight(
                    memory_type=memory_type,
                    access_count=0,
                    created_at=datetime.datetime.now()
                )
            
            # Prepare metadata JSON
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Insert into SQL (but don't commit yet)
            cursor.execute("""
                INSERT INTO memories 
                (content, memory_type, source, weight, tags, tracking_id, access_count, created_at, last_accessed, metadata) 
                VALUES (?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """, (
                content,
                memory_type,
                metadata.get("source", "unknown"),
                initial_weight,
                tags_str,
                memory_id,
                metadata_json
            ))
            
            row_id = cursor.lastrowid
            logging.info(f"SQL insert prepared (not committed) for tracking_id={memory_id}, row_id={row_id}")
            
        except sqlite3.IntegrityError as integrity_error:
            # Handle constraint violations (e.g., duplicate tracking_id)
            logging.error(f"SQL integrity constraint violation: {integrity_error}")
            if conn:
                conn.rollback()
                conn.close()
            return False, None
            
        except Exception as e:
            logging.error(f"Error preparing SQL insert: {e}", exc_info=True)
            if conn:
                conn.rollback()
                conn.close()
            return False, None

        # ✅ PHASE 2: Try Vector DB storage
        metadata["memory_id"] = memory_id
        vector_success = False
        failure_reason = None
        
        for attempt in range(max_retries):
            try:
                backoff_time = 0.5 * (2 ** attempt) if attempt > 0 else 0
                if attempt > 0:
                    logging.info(f"VectorDB retry attempt {attempt+1}/{max_retries} after {backoff_time:.2f}s")
                    time.sleep(backoff_time)
                
                logging.debug(f"Attempting VectorDB add_text: memory_id={memory_id}, metadata={metadata}")
                
                # ================================================================
                # FIXED: Pass duplicate_threshold to vector_db.add_text()
                # This allows conversation summaries to use a stricter threshold
                # ================================================================
                vector_success, reason = self.vector_db.add_text(
                    text=content,
                    metadata=metadata,
                    memory_id=memory_id,
                    duplicate_threshold=duplicate_threshold
                )
                failure_reason = reason
                
                if vector_success:
                    logging.info(f"VectorDB add_text succeeded for memory_id={memory_id}")
                    break
                elif reason == "duplicate":
                    # If it's a duplicate, stop retrying immediately
                    logging.info(f"Duplicate detected in VectorDB, stopping retries for memory_id={memory_id}")
                    break
                else:
                    logging.warning(f"VectorDB add_text returned False on attempt {attempt+1}, reason: {reason}")
                    if attempt == max_retries - 1:
                        logging.error(f"All {max_retries} VectorDB attempts failed")
                        
            except Exception as e:
                logging.error(f"VectorDB error on attempt {attempt+1}: {e}", exc_info=True)
                failure_reason = "error"
                if attempt == max_retries - 1:
                    logging.error(f"Final VectorDB attempt failed: {e}")

        # ✅ PHASE 3: Commit or Rollback based on Vector DB result
        try:
            if not vector_success:
                # Vector DB failed - rollback SQL transaction
                conn.rollback()
                conn.close()
                
                if failure_reason == "duplicate":
                    logging.info(f"Duplicate detected in VectorDB - rolled back SQL transaction: {content[:50]}...")
                    # Don't queue for retry - this is expected behavior
                    return False, None
                else:
                    logging.warning(f"VectorDB storage failed (reason: {failure_reason}) - rolled back SQL transaction: {content[:50]}...")
                    # Queue for retry on actual errors
                    try:
                        self.memory_db.queue_for_deletion(memory_id)
                    except Exception as queue_error:
                        logging.error(f"Error queueing for retry: {queue_error}")
                    return False, None
            else:
                # Vector DB succeeded - commit SQL transaction
                conn.commit()
                conn.close()
                logging.info(f"✅ Successfully stored memory in BOTH databases with ID {memory_id}: {content[:50]}...")
                self._last_memory_id = memory_id
                return True, memory_id
                
        except Exception as commit_error:
            logging.error(f"Error during commit/rollback phase: {commit_error}", exc_info=True)
            try:
                if conn:
                    conn.rollback()
                    conn.close()
            except:
                pass
            return False, None

    def _fetch_full_summary_content(self, memory_id: str) -> Optional[str]:
        """
        Fetch the full content of a conversation summary from SQL by memory_id.

        Vector DB stores conversation summaries as multiple chunks
        (each labeled "[Part X of Y]"), but SQL stores the complete summary
        in one row keyed by either tracking_id (UUID, modern) or id
        (integer PK, legacy). This method retrieves the canonical full
        text so the search-result formatters can display the whole
        summary instead of a fragment.

        Lookup strategy:
          1. UUID path  → SELECT content FROM memories WHERE tracking_id = ?
          2. Integer fallback → SELECT content FROM memories WHERE id = ?
             (only attempted if memory_id parses as an int)

        Args:
            memory_id (str): The 'memory_id' value from a Qdrant chunk's
                             metadata. Can be a UUID string or numeric
                             string. Empty/None returns None.

        Returns:
            Optional[str]: Full summary text from SQL, or None if neither
                           lookup path finds a row, or on SQL error.
        """
        # Validate input — empty or None returns immediately
        if not memory_id:
            return None

        try:
            # Use a context manager so the connection always closes,
            # even on exception. Read-only — no transaction needed.
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()

                # ── Path 1: UUID via tracking_id (modern entries) ─────
                cursor.execute(
                    "SELECT content FROM memories WHERE tracking_id = ?",
                    (str(memory_id),)
                )
                row = cursor.fetchone()
                if row and row[0]:
                    return row[0]

                # ── Path 2: integer id PK (legacy entries) ────────────
                # Only attempt if memory_id is purely numeric — avoids
                # spurious queries for malformed UUIDs.
                try:
                    int_id = int(memory_id)
                    cursor.execute(
                        "SELECT content FROM memories WHERE id = ?",
                        (int_id,)
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        return row[0]
                except (ValueError, TypeError):
                    # Not numeric — UUID path was already tried above
                    pass

                # Neither path found anything — log at debug so it
                # doesn't spam logs but is available when needed
                logging.debug(
                    f"FULL_SUMMARY_LOOKUP: No SQL row found for "
                    f"memory_id={memory_id} (tried tracking_id and id)"
                )
                return None

        except sqlite3.Error as sql_err:
            logging.error(
                f"FULL_SUMMARY_LOOKUP: SQL error for memory_id={memory_id}: "
                f"{sql_err}"
            )
            return None
        except Exception as e:
            # Catch-all so a lookup failure never breaks the search flow
            logging.error(
                f"FULL_SUMMARY_LOOKUP: Unhandled error for "
                f"memory_id={memory_id}: {e}",
                exc_info=True
            )
            return None
        
    def _fetch_created_at_by_memory_id(self, memory_id: str) -> Optional[datetime.datetime]:
        """
        Fetch the SQL `created_at` timestamp for a memory by memory_id.

        Some memory types (notably document_summary) don't store date
        fields in Qdrant metadata. SQL has the timestamp as a table
        column, so this helper retrieves it on-demand for the age
        filter to use as a fallback.

        Lookup strategy mirrors _fetch_full_summary_content:
          1. UUID path  → SELECT created_at FROM memories WHERE tracking_id = ?
          2. Integer fallback → SELECT created_at FROM memories WHERE id = ?
             (only attempted if memory_id parses as an int)

        Args:
            memory_id (str): The 'memory_id' or 'tracking_id' value from
                             a Qdrant chunk's metadata. Can be a UUID
                             string or numeric string. Empty/None
                             returns None.

        Returns:
            Optional[datetime.datetime]: Parsed timestamp from SQL, or
                                          None if no row found, no
                                          timestamp on the row, or any
                                          error during parse/SQL.
        """
        # Validate input — empty or None returns immediately
        if not memory_id:
            return None

        try:
            # Read-only — context manager handles connection cleanup
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()

                # ── Path 1: UUID via tracking_id (modern entries) ─────
                cursor.execute(
                    "SELECT created_at FROM memories WHERE tracking_id = ?",
                    (str(memory_id),)
                )
                row = cursor.fetchone()
                raw_ts = row[0] if row else None

                # ── Path 2: integer id PK (legacy entries) ────────────
                if not raw_ts:
                    try:
                        int_id = int(memory_id)
                        cursor.execute(
                            "SELECT created_at FROM memories WHERE id = ?",
                            (int_id,)
                        )
                        row = cursor.fetchone()
                        raw_ts = row[0] if row else None
                    except (ValueError, TypeError):
                        # memory_id wasn't numeric — UUID path already tried
                        pass

                if not raw_ts:
                    # Neither path found a timestamp — log at debug
                    logging.debug(
                        f"CREATED_AT_LOOKUP: No SQL row for memory_id={memory_id}"
                    )
                    return None

                # ── Parse the timestamp ───────────────────────────────
                # SQLite stores TIMESTAMP as text. Common forms:
                #   '2026-04-12 17:18:08'        (default)
                #   '2026-04-12T17:18:08.123456' (ISO with T)
                # Try both — fall through to None on failure.
                ts_str = str(raw_ts)
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                    try:
                        return datetime.datetime.strptime(ts_str, fmt)
                    except ValueError:
                        continue

                # Last-ditch: ISO format via fromisoformat
                try:
                    return datetime.datetime.fromisoformat(
                        ts_str.replace('Z', '').split('+')[0]
                    )
                except (ValueError, TypeError):
                    pass

                # All parse paths failed — give up cleanly
                logging.warning(
                    f"CREATED_AT_LOOKUP: Found row but could not parse "
                    f"timestamp '{ts_str}' for memory_id={memory_id}"
                )
                return None

        except sqlite3.Error as sql_err:
            logging.error(
                f"CREATED_AT_LOOKUP: SQL error for memory_id={memory_id}: {sql_err}"
            )
            return None
        except Exception as e:
            # Catch-all — never let a lookup failure break the search flow
            logging.error(
                f"CREATED_AT_LOOKUP: Unhandled error for memory_id={memory_id}: {e}",
                exc_info=True
            )
            return None
                
    def get_unified_token_count(self) -> tuple[int, int, float]:
        """
        Get current context window pressure - READ ONLY, no side effects on the field.
        Returns: (last_sent_prompt_tokens, max_tokens, percentage)

        BEHAVIOR (updated 2026-05-14):
        Reports the token count of the most recently sent prompt as a percentage
        of num_ctx — the actual context window pressure metric that determines
        when Ollama starts truncating older content from the prompt.
        
        Logging changed to DEBUG level (was INFO) so this method can be called
        freely from UI render paths without flooding the logs. The former 
        get_token_stats_readonly() has been removed — this method now serves both
        the canonical-read and the side-effect-free-read use cases.

        Percentage CAN exceed 100% if a single prompt exceeds num_ctx (e.g., a
        search returning more tokens than the window holds). When that happens,
        Ollama silently truncates older content from the prompt before the model
        sees it. The 100%+ tier of get_token_usage_warning() handles this as an
        OVERFLOW condition with appropriately urgent wording.
        """
        try:
            # Pull max tokens from config — no hardcoded fallback. If MODEL_PARAMS
            # is missing num_ctx the except branch returns safe zeros and logs the error.
            max_tokens = MODEL_PARAMS["num_ctx"]

            # Initialize tracking variable if needed (defensive — survives hot reload)
            if not hasattr(self, '_last_sent_prompt_tokens'):
                self._last_sent_prompt_tokens = 0

            # The displayed pressure value — most recent prompt's token count.
            # Read directly off the field with no transformation.
            last_sent = self._last_sent_prompt_tokens

            # Calculate overflow for diagnostic logging only (display path uses percentage)
            overflow = max(0, last_sent - max_tokens)
            if overflow > 0:
                logging.info(f"TOKEN_TRACKING: Last prompt exceeded window by {overflow:,} tokens (Ollama truncating)")

            # Calculate percentage (can exceed 100% in overflow state)
            percentage = (last_sent / max_tokens) * 100

            # DEBUG level (was INFO) — quiet enough for UI render-path calls.
            # TOKEN_READ tag retained for log-grep compatibility with existing tooling.
            logging.debug(f"TOKEN_READ: {last_sent:,}/{max_tokens:,} ({percentage:.2f}%)")

            return last_sent, max_tokens, percentage

        except KeyError as ke:
            # Specific catch for the new no-fallback config access
            logging.error(f"TOKEN_COUNT: MODEL_PARAMS missing required key: {ke}")
            return 0, 0, 0.0
        except Exception as e:
            # Defensive: never let a token-read failure break the chat flow
            logging.error(f"TOKEN_COUNT: Error - {e}", exc_info=True)
            return 0, 0, 0.0

    def get_token_statistics(self) -> dict:
        """
        Get comprehensive token statistics for detailed UI display.

        BEHAVIOR (updated 2026-05-14):
        Returns the most recently sent prompt's token count and its percentage
        of the context window. See get_unified_token_count() for full rationale.

        Returns:
            dict: {
                'last_sent_tokens':  Tokens in the most recently sent prompt,
                'max_tokens':        Context window limit (num_ctx),
                'percentage_active': Usage percentage (>100% means overflow),
                'overflow':          Tokens beyond limit (0 when within window)
            }
        """
        try:
            # Pull max tokens from config — no hardcoded fallback
            max_tokens = MODEL_PARAMS["num_ctx"]

            # Use get_unified_token_count() for the canonical read.
            # Logs at DEBUG level so this call chain is quiet enough for any caller.
            last_sent, _, percentage = self.get_unified_token_count()

            # Overflow is the only derived value worth exposing in the dict.
            overflow = max(0, last_sent - max_tokens)

            return {
                'last_sent_tokens': last_sent,
                'max_tokens': max_tokens,
                'percentage_active': percentage,
                'overflow': overflow
            }

        except KeyError as ke:
            # Specific catch for the new no-fallback config access
            logging.error(f"TOKEN_STATS: MODEL_PARAMS missing required key: {ke}")
            return {
                'last_sent_tokens': 0,
                'max_tokens': 0,
                'percentage_active': 0.0,
                'overflow': 0
            }
        except Exception as e:
            # Defensive fallback — never let a stat fetch break the chat flow
            logging.error(f"Error getting token statistics: {e}", exc_info=True)
            return {
                'last_sent_tokens': 0,
                'max_tokens': 0,
                'percentage_active': 0.0,
                'overflow': 0
            }
        
    
    def get_token_usage_warning(self) -> str:
        """
        Generate context-aware token usage warnings for the model.

        ARCHITECTURE NOTE (updated 2026-05-20):
        QWEN owns context housekeeping entirely. main.py's auto-summary trigger
        has been disabled. This method generates the only signal QWEN sees about
        her own context state — she chooses when to run [SUMMARIZE_CONVERSATION].
        The token-counter reset on summarization is handled defensively in
        deepseek.py's command handlers so the warning naturally clears whenever
        QWEN issues the command.

        Thresholds (percentage-based, dynamic against any num_ctx value):
        -   0-74%:  Silent — no warning needed
        -  75-94%:  Gentle advisory — QWEN's primary window to act autonomously
        -  95-99%:  Critical — strong nudge to act on the next turn
        - 100%+:    Overflow — Ollama is silently truncating older content

        Rationale for thresholds (revised 2026-05-20):
        The 75% gentle floor reflects empirical behavior from the 2026-05-09
        logs: single-turn jumps driven by large SEARCH results regularly
        traverse 15+ percentage points in one turn boundary, which means an
        85% floor gets skipped past on the way to OVERFLOW. 75% on a 65K
        window gives ~16K tokens of runway before OVERFLOW — roughly one
        search-bloated turn of buffer. The 95% critical leaves ~3.3K tokens
        of headroom on 65K, a last-call signal. The 100% tier exists as a
        distinct "you've already overflowed" signal so QWEN can distinguish
        "act soon" from "act NOW, content is being lost."

        Thresholds auto-scale with MODEL_PARAMS['num_ctx']:
            65K  → gentle at ~49,152 tokens, critical at ~62,259 tokens
            128K → gentle at ~96,000 tokens, critical at ~121,600 tokens

        UI color tiers in main.py mirror these thresholds (yellow at 75%,
        orange at 85%, red at 95%, darkred at 100%+) so the visual cue and
        the textual warning align at every breakpoint.

        Returns:
            str: Warning message injected into prompt, empty string if no warning needed
        """
        try:
            # Pull current usage from the unified statistics method
            # (which now reports honest cumulative %, no cap)
            stats = self.get_token_statistics()
            percentage = stats['percentage_active']
            current = stats['last_sent_tokens']
            max_tokens = stats['max_tokens']

            # Helper closure: emit the diagnostic log line and return the text.
            # Centralized so every fire path produces the same TOKEN_WARNING_FIRED
            # record — useful later for confirming warnings actually reached QWEN.
            def _emit(tier: str, text: str) -> str:
                logging.info(
                    f"TOKEN_WARNING_FIRED: tier={tier} pct={percentage:.1f}% "
                    f"len={len(text)} chars"
                )
                return text

            # ── 100%+ : OVERFLOW — content being lost RIGHT NOW ──────────────
            # Distinct urgency from 95% critical: this isn't "near the edge,"
            # it's "past the edge." Ollama silently truncates older content
            # when the prompt exceeds num_ctx, so older history is already
            # being dropped at the model's input layer.
            if percentage >= 100:
                return _emit("OVERFLOW", (
                    f"\n**🚨 OVERFLOW: Context window exceeded — {percentage:.1f}% "
                    f"({current:,}/{max_tokens:,} tokens). "
                    f"Ollama is silently truncating older content from this prompt. "
                    f"Run [SUMMARIZE_CONVERSATION] NOW to preserve what's left "
                    f"before more context is lost.**\n"
                ))

            # ── 95-99% : CRITICAL — strong urgency, but QWEN still decides ───
            # Fires every turn at 95%+ until QWEN runs [SUMMARIZE_CONVERSATION].
            # The reset inside deepseek's command handler clears this naturally.
            if percentage >= 95:
                return _emit("CRITICAL", (
                    f"\n**⚠️ CRITICAL: Context window at {percentage:.1f}% full "
                    f"({current:,}/{max_tokens:,} tokens). "
                    f"Run [SUMMARIZE_CONVERSATION] immediately to preserve this "
                    f"conversation before context is lost. "
                    f"This warning will clear once the summary is stored.**\n"
                ))

            # ── 75-94% : Gentle advisory — QWEN's primary window to act ──────
            # Twenty-percentage-point runway before critical, sized to absorb
            # a single search-bloated turn on a 65K window. Wording is firmer
            # than "consider" because the window past 75% is finite.
            if percentage >= 75:
                return _emit("GENTLE", (
                    f"\n📊 Context window at {percentage:.1f}% "
                    f"({current:,}/{max_tokens:,} tokens). "
                    f"Recommend running [SUMMARIZE_CONVERSATION] soon to preserve "
                    f"this conversation.\n"
                ))

            # ── Below 75% : Silent (no log entry — quiet path stays quiet) ───
            return ""

            # ═══════════════════════════════════════════════════════════════════
            # COMMENTED OUT: Original auto-summarization trigger (85% threshold)
            # Preserved for rollback. To re-enable: remove the return "" above,
            # uncomment this block, and remove the tiered blocks above.
            # ═══════════════════════════════════════════════════════════════════
            #
            # if percentage >= 85:
            #     # Ensure flag exists (also set in __init__ now via Bug #6 fix)
            #     if not hasattr(self, '_auto_summary_triggered'):
            #         self._auto_summary_triggered = False
            #
            #     if not self._auto_summary_triggered and percentage < 100:
            #         # Set flag to prevent repeated triggers across turns
            #         self._auto_summary_triggered = True
            #         return (
            #             f"\n🔄 AUTO-SUMMARIZATION INITIATED: You are at "
            #             f"{current:,}/{max_tokens:,} tokens ({percentage:.1f}%). "
            #             f"Your system is auto-summarizing the current conversation now "
            #             f"to maintain continuous context. The summary will be injected "
            #             f"into the current conversation so you may continue seamlessly.\n"
            #         )
            #
            #     # 100%+ means auto-summarization should have fired but didn't
            #     if percentage >= 100:
            #         return (
            #             f"\n⚠️ CRITICAL: Auto-summarization should have triggered at 85% "
            #             f"but failed. Please ask Ken to check the system logs. "
            #             f"You are currently at {current:,}/{max_tokens:,} tokens "
            #             f"({percentage:.1f}%).\n"
            #         )
            #
            # elif percentage >= 80:
            #     return (
            #         f"\n📊 CONTEXT AWARENESS [INTERNAL]: Your context window is at "
            #         f"{current:,}/{max_tokens:,} tokens ({percentage:.1f}%). "
            #         f"Auto-summarization will trigger at 85%. "
            #         f"Please be concise in your responses and avoid large STORE "
            #         f"operations until summarization occurs.\n"
            #     )
            # ═══════════════════════════════════════════════════════════════════

        except Exception as e:
            # Defensive: never let a warning failure break the prompt build.
            # Empty string returned silently keeps the conversation flowing.
            logging.error(f"Error generating token usage warning: {e}", exc_info=True)
            return ""

    def reset_token_counter_after_summary(self, keep_lifetime_stats=True):
        """
        Reset the pressure counter after conversation summarization.
        
        Behavior (updated 2026-05-14):
        ONLY resets _last_sent_prompt_tokens (current context window pressure).
        The _session_total_tokens_sent counter is INTENTIONALLY NOT TOUCHED —
        it's a true running total of session work and surviving the summary
        reset is the whole point of the new design.
        
        After reset, _last_sent_prompt_tokens is 0. The next prompt build + send
        cycle will set it to the new (post-summary) prompt's actual size, which
        will be much smaller because the conversation summary replaces the full
        message history in subsequent prompts.

        Args:
            keep_lifetime_stats (bool): Retained for backward compatibility with
                existing callers. No longer has any effect — the session counter
                always persists across summarizations under the new design.
        
        Returns:
            bool: Success status
        """
        try:
            # Log the pre-reset state for diagnostic clarity
            pre_reset_pressure = getattr(self, '_last_sent_prompt_tokens', 0)
            session_total = getattr(self, '_session_total_tokens_sent', 0)
            
            logging.info(
                f"TOKEN_RESET: Pre-reset state — "
                f"pressure: {pre_reset_pressure:,}, "
                f"session_total: {session_total:,} (preserved)"
            )
            
            # Reset ONLY the pressure counter — session total is preserved.
            # Rationale: pressure represents "what's currently in the window."
            # After summarization the conversation history collapses to a summary,
            # so the next prompt will be much smaller. The session counter, however,
            # represents true work done and shouldn't be affected by summarization.
            self._last_sent_prompt_tokens = 0
            self._last_prompt_tokens = 0
            
            # Clear dedup key so the next prompt (post-summary) is properly counted
            self._last_counted_prompt_text = None

            logging.info("TOKEN_RESET: Pressure counter reset to 0 — next prompt will set actual value")
            logging.info(f"TOKEN_RESET: Session total preserved at {session_total:,} tokens")
            
            return True
            
        except Exception as e:
            # Defensive: never let a counter-reset failure break the summarization flow
            logging.error(f"Error resetting token counter: {e}", exc_info=True)
            return False

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (chatbot.py cleanup pass).
    # Auto-summarization wrapper — actual flow goes through conversation_summary_manager.summarize_if_needed.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_trigger_auto_summarization(self) -> dict:
        """
        Automatically summarize conversation when token threshold is reached.
        Does NOT clear conversation - only resets token counter and injects summary.
        
        Returns:
            dict: {
                'success': bool,
                'summary': str (if successful),
                'error': str (if failed),
                'tokens_before': int,
                'tokens_after': int
            }
        """
        try:
            logging.info("🔄 AUTO_SUMMARIZATION: Starting automatic conversation summarization")
            
            # Get current token stats
            stats_before = self.get_token_statistics()
            tokens_before = stats_before['last_sent_tokens']
            
            # Step 1: Get conversation from Streamlit
            import streamlit as st
            if not hasattr(st, 'session_state') or 'messages' not in st.session_state:
                logging.error("AUTO_SUMMARIZATION: No conversation available in Streamlit")
                return {
                    'success': False,
                    'error': 'No conversation found',
                    'tokens_before': tokens_before,
                    'tokens_after': tokens_before
                }
            
            conversation = st.session_state.messages
            
            if len(conversation) < 10:
                logging.warning("AUTO_SUMMARIZATION: Conversation too short to summarize")
                return {
                    'success': False,
                    'error': 'Conversation too short',
                    'tokens_before': tokens_before,
                    'tokens_after': tokens_before
                }
            
            # Step 2: Generate and store summary
            logging.info(f"🔄 AUTO_SUMMARIZATION: Generating summary from {len(conversation)} messages")
            success = self._generate_and_store_conversation_summary(conversation)
            
            if not success:
                logging.error("AUTO_SUMMARIZATION: Failed to generate summary")
                return {
                    'success': False,
                    'error': 'Summary generation failed',
                    'tokens_before': tokens_before,
                    'tokens_after': tokens_before
                }
            
            # Step 3: Reset token counter (but keep conversation in UI)
            # This resets our ESTIMATE, not Ollama's actual sliding window
            self.reset_token_counter_after_summary(keep_lifetime_stats=True)
            
            # Step 4: Reset the auto-summary trigger flag for next cycle
            self._auto_summary_triggered = False
            
            # Get new token stats
            stats_after = self.get_token_statistics()
            tokens_after = stats_after['last_sent_tokens']
            
            logging.info(f"🔄 AUTO_SUMMARIZATION: Complete! Token estimate reset: {tokens_before:,} → {tokens_after:,}")
            logging.info(f"🔄 AUTO_SUMMARIZATION: Note - Ollama maintains its own sliding window independently")
            
            return {
                'success': True,
                'tokens_before': tokens_before,
                'tokens_after': tokens_after
            }
            
        except Exception as e:
            logging.error(f"AUTO_SUMMARIZATION: Error during auto-summarization: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'tokens_before': 0,
                'tokens_after': 0
            }
    
    def _identify_search_mode(self, user_input: str) -> tuple[str, str]:
        """
        Simplified search mode identification - just two modes: default and comprehensive.
        The model should decide which to use via its memory commands.
    
        Args:
            user_input (str): The user input to analyze
        
        Returns:
            tuple[str, str]: (search_mode, processed_input)
        """
        # Check for null or empty input
        if not user_input or not user_input.strip():
            return "default", ""  # Return empty string for empty input
    
        cleaned_input = user_input.lower()

        # Check for explicit comprehensive search requests
        comprehensive_prefixes = ("deep search:", "comprehensive search:", "search thoroughly", 
                                  "find all", "tell me everything about")
        for prefix in comprehensive_prefixes:
            if cleaned_input.startswith(prefix):
                return "comprehensive", user_input[len(prefix):].strip()

        # Default search mode for everything else - let the model decide when to use [COMPREHENSIVE_SEARCH]
        return "default", user_input

    def _gather_relevant_context(self, user_input: str) -> str:
        """
        Gather relevant context from memory based on the user input.

        Args:
            user_input (str): The user's input

        Returns:
            str: Formatted memory context
        """
        try:
            if not user_input or not isinstance(user_input, str) or not user_input.strip():
                logging.warning("Empty user_input provided to _gather_relevant_context")
                return ""
        
            # Determine search mode based on user input
            search_mode, processed_input = self._identify_search_mode(user_input)
        
            # Set search parameters based on mode
            if search_mode == "comprehensive":
                k = 20          # More results for comprehensive
                threshold = 0.30  # Lower threshold for better recall
            else:               # default — balanced results with moderate precision
                k = 5
                threshold = 0.35
        
            # Get relevant memories based on user input
            results = self.vector_db.search(
                query=processed_input,
                mode=search_mode,
                k=k
            )
        
            # Get conversation context if available
            conversation_context = ""
            if hasattr(self, 'conversation_manager'):
                conversation_context = self.conversation_manager.get_formatted_context()
        
            if not results:
                logging.info(f"No relevant memories found for: {processed_input[:50]}...")
                # Return conversation context even if no memory results found
                return conversation_context
        
            # Filter by threshold
            filtered_results = [r for r in results if r.get('similarity_score', 0) >= threshold]
        
            if not filtered_results:
                logging.info(f"No memories passed threshold {threshold} for: {processed_input[:50]}...")
                # Return conversation context even if no filtered results
                return conversation_context
        
            # Format memory context for inclusion in prompt
            formatted_context = []
            for i, result in enumerate(filtered_results[:5]):  # Limit to top 5
                memory_content = result.get('content', '')
                if not memory_content:
                    continue
                
                score = result.get('similarity_score', 0)
                memory_type = result.get('metadata', {}).get('type', 'general')
                source = result.get('metadata', {}).get('source', 'Unknown')
            
                # Assign type prefixes for better readability
                type_prefix = {
                    "important": "[Important Memory]",
                    "document": "[Document Memory]", 
                    "general": "[Memory]",
                    "conversation": "[Conversation Summary]"
                }.get(memory_type, "[Memory]")
            
                formatted_context.append(
                    f"{type_prefix} {memory_content} (Source: {source}, Relevance: {score:.2f})"
                )
        
            # Log the automatic search
            logging.info(f"Automatic search for '{processed_input[:30]}...' found {len(filtered_results)} relevant memories")
       
            # Increment retrieve counter for auto-search
            if hasattr(self, 'deepseek_enhancer'):
                self.deepseek_enhancer.lifetime_counters.increment('retrieve')
                # Update session counters - safely check for streamlit
                try:
                    # Try to import streamlit if it's available
                    import streamlit as st_local
                    if hasattr(st_local, 'session_state') and 'memory_command_counts' in st_local.session_state:
                        if 'retrieve' not in st_local.session_state.memory_command_counts:
                            st_local.session_state.memory_command_counts['retrieve'] = 0
                        st_local.session_state.memory_command_counts['retrieve'] += 1
                except (ImportError, ModuleNotFoundError):
                    # Streamlit not available, skip counter update
                    logging.info("Streamlit not available, skipping session counter update")
                    
            # Return formatted context string
            if formatted_context:
                memory_str = "\n\n".join(formatted_context)
                logging.info(f"Auto-retrieved {len(formatted_context)} memories for context")
                
                # Combine conversation context with memory results
                if conversation_context:
                    return conversation_context + memory_str
                else:
                    return memory_str
            else:
                # Return conversation context if no memory context
                return conversation_context
            
        except Exception as e:
            logging.error(f"Error gathering relevant context: {e}")
            return ""

    def search_ai_learned_content(self, topic: str) -> str:
        """
        Search for content learned through AI-driven web processing.
        
        Filters results to only return memories tagged as web_learning content
        that was processed by the autonomous AI-driven selection pipeline.
        This distinguishes autonomously acquired knowledge from conversation
        memories or user-provided facts.
        
        Args:
            topic (str): The topic to search for in AI-learned content
            
        Returns:
            str: Formatted search results or a message if nothing found
        """
        try:
            # Validate input
            if not topic or not isinstance(topic, str) or not topic.strip():
                logging.warning("search_ai_learned_content called with empty topic")
                return "No topic provided for AI-learned content search."
            
            logging.info(f"Searching AI-learned web content for topic: '{topic[:50]}'")
            
            # Search with metadata filters — restricts to autonomous web learning only
            # Excludes conversation memories, user-told facts, and manually stored content
            results = self.vector_db.search(
                query=topic,
                mode="comprehensive",
                k=10,
                metadata_filters={
                    "type": "web_learning",
                    "processed_by": "ai_driven_selection"
                }
            )
            
            if not results:
                logging.info(f"No AI-learned content found for topic: '{topic[:50]}'")
                return f"No AI-learned web content found for '{topic}'"
            
            # Format results for display
            formatted_results = [f"\n**AI-LEARNED WEB CONTENT ABOUT '{topic.upper()}'**\n"]
            
            for i, result in enumerate(results, 1):
                content = result.get('content', '')
                source = result.get('metadata', {}).get('source', 'Unknown')
                score = result.get('similarity_score', 0)
                
                formatted_results.append(f"**[{i}]** (Score: {score:.2f}) Source: {source}")
                formatted_results.append(f"{content[:300]}...")
                formatted_results.append("")
            
            logging.info(f"Found {len(results)} AI-learned content results for '{topic[:50]}'")
            return "\n".join(formatted_results)
        
        except Exception as e:
            logging.error(f"Error searching AI-learned content for '{topic[:50]}': {e}", exc_info=True)
            return f"Error searching AI-learned content: {str(e)}"
          
                    
    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (chatbot.py cleanup pass).
    # Legacy: extracts <model_reasoning> tags, but Qwen3 uses <think> tags now — pattern never matches.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_format_response_with_reasoning(self, response):
        """Extract and format model reasoning for display."""
        reasoning_pattern = re.compile(r'<model_reasoning>(.*?)</model_reasoning>', re.DOTALL)
        reasoning_match = reasoning_pattern.search(response)
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            final_response = response.replace(reasoning_match.group(0), '').strip()
            
            # Format with clear separation
            return f"MODEL REASONING:\n{reasoning}\n\nFINAL RESPONSE:\n{final_response}"
        
        return response  # If no reasoning found, return as-is
        
    def _generate_and_store_conversation_summary(self, conversation: List[Dict]) -> bool:
        """Generate and store a conversation summary using the conversation summary manager."""
        try:
            # Skip if conversation is too short
            if len(conversation) < 10:
                logging.info("Conversation too short for summarization")
                return False
            
            # Use the conversation_summary_manager to generate and store the summary
            if not hasattr(self, 'conversation_summary_manager'):
                logging.error("Conversation summary manager not initialized")
                return False
            
            # Generate summary using the conversation summary manager
            summary = self.conversation_summary_manager.generate_summary(conversation)
            
            if not summary:
                logging.warning("Failed to generate conversation summary")
                return False
            
            logging.info(f"Generated conversation summary: {len(summary)} characters")
            
            # Store using conversation state manager
            state_success = False
            if hasattr(self, 'conversation_manager'):
                state_success = self.conversation_manager.update_summary(
                    summary, 
                    memory_db=self.memory_db
                )
                logging.info(f"Stored summary in conversation_manager: {state_success}")
            
            return state_success
            
        except Exception as e:
            logging.error(f"Error generating and storing summary: {e}", exc_info=True)
            return False
            
    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (chatbot.py cleanup pass).
    # BROKEN: calls self.get_command_usage_summary() which does not exist — would AttributeError if reached.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_log_command_usage_summary(self):
        """Log the current command usage summary."""
        try:
            summary = self.get_command_usage_summary()
            logging.info("=== COMMAND USAGE SUMMARY ===")
            logging.info(f"Session ID: {summary.get('session_id', 'Unknown')}")
            
            if 'lifetime_counters' in summary:
                total_lifetime = summary['lifetime_counters'].get('total', 0)
                logging.info(f"Total Lifetime Commands: {total_lifetime}")
                
                # Log top 5 most used commands
                lifetime_sorted = sorted(
                    [(k, v) for k, v in summary['lifetime_counters'].items() if k != 'total' and v > 0],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                if lifetime_sorted:
                    logging.info("Top Lifetime Commands:")
                    for cmd, count in lifetime_sorted:
                        logging.info(f"  {cmd}: {count}")
            
            if 'session_counters' in summary:
                total_session = sum([v for v in summary['session_counters'].values() if isinstance(v, int)])
                logging.info(f"Total Session Commands: {total_session}")
                
                # Log session commands with counts > 0
                session_active = [(k, v) for k, v in summary['session_counters'].items() if v > 0]
                if session_active:
                    logging.info("Active Session Commands:")
                    for cmd, count in session_active:
                        logging.info(f"  {cmd}: {count}")
            
            logging.info("=== END COMMAND USAGE SUMMARY ===")
            
        except Exception as e:
            logging.error(f"Error logging command usage summary: {e}")

    def check_and_repair_database_sync(self):
        """Check for and repair inconsistencies between MemoryDB and VectorDB."""
        try:
            logging.info("Starting database synchronization check...")
        
            # Import QDRANT_COLLECTION_NAME directly from config
            from config import QDRANT_COLLECTION_NAME
    
            # Track statistics for reporting
            stats = {
                "memory_db_total": 0,
                "vector_db_total": 0,
                "memory_missing_in_vector": 0,
                "vector_missing_in_memory": 0,
                "repairs_attempted": 0,
                "repairs_succeeded": 0
            }
    
            # Step 1: Get count of items in both databases for reference
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories")
                stats["memory_db_total"] = cursor.fetchone()[0]
    
            # Get vector DB count - handle properly with error checking
            try:
                # Use QDRANT_COLLECTION_NAME directly instead of self.vector_db.client.collection_name
                vector_count = self.vector_db.client.count(
                    collection_name=QDRANT_COLLECTION_NAME,
                    count_filter=None  # Count all points
                )
                stats["vector_db_total"] = vector_count.count
            except Exception as e:
                logging.error(f"Error getting vector count: {e}")
                stats["vector_db_total"] = "unknown"
    
            # Step 2: Check MemoryDB items in VectorDB using tracking_id (more reliable than content)
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                # Use limit and pagination to handle large databases
                batch_size = 50
                offset = 0
        
                while True:
                    cursor.execute("""
                        SELECT content, memory_type, tracking_id, id
                        FROM memories 
                        ORDER BY id
                        LIMIT ? OFFSET ?
                    """, (batch_size, offset))
            
                    memories = cursor.fetchall()
                    if not memories:
                        break  # No more records
            
                    for content, memory_type, tracking_id, memory_id in memories:
                        # Skip if content is empty
                        if not content or not content.strip():
                            continue
                
                        # First try to find by tracking_id if available
                        memory_exists = False
                        if tracking_id:
                            # Search for exact tracking_id match in vector_db
                            try:
                                # Use vector search with filter to find by tracking_id
                                results = self.vector_db.search(
                                    query=content[:50],  # Use partial content as query
                                    mode="selective",
                                    k=3
                                )
                        
                                # Check if any result has the matching tracking_id
                                for result in results:
                                    result_tracking_id = result.get('metadata', {}).get('memory_id')
                                    if result_tracking_id == tracking_id:
                                        memory_exists = True
                                        break
                            except Exception as e:
                                logging.warning(f"Error searching by tracking_id: {e}")
                
                        # If no match by tracking_id, try content-based search
                        if not memory_exists:
                            results = self.vector_db.search(
                                query=content,
                                mode="selective",
                                k=3
                            )
                    
                            for result in results:
                                # Check for exact content match or very high similarity
                                if (result['content'] == content or 
                                    (result.get('similarity_score', 0) > 0.95 and content in result['content'])):
                                    memory_exists = True
                                    break
                
                        # If memory doesn't exist in VectorDB, add it
                        if not memory_exists:
                            logging.warning(f"Memory found in MemoryDB but missing in VectorDB (ID: {memory_id})")
                            stats["memory_missing_in_vector"] += 1
                    
                            # Add to VectorDB with original metadata
                            try:
                                # Get all metadata from the memory record
                                cursor.execute("""
                                    SELECT source, tags, weight, created_at
                                    FROM memories WHERE id = ?
                                """, (memory_id,))
                                meta_row = cursor.fetchone()
                        
                                if meta_row:
                                    source, tags, weight, created_at = meta_row
                                    metadata = {
                                        "metadata.type": memory_type, 
                                        "metadata.source": source or "sync_repair", 
                                        "metadata.memory_id": tracking_id or str(uuid.uuid4()),
                                        "metadata.tags": tags
                                    }
                            
                                    vector_success = self.vector_db.add_text(
                                        text=content, 
                                        metadata=metadata
                                    )
                            
                                    if vector_success:
                                        stats["repairs_succeeded"] += 1
                                        logging.info(f"Successfully repaired missing vector for memory ID {memory_id}")
                                    else:
                                        logging.error(f"Failed to add missing vector for memory ID {memory_id}")
                        
                                stats["repairs_attempted"] += 1
                            except Exception as repair_error:
                                logging.error(f"Error repairing memory: {repair_error}")
            
                    # Move to next batch
                    offset += batch_size
                    logging.info(f"Processed {offset} memories from MemoryDB")
    
            # Step 3: Sample check for orphaned vectors (in VectorDB but not in MemoryDB)
            try:
                # We'll sample a portion of vectors to keep this efficient
                vector_sample_limit = 100
        
                # Get a sample of vectors using search with a broad query
                sample_results = self.vector_db.search(
                    query="memory data information",  # Generic terms to get diverse results
                    mode="comprehensive",
                    k=vector_sample_limit
                )
        
                if sample_results:
                    # Open ONE connection for all orphan checks — not one per result
                    # Bug #7 fix: was opening sqlite3.connect() inside the loop (up to 100x)
                    with sqlite3.connect(self.memory_db.db_path) as conn:
                        cursor = conn.cursor()
                        
                        for result in sample_results:
                            content = result.get('content', '')
                            if not content:
                                continue
                            
                            # Get memory_id if available
                            memory_id = result.get('metadata', {}).get('memory_id')
                            
                            # Check if exists in MemoryDB — reset flag each iteration
                            memory_exists = False
                            
                            # First try by memory_id if available — fastest lookup
                            if memory_id:
                                cursor.execute("""
                                    SELECT COUNT(*) FROM memories 
                                    WHERE tracking_id = ?
                                """, (memory_id,))
                                count = cursor.fetchone()[0]
                                if count > 0:
                                    memory_exists = True
                            
                            # If not found by ID, fall back to content match
                            if not memory_exists:
                                cursor.execute("""
                                    SELECT COUNT(*) FROM memories 
                                    WHERE content = ?
                                """, (content,))
                                count = cursor.fetchone()[0]
                                if count > 0:
                                    memory_exists = True
                            
                            # If vector exists but memory doesn't, report it
                            if not memory_exists:
                                stats["vector_missing_in_memory"] += 1
                                logging.warning(
                                    f"Vector found in VectorDB but missing in MemoryDB: "
                                    f"{content[:100]}..."
                                )
                    
                            
            except Exception as e:
                logging.error(f"Error checking for orphaned vectors: {e}")
    
            # Step 4: Generate final report
            logging.info(f"Database sync check completed. Stats: {stats}")
    
            result = {
                "found": stats["memory_db_total"],
                "repaired": stats["repairs_succeeded"],
                "missing_in_vector": stats["memory_missing_in_vector"],
                "missing_in_memory": stats["vector_missing_in_memory"],
                "stats": stats
            }
    
            # Store the sync report in a dedicated file for tracking
            try:
                sync_report_path = os.path.join(os.path.dirname(self.memory_db.db_path), "sync_reports.json")
        
                existing_reports = []
                if os.path.exists(sync_report_path):
                    try:
                        with open(sync_report_path, 'r') as f:
                            existing_reports = json.load(f)
                    except:
                        existing_reports = []
        
                # Add timestamp to this report
                report_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "report": result
                }
        
                # Keep only the last 10 reports
                existing_reports.append(report_entry)
                if len(existing_reports) > 10:
                    existing_reports = existing_reports[-10:]
        
                with open(sync_report_path, 'w') as f:
                    json.dump(existing_reports, f, indent=2)
            except Exception as e:
                logging.error(f"Error storing sync report: {e}")
    
            return result
    
        except Exception as e:
            logging.error(f"Error in database sync check: {e}")
            return {"error": str(e), "found": 0, "repaired": 0}
        
           
    def check_due_reminders(self):
        """
        Check for reminders that are due by delegating to the reminder_manager.
        
        Returns:
            list: List of due reminders
        """
        try:
            if hasattr(self, 'reminder_manager') and self.reminder_manager:
                # Delegate to the existing method in the reminder manager
                return self.reminder_manager.get_due_reminders()
            else:
                logging.warning("ReminderManager not available, cannot check due reminders")
                return []
        except Exception as e:
            logging.error(f"Error checking due reminders: {e}")
            return []
    
    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (chatbot.py cleanup pass).
    # Memory correction utility, fully implemented but never wired into the command dispatcher.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_correct_memory(self, original_content: str, new_content: str) -> str:
        try:
            # First, check if the original memory exists or something similar exists
            logging.info(f"Checking if memory exists: '{original_content[:50]}...'")
            memory_exists = self.memory_db.contains(original_content)
            
            if not memory_exists:
                logging.warning(f"Exact memory not found in memory_db: '{original_content[:50]}...'")
                # Try to find similar content in memory_db instead of just failing
                similar_memories = self.memory_db.search_similar(original_content)
                if similar_memories and len(similar_memories) > 0:
                    original_content = similar_memories[0]['content']
                    logging.info(f"Found similar memory in memory_db: '{original_content[:50]}...'")
                    memory_exists = True
                    
            if not memory_exists:
                return f"Memory not found: '{original_content[:50]}...'"
            
            # Get the original memory metadata using search_with_ids
            logging.info(f"Finding memory metadata for: '{original_content[:50]}...'")
            memory = None
            
            # Use search_with_ids to get proper vector IDs
            original_memories = self.vector_db.search_with_ids(original_content, mode="selective", k=5)
            logging.info(f"Found {len(original_memories)} potential matches in vector_db")

            # First try exact match
            for mem in original_memories:
                logging.info(f"Comparing: '{mem['content'][:50]}...' to '{original_content[:50]}...'")
                if mem['content'] == original_content:
                    memory = mem
                    logging.info("Found exact match")
                    break
            
            # If no exact match, try best similar match above threshold
            if not memory and original_memories:
                best_match = None
                highest_score = 0
                for mem in original_memories:
                    if mem['similarity_score'] >= 0.40 and mem['similarity_score'] > highest_score:
                        best_match = mem
                        highest_score = mem['similarity_score']
                
                if best_match:
                    memory = best_match
                    original_content = best_match['content']  # Update original_content to what we found
                    logging.info(f"Found similar match with score {highest_score}: '{original_content[:50]}...'")
                    
            if not memory:
                logging.warning(f"No suitable match found in vector_db for: '{original_content[:50]}...'")
                return f"Unable to retrieve full metadata for memory: '{original_content[:50]}...'"

            # Get metadata from the original memory BEFORE deletion
            metadata = memory.get('metadata', {})
            memory_type = metadata.get('type', 'general')
            source = metadata.get('source', '')
            confidence = metadata.get('confidence', 0.5)

            # Add note about correction
            if source:
                source = f"{source} (corrected)"
            else:
                source = "corrected"
        
            # Update metadata
            metadata["source"] = source

            # CRITICAL FIX: Use coordinated deletion instead of separate calls
            logging.info(f"Using coordinated deletion for: '{original_content[:50]}...'")
            deletion_success = self.delete_memory_with_coordination(original_content)
            
            if not deletion_success:
                logging.error(f"Coordinated deletion failed for: '{original_content[:50]}...'")
                return f"Error deleting original memory: '{original_content[:50]}...'"
            
            logging.info(f"Coordinated deletion successful, now storing corrected content")
            
            # Store the corrected memory using transaction coordination
            success, memory_id = self.store_memory_with_transaction(
                content=new_content,
                memory_type=memory_type,
                metadata=metadata,
                confidence=confidence
            )

            if success:
                logging.info(f"Successfully corrected memory with new ID: {memory_id}")
                return f"Successfully corrected memory:\nOriginal: '{original_content[:100]}...'\nCorrected: '{new_content[:100]}...'"
            else:
                logging.error(f"Failed to store corrected memory, but original was already deleted!")
                # This is a critical state - original is deleted but new content failed to store
                return f"CRITICAL ERROR: Original memory deleted but failed to store corrected version: '{new_content[:50]}...'"
        
        except Exception as e:
            logging.error(f"Error correcting memory: {e}", exc_info=True)
            return f"Error correcting memory: {str(e)}"
        
    def delete_memory_with_coordination(self, content: str) -> bool:
        """
        Delete a memory ensuring both SQL and Vector databases stay synchronized.
        Uses transaction-like behavior with rollback capability.
        """
        try:
            logging.info(f"DELETE_COORDINATION: Starting synchronized deletion for: '{content[:100]}...'")
            
            # Step 1: Find the memory using vector search WITH IDs
            vector_results = self.vector_db.search_with_ids(  # Use the new method
                query=content,
                mode="comprehensive",
                k=15
            )
            
            if not vector_results:
                logging.warning("DELETE_COORDINATION: No vector results found")
                return False
            
            # Step 2: Find best match and extract identifiers
            best_match = self._find_best_vector_match(content, vector_results)
            if not best_match:
                logging.warning("DELETE_COORDINATION: No suitable vector match found")
                return False
            
            vector_id = best_match.get('id')  # This should now have a value!
            vector_content = best_match.get('content', '')
            metadata = best_match.get('metadata', {})
            tracking_id = metadata.get('tracking_id') or metadata.get('memory_id')
            
            logging.info(f"DELETE_COORDINATION: Best match - Vector ID: {vector_id}, Tracking ID: {tracking_id}")

            
            # Step 3: Backup information for potential rollback
            # DEAD CODE TEST 2026-05-17: backup_info built but never used; _rollback_sql_deletion is never called with this dict. SQL rollback uses sql_memory_backup below instead. (ruff F841)
            # backup_info = {
            #     'vector_id': vector_id,
            #     'vector_content': vector_content,
            #     'tracking_id': tracking_id,
            #     'metadata': metadata
            # }
            
            # Step 4: Delete from SQL database (with backup for rollback)
            sql_memory_backup = None
            sql_success = False

            # Find and backup the SQL memory before deletion
            sql_memory_backup = self._backup_sql_memory(tracking_id, vector_content, content)

            if sql_memory_backup:
                # Found SQL record - proceed with normal deletion
                sql_success = self._delete_sql_with_identifiers(tracking_id, vector_content, content)
                logging.info(f"DELETE_COORDINATION: SQL deletion: {'success' if sql_success else 'failed'}")
            else:
                # No SQL record found - this might be an orphaned vector entry
                logging.warning("DELETE_COORDINATION: No SQL record found - treating as orphaned vector entry")
                sql_success = True  # Consider SQL "deletion" successful since there's nothing to delete
                
                # Create a dummy backup for consistency
                sql_memory_backup = {
                    'orphaned_entry': True,
                    'tracking_id': tracking_id,
                    'vector_content': vector_content
                }
            
            # Step 5: If SQL deletion failed, abort
            if not sql_success:
                logging.error("DELETE_COORDINATION: SQL deletion failed - aborting")
                return False

            # Step 6: Delete from Vector database
            vector_success = self._delete_vector_with_identifiers(vector_id, tracking_id, vector_content)
            logging.info(f"DELETE_COORDINATION: Vector deletion: {'success' if vector_success else 'failed'}")

            # Step 7: Handle rollback if vector deletion failed
            if not vector_success:
                # Only attempt SQL rollback if we actually deleted something from SQL
                if not sql_memory_backup.get('orphaned_entry', False):
                    logging.warning("DELETE_COORDINATION: Vector deletion failed - attempting SQL rollback")
                    rollback_success = self._rollback_sql_deletion(sql_memory_backup)
                    
                    if rollback_success:
                        logging.info("DELETE_COORDINATION: Successfully rolled back SQL deletion")
                    else:
                        logging.error("DELETE_COORDINATION: CRITICAL - SQL rollback failed! Databases are out of sync!")
                else:
                    logging.info("DELETE_COORDINATION: Vector deletion failed but no SQL rollback needed (orphaned entry)")
                
                return False

            # Step 8: Success - both databases updated (or orphaned vector entry cleaned up)
            if sql_memory_backup.get('orphaned_entry', False):
                logging.info("DELETE_COORDINATION: Successfully cleaned up orphaned vector entry")
            else:
                logging.info("DELETE_COORDINATION: Successfully deleted from both SQL and Vector databases")

            return True
            
        except Exception as e:
            logging.error(f"DELETE_COORDINATION: Exception during deletion: {e}", exc_info=True)
            return False
        
    def delete_memory_by_id(self, memory_id: str) -> tuple[bool, str]:
        """
        Delete a memory by its memory_id (tracking_id) with full SQL+Vector
        coordination and rollback semantics.
        
        Mirrors delete_memory_with_coordination() but skips the content-based
        vector search step — the tracking_id is provided directly. This is
        the clean path for deleting long-form memories like image_analysis
        and document_summary where text-based matching is unreliable
        (long content, multi-chunk, embedded newlines, display-formatting
        contamination).
        
        Transaction order (matches existing delete_memory_with_coordination):
            1. Validate memory_id format (UUID-shaped)
            2. Backup SQL row (for potential rollback) via _backup_sql_memory
            3. Delete SQL row via _delete_sql_with_identifiers
            4. Delete ALL associated Qdrant points via vector_db.delete_by_memory_id
            5. If vector delete fails, rollback SQL via _rollback_sql_deletion
        
        Args:
            memory_id (str): The UUID tracking_id of the memory to delete.
        
        Returns:
            tuple[bool, str]: (success, message)
                On success: message includes count of vector points deleted.
                On failure: message describes which step failed.
        """
        import re
        
        # ============================================================
        # STEP 1: Validate memory_id format (must be UUID-shaped)
        # ============================================================
        # Rejects malformed input early so we don't waste DB calls on
        # obviously bad data. Pattern matches standard 8-4-4-4-12 hex UUID.
        if not memory_id or not isinstance(memory_id, str):
            logging.error(
                f"DELETE_BY_ID: Invalid memory_id (None or non-string): {memory_id}"
            )
            return False, "Invalid memory ID: must be a non-empty UUID string"
        
        memory_id = memory_id.strip()
        
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        if not re.match(uuid_pattern, memory_id):
            logging.error(
                f"DELETE_BY_ID: Memory ID is not a valid UUID: '{memory_id[:60]}'"
            )
            return False, f"Invalid memory ID format: '{memory_id[:40]}' is not a valid UUID"
        
        logging.info(f"DELETE_BY_ID: Starting ID-based deletion for memory_id={memory_id}")
        
        try:
            # ============================================================
            # STEP 2: Backup SQL row for potential rollback
            # ============================================================
            # Reuse the existing helper. Empty vector_content/original_content
            # args are safe — _backup_sql_memory tries tracking_id FIRST
            # (line 3837-3852) and only falls back to content matching
            # when tracking_id lookup fails. The fallback loop already
            # skips empty strings (`if not content_to_try: continue`).
            sql_memory_backup = self._backup_sql_memory(
                tracking_id=memory_id,
                vector_content="",
                original_content=""
            )
            
            if not sql_memory_backup:
                # No SQL row found. Could mean:
                #   (a) ID doesn't exist anywhere
                #   (b) SQL was already cleaned but vector points remain (orphan)
                # Attempt vector cleanup either way — it's idempotent.
                logging.warning(
                    f"DELETE_BY_ID: No SQL row found for memory_id={memory_id} — "
                    f"attempting orphan cleanup in vector DB"
                )
                
                vector_success, count = self.vector_db.delete_by_memory_id(memory_id)
                
                if vector_success and count > 0:
                    logging.info(
                        f"DELETE_BY_ID: Cleaned up {count} orphaned vector point(s) "
                        f"for memory_id={memory_id}"
                    )
                    return True, (
                        f"Cleaned up {count} orphaned vector point(s) "
                        f"for ID {memory_id} (no SQL row existed)"
                    )
                else:
                    logging.info(
                        f"DELETE_BY_ID: No memory found anywhere for memory_id={memory_id}"
                    )
                    return False, f"No memory found with ID {memory_id}"
            
            logging.info(
                f"DELETE_BY_ID: SQL backup successful for memory_id={memory_id} "
                f"(row_id={sql_memory_backup.get('id')}, "
                f"memory_type={sql_memory_backup.get('memory_type')})"
            )
            
            # ============================================================
            # STEP 3: Delete from SQL database
            # ============================================================
            # Reuse existing _delete_sql_with_identifiers — already tries
            # tracking_id FIRST (line 3911: WHERE tracking_id = ?).
            sql_success = self._delete_sql_with_identifiers(
                tracking_id=memory_id,
                vector_content="",
                original_content=""
            )
            
            if not sql_success:
                logging.error(
                    f"DELETE_BY_ID: SQL deletion failed for memory_id={memory_id}"
                )
                return False, f"Failed to delete SQL row for memory ID {memory_id}"
            
            logging.info(f"DELETE_BY_ID: SQL deletion succeeded for memory_id={memory_id}")
            
            # ============================================================
            # STEP 4: Delete ALL Qdrant points sharing this memory_id
            # ============================================================
            # Uses the new vector_db.delete_by_memory_id helper which:
            #   - Scrolls Qdrant by metadata.tracking_id (indexed)
            #   - Handles chunked memories (multiple points per memory_id)
            #   - Returns (success, count_deleted)
            vector_success, count_deleted = self.vector_db.delete_by_memory_id(memory_id)
            
            if not vector_success:
                # Vector deletion failed — rollback SQL to keep DBs in sync
                logging.warning(
                    f"DELETE_BY_ID: Vector deletion failed for memory_id={memory_id} — "
                    f"attempting SQL rollback"
                )
                
                rollback_success = self._rollback_sql_deletion(sql_memory_backup)
                
                if rollback_success:
                    logging.info(
                        f"DELETE_BY_ID: SQL rollback succeeded for memory_id={memory_id}"
                    )
                    return False, (
                        f"Vector deletion failed for memory ID {memory_id}. "
                        f"SQL deletion was rolled back — database state unchanged."
                    )
                else:
                    logging.error(
                        f"DELETE_BY_ID: CRITICAL — SQL rollback FAILED after vector "
                        f"delete failure for memory_id={memory_id}. "
                        f"Databases may be out of sync!"
                    )
                    return False, (
                        f"CRITICAL: Vector deletion failed AND SQL rollback failed "
                        f"for memory ID {memory_id}. Databases may be out of sync — "
                        f"manual investigation required."
                    )
            
            # ============================================================
            # STEP 5: Success — both databases updated
            # ============================================================
            logging.info(
                f"DELETE_BY_ID: Successfully deleted memory_id={memory_id} "
                f"(SQL row + {count_deleted} vector point(s))"
            )
            return True, (
                f"Successfully deleted memory ID {memory_id} "
                f"(SQL row + {count_deleted} vector chunk(s))"
            )
        
        except Exception as e:
            logging.error(
                f"DELETE_BY_ID: Unexpected exception for memory_id={memory_id}: {e}",
                exc_info=True
            )
            return False, f"Error deleting memory ID {memory_id}: {str(e)}"

    def _find_best_vector_match(self, content: str, vector_results: list = None) -> dict:
        """Find the best matching vector result, using direct search if needed."""
        best_match = None
        best_score = 0
        
        # If no vector_results provided, do a direct search with IDs
        if not vector_results:
            try:
                vector_results = self.vector_db.search_with_ids(
                    query=content,
                    mode="comprehensive",
                    k=10
                )
            except Exception as e:
                logging.error(f"Error in direct search with IDs: {e}")
                return None
        
        for result in vector_results:
            score = result.get('similarity_score', 0)
            result_content = result.get('content', '')
            
            # Calculate word overlap
            content_words = set(content.lower().split())
            result_words = set(result_content.lower().split())
            
            if content_words and result_words:
                word_overlap = len(content_words.intersection(result_words)) / len(content_words)
                combined_score = (score * 0.7) + (word_overlap * 0.3)
                
                is_good_match = (
                    score >= 0.45 or
                    word_overlap >= 0.40 or
                    content.lower() in result_content.lower() or
                    result_content.lower() in content.lower()
                )
                
                if is_good_match and combined_score > best_score:
                    best_match = result
                    best_score = combined_score
                    logging.info(f"BEST_MATCH: Score {score:.2f}, Overlap {word_overlap:.2f}, Combined {combined_score:.2f}, ID: {result.get('id', 'None')}")
        
        return best_match

    def _backup_sql_memory(self, tracking_id: str, vector_content: str, original_content: str) -> dict:
        """Backup SQL memory before deletion, handling cases where SQL record might not exist."""
        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Try to find by tracking_id first
                if tracking_id:
                    cursor.execute("""
                        SELECT id, content, memory_type, source, weight, access_count, 
                            created_at, last_accessed, tags, metadata, tracking_id
                        FROM memories WHERE tracking_id = ?
                    """, (tracking_id,))
                    result = cursor.fetchone()
                    
                    if result:
                        logging.info(f"BACKUP_SQL: Found memory by tracking_id: {tracking_id}")
                        return {
                            'id': result[0], 'content': result[1], 'memory_type': result[2],
                            'source': result[3], 'weight': result[4], 'access_count': result[5],
                            'created_at': result[6], 'last_accessed': result[7], 'tags': result[8],
                            'metadata': result[9], 'tracking_id': result[10]
                        }
                    else:
                        logging.warning(f"BACKUP_SQL: No SQL record found for tracking_id: {tracking_id}")
                
                # Try to find by content matching (fuzzy approach)
                for content_to_try in [vector_content, original_content]:
                    if not content_to_try:
                        continue
                        
                    # Try exact match first
                    cursor.execute("""
                        SELECT id, content, memory_type, source, weight, access_count, 
                            created_at, last_accessed, tags, metadata, tracking_id
                        FROM memories WHERE content = ?
                    """, (content_to_try,))
                    result = cursor.fetchone()
                    
                    if result:
                        logging.info(f"BACKUP_SQL: Found memory by exact content match")
                        return {
                            'id': result[0], 'content': result[1], 'memory_type': result[2],
                            'source': result[3], 'weight': result[4], 'access_count': result[5],
                            'created_at': result[6], 'last_accessed': result[7], 'tags': result[8],
                            'metadata': result[9], 'tracking_id': result[10]
                        }
                    
                    # Try partial match
                    cursor.execute("""
                        SELECT id, content, memory_type, source, weight, access_count, 
                            created_at, last_accessed, tags, metadata, tracking_id
                        FROM memories WHERE content LIKE ?
                    """, (f'%{content_to_try}%',))
                    result = cursor.fetchone()
                    
                    if result:
                        logging.info(f"BACKUP_SQL: Found memory by partial content match")
                        return {
                            'id': result[0], 'content': result[1], 'memory_type': result[2],
                            'source': result[3], 'weight': result[4], 'access_count': result[5],
                            'created_at': result[6], 'last_accessed': result[7], 'tags': result[8],
                            'metadata': result[9], 'tracking_id': result[10]
                        }
                
                # If no SQL record found, this might be an orphaned vector entry
                logging.warning("BACKUP_SQL: No matching SQL memory found - this may be an orphaned vector entry")
                return None
                
        except Exception as e:
            logging.error(f"BACKUP_SQL: Error backing up memory: {e}")
            return None

    def _delete_sql_with_identifiers(self, tracking_id: str, vector_content: str, original_content: str) -> bool:
        """Delete from SQL using multiple identifier approaches."""
        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Try tracking_id first (most reliable)
                if tracking_id:
                    cursor.execute("DELETE FROM memories WHERE tracking_id = ?", (tracking_id,))
                    if cursor.rowcount > 0:
                        conn.commit()
                        logging.info(f"SQL_DELETE: Success via tracking_id: {tracking_id}")
                        return True
                
                # Try exact content matches
                for content_to_try in [vector_content, original_content]:
                    if not content_to_try:
                        continue
                        
                    cursor.execute("DELETE FROM memories WHERE content = ?", (content_to_try,))
                    if cursor.rowcount > 0:
                        conn.commit()
                        logging.info(f"SQL_DELETE: Success via content match")
                        return True
                
                logging.warning("SQL_DELETE: No matches found for deletion")
                return False
                
        except Exception as e:
            logging.error(f"SQL_DELETE: Error: {e}")
            return False

    def _delete_vector_with_identifiers(self, vector_id: str, tracking_id: str, vector_content: str) -> bool:
        """Delete from Vector database using multiple approaches."""
        try:
            from qdrant_client.http import models as rest
            
            # Approach 1: Delete by vector ID
            if vector_id:
                try:
                    self.vector_db.delete_by_id(vector_id)
                    logging.info(f"VECTOR_DELETE: Success via vector ID: {vector_id}")
                    return True
                except Exception as e:
                    logging.warning(f"VECTOR_DELETE: Failed via vector ID: {e}")
            
            # Approach 2: Delete by tracking_id filter
            if tracking_id:
                try:
                    delete_filter = rest.Filter(
                        must=[rest.FieldCondition(
                            key="tracking_id", 
                            match=rest.MatchValue(value=tracking_id)
                        )]
                    )
                    
                    self.vector_db.delete(
                        collection_name=QDRANT_COLLECTION_NAME,
                        points_selector=rest.FilterSelector(filter=delete_filter)
                    )
                    logging.info(f"VECTOR_DELETE: Success via tracking_id filter: {tracking_id}")
                    return True
                except Exception as e:
                    logging.warning(f"VECTOR_DELETE: Failed via tracking_id filter: {e}")
            
            # Approach 3: Search and delete by content
            if vector_content:
                try:
                    search_results = self.vector_db.search(
                        query=vector_content,
                        mode="selective",
                        k=3
                    )
                    
                    for result in search_results:
                        result_id = result.get('id')
                        similarity = result.get('similarity_score', 0)
                        
                        if result_id and similarity > 0.9:  # High confidence match
                            try:
                                self.vector_db.delete_by_id(result_id)
                                logging.info(f"VECTOR_DELETE: Success via content search, ID: {result_id}")
                                return True
                            except Exception as del_e:
                                logging.warning(f"VECTOR_DELETE: Failed to delete ID {result_id}: {del_e}")
                    
                except Exception as e:
                    logging.warning(f"VECTOR_DELETE: Failed via content search: {e}")
            
            logging.error("VECTOR_DELETE: All approaches failed")
            return False
            
        except Exception as e:
            logging.error(f"VECTOR_DELETE: Critical error: {e}")
            return False

    def _rollback_sql_deletion(self, backup_info: dict) -> bool:
        """Rollback SQL deletion by restoring the backed-up memory."""
        try:
            if not backup_info:
                logging.error("ROLLBACK_SQL: No backup info available")
                return False
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Restore the memory
                cursor.execute("""
                    INSERT INTO memories 
                    (content, memory_type, source, weight, access_count, created_at, 
                    last_accessed, tags, metadata, tracking_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backup_info['content'], backup_info['memory_type'], backup_info['source'],
                    backup_info['weight'], backup_info['access_count'], backup_info['created_at'],
                    backup_info['last_accessed'], backup_info['tags'], backup_info['metadata'],
                    backup_info['tracking_id']
                ))
                
                conn.commit()
                logging.info("ROLLBACK_SQL: Successfully restored memory")
                return True
                
        except Exception as e:
            logging.error(f"ROLLBACK_SQL: Failed to rollback: {e}")
            return False
    
    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (chatbot.py cleanup pass).
    # Superseded by the separate delete_memory_by_id + delete_memory_with_coordination methods (both alive).
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_delete_memory_by_id_with_coordination(self, memory_id):
        """Delete a memory by ID with coordination between both databases.
        
        Args:
            memory_id: The database ID of the memory
            
        Returns:
            bool: Success status
        """
        try:
            # Convert memory_id to integer if it's a string
            if isinstance(memory_id, str) and memory_id.isdigit():
                memory_id = int(memory_id)
                
            if not isinstance(memory_id, int):
                logging.error(f"Invalid memory_id type: {type(memory_id)}. Expected int.")
                return False
                
            # Step 1: Get the memory details from SQLite
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT content, tracking_id FROM memories
                    WHERE id = ?
                """, (memory_id,))
                
                result = cursor.fetchone()
                if not result:
                    logging.warning(f"No memory found with ID {memory_id}")
                    return False
                    
                content, tracking_id = result
                
                # Step 2: Delete from SQLite
                cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                sqlite_success = cursor.rowcount > 0
                conn.commit()
                
                if not sqlite_success:
                    logging.warning(f"Failed to delete memory {memory_id} from SQLite")
                    return False
                    
                # Step 3: Delete from vector database using tracking_id or content
                vector_success = False
                if tracking_id and hasattr(self, 'vector_db') and self.vector_db:
                    try:
                        # Try to delete by tracking_id
                        self.vector_db.delete_by_metadata({"tracking_id": tracking_id})
                        vector_success = True
                        logging.info(f"Deleted memory with tracking_id {tracking_id} from vector database")
                    except Exception as e:
                        logging.error(f"Failed to delete from vector database by tracking_id: {e}")
                        
                        # Fall back to content-based deletion
                        try:
                            if content:
                                self.vector_db.delete_text(content)
                                vector_success = True
                                logging.info(f"Deleted memory with content from vector database")
                        except Exception as content_error:
                            logging.error(f"Failed to delete from vector database by content: {content_error}")
                
                return sqlite_success or vector_success
                
        except Exception as e:
            logging.error(f"Error deleting memory by ID: {e}", exc_info=True)
            return False

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (chatbot.py cleanup pass).
    # Debug utility from an earlier tracking_id sync investigation — unwired.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_debug_tracking_id_sync(self, tracking_id: str):
        """Debug method to check tracking_id sync between databases."""
        try:
            logging.info(f"DEBUG_SYNC: Checking tracking_id: {tracking_id}")
            
            # Check SQL
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, content FROM memories WHERE tracking_id = ?", (tracking_id,))
                sql_result = cursor.fetchone()
                
                if sql_result:
                    logging.info(f"DEBUG_SYNC: SQL found - ID: {sql_result[0]}, Content: {sql_result[1][:50]}...")
                else:
                    logging.warning(f"DEBUG_SYNC: SQL NOT found for tracking_id: {tracking_id}")
            
            # Check Vector
            from qdrant_client.http import models as rest
            
            search_filter = rest.Filter(
                must=[rest.FieldCondition(
                    key="tracking_id", 
                    match=rest.MatchValue(value=tracking_id)
                )]
            )
            
            vector_results = self.vector_db.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=search_filter,
                limit=5
            )
            
            if vector_results[0]:  # scroll returns (points, next_page_offset)
                logging.info(f"DEBUG_SYNC: Vector found {len(vector_results[0])} entries")
                for point in vector_results[0]:
                    logging.info(f"DEBUG_SYNC: Vector point ID: {point.id}, payload: {point.payload}")
            else:
                logging.warning(f"DEBUG_SYNC: Vector NOT found for tracking_id: {tracking_id}")
                
        except Exception as e:
            logging.error(f"DEBUG_SYNC: Error checking sync: {e}")
    
    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (chatbot.py cleanup pass).
    # Superseded by deepseek.py _handle_modify_system_prompt_command + chatbot.update_llm_system_prompt (both alive).
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_update_system_prompt(self, additional_prompt: str) -> str:
        """Update the system prompt file."""
        try:
            cleaned_prompt = additional_prompt.strip()
            with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
                current_content = f.read()
            separator = "\n\n" if not current_content.endswith("\n\n") else "\n"
            with open(self.system_prompt_file, 'a', encoding='utf-8') as f:
                f.write(f"{separator}{cleaned_prompt}")
            self._initialize_system_prompt()
            self.llm = self._initialize_llm()
            logging.info(f"System prompt updated with: {cleaned_prompt}")
            return f"Successfully updated system prompt with: {cleaned_prompt}"
        except Exception as e:
            logging.error(f"Error updating system prompt: {e}")
            return f"Failed to update system prompt: {str(e)}"
    
    def _parse_params(self, params_str: str) -> dict:
        """Parse parameter string into a dictionary."""
        params = {}
        if params_str:
            param_parts = params_str.split('|')
            for part in param_parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    params[key.strip()] = value.strip()
        return params

    def _parse_confidence(self, confidence_str: str) -> float:
        """Parse confidence value, ensuring it's between 0.1 and 1.0."""
        try:
            confidence = float(confidence_str)
            return max(0.1, min(1.0, confidence))
        except (ValueError, TypeError):
            return 0.5

    # QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (chatbot.py cleanup pass).
    # Date formatting helper, unwired.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__format_summary_date(self, date_str: str) -> str:
        """Format a date string with robust error handling for display."""
        if not date_str or not isinstance(date_str, str):
            return "Unknown date"
        
        try:
            # Try different date formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                try:
                    date_obj = dt.strptime(date_str.split('.')[0], fmt)
                    return date_obj.strftime("%b %d, %Y")
                except ValueError:
                    continue
                
            # If we couldn't parse with standard formats, check if it's an ISO format
            if 'T' in date_str:
                try:
                    date_parts = date_str.split('T')[0]
                    date_obj = dt.strptime(date_parts, '%Y-%m-%d')
                    return date_obj.strftime("%b %d, %Y")
                except ValueError:
                    pass
                
            # Return original if we can't parse it
            return date_str
        except Exception as e:
            logging.error(f"Error formatting date: {e}")
            return "Unknown date"

    def _assess_for_memory_storage(self, user_input: str, response: str) -> None:
        """
        Forward memory storage commands to deepseek_enhancer for processing.
        This is now a simple wrapper to maintain backward compatibility and 
        ensure all commands are processed through a single path.
        """
        try:
            # Guard against None or invalid response
            if response is None or not isinstance(response, str) or not response.strip():
                logging.warning("Received None, non-string, or empty response in _assess_for_memory_storage")
                return
    
            # No-op: Simply log that we're redirecting to deepseek_enhancer
            logging.info("_assess_for_memory_storage called - all command processing now handled by deepseek_enhancer")
        
            # All actual processing has already been done in deepseek_enhancer.process_response
            # This method now exists only for backward compatibility
        
        except Exception as e:
            logging.error(f"Error in _assess_for_memory_storage wrapper: {e}", exc_info=True)