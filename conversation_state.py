# conversation_state.py
"""Conversation state management for maintaining context across interactions."""

import logging
import sqlite3
import datetime
import uuid
import json 
from typing import Dict, Optional, Tuple, List

class ConversationStateManager:
    """Manages conversation state persistence and retrieval."""
    
    def __init__(self, db_path: str):
        """Initialize with the path to the SQLite database."""
        self.db_path = db_path
        self.current_session_id = None
        self.current_summary = None
        self.ensure_schema()
        logging.info("Conversation state manager initialized")
        
    
    def ensure_schema(self) -> bool:
        """Create necessary database schema if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create the main conversation state table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_state (
                        id INTEGER PRIMARY KEY,
                        session_id TEXT UNIQUE,
                        active INTEGER DEFAULT 1,
                        last_updated TEXT,
                        summary TEXT,
                        turn_count INTEGER DEFAULT 0
                    )
                """)
                
                # Add index for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conv_session_id 
                    ON conversation_state(session_id)
                """)
                
                conn.commit()
                logging.info("Conversation state schema created/verified")
                return True
                
        except Exception as e:
            logging.error(f"Error ensuring conversation schema: {e}", exc_info=True)
            return False
    
    def initialize_session(self, auto_retrieve_summary=True) -> Optional[str]:
        """
        Initialize or retrieve the current conversation session with enhanced summary retrieval.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the most recent active conversation
                cursor.execute("""
                    SELECT session_id, summary, turn_count
                    FROM conversation_state
                    WHERE active = 1
                    ORDER BY last_updated DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                
                if result:
                    # Restore existing conversation
                    self.current_session_id = result[0]
                    self.current_summary = result[1]
                    turn_count = result[2]
                    logging.info(f"Restored conversation session {self.current_session_id} with {turn_count} turns")
                    
                    # If auto_retrieve_summary is True, explicitly call our existing method
                    if auto_retrieve_summary:
                        # Call our existing method to retrieve the latest summary
                        summary_found = self.retrieve_latest_summary()
                        if summary_found:
                            logging.info(f"Auto-retrieved latest conversation summary: {self.current_summary[:50]}...")
                            
                            # Also update the conversation_state table with this better summary
                            cursor.execute("""
                                UPDATE conversation_state
                                SET summary = ?
                                WHERE session_id = ?
                            """, (self.current_summary, self.current_session_id))
                            conn.commit()
                            logging.info("Updated conversation state with latest summary")
                            

                else:
                    # Create new conversation session
                    self.current_session_id = f"conv_{uuid.uuid4().hex[:10]}"
                    self.current_summary = "Conversation just started."
                    
                    # Store new session
                    cursor.execute("""
                        INSERT INTO conversation_state 
                        (session_id, last_updated, summary, turn_count)
                        VALUES (?, ?, ?, ?)
                    """, (
                        self.current_session_id, 
                        datetime.datetime.now().isoformat(),
                        self.current_summary,
                        0
                    ))
                    conn.commit()
                    logging.info(f"Created new conversation session {self.current_session_id}")
                
                return self.current_session_id
                
        except Exception as e:
            logging.error(f"Error initializing conversation session: {e}", exc_info=True)
        return None
        
    def retrieve_latest_summary(self):
        """
        Retrieve the latest conversation summary from the memory system.
        """
        try:
            # Connect to the database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Query for the most recent conversation summary - also get metadata
                cursor.execute(
                    "SELECT content, metadata FROM memories WHERE memory_type='conversation_summary' ORDER BY created_at DESC LIMIT 1"
                )
                
                result = cursor.fetchone()
                
                if result:
                    # Update the current summary with results
                    content, metadata_str = result
                    self.current_summary = content
                    
                    # Extract date for display if available
                    summary_date = "Unknown date"
                    if metadata_str:
                        try:
                            metadata = json.loads(metadata_str)
                            summary_date = metadata.get('date', metadata.get('summary_date', 'Unknown date'))
                        except json.JSONDecodeError:
                            logging.warning("Could not parse summary metadata JSON")
                        
                    # Format the summary with date for better display
                    self.formatted_summary = f"PREVIOUS CONVERSATION SUMMARY (from {summary_date}):\n\n{content}\n\nEND OF PREVIOUS CONVERSATION SUMMARY"
                    
                    logging.info(f"Retrieved latest conversation summary from {summary_date}: {self.current_summary[:50]}...")
                    return True
                else:
                    logging.info("No previous conversation summaries found")
                    return False
                    
        except Exception as e:
            logging.error(f"Error retrieving latest summary: {e}")
            return False
    
    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 6 utility cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Simple getter for self.current_summary. The active flow uses get_formatted_context()
    # for context-building (which formats with PREVIOUS CONVERSATION SUMMARY framing) or
    # reads the attribute directly. This bare getter is never called.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_get_current_summary(self) -> str:
        """Get the current conversation summary."""
        return self.current_summary or ""
    
    def update_summary(self, new_summary: str, memory_db=None, vector_db=None) -> bool:
        """
        Update the conversation summary for the current session.
        Uses transaction coordination to ensure consistency across databases.
        
        Args:
            new_summary (str): The new conversation summary
            memory_db: Optional reference to memory_db for storage
            vector_db: Optional reference to vector_db for storage
            
        Returns:
            bool: Success status
        """
        if not self.current_session_id:
            logging.warning("No active session when updating summary")
            return False
            
        try:
            # First update the local database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update the session summary
                cursor.execute("""
                    UPDATE conversation_state
                    SET summary = ?, last_updated = ?
                    WHERE session_id = ?
                """, (
                    new_summary,
                    datetime.datetime.now().isoformat(),
                    self.current_session_id
                ))
                
                # Update turn count
                cursor.execute("""
                    UPDATE conversation_state
                    SET turn_count = turn_count + 1
                    WHERE session_id = ?
                """, (self.current_session_id,))
                
                conn.commit()
                
            # Update local state
            self.current_summary = new_summary
            logging.info(f"Updated summary for session {self.current_session_id}")
            
            # Store in memory system using transaction coordination if both DBs are provided
            if memory_db and hasattr(memory_db, 'store_memory_with_transaction'):
                try:
                    # Prepare metadata for the summary
                    # Get current date/time for consistent formatting
                    now = datetime.datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")

                    metadata = {
                        "source": f"session_{self.current_session_id}", 
                        "type": "conversation_summary",
                        "created_at": now.isoformat(),
                        "date": date_str,  # ✅ ADD: Standardized date field
                        "time": time_str,  # ✅ ADD: Time field
                        "tags": ["conversation_summary", f"date={date_str}"],  # ✅ CHANGE: Array with date tag
                        "session_id": self.current_session_id
                    }
                    
                    # Use the transaction coordinator to ensure consistency
                    success, memory_id = memory_db.store_memory_with_transaction(
                        content=new_summary,
                        memory_type="conversation_summary",  # Changed from "conversation" to "conversation_summary"
                        metadata=metadata,
                        confidence=0.5  # Conversation summaries are relatively confident
                    )
                    
                    if success:
                        logging.info(f"Stored conversation summary with ID {memory_id} using transaction coordination")
                    else:
                        logging.warning(f"Failed to store conversation summary with transaction coordination")
                        
                except Exception as mem_error:
                    logging.error(f"Error storing summary with transaction coordination: {mem_error}", exc_info=True)
                    return False
                    
            # If memory_db is provided but transaction method is missing, log error
            elif memory_db:
                logging.error("Memory DB doesn't have store_memory_with_transaction method - cannot safely store summary")
                return False  # Don't proceed with inconsistent storage
                
            return True
                
        except Exception as e:
            logging.error(f"Error updating conversation summary: {e}", exc_info=True)
            return False
        
    def get_formatted_context(self) -> str:
        """Get formatted conversation context for inclusion in prompts."""
        # Use the formatted summary if available, otherwise fall back to current_summary
        if hasattr(self, 'formatted_summary') and self.formatted_summary:
            return f"{self.formatted_summary}\n\n"
        elif self.current_summary:
            return f"Previous conversation summary: {self.current_summary}\n\n"
        else:
            return ""
    
    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 6 utility cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Distinct from SessionManager.end_session (which was removed in earlier cleanup).
    # This is ConversationStateManager.end_session that marks sessions as active=0 in the
    # conversation_state table. Sessions are never explicitly ended in the active flow — they
    # stay active=1 indefinitely. Consider wiring if conversation lifecycle tracking matters
    # (e.g., to support multiple distinct sessions per user vs. one running thread).
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_end_session(self) -> bool:
        """Mark the current session as inactive."""
        if not self.current_session_id:
            return False
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE conversation_state
                    SET active = 0
                    WHERE session_id = ?
                """, (self.current_session_id,))
                conn.commit()
                
                logging.info(f"Ended conversation session {self.current_session_id}")
                self.current_session_id = None
                self.current_summary = None
                return True
                
        except Exception as e:
            logging.error(f"Error ending conversation session: {e}", exc_info=True)
            return False