# Implementation of lifetime memory counters for DeepSeek using a separate database

import sqlite3
import logging
import os
from typing import Dict

class LifetimeCounters:
    """Tracks and maintains lifetime counters for memory commands across sessions."""
    
    def __init__(self, db_directory: str = None):
        """Initialize the lifetime counters using a separate database."""
        logging.info("Initializing LifetimeCounters with separate database")
        try:
            # Use the specified directory or default to the main project directory
            if db_directory is None:
                db_directory = r"C:\Users\kenba\source\repos\Ollama3"
            
            # Create the database path for lifetime counters
            self.db_path = os.path.join(db_directory, "LifetimeCounters.db")
            
            logging.info(f"LifetimeCounters database path: {self.db_path}")
            
            # Ensure the directory exists
            os.makedirs(db_directory, exist_ok=True)
            
            self._initialize_counters_database()
        except Exception as e:
            logging.error(f"LifetimeCounters initialization error: {e}")
            raise

    def _initialize_counters_database(self):
        """Initialize the lifetime counters database and table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create the lifetime_counters table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS lifetime_counters (
                        command_type TEXT PRIMARY KEY,
                        count INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create session_counters table for current session tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS session_counters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        command_type TEXT NOT NULL,
                        count INTEGER DEFAULT 0,
                        session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(session_id, command_type)
                    )
                ''')
                
                # Initialize counters for ALL command types used in your system
                command_types = [
                    "store", 
                    "store_important", 
                    "search", 
                    "comprehensive_search",
                    "reflect", 
                    "reflect_concept",
                    "forget", 
                    "self",
                    "summarize",
                    "reminder",
                    "reminder_complete",
                    "discuss_with_claude",
                    "self_dialogue",
                    "research_dialogue",
                    "web_search",
                    "cognitive_state",
                    "show_system_prompt",
                    "modify_system_prompt",
                    "help",  
                    "total"
                ]
                
                for cmd_type in command_types:
                    cursor.execute('''
                        INSERT OR IGNORE INTO lifetime_counters (command_type, count)
                        VALUES (?, 0)
                    ''', (cmd_type,))
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_counters_session_id ON session_counters(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_counters_command_type ON session_counters(command_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_lifetime_counters_command_type ON lifetime_counters(command_type)')
                    
                conn.commit()
                logging.info(f"LifetimeCounters database initialized successfully at {self.db_path}")
                
        except Exception as e:
            logging.error(f"Error initializing lifetime counters database: {e}")
            raise
    
    def increment_counter(self, command_type: str, session_id: str = None) -> bool:
        """
        Increment both lifetime and session counters for the specified command type.
        
        Args:
            command_type (str): The type of command
            session_id (str): Optional session ID for session tracking
                                
        Returns:
            bool: True if successful, False otherwise
        """
        try:
             # Define all valid command types
            valid_command_types = [
                "store", "store_important", "search", "comprehensive_search",
                "reflect", "reflect_concept",
                "forget", "self", "summarize", "reminder", "reminder_complete", 
                "discuss_with_claude", "self_dialogue", "research_dialogue",
                "web_search", "cognitive_state",
                "show_system_prompt", "modify_system_prompt",
                "help"  
            ]
            
            if command_type not in valid_command_types:
                logging.warning(f"Invalid command type: {command_type}")
                return False
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert the command type if it doesn't exist (safety measure)
                cursor.execute('''
                    INSERT OR IGNORE INTO lifetime_counters (command_type, count)
                    VALUES (?, 0)
                ''', (command_type,))
                
                # Increment the specific command counter
                cursor.execute('''
                    UPDATE lifetime_counters
                    SET count = count + 1,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE command_type = ?
                ''', (command_type,))
                
                # Also increment the total counter
                cursor.execute('''
                    INSERT OR IGNORE INTO lifetime_counters (command_type, count)
                    VALUES ('total', 0)
                ''')
                
                cursor.execute('''
                    UPDATE lifetime_counters
                    SET count = count + 1,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE command_type = 'total'
                ''')
                
                # Update session counters if session_id is provided
                if session_id:
                    cursor.execute('''
                        INSERT OR REPLACE INTO session_counters 
                        (session_id, command_type, count, last_updated)
                        VALUES (?, ?, 
                               COALESCE((SELECT count FROM session_counters 
                                       WHERE session_id = ? AND command_type = ?), 0) + 1,
                               CURRENT_TIMESTAMP)
                    ''', (session_id, command_type, session_id, command_type))
                
                conn.commit()
                logging.info(f"Incremented lifetime {command_type} counter")
                return True
                
        except Exception as e:
            logging.error(f"Error incrementing lifetime counter for {command_type}: {e}")
            return False
    
    def get_counters(self) -> Dict[str, int]:
        """
        Get all lifetime memory command counters.
        
        Returns:
            Dict[str, int]: Dictionary of command types and their counts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT command_type, count
                    FROM lifetime_counters
                ''')
                
                counters = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Ensure all expected counters exist with default values
                default_counters = {
                    "store": 0,
                    "store_important": 0,
                    "search": 0,
                    "comprehensive_search": 0,  # NEW
                    "reflect": 0,
                    "reflect_concept": 0,
                    "forget": 0,
                    "self": 0,
                    "summarize": 0,
                    "reminder": 0,
                    "reminder_complete": 0,
                    "discuss_with_claude": 0,
                    "self_dialogue": 0,      
                    "web_search": 0,
                    "cognitive_state": 0,
                    "total": 0
                }
                
                # Merge with defaults
                for key, default_value in default_counters.items():
                    if key not in counters:
                        counters[key] = default_value
                
                return counters
                
        except Exception as e:
            logging.error(f"Error getting lifetime counters: {e}")
            return {
                "store": 0,
                "store_important": 0,
                "search": 0,
                "reflect": 0,
                "reflect_concept": 0,
                "forget": 0,
                "self": 0,
                "summarize": 0,
                "reminder": 0,
                "reminder_complete": 0,
                "discuss_with_claude": 0,
                "total": 0
            }
    
    def get_session_counters(self, session_id: str) -> Dict[str, int]:
        """
        Get counters for a specific session.
        
        Args:
            session_id (str): The session ID to get counters for
            
        Returns:
            Dict[str, int]: Dictionary of command types and their counts for this session
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT command_type, count
                    FROM session_counters
                    WHERE session_id = ?
                ''', (session_id,))
                
                counters = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Ensure all expected counters exist with default values
                default_counters = {
                    "store": 0,
                    "store_important": 0,
                    "search": 0,
                    "reflect": 0,
                    "reflect_concept": 0,
                    "forget": 0,
                    "self": 0,
                    "summarize": 0,
                    "reminder": 0,
                    "reminder_complete": 0,
                    "discuss_with_claude": 0,
                    "self_dialogue": 0,      
                    "web_search": 0,
                    "cognitive_state": 0,
                    "show_system_prompt": 0,
                    "modify_system_prompt": 0,
                    "research_dialogue": 0,
                    "total": 0
                }
                
                # Merge with defaults
                for key, default_value in default_counters.items():
                    if key not in counters:
                        counters[key] = default_value
                
                return counters
                
        except Exception as e:
            logging.error(f"Error getting session counters for {session_id}: {e}")
            return {}
    
       
    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 6 utility cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Manual session counter cleanup utility. The session_counters table grows indefinitely
    # because nothing clears it. Could be useful for long-running deployments to bound table
    # growth, but currently unwired.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_clear_session_counters(self, session_id: str = None) -> bool:
        """
        Clear session counters for a specific session or all sessions.
        
        Args:
            session_id (str): Optional session ID. If None, clears all sessions.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if session_id:
                    cursor.execute('DELETE FROM session_counters WHERE session_id = ?', (session_id,))
                    logging.info(f"Cleared session counters for session {session_id}")
                else:
                    cursor.execute('DELETE FROM session_counters')
                    logging.info("Cleared all session counters")
                
                conn.commit()
                return True
                
        except Exception as e:
            logging.error(f"Error clearing session counters: {e}")
            return False

    def get_database_path(self) -> str:
        """Get the path to the lifetime counters database."""
        return self.db_path