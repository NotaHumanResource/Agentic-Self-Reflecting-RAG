# reminders.py
"""Dedicated module for managing reminders through SQL database only."""
import datetime
import logging
import sqlite3
import uuid
import json
import re
from typing import List, Dict, Optional, Any, Tuple

class ReminderManager:
    """Manages reminders exclusively through SQL database without vector DB integration."""
    
    def __init__(self, db_path: str):
        """Initialize the reminder manager with database path.
        
        Args:
            db_path (str): Path to the SQLite database
        """
        self.db_path = db_path
        logging.info(f"Initialized ReminderManager with database: {db_path}")
        
        # Ensure the reminders table exists with proper schema
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the reminders table exists in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if we need to create a dedicated reminders table
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='reminders'
                """)
                
                if not cursor.fetchone():
                    # Create a dedicated reminders table
                    cursor.execute("""
                        CREATE TABLE reminders (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            content TEXT NOT NULL,
                            due_date TEXT,
                            status TEXT DEFAULT 'active',
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            metadata TEXT
                        )
                    """)
                    logging.info("Created dedicated reminders table")
                    conn.commit()
                
        except sqlite3.Error as e:
            logging.error(f"Database error ensuring reminders table: {e}")
            raise
    
    def create_reminder(self, content: str, due_date: str = None, metadata: Dict = None) -> Tuple[bool, int]:
        """Create a new reminder in SQL database only.
        
        Args:
            content (str): The reminder text content
            due_date (str, optional): Due date in ISO format (YYYY-MM-DD)
            metadata (Dict, optional): Additional metadata for the reminder
            
        Returns:
            Tuple[bool, int]: (success status, reminder ID if successful)
        """
        try:
            # Validate inputs
            if not content or not content.strip():
                logging.warning("Attempted to create empty reminder")
                return False, None
                
            content = content.strip()
            
            # Process due date
            if due_date:
                due_date = self._standardize_due_date(due_date)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
                
            # Add a reminder_id to metadata for compatibility with existing code
            reminder_id = str(uuid.uuid4())
            metadata["reminder_id"] = reminder_id
            metadata["id"] = reminder_id  # For backward compatibility
            metadata_json = json.dumps(metadata)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO reminders (content, due_date, metadata) 
                    VALUES (?, ?, ?)
                """, (content, due_date, metadata_json))
                
                # Get the auto-generated ID
                reminder_id = cursor.lastrowid
                
                conn.commit()
                
                logging.info(f"Created reminder with ID {reminder_id}: {content[:50]}...")
                return True, reminder_id
                
        except Exception as e:
            logging.error(f"Error creating reminder: {e}")
            return False, None
    
    def get_reminders(self, limit: int = None, status: str = "active") -> List[Dict]:
        """Get reminders from the database.
        
        Args:
            limit (int, optional): Maximum number of reminders to return
            status (str, optional): Filter by status (active, completed, etc.)
            
        Returns:
            List[Dict]: List of reminder dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
                cursor = conn.cursor()
                
                # Build the query
                query = "SELECT * FROM reminders WHERE status = ?"
                params = [status]
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries with parsed metadata
                reminders = []
                for row in rows:
                    reminder = dict(row)
                    
                    # Parse metadata JSON
                    if reminder.get('metadata'):
                        try:
                            reminder['metadata'] = json.loads(reminder['metadata'])
                        except json.JSONDecodeError:
                            reminder['metadata'] = {}
                    
                    reminders.append(reminder)
                
                logging.info(f"Retrieved {len(reminders)} reminders with status '{status}'")
                return reminders
                
        except Exception as e:
            logging.error(f"Error retrieving reminders: {e}")
            return []
        
    # QUARANTINED 2026-05-19: 0 callers AND broken (batch 6 utility cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # This method belongs to ReminderManager class but calls self.reminder_manager.get_due_reminders()
    # — there is NO self.reminder_manager attribute on ReminderManager. Would throw AttributeError
    # if invoked. The active 'check_due_reminders' is chatbot.py L3595 which delegates correctly
    # to self.reminder_manager.get_due_reminders() (chatbot HAS a reminder_manager attribute pointing
    # to a ReminderManager instance).
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_check_due_reminders(self):
        """
        Check for any reminders that are due and return them for notification.
        
        Returns:
            List[Dict[str, Any]]: List of due reminders
        """
        try:
            # Use the reminder manager to get due reminders
            due_reminders = self.reminder_manager.get_due_reminders()
            if due_reminders:
                logging.info(f"Found {len(due_reminders)} reminders due today or earlier")
            return due_reminders
        except Exception as e:
            logging.error(f"Error checking for due reminders: {e}")
            return []
    
    def get_due_reminders(self) -> List[Dict]:
        """Get reminders that are due today or in the past.
        
        Returns:
            List[Dict]: List of due reminder dictionaries
        """
        try:
            today = datetime.date.today().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM reminders 
                    WHERE status = 'active' 
                    AND (due_date <= ? OR due_date IS NULL OR due_date = '')
                    ORDER BY due_date ASC
                """, (today,))
                
                rows = cursor.fetchall()
                
                # Convert to dictionaries with parsed metadata
                reminders = []
                for row in rows:
                    reminder = dict(row)
                    
                    # Parse metadata JSON
                    if reminder.get('metadata'):
                        try:
                            reminder['metadata'] = json.loads(reminder['metadata'])
                        except json.JSONDecodeError:
                            reminder['metadata'] = {}
                    
                    reminders.append(reminder)
                
                logging.info(f"Found {len(reminders)} reminders due today or earlier")
                return reminders
                
        except Exception as e:
            logging.error(f"Error retrieving due reminders: {e}")
            return []
    
    def update_reminder_status(self, reminder_id: int, status: str = "completed") -> bool:
        """Update a reminder's status.
        
        Args:
            reminder_id (int): ID of the reminder to update
            status (str): New status ('completed', 'cancelled', etc.)
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE reminders 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, reminder_id))
                
                success = cursor.rowcount > 0
                conn.commit()
                
                if success:
                    logging.info(f"Updated reminder {reminder_id} status to '{status}'")
                else:
                    logging.warning(f"Failed to update reminder {reminder_id} - not found")
                
                return success
                
        except Exception as e:
            logging.error(f"Error updating reminder status: {e}")
            return False
    
    def delete_reminder(self, reminder_id: int) -> bool:
        """Delete a reminder by ID.
        
        Args:
            reminder_id (int): ID of the reminder to delete
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # For safety, first check if the reminder exists
                cursor.execute("SELECT id FROM reminders WHERE id = ?", (reminder_id,))
                if not cursor.fetchone():
                    logging.warning(f"Attempted to delete non-existent reminder with ID {reminder_id}")
                    return False
                
                # Delete the reminder
                cursor.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
                conn.commit()
                
                logging.info(f"Deleted reminder with ID {reminder_id}")
                return True
                
        except Exception as e:
            logging.error(f"Error deleting reminder: {e}")
            return False
    
    def delete_reminder_by_content(self, content: str) -> bool:
        """Delete a reminder by matching its content.
        
        Args:
            content (str): Content to match for deletion
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find reminders with matching content
                cursor.execute("SELECT id FROM reminders WHERE content LIKE ?", (f"%{content}%",))
                matches = cursor.fetchall()
                
                if not matches:
                    logging.warning(f"No reminders found with content like '{content[:50]}...'")
                    return False
                
                # Delete the first matching reminder
                reminder_id = matches[0][0]
                cursor.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
                conn.commit()
                
                logging.info(f"Deleted reminder with ID {reminder_id} matching content '{content[:50]}...'")
                return True
                
        except Exception as e:
            logging.error(f"Error deleting reminder by content: {e}")
            return False
    
    def search_reminders(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for reminders matching the query.
        
        Args:
            query (str): Search query
            limit (int, optional): Maximum number of results
            
        Returns:
            List[Dict]: List of matching reminders
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Search in content field
                cursor.execute("""
                    SELECT * FROM reminders 
                    WHERE content LIKE ? AND status = 'active'
                    ORDER BY due_date ASC
                    LIMIT ?
                """, (f"%{query}%", limit))
                
                rows = cursor.fetchall()
                
                # Convert to dictionaries with parsed metadata
                reminders = []
                for row in rows:
                    reminder = dict(row)
                    
                    # Parse metadata JSON
                    if reminder.get('metadata'):
                        try:
                            reminder['metadata'] = json.loads(reminder['metadata'])
                        except json.JSONDecodeError:
                            reminder['metadata'] = {}
                    
                    reminders.append(reminder)
                
                logging.info(f"Found {len(reminders)} reminders matching '{query}'")
                return reminders
                
        except Exception as e:
            logging.error(f"Error searching reminders: {e}")
            return []
    
    def _standardize_due_date(self, due_date_raw: str) -> str:
        """Convert various date formats to a standardized date format (YYYY-MM-DD).
        
        Args:
            due_date_raw (str): The raw date string from user input
            
        Returns:
            str: Standardized date string or original if unparseable
        """
        try:
            # Return original if empty
            if not due_date_raw or not due_date_raw.strip():
                return ""
                
            # Get current date for relative calculations
            today = datetime.date.today()
            
            # Handle common relative date terms
            lower_date = due_date_raw.lower().strip()
            
            # Handle "today", "now", etc.
            if lower_date in ("today", "now", "asap", "immediately"):
                return today.isoformat()
                
            # Handle "tomorrow"
            if lower_date == "tomorrow":
                tomorrow = today + datetime.timedelta(days=1)
                return tomorrow.isoformat()
                
            # Handle "next week"
            if "next week" in lower_date:
                next_week = today + datetime.timedelta(days=7)
                return next_week.isoformat()
                
            # Handle "next month"
            if "next month" in lower_date:
                # Simple approximation - more accurate would check actual month length
                next_month = today + datetime.timedelta(days=30)
                return next_month.isoformat()
                
            # Handle "in X days"
            match = re.search(r"in\s+(\d+)\s+days?", lower_date)
            if match:
                days = int(match.group(1))
                future_date = today + datetime.timedelta(days=days)
                return future_date.isoformat()
                
            # Try to parse various date formats
            for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%b %d, %Y', '%m-%d-%Y', '%d-%m-%Y'):
                try:
                    parsed_date = datetime.datetime.strptime(due_date_raw, fmt).date()
                    return parsed_date.isoformat()
                except ValueError:
                    continue
                    
            # If we couldn't parse it as a date, return the original
            logging.info(f"Could not standardize date format for '{due_date_raw}', using as-is")
            return due_date_raw
                
        except Exception as e:
            logging.error(f"Error in _standardize_due_date: {e}")
            # Return the original string if any errors occur
            return due_date_raw
    
    def format_reminder_notification(self, reminders: List[Dict]) -> str:
        """Format a list of due reminders into a notification message.
        
        Args:
            reminders (List[Dict]): List of reminder dictionaries
            
        Returns:
            str: Formatted reminder notification message
        """
        if not reminders:
            return ""
        
        # Create a reminder notification message
        message = "📅 **Reminder Notification**\n\n"
        
        for i, reminder in enumerate(reminders, 1):
            content = reminder.get('content', '')
            due_date = reminder.get('due_date', 'Today')
            reminder_id = reminder.get('id')
            
            message += f"{i}. {content}\n"
            message += f"   Due: {due_date}\n"
            # Add invisible marker with reminder ID for later reference
            message += f"<!-- reminder_id:{reminder_id} -->\n\n"
        
        message += "\nYou can mark these as completed by replying with 'complete reminder X' where X is the reminder number."
        
        return message
    
    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 6 utility cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # One-shot migration utility moving legacy reminders from the 'memories' table into the
    # dedicated 'reminders' table. Migration has presumably been completed or skipped. No
    # callers — was likely run manually once via Python REPL or removed call site.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_migrate_from_memories_table(self) -> Tuple[int, int]:
        """Migrate reminders from the memories table to the dedicated reminders table.
        
        Returns:
            Tuple[int, int]: (number of reminders found, number successfully migrated)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if memories table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='memories'
                """)
                
                if not cursor.fetchone():
                    logging.warning("No memories table found, nothing to migrate")
                    return 0, 0
                
                # Get all reminders from memories table
                cursor.execute("""
                    SELECT id, content, metadata, tracking_id, active 
                    FROM memories 
                    WHERE memory_type = 'reminder'
                """)
                
                reminders = cursor.fetchall()
                logging.info(f"Found {len(reminders)} reminders in memories table")
                
                migrated = 0
                
                for rem in reminders:
                    mem_id, content, metadata_json, tracking_id, active = rem
                    
                    # Parse metadata
                    try:
                        metadata = json.loads(metadata_json) if metadata_json else {}
                    except:
                        metadata = {}
                    
                    # Get due date and status
                    due_date = metadata.get('due_date', '')
                    status = 'active' if active == 1 else 'completed'
                    
                    # Make sure we have a reminder_id
                    if not tracking_id:
                        tracking_id = str(uuid.uuid4())
                    
                    metadata['migrated_from_id'] = mem_id
                    metadata['reminder_id'] = tracking_id
                    new_metadata_json = json.dumps(metadata)
                    
                    # Insert into reminders table
                    cursor.execute("""
                        INSERT INTO reminders 
                        (content, due_date, status, metadata) 
                        VALUES (?, ?, ?, ?)
                    """, (content, due_date, status, new_metadata_json))
                    
                    migrated += 1
                
                conn.commit()
                logging.info(f"Successfully migrated {migrated} reminders to dedicated table")
                return len(reminders), migrated
                
        except Exception as e:
            logging.error(f"Error migrating reminders: {e}")
            return 0, 0