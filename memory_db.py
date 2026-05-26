"""SQLite-based memory storage with weighting system and improved content handling."""
import time
import uuid
import sqlite3
import logging
import math
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from config import DB_PATH
from utils import RateLimiter
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

logging.debug(f"timedelta imported successfully: {timedelta}")

class MemoryDB:
    """Handles persistent storage and retrieval of memories with weighting system."""
    
    def __init__(self):
        """Initialize the MemoryDB class with SQLite database and weighting system."""
        logging.info("Initializing MemoryDB with weighting system")
        try:
            self.db_path = DB_PATH
            self._upgrade_db_schema()
            self.vector_db = QdrantClient(host="localhost", port=6333)
            logging.info("Initialized VectorDB (Qdrant client)")
            self._ensure_qdrant_collection()
            # Callback invoked when an item is added to the deletion queue.
            # Wired up externally (main.py) to start the demand-driven processor thread.
            # None by default so memory_db.py stays decoupled from threading.
            self.on_item_queued = None
        except Exception as e:
            logging.error(f"MemoryDB initialization error: {e}")
            raise

   
    def _upgrade_db_schema(self):
        """Ensure all required tables exist and upgrade schema if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create memories table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        memory_type TEXT DEFAULT 'general',
                        source TEXT DEFAULT 'unknown',
                        weight REAL DEFAULT 0.5,
                        access_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        tags TEXT,
                        metadata TEXT,
                        tracking_id TEXT,
                        confidence REAL DEFAULT 0.5
                    )
                ''')
                
                # Create reminders table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS reminders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        due_date TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed BOOLEAN DEFAULT 0,
                        metadata TEXT,
                        status TEXT DEFAULT 'active'
                    )
                ''')
                
                # Create deletion_queue table with ALL necessary columns including attempt_count
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS deletion_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id TEXT NOT NULL,
                        tracking_id TEXT,
                        content TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT 0,
                        attempt_count INTEGER DEFAULT 0,
                        last_attempt TIMESTAMP,
                        error_message TEXT
                    )
                ''')
                
                # Check and add missing columns to existing tables
                
                # Check reminders table
                cursor.execute("PRAGMA table_info(reminders)")
                reminder_columns = [column[1] for column in cursor.fetchall()]
                
                if 'status' not in reminder_columns:
                    logging.info("Adding missing 'status' column to reminders table")
                    cursor.execute('ALTER TABLE reminders ADD COLUMN status TEXT DEFAULT "active"')
                
                # Check memories table
                cursor.execute("PRAGMA table_info(memories)")
                memory_columns = [column[1] for column in cursor.fetchall()]
                
                missing_memory_columns = {
                    'tracking_id': 'ALTER TABLE memories ADD COLUMN tracking_id TEXT',
                    'confidence': 'ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 0.5',
                    'metadata': 'ALTER TABLE memories ADD COLUMN metadata TEXT'
                }
                
                for col_name, alter_sql in missing_memory_columns.items():
                    if col_name not in memory_columns:
                        logging.info(f"Adding missing '{col_name}' column to memories table")
                        cursor.execute(alter_sql)
                
                # Check deletion_queue table for missing columns
                cursor.execute("PRAGMA table_info(deletion_queue)")
                deletion_columns = [column[1] for column in cursor.fetchall()]
                
                missing_deletion_columns = {
                    'tracking_id': 'ALTER TABLE deletion_queue ADD COLUMN tracking_id TEXT',
                    'attempt_count': 'ALTER TABLE deletion_queue ADD COLUMN attempt_count INTEGER DEFAULT 0',
                    'last_attempt': 'ALTER TABLE deletion_queue ADD COLUMN last_attempt TIMESTAMP',
                    'error_message': 'ALTER TABLE deletion_queue ADD COLUMN error_message TEXT'
                }
                
                for col_name, alter_sql in missing_deletion_columns.items():
                    if col_name not in deletion_columns:
                        logging.info(f"Adding missing '{col_name}' column to deletion_queue table")
                        cursor.execute(alter_sql)
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_content ON memories(content)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_tracking_id ON memories(tracking_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_reminders_due_date ON reminders(due_date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_deletion_queue_processed ON deletion_queue(processed)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_deletion_queue_memory_id ON deletion_queue(memory_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_deletion_queue_tracking_id ON deletion_queue(tracking_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_deletion_queue_attempt_count ON deletion_queue(attempt_count)')
                
                conn.commit()
                logging.info("Database schema upgraded successfully - all tables and columns ensured")
                
        except Exception as e:
            logging.error(f"Error upgrading database schema: {e}")
            raise

    def _ensure_qdrant_collection(self):
        """Ensure the Qdrant collection for memories exists."""
        try:
            collection_name = "memories"
            collections = self.vector_db.get_collections().collections
            if not any(collection.name == collection_name for collection in collections):
                self.vector_db.recreate_collection(
                    collection_name=collection_name,
                    vectors_config={"size": 4096, "distance": "Cosine"}
                )
                logging.info(f"Created Qdrant collection: {collection_name}")
            else:
                logging.info(f"Qdrant collection {collection_name} already exists")
        except Exception as e:
            logging.error(f"Error ensuring Qdrant collection: {e}")
            raise

                    
    def calculate_memory_weight(self, memory_type: str, access_count: int, created_at: datetime = None, last_accessed: datetime = None) -> float:
        """
        Calculate memory weight based on type and access count.
        Weights do not decay over time - memories persist based on their confidence type.
        Uses 0.0-1.0 scale to align with user-facing confidence levels.
        
        Args:
            memory_type (str): Type of memory (important, conversation, self, etc.)
            access_count (int): Number of times the memory has been accessed
            created_at (datetime): Creation timestamp (kept for backwards compatibility but not used)
            last_accessed (datetime): Last access timestamp (kept for backwards compatibility but not used)
        
        Returns:
            float: Calculated weight value (0.6-1.0 range)
        """
        try:
            # Define base weights for each memory type
            # Using 0.0-1.0 scale to match user-facing confidence system
            # Aligns with system prompt confidence guidelines:
            # 0.9-1.0 = Critical, 0.6-0.8 = High, 0.5 = Medium, 0.3-0.5 = Medium-Low
            type_weights = {
                "important": 1.0,      # Critical information (matches 0.9-1.0 Critical in system prompt)
                "document": 0.5,       # Document-sourced knowledge (high confidence)
                "conversation": 0.7,   # Conversation summaries (0.6-0.8 High)
                "self": 0.7,          # Self-knowledge and identity (0.6-0.8 High)
                "reflection": 0.7,     # Self-reflections and insights (0.6-0.8 High)
                "general": 0.5,        # General memories (0.5 Medium baseline)
                "reminder": 1.0        # Reminders (0.3-0.5 Medium-Low)
            }
            
            # Get base weight for this memory type (default to 0.7 if type not found)
            base_weight = type_weights.get(memory_type, 0.7)
            
            # Calculate access bonus (rewards frequently accessed memories)
            # Uses logarithmic scale to prevent runaway growth
            # Scaled to 0.01 increments to match 0.0-1.0 range
            # Example: 0 accesses = 0.0, 10 accesses = 0.033, 100 accesses = 0.066
            access_weight = math.log(access_count + 1, 2) * 0.01
            
            # Final weight is base + access bonus
            # No time decay - memories maintain their confidence indefinitely
            final_weight = base_weight + access_weight
            
            # Cap at 1.0 to maintain scale consistency
            final_weight = min(final_weight, 1.0)
            
            return round(final_weight, 3)
            
        except Exception as e:
            logging.error(f"Error calculating memory weight: {e}")
            # Return a safe default weight (general type = medium confidence)
            return 0.5
        
    def update_memory_access(self, memory_id: int):
        """Update access count and last accessed time for a memory."""
        try:
            # Guard against None or invalid memory_id
            if memory_id is None or not isinstance(memory_id, int) or memory_id <= 0:
                logging.warning(f"Invalid memory_id provided to update_memory_access: {memory_id}")
                return
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE memories 
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (memory_id,))
                conn.commit()
        except Exception as e:
            logging.error(f"Error updating memory access: {e}")

     

    #Used by Self Reflect
    def search_similar(self, content: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for memories with similar content.
        
        Args:
            content (str): The content to search for
            threshold (float): Minimum similarity threshold (0-1)
            
        Returns:
            List[Dict[str, Any]]: List of similar memories
        """
        try:
            # Guard against None or empty content
            if not content or not isinstance(content, str) or not content.strip():
                logging.warning("Attempted to search for similar memories with None or empty content")
                return []
                
            # First try to get exact match
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, content, memory_type, source, weight, access_count, "
                    "created_at, last_accessed, tags "
                    "FROM memories WHERE content = ?", 
                    (content,)
                )
                exact_match = cursor.fetchone()
                
                if exact_match:
                    # Format and return the exact match
                    memory_dict = self._format_db_row_to_dict(exact_match)
                    return [memory_dict]  # Return as a list for consistency
            
            # No exact match, get potential matches using get_memories_by_weight
            # This aligns with the approach in _find_similar_content
            weighted_memories = self.get_memories_by_weight(content)
            
            # If we got results, filter and score them by similarity
            if weighted_memories:
                # Calculate similarity for each memory
                similar_memories = []
                content_words = set(content.lower().split())
                
                for memory in weighted_memories:
                    memory_content = memory.get('content', '')
                    if not memory_content:
                        continue
                        
                    # Calculate similarity using same method as _is_similar_content
                    memory_words = set(memory_content.lower().split())
                    if not memory_words or not content_words:
                        continue
                        
                    intersection = len(content_words.intersection(memory_words))
                    union = len(content_words.union(memory_words))
                    similarity = intersection / union if union > 0 else 0
                    
                    # Add similarity score to the memory
                    memory['similarity_score'] = similarity
                    
                    # Only include memories above threshold
                    if similarity >= threshold:
                        similar_memories.append(memory)
                
                # Sort by similarity score
                similar_memories.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                return similar_memories
                
            return []  # No similar memories found
            
        except Exception as e:
            logging.error(f"Error searching for similar memories: {e}", exc_info=True)
            return []
        
    #Used by Self Reflection
    def _format_db_row_to_dict(self, row):
        """
        Format a database row into a memory dictionary.
        
        Args:
            row: Database query result row
            
        Returns:
            Dict: Formatted memory dictionary
        """
        if not row:
            return {}
            
        id_val, content, memory_type, source, weight, access_count, created_at, last_accessed, tags = row
        
        memory_dict = {
            'id': id_val,
            'content': content,
            'memory_type': memory_type or 'general',
            'source': source or '',
            'weight': weight or 1.0,
            'access_count': access_count or 0,
            'created_at': created_at,
            'last_accessed': last_accessed,
            'tags': tags,
            'similarity_score': 1.0  # Exact matches get perfect score
        }
        
        return memory_dict

    def store_memory(self, content, memory_type="general", source="unknown", 
                confidence=0.5, tags=None, additional_metadata=None, retry_count=3):
        """
        Store a memory in the database with transaction protection and retries.
        
        This method now properly supports transaction coordination by using provided
        tracking_id from additional_metadata when available, ensuring UUID consistency
        across SQL and Vector databases.

        Args:
            content (str): The content to store
            memory_type (str): Type of memory (general, important, etc.)
            source (str): Source of the memory
            confidence (float, optional): Confidence value from 0.0-1.0
            tags (str, optional): Tags for the memory
            retry_count (int): Number of retries on failure
            additional_metadata (dict, optional): Additional metadata key-value pairs to store.
                                                If contains 'tracking_id', that UUID will be used
                                                instead of generating a new one.

        Returns:
            bool: Success status
        """
        # Guard against None or empty content
        if content is None or not isinstance(content, str) or not content.strip():
            logging.warning("Attempted to store empty memory")
            return False
            
        # Guard against None memory_type
        if memory_type is None:
            memory_type = "general"
            logging.warning("None memory_type provided, using 'general' as default")

        # Calculate confidence weight
        if confidence is not None:
            initial_weight = confidence
        else:
            initial_weight = self.calculate_memory_weight(
                memory_type=memory_type,
                access_count=0,
                created_at=datetime.now()
            )

        # Process additional metadata for JSON storage
        metadata_json = None
        if additional_metadata and isinstance(additional_metadata, dict):
            try:
                import json
                metadata_json = json.dumps(additional_metadata)
                logging.debug(f"Prepared additional metadata: {metadata_json[:100]}...")
            except Exception as e:
                logging.warning(f"Failed to serialize additional metadata: {e}")
                # Continue with metadata_json = None rather than failing

        # CRITICAL FIX: Use provided tracking_id for transaction coordination consistency
        # This ensures the same UUID is used across SQL and Vector databases
        if additional_metadata and isinstance(additional_metadata, dict) and 'tracking_id' in additional_metadata:
            memory_id = str(additional_metadata['tracking_id'])  # Ensure it's a string
            logging.debug(f"Transaction coordination: Using provided tracking_id: {memory_id}")
        else:
            memory_id = str(uuid.uuid4())
            logging.debug(f"Standalone storage: Generated new memory_id: {memory_id}")

        # Validate the memory_id format (basic UUID validation)
        try:
            # This will raise ValueError if memory_id is not a valid UUID format
            uuid.UUID(memory_id)
        except ValueError as e:
            logging.error(f"Invalid UUID format for memory_id '{memory_id}': {e}")
            # Generate a new valid UUID as fallback
            memory_id = str(uuid.uuid4())
            logging.warning(f"Generated fallback UUID: {memory_id}")

        # Add retry loop with exponential backoff for database resilience
        last_error = None
        for attempt in range(retry_count):
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Enable WAL mode for better concurrency (allows concurrent reads during writes)
                    conn.execute("PRAGMA journal_mode=WAL")
                    
                    # Begin explicit transaction for atomicity
                    # IMMEDIATE prevents other writers from starting but allows readers
                    conn.execute("BEGIN IMMEDIATE TRANSACTION")
            
                    cursor = conn.cursor()
                    
                    # Ensure database schema is up to date
                    try:
                        # Check if required columns exist
                        cursor.execute("PRAGMA table_info(memories)")
                        columns = [column[1] for column in cursor.fetchall()]
                        
                        # Add tags column if it doesn't exist (backward compatibility)
                        if 'tags' not in columns:
                            cursor.execute('ALTER TABLE memories ADD COLUMN tags TEXT')
                            logging.info("Added tags column to memories table")
                    
                        # Add metadata column if it doesn't exist (backward compatibility)
                        if 'metadata' not in columns:
                            cursor.execute('ALTER TABLE memories ADD COLUMN metadata TEXT DEFAULT "{}"')
                            logging.info("Added metadata column to memories table")
                            
                    except sqlite3.Error as schema_error:
                        logging.error(f"Schema update failed: {schema_error}")
                        # Continue anyway - the INSERT might still work with existing schema
            
                    # Store memory with all available data
                    # tracking_id now consistently uses the same UUID as transaction coordinator
                    try:
                        cursor.execute("""
                            INSERT INTO memories 
                            (content, memory_type, source, weight, tags, tracking_id, access_count, created_at, last_accessed, metadata) 
                            VALUES (?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                        """, (content, memory_type, source, initial_weight, tags, memory_id, metadata_json))
                        
                        # Get the auto-generated row ID (different from our UUID tracking_id)
                        row_id = cursor.lastrowid
                        
                        # Commit transaction atomically
                        conn.commit()
                        
                        # Log success with both IDs for debugging
                        logging.info(f"[Attempt {attempt+1}] Successfully stored memory with tracking_id {memory_id} " 
                                    f"(row_id: {row_id}) and weight {initial_weight}: {content[:50]}...")
                        
                        # Store references for potential rollback operations
                        self._last_added_memory_id = memory_id  # Our UUID
                        self._last_added_row_id = row_id        # SQLite's auto-increment ID
                        
                        # SUCCESS: Return True to indicate successful storage
                        return True
                        
                    except sqlite3.IntegrityError as integrity_error:
                        # Handle constraint violations (e.g., duplicate tracking_id)
                        logging.error(f"Integrity constraint violation: {integrity_error}")
                        if "tracking_id" in str(integrity_error).lower() or "unique" in str(integrity_error).lower():
                            logging.error(f"Duplicate tracking_id detected: {memory_id}")
                            # For transaction coordination, this is a serious error
                            return False
                        raise  # Re-raise other integrity errors for retry logic
                        
            except sqlite3.OperationalError as op_error:
                # Handle database locked, disk full, etc.
                last_error = op_error
                if "database is locked" in str(op_error).lower():
                    if attempt < retry_count - 1:
                        backoff = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s...
                        logging.warning(f"Database locked on attempt {attempt+1}/{retry_count}. "
                                    f"Retrying in {backoff:.1f}s...")
                        time.sleep(backoff)
                        continue
                    else:
                        logging.error(f"Database remained locked after {retry_count} attempts")
                        return False
                else:
                    logging.error(f"Operational error on attempt {attempt+1}: {op_error}")
                    raise  # Re-raise for general retry logic
                    
            except sqlite3.Error as db_error:
                # Handle other database errors with retry logic
                last_error = db_error
                if attempt < retry_count - 1:
                    # Calculate exponential backoff (0.5s, 1s, 2s...)
                    backoff = 0.5 * (2 ** attempt)
                    logging.warning(f"Database error on attempt {attempt+1}/{retry_count}: {db_error}. "
                                f"Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    logging.error(f"Failed to store memory after {retry_count} attempts. "
                                f"Final error: {db_error}")
                    return False
                    
            except Exception as unexpected_error:
                # Handle unexpected errors (programming errors, etc.)
                logging.error(f"Unexpected error in store_memory attempt {attempt+1}: {unexpected_error}", 
                            exc_info=True)
                if attempt < retry_count - 1:
                    backoff = 1.0 * (2 ** attempt)  # Longer backoff for unexpected errors
                    logging.warning(f"Retrying after unexpected error in {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    logging.error(f"Failed to store memory after {retry_count} attempts due to unexpected errors")
                    return False

        # If we reach here, all retry attempts failed
        logging.error(f"All {retry_count} storage attempts failed for memory: {content[:50]}...")
        if last_error:
            logging.error(f"Last error was: {last_error}")
        return False


    def contains(self, content: str) -> bool:
        """Check if a specific content chunk already exists in the database."""
        try:
            # Guard against None or empty content
            if content is None or not isinstance(content, str) or not content.strip():
                logging.warning("Attempted to check existence of None or empty content")
                return False
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM memories WHERE content = ?",
                    (content,)
                )
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logging.error(f"Error checking content existence: {e}")
            return False
        
    
    def get_memories_by_weight(self, query: str = None, comprehensive: bool = True) -> List[Dict[str, Any]]:
        """Retrieve memories ordered by relevance to the query, with optional comprehensive results."""
        try:
            # Guard against None but allow empty query for retrieving all memories
            if query is None:
                query = ""
                logging.warning("None query provided to get_memories_by_weight, using empty string")
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if query:
                    if comprehensive:
                        sql = """
                            SELECT id, content, memory_type, source, weight, access_count, 
                                   created_at, last_accessed 
                            FROM memories 
                            WHERE content LIKE ? 
                            ORDER BY CASE 
                                WHEN content LIKE ? THEN 2  
                                ELSE 1 END DESC, weight DESC
                        """
                        cursor.execute(sql, (f"%{query}%", f"%{query}%"))
                    else:
                        sql = """
                            SELECT id, content, memory_type, source, weight, access_count, 
                                   created_at, last_accessed 
                            FROM memories 
                            WHERE content LIKE ? 
                            ORDER BY weight DESC 
                            LIMIT 10
                        """
                        cursor.execute(sql, (f"%{query}%",))
                else:
                    sql = """
                        SELECT id, content, memory_type, source, weight, access_count, 
                               created_at, last_accessed 
                        FROM memories 
                        ORDER BY weight DESC 
                        LIMIT 10
                    """
                    cursor.execute(sql)
                
                memories = cursor.fetchall()
                
                # Guard against None result (shouldn't happen with fetchall but just in case)
                if memories is None:
                    return []
                    
                for memory in memories:
                    self.update_memory_access(memory[0])
                
                formatted_memories = []
                for memory in memories:
                    memory_dict = {
                        'id': memory[0],
                        'content': memory[1],
                        'memory_type': memory[2],
                        'source': memory[3],
                        'weight': memory[4],
                        'access_count': memory[5],
                        'created_at': memory[6],
                        'last_accessed': memory[7]
                    }
                    formatted_memories.append(self.format_memory_result(memory_dict))
                
                return formatted_memories
                
        except Exception as e:
            logging.error(f"Error retrieving weighted memories: {e}")
            return []

    def get_memories_by_type(self, memory_type: str, limit: int = None, order_by: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve all memories of a specific type with optional ordering and limit.
        
        Args:
            memory_type (str): Type of memories to retrieve
            limit (int, optional): Maximum number of memories to retrieve
            order_by (str, optional): Ordering specification (e.g., "created_at DESC", "weight ASC")
                                    Default is "created_at DESC" if not specified
        
        Returns:
            List[Dict[str, Any]]: List of memory dictionaries
        """
        try:
            # Guard against None or empty memory_type
            if memory_type is None or not isinstance(memory_type, str) or not memory_type.strip():
                logging.warning("Attempted to get memories with None or empty memory_type")
                return []
            
            # Parse order_by parameter
            if order_by is None:
                # Default ordering is by creation date (newest first)
                order_field = "created_at"
                order_direction = "DESC"
            else:
                # Split the order_by string to get field and direction
                order_parts = order_by.strip().split()
                order_field = order_parts[0]
                # If direction is specified, use it; otherwise default to ASC
                order_direction = order_parts[1].upper() if len(order_parts) > 1 else "ASC"
                
                # Validate the field name to prevent SQL injection
                valid_fields = ["id", "content", "memory_type", "source", "weight", 
                            "access_count", "created_at", "last_accessed"]
                if order_field not in valid_fields:
                    logging.warning(f"Invalid order field: {order_field}. Using default: created_at")
                    order_field = "created_at"
                    order_direction = "DESC"
                
                # Validate the direction
                if order_direction not in ["ASC", "DESC"]:
                    logging.warning(f"Invalid order direction: {order_direction}. Using default: DESC")
                    order_direction = "DESC"
            
            logging.info(f"Retrieving memories by type: {memory_type} with ordering: {order_field} {order_direction}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                sql = f"""
                    SELECT id, content, memory_type, source, weight, access_count, 
                        created_at, last_accessed 
                    FROM memories 
                    WHERE memory_type = ? 
                    ORDER BY {order_field} {order_direction}
                """
                
                if limit is not None:
                    sql += " LIMIT ?"
                    cursor.execute(sql, (memory_type, limit))
                else:
                    cursor.execute(sql, (memory_type,))
                
                memories = cursor.fetchall()
            
                # Guard against None result
                if memories is None:
                    return []
                
                formatted_memories = []
                for memory in memories:
                    memory_dict = {
                        'id': memory[0],
                        'content': memory[1] if memory[1] is not None else "",
                        'memory_type': memory[2] if memory[2] is not None else "general",
                        'source': memory[3] if memory[3] is not None else "",
                        'weight': memory[4] if memory[4] is not None else 1.0,
                        'access_count': memory[5] if memory[5] is not None else 0,
                        'created_at': memory[6],
                        'last_accessed': memory[7]
                    }
                    formatted_memories.append(self.format_memory_result(memory_dict))
            
                logging.info(f"Retrieved {len(formatted_memories)} memories of type '{memory_type}'")
                return formatted_memories
            
        except Exception as e:
            logging.error(f"Error retrieving memories by type: {e}", exc_info=True)
            return []
        
    def get_memory_by_id(self, memory_id: int) -> dict:
        """Get a memory by its ID from the database.
        
        Args:
            memory_id (int): The ID of the memory to retrieve
            
        Returns:
            dict: The memory object or None if not found
        """
        try:
            # Log the ID being searched for debugging
            logging.debug(f"Searching for memory with ID: {memory_id}, Type: {type(memory_id)}")
            
            # Make sure we have an integer
            if not isinstance(memory_id, int):
                try:
                    memory_id = int(memory_id)
                except (ValueError, TypeError):
                    logging.error(f"Invalid memory_id: {memory_id}. Must be an integer.")
                    return None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, content, memory_type, source, weight, access_count, 
                        created_at, last_accessed, tracking_id, metadata
                    FROM memories 
                    WHERE id = ?
                """, (memory_id,))
                
                row = cursor.fetchone()
                
                if not row:
                    logging.warning(f"No memory found with ID {memory_id}")
                    return None
                    
                # Extract metadata from JSON if present
                metadata_raw = row[9]
                metadata = {}
                if metadata_raw:
                    try:
                        if isinstance(metadata_raw, dict):
                            metadata = metadata_raw
                        else:
                            # Try to parse as JSON
                            metadata = json.loads(metadata_raw)
                    except Exception as e:
                        logging.warning(f"Failed to parse metadata for memory {memory_id}: {e}")
                
                memory = {
                    'id': row[0],
                    'content': row[1],
                    'memory_type': row[2],
                    'source': row[3],
                    'weight': row[4],
                    'access_count': row[5],
                    'created_at': row[6],
                    'last_accessed': row[7],
                    'tracking_id': row[8],
                    'metadata': metadata
                }
                
                return memory
        except Exception as e:
            logging.error(f"Error retrieving memory by ID {memory_id}: {e}", exc_info=True)
            return None
        
    def get_memories_since(self, days: int = 1, limit: int = 50) -> list:
        """
        Retrieve memories created within the past N days, returning structured dicts.
        Used by the reflection engine to provide time-appropriate memory windows
        for daily, weekly, and monthly reflection cycles.
        
        Args:
            days (int): Number of days to look back (1=daily, 7=weekly, 30=monthly)
            limit (int): Maximum number of memories to return
            
        Returns:
            list: List of dicts with keys: content, memory_type, weight, 
                created_at, source, confidence
        """
        try:
            # Guard against invalid parameters
            if not isinstance(days, int) or days <= 0:
                logging.warning(f"Invalid days value: {days}, defaulting to 1")
                days = 1
            if not isinstance(limit, int) or limit <= 0:
                logging.warning(f"Invalid limit value: {limit}, defaulting to 50")
                limit = 50

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Use SQLite native datetime math on confirmed YYYY-MM-DD HH:MM:SS format
                # Concatenation approach avoids parameter binding issues with interval strings
                cursor.execute("""
                    SELECT 
                        content,
                        memory_type,
                        weight,
                        created_at,
                        source,
                        confidence
                    FROM memories
                    WHERE created_at >= datetime('now', '-' || ? || ' days')
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (days, limit))
                
                rows = cursor.fetchall()
                
                if not rows:
                    logging.info(f"[MEMORY_DB] get_memories_since: No memories found in past {days} days")
                    return []
                
                # Return clean structured dicts — no string formatting needed
                # reflection_engine._get_recent_memories expects content and memory_type keys
                memories = []
                for row in rows:
                    memories.append({
                        'content':     row[0] if row[0] is not None else "",
                        'memory_type': row[1] if row[1] is not None else "general",
                        'weight':      row[2],
                        'created_at':  row[3],
                        'source':      row[4],
                        'confidence':  row[5]
                    })
                
                logging.info(f"[MEMORY_DB] get_memories_since: Retrieved {len(memories)} memories "
                            f"from past {days} days (limit={limit})")
                return memories
                
        except Exception as e:
            logging.error(f"[MEMORY_DB] Error in get_memories_since (days={days}, limit={limit}): {e}", 
                        exc_info=True)
            return []
       
            
    def format_memory_result(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Format a memory result with enhanced metadata and confidence."""
        try:
            # Guard against None input
            if memory is None:
                logging.warning("Attempted to format None memory")
                return {
                    'content': '',
                    'confidence': {'level': 'Low', 'score': 0.1},
                    'metadata': {'type': 'general', 'source': 'Unknown', 'weight': 1.0, 'age_days': 0}
                }
        
            # Get content with safe default
            content = memory.get('content', '')
            if content is None:
                content = ''
        
            # Get memory type with safe default
            memory_type = memory.get('memory_type', memory.get('type', 'general'))
            if memory_type is None:
                memory_type = 'general'
        
            # Define date_formats OUTSIDE of any conditional blocks
            # This ensures it's always available when needed later
            date_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S.%f'
            ]
            
            # Improved date parsing with multiple format support
            created_date = None
            created_at_raw = memory.get('created_at')
            if created_at_raw:
                if isinstance(created_at_raw, datetime):
                    created_date = created_at_raw
                elif isinstance(created_at_raw, str):
                    # Try multiple date formats
                    for fmt in date_formats:
                        try:
                            created_date = datetime.strptime(created_at_raw.split('.')[0], fmt)
                            break
                        except ValueError:
                            continue
            
            # If all parsing failed, use current time
            if created_date is None:
                created_date = datetime.now()
                logging.warning(f"Failed to parse created_at date '{created_at_raw}', using current time")
            
            # Calculate age with safe computation
            try:
                if created_date:
                    # Handle potential future date issues
                    now = datetime.now()
                    is_future = created_date > now
                    
                    # Check if same calendar day (regardless of time)
                    is_same_day = (
                        created_date.year == now.year and 
                        created_date.month == now.month and
                        created_date.day == now.day
                    )
                    
                    # Calculate time difference
                    if is_future:
                        time_diff = created_date - now
                        # If it's the same calendar day or within 24 hours, treat as current time
                        if is_same_day or time_diff.total_seconds() < 86400:  # 86400 seconds = 24 hours
                            age_days = 0
                        else:
                            # Only log warnings for dates more than a day in the future
                            logging.warning(f"Future date detected: {created_date}, treating as current day")
                            age_days = 0
                    else:
                        # Calculate age normally for past dates
                        age_days = (now - created_date).days
                else:
                    age_days = 0
            except (TypeError, AttributeError, ValueError) as e:
                logging.warning(f"Invalid created_date format: {created_date}, error: {e}")
                age_days = 0
        
            # Get weight with safe fallback
            try:
                weight = float(memory.get('weight', 1.0))
                if weight <= 0 or weight > 10:  # Validate weight is reasonable
                    logging.warning(f"Invalid weight value: {weight}, using default")
                    weight = 1.0
            except (TypeError, ValueError):
                weight = 1.0

            # Define confidence based on memory type
            type_confidence = {
                'important': 0.8,      # Highest confidence for explicitly marked important memories
                'document': 0.7,       # High confidence for document-sourced information
                'self': 0.75,          # High confidence for self-knowledge/identity information
                'reflection': 0.7,     # High confidence for reflections (they represent system's understanding)
                'conversation': 0.65,  # Medium-high for conversation summaries
                'reminder': 0.75,      # High confidence for reminders (they're explicitly created)
                'general': 0.6         # Standard confidence for general memories
            }
            base_confidence = type_confidence.get(memory_type, 0.6)

            # Calculate age factor with safe math - special cases for different memory types
            if memory_type in ['important', 'document', 'self']:
                # These types should decay more slowly over time
                age_factor = 1.0 / (1.0 + (age_days / 180))
            elif memory_type == 'reminder':
                # Reminders should maintain high confidence until their due date
                age_factor = 1.0  # Could implement due-date based confidence in the future
            else:
                # Standard decay for other memory types
                age_factor = 1.0 / (1.0 + (age_days / 30))
        
            confidence_score = base_confidence * age_factor * min(weight, 2.0)  # Cap weight influence
            confidence_score = max(0.1, min(1.0, confidence_score))  # Ensure in range 0.1-1.0

            confidence_level = "High" if confidence_score >= 0.7 else "Medium" if confidence_score >= 0.4 else "Low"
        
            # Get source with safe default
            source = memory.get('source', '')
            if source is None:
                source = ''
        
            # Improved source formatting with additional types
            if memory_type == 'document' and source:
                source = f"Document: {source}"
            elif memory_type == 'conversation':
                source = "Conversation Summary"
            elif memory_type == 'self':
                source = "Self-Knowledge"
            elif memory_type == 'reflection':
                source = f"Reflection: {source}" if source else "Reflection"
            elif memory_type == 'reminder':
                source = f"Reminder: {source}" if source else "Reminder"
            else:
                source = "User Memory" if not source else source
        
            # Get access count with safe default
            try:
                access_count = int(memory.get('access_count', 0))
                if access_count < 0:  # Validate count is non-negative
                    access_count = 0
            except (TypeError, ValueError):
                access_count = 0
        
            # Parse last_accessed with improved handling
            last_accessed_iso = None
            last_accessed = memory.get('last_accessed')
            if last_accessed is not None:
                try:
                    if isinstance(last_accessed, datetime):
                        last_accessed_iso = last_accessed.isoformat()
                    elif isinstance(last_accessed, str):
                        # Try multiple formats, similar to created_at parsing
                        for fmt in date_formats:
                            try:
                                last_accessed_dt = datetime.strptime(last_accessed.split('.')[0], fmt)
                                last_accessed_iso = last_accessed_dt.isoformat()
                                break
                            except ValueError:
                                continue
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Error parsing last_accessed: {e}")
                    last_accessed_iso = None
        
            # Get tags with safe default
            tags = memory.get('tags', None)
            if tags is not None and not isinstance(tags, str):
                try:
                    tags = str(tags)
                except Exception:
                    tags = None
                
            formatted_memory = {
                'content': content,
                'confidence': {'level': confidence_level, 'score': round(confidence_score, 2)},
                'metadata': {
                    'type': memory_type,
                    'source': source,
                    'age_days': age_days,
                    'weight': weight,
                    'created_at': created_date.isoformat() if created_date else None,
                    'last_accessed': last_accessed_iso,
                    'access_count': access_count,
                    'tags': tags,
                    'id': int(memory.get('id', 0)) if memory.get('id') else None  # Ensure ID is an integer
                }
            }

            return formatted_memory

        except Exception as e:
            logging.error(f"Error formatting memory result: {str(e)}", exc_info=True)
            return {
                'content': memory.get('content', ''),
                'confidence': {'level': 'Low', 'score': 0.1},
                'metadata': {'type': 'general', 'source': 'Unknown', 'weight': 1.0, 'age_days': 0}
            }
    
                
    def check_document_processed(self, filename: str) -> bool:
        """Check if a document has already been processed."""
        try:
            # Guard against None or empty filename
            if filename is None or not isinstance(filename, str) or not filename.strip():
                logging.warning(f"Invalid filename provided to check_document_processed: {filename}")
                return False
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM memories WHERE source = ?",
                    (filename,)
                )
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logging.error(f"Error checking document status: {e}")
            return False
      
    
    def delete_memory(self, content: str) -> bool:
        """Delete a memory containing the specified content."""
        try:
            # Guard against None or empty content
            if content is None or not isinstance(content, str) or not content.strip():
                logging.warning("Attempted to delete None or empty memory content")
                return False
            
            with sqlite3.connect(self.db_path) as conn:
                # Explicitly start a transaction
                conn.execute("BEGIN TRANSACTION")
            
                cursor = conn.cursor()
        
                # Try exact match first
                cursor.execute(
                    "DELETE FROM memories WHERE content = ?",
                    (content,)
                )
        
                deleted = cursor.rowcount > 0
        
                # If no exact match, try partial match
                if not deleted:
                    cursor.execute(
                        "DELETE FROM memories WHERE content LIKE ?",
                        (f"%{content}%",)
                    )
                    deleted = cursor.rowcount > 0
        
                # Commit or rollback transaction based on success
                if deleted:
                    conn.commit()
                    logging.info(f"Successfully deleted memory containing: {content[:100]}...")
                else:
                    conn.rollback()
                    logging.info(f"No memory found containing: {content[:100]}...")
            
                return deleted
        except Exception as e:
            logging.error(f"Error deleting memory: {e}")
            return False

    def delete_memory_by_tracking_id(self, tracking_id: str) -> bool:
        """
        Delete a memory by its tracking ID from both SQL and vector databases with proper error handling.
        
        Args:
            tracking_id (str): The tracking ID of the memory to delete
            
        Returns:
            bool: True if successfully deleted from both databases, False otherwise
        """
        try:
            logging.info(f"Starting deletion of memory with tracking_id {tracking_id}")
            
            # Step 1: Check SQLite
            sqlite_success = False
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, content, memory_type FROM memories WHERE tracking_id = ?", (tracking_id,))
                memory_info = cursor.fetchone()
                if memory_info:
                    mem_id, content, mem_type = memory_info
                    logging.info(f"Found memory to delete: ID={mem_id}, Type={mem_type}, Content={content[:50]}...")
                    
                    cursor.execute("DELETE FROM memories WHERE tracking_id = ?", (tracking_id,))
                    sqlite_success = cursor.rowcount > 0
                    conn.commit()
                    logging.info(f"SQLite deletion {'succeeded' if sqlite_success else 'failed'}")
                else:
                    logging.warning(f"No memory found with tracking_id {tracking_id} in SQLite")
            
            # Step 2: Check vector store
            vector_success = False
            vector_db_available = hasattr(self, 'vector_db') and self.vector_db is not None
            if vector_db_available:
                try:
                    # Use the correct Qdrant API method for deleting points by filter condition
                    collection_name = "memories"
                    delete_filter = rest.Filter(
                        must=[rest.FieldCondition(
                            key="tracking_id", 
                            match=rest.MatchValue(value=tracking_id)
                        )]
                    )
                    
                    # Delete from Qdrant with proper filter
                    delete_result = self.vector_db.delete(
                        collection_name=collection_name,
                        points_selector=rest.FilterSelector(filter=delete_filter)
                    )
                    
                    # Check if deletion was successful
                    if hasattr(delete_result, 'status') and delete_result.status == 'ok':
                        vector_success = True
                        logging.info(f"Deleted memory with tracking_id {tracking_id} from vector database")
                    else:
                        logging.warning(f"Vector database deletion for tracking_id {tracking_id} may have failed")
                except Exception as e:
                    logging.error(f"Failed to delete from vector database: {e}", exc_info=True)
            
            # Step 3: Handle synchronization
            overall_success = sqlite_success or not memory_info  # Success if deleted or didn't exist
            if vector_db_available:
                overall_success = overall_success and vector_success
            
            if not overall_success:
                logging.warning(f"Deletion failed for tracking_id {tracking_id}, queuing for retry")
                self.queue_for_deletion(tracking_id)
            
            return overall_success
        
        except Exception as e:
            logging.error(f"Error deleting memory with tracking_id {tracking_id}: {e}", exc_info=True)
            self.queue_for_deletion(tracking_id)
            return False
        
        
    def queue_for_deletion(self, tracking_id: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO deletion_queue (tracking_id, attempt_count, last_attempt)
                    VALUES (?, 0, NULL)
                """, (tracking_id,))
                conn.commit()
                logging.info(f"Queued tracking_id {tracking_id} for deletion retry")
                # Signal the demand-driven processor to start if wired up
                if self.on_item_queued:
                    self.on_item_queued()
        except Exception as e:
            logging.error(f"Error queuing tracking_id {tracking_id} for deletion: {e}")

    def process_deletion_queue(self, max_attempts: int = 2, retry_interval_minutes: int = 5, max_duration_minutes: int = 15) -> None:
        """Process items in the deletion queue with time-based retry logic."""
        from datetime import timedelta  # Local import as a failsafe
        try:
            now = datetime.utcnow()
            
            # Cleanup: Remove items older than 15 minutes
            cutoff_time = now - timedelta(minutes=max_duration_minutes)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM deletion_queue WHERE created_at < ?", (cutoff_time.strftime('%Y-%m-%d %H:%M:%S'),))
                removed_count = cursor.rowcount
                conn.commit()
                if removed_count > 0:
                    logging.warning(f"Removed {removed_count} items from deletion_queue that exceeded {max_duration_minutes} minutes")
            
            # Select items eligible for retry: < 3 attempts and last_attempt > 5 minutes ago (or NULL)
            retry_cutoff = now - timedelta(minutes=retry_interval_minutes)
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT tracking_id, attempt_count, last_attempt
                    FROM deletion_queue
                    WHERE attempt_count < ?
                    AND (last_attempt IS NULL OR last_attempt < ?)
                """, (max_attempts, retry_cutoff.strftime('%Y-%m-%d %H:%M:%S')))
                queued_items = cursor.fetchall()
            
            if not queued_items:
                logging.debug("No items in deletion queue eligible for retry")  # was INFO
                return
                        
            logging.info(f"Processing {len(queued_items)} items in deletion queue")
            for item in queued_items:
                tracking_id = item['tracking_id']
                attempt_count = item['attempt_count'] + 1
                
                # Update attempt count and last attempt timestamp
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE deletion_queue
                        SET attempt_count = ?, last_attempt = ?
                        WHERE tracking_id = ?
                    """, (attempt_count, now.strftime('%Y-%m-%d %H:%M:%S'), tracking_id))
                    conn.commit()
                
                # Attempt deletion
                success = self.delete_memory_by_tracking_id(tracking_id)
                
                if success:
                    # Remove from queue if successful
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM deletion_queue WHERE tracking_id = ?", (tracking_id,))
                        conn.commit()
                        logging.info(f"Successfully deleted tracking_id {tracking_id}")
                else:
                    if attempt_count >= max_attempts:
                        # Store failure memory
                        failure_content = f"Failed to delete memory with tracking_id {tracking_id} after {max_attempts} attempts"
                        self.store_memory(
                            content=failure_content,
                            memory_type="deletion_failure",
                            source="system",
                            additional_metadata={"tracking_id": tracking_id}
                        )
                        # Remove from queue
                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM deletion_queue WHERE tracking_id = ?", (tracking_id,))
                            conn.commit()
                        logging.error(f"Max attempts reached for tracking_id {tracking_id}, stored failure memory and removed from queue")
        
        except Exception as e:
            logging.error(f"Error processing deletion queue: {e}")
   
                