# knowledge_gap.py - Updated version with semantic similarity checking
"""Knowledge gap tracking and management for autonomous learning with duplicate prevention."""

import sqlite3
import logging
import datetime
import uuid
from typing import List, Tuple, Dict, Any, Optional

# Qdrant imports for semantic similarity
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_ollama import OllamaEmbeddings
# Import EMBEDDING_MODEL with alias to avoid shadowing the class-level attribute below.
# OLLAMA_MODEL and OLLAMA_BASE_URL imported normally — they aren't shadowed anywhere.
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, EMBEDDING_MODEL as _CONFIG_EMBEDDING_MODEL


class KnowledgeGapQueue:
    """Manages and prioritizes identified knowledge gaps with semantic duplicate prevention."""
    
    # Class-level constants for semantic similarity
    SIMILARITY_THRESHOLD = 0.80  # 80% similarity = duplicate
    # Embedding model name sourced from config.py for project-wide consistency.
    # Aliased import prevents shadowing — class attribute remains the public API
    # for methods using self.EMBEDDING_MODEL, but the value comes from one place.
    EMBEDDING_MODEL = _CONFIG_EMBEDDING_MODEL
    
    def __init__(self, db_path, vector_db_url: str = "http://localhost:6333", 
                 gaps_collection_name: str = None):
        """
        Initialize with the database path and optional Qdrant configuration.
        
        Args:
            db_path: Path to SQLite database
            vector_db_url: URL for Qdrant server (default: localhost:6333)
            gaps_collection_name: Name of Qdrant collection for gap embeddings
        """
        self.db_path = db_path
        self.vector_db_url = vector_db_url
        
        # Import collection name from config, with fallback
        try:
            from config import QDRANT_GAPS_COLLECTION_NAME, OLLAMA_BASE_URL
            self.gaps_collection_name = gaps_collection_name or QDRANT_GAPS_COLLECTION_NAME
            self.ollama_base_url = OLLAMA_BASE_URL
        except ImportError:
            # Fallback defaults if config not available
            self.gaps_collection_name = gaps_collection_name or "knowledge_gaps_embeddings"
            self.ollama_base_url = "http://localhost:11434"
            logging.warning("Could not import from config, using defaults")
        
        # Initialize SQL database
        self._initialize_db()
        
        # Initialize Qdrant client and embeddings (lazy loading)
        self._qdrant_client = None
        self._embeddings = None
        self._collection_initialized = False
        
    @property
    def qdrant_client(self):
        """Lazy initialization of Qdrant client."""
        if self._qdrant_client is None:
            try:
                self._qdrant_client = QdrantClient(url=self.vector_db_url, timeout=30.0)
                logging.info(f"Connected to Qdrant at {self.vector_db_url}")
            except Exception as e:
                logging.error(f"Failed to connect to Qdrant: {e}")
                raise
        return self._qdrant_client
    
    @property
    def embeddings(self):
        """Lazy initialization of embeddings model."""
        if self._embeddings is None:
            try:
                self._embeddings = OllamaEmbeddings(
                    model=self.EMBEDDING_MODEL,
                    base_url=self.ollama_base_url
                )
                logging.info(f"Initialized {self.EMBEDDING_MODEL} embeddings")
            except Exception as e:
                logging.error(f"Failed to initialize embeddings: {e}")
                raise
        return self._embeddings
        
    def _initialize_db(self):
        """Create the knowledge gaps table if it doesn't exist and ensure all columns are present."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create the base table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS knowledge_gaps (
                        id INTEGER PRIMARY KEY,
                        topic TEXT NOT NULL,
                        description TEXT,
                        priority FLOAT DEFAULT 0.5,
                        status TEXT DEFAULT 'pending',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        fulfilled_at DATETIME
                    )
                ''')
                
                # Check existing columns
                cursor.execute("PRAGMA table_info(knowledge_gaps)")
                existing_columns = {row[1] for row in cursor.fetchall()}
                
                # Add missing columns if they don't exist.
                # Existing databases are migrated automatically on next startup.
                required_columns = {
                    'items_acquired': 'INTEGER DEFAULT 0',
                    'last_attempt_at': 'DATETIME',
                    'vector_id': 'TEXT',       # Qdrant point ID for embedding cleanup
                    'attempt_count': 'INTEGER DEFAULT 0'  # Tracks fill attempts for retry limit
                }
                
                for column_name, column_def in required_columns.items():
                    if column_name not in existing_columns:
                        try:
                            cursor.execute(f'ALTER TABLE knowledge_gaps ADD COLUMN {column_name} {column_def}')
                            logging.info(f"Added missing column '{column_name}' to knowledge_gaps table")
                        except Exception as e:
                            logging.warning(f"Could not add column '{column_name}': {e}")
                
                # Add indexes for faster querying
                try:
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_status_priority
                        ON knowledge_gaps(status, priority)
                    ''')
                except Exception as e:
                    logging.warning(f"Could not create status_priority index: {e}")
                
                try:
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_topic
                        ON knowledge_gaps(topic)
                    ''')
                except Exception as e:
                    logging.warning(f"Could not create topic index: {e}")
                
                conn.commit()
                logging.info("Knowledge gaps table initialized successfully with all required columns")
                
        except Exception as e:
            logging.error(f"Error initializing knowledge gaps table: {e}")
            raise

    def _ensure_gaps_collection_exists(self) -> bool:
        """
        Ensure the Qdrant collection for knowledge gap embeddings exists.
        Creates it if necessary with proper vector configuration.
        
        Returns:
            bool: True if collection exists or was created successfully
        """
        if self._collection_initialized:
            return True
            
        try:
            # Check if collection already exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.gaps_collection_name for c in collections)
            
            if collection_exists:
                logging.info(f"Knowledge gaps collection '{self.gaps_collection_name}' already exists")
                self._collection_initialized = True
                return True
            
            # Get embedding dimension from a sample embedding
            sample_embedding = self.embeddings.embed_documents(["test dimension check"])
            vector_dimension = len(sample_embedding[0])
            
            # Create the collection
            self.qdrant_client.create_collection(
                collection_name=self.gaps_collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_dimension,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            
            logging.info(f"Created knowledge gaps collection '{self.gaps_collection_name}' "
                        f"with dimension {vector_dimension}")
            self._collection_initialized = True
            return True
            
        except Exception as e:
            logging.error(f"Error ensuring gaps collection exists: {e}")
            return False

    def _generate_embedding(self, topic: str, description: str) -> Optional[List[float]]:
        """
        Generate an embedding for a knowledge gap.
        
        Args:
            topic: The gap topic
            description: The gap description
            
        Returns:
            List[float]: The embedding vector, or None on failure
        """
        try:
            # Combine topic and description for richer semantic representation
            combined_text = f"{topic}: {description}"
            
            # Generate embedding
            embedding = self.embeddings.embed_documents([combined_text])
            
            if embedding and len(embedding) > 0:
                return embedding[0]
            else:
                logging.warning("Empty embedding returned")
                return None
                
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None

    def check_semantic_similarity(self, topic: str, description: str,
                                   threshold: float = None) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a semantically similar knowledge gap already exists using vector similarity.
        
        M1 FIX: threshold is tested with 'is not None' rather than truthiness so that
        an explicit threshold=0.0 (catch-all mode) is not silently overridden by the
        class default.
        
        Args:
            topic: The topic of the new gap
            description: The description of the new gap
            threshold: Similarity threshold (default: class SIMILARITY_THRESHOLD)
            
        Returns:
            Tuple[bool, Optional[Dict]]:
                - (True, similar_gap_info) if similar gap exists
                - (False, None) if no similar gap found
        """
        # M1 FIX: use 'is not None' so threshold=0.0 is respected, not treated as falsy
        threshold = threshold if threshold is not None else self.SIMILARITY_THRESHOLD
        
        try:
            # Ensure collection exists before searching
            if not self._ensure_gaps_collection_exists():
                logging.warning("Could not ensure gaps collection exists, skipping semantic check")
                return False, None
            
            # Short-circuit if collection is empty — no duplicates possible
            collection_info = self.qdrant_client.get_collection(self.gaps_collection_name)
            if collection_info.points_count == 0:
                logging.debug("Gaps collection is empty, no semantic duplicates possible")
                return False, None
            
            # Generate embedding for the candidate gap
            embedding = self._generate_embedding(topic, description)
            if embedding is None:
                logging.warning("Could not generate embedding, skipping semantic check")
                return False, None
            
            # Search for semantically similar existing gaps
            search_results = self.qdrant_client.search(
                collection_name=self.gaps_collection_name,
                query_vector=embedding,
                limit=3,          # Check top 3 candidates
                score_threshold=threshold,
                with_payload=True
            )
            
            if search_results:
                top_match = search_results[0]
                similar_gap_info = {
                    'score': top_match.score,
                    'topic': top_match.payload.get('topic', 'Unknown'),
                    'description': top_match.payload.get('description', ''),
                    'gap_id': top_match.payload.get('gap_id'),
                    'vector_id': str(top_match.id)
                }
                
                logging.info(f"🔍 Semantic duplicate detected! "
                           f"New: '{topic}' similar to existing: '{similar_gap_info['topic']}' "
                           f"(similarity: {top_match.score:.2%})")
                
                return True, similar_gap_info
            
            logging.debug(f"No semantic duplicates found for '{topic}'")
            return False, None
            
        except Exception as e:
            logging.error(f"Error checking semantic similarity: {e}")
            # Fail open — allow the gap to be created if the check itself errors
            return False, None
 
 


    def _store_gap_embedding(self, gap_id: int, topic: str, description: str) -> Optional[str]:
        """
        Store the embedding for a knowledge gap in Qdrant.
        
        Args:
            gap_id: The SQL ID of the gap
            topic: The gap topic
            description: The gap description
            
        Returns:
            str: The Qdrant point ID, or None on failure
        """
        try:
            # Ensure collection exists
            if not self._ensure_gaps_collection_exists():
                logging.warning("Could not ensure gaps collection exists")
                return None
            
            # Generate embedding
            embedding = self._generate_embedding(topic, description)
            if embedding is None:
                return None
            
            # Create unique point ID
            vector_id = str(uuid.uuid4())
            
            # Store in Qdrant with metadata
            self.qdrant_client.upsert(
                collection_name=self.gaps_collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=vector_id,
                        vector=embedding,
                        payload={
                            'gap_id': gap_id,
                            'topic': topic,
                            'description': description[:500],  # Truncate for payload size
                            'status': 'pending',
                            'created_at': datetime.datetime.now().isoformat()
                        }
                    )
                ]
            )
            
            logging.info(f"Stored embedding for gap {gap_id} with vector_id {vector_id}")
            return vector_id
            
        except Exception as e:
            logging.error(f"Error storing gap embedding: {e}")
            return None

    def _remove_gap_embedding(self, vector_id: str) -> bool:
        """
        Remove a gap embedding from Qdrant.
        
        Args:
            vector_id: The Qdrant point ID to remove
            
        Returns:
            bool: True if successful
        """
        try:
            if not vector_id:
                return True  # Nothing to remove
                
            self.qdrant_client.delete(
                collection_name=self.gaps_collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[vector_id]
                )
            )
            
            logging.info(f"Removed gap embedding with vector_id {vector_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error removing gap embedding: {e}")
            return False

    def add_gap(self, topic: str, description: str, priority: float = 0.5,
                skip_semantic_check: bool = False) -> int:
        """
        Add a new knowledge gap to the queue with semantic duplicate prevention.
        
        D1 FIX: The fast SQL exact-match check now runs BEFORE the slow semantic
        embedding check. This avoids an unnecessary embedding model roundtrip
        whenever an exact topic duplicate already exists in the pending queue.
        
        Args:
            topic: The knowledge topic (main subject)
            description: Detailed description of what needs to be learned
            priority: Importance value from 0.0-1.0
            skip_semantic_check: If True, skip the semantic similarity check
            
        Returns:
            int: ID of the newly created gap, -1 if semantic duplicate, -2 on error
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # STEP 1: Fast exact-match SQL check (indexed — no embedding needed)
                # This runs FIRST to avoid a costly embedding call for obvious duplicates
                cursor.execute('''
                    SELECT id, priority FROM knowledge_gaps
                    WHERE topic = ? AND status = 'pending'
                ''', (topic,))
                
                existing = cursor.fetchone()
                if existing:
                    gap_id, existing_priority = existing
                    # Keep the highest priority and update description in case it evolved
                    new_priority = max(existing_priority, priority)
                    
                    cursor.execute('''
                        UPDATE knowledge_gaps 
                        SET priority = ?, description = ?, last_attempt_at = NULL
                        WHERE id = ?
                    ''', (new_priority, description, gap_id))
                    
                    logging.info(f"Updated existing knowledge gap '{topic}' (ID: {gap_id}) "
                               f"with priority {new_priority}")
                    conn.commit()
                    return gap_id
                
                # STEP 2: Semantic duplicate check (slower — embedding model roundtrip)
                # Only reached when no exact topic match was found above
                if not skip_semantic_check:
                    is_duplicate, similar_info = self.check_semantic_similarity(topic, description)
                    if is_duplicate:
                        logging.info(f"⚠️ Rejected duplicate gap '{topic}' — "
                                   f"similar to existing gap '{similar_info['topic']}' "
                                   f"(similarity: {similar_info['score']:.2%})")
                        
                        # Boost priority of the existing similar gap since this topic
                        # is appearing again — it's getting more important
                        if similar_info.get('gap_id'):
                            self._boost_gap_priority(similar_info['gap_id'], priority)
                        
                        return -1  # Semantic duplicate — expected, not an error
                
                # STEP 3: Insert new unique gap into SQL
                cursor.execute('''
                    INSERT INTO knowledge_gaps (topic, description, priority)
                    VALUES (?, ?, ?)
                ''', (topic, description, priority))
                gap_id = cursor.lastrowid
                
                # STEP 4: Store embedding in Qdrant for future semantic checks
                vector_id = self._store_gap_embedding(gap_id, topic, description)
                
                # Record the vector_id so the embedding can be cleaned up on fulfillment
                if vector_id:
                    cursor.execute('''
                        UPDATE knowledge_gaps SET vector_id = ? WHERE id = ?
                    ''', (vector_id, gap_id))
                
                conn.commit()
                logging.info(f"✅ Added new knowledge gap '{topic}' with ID {gap_id} "
                           f"and priority {priority}")
                
                return gap_id
                
        except Exception as e:
            logging.error(f"Error adding knowledge gap '{topic}': {e}")
            return -2  # Actual storage error
 
 
 

    def _boost_gap_priority(self, gap_id: int, boost_priority: float):
        """
        Boost the priority of an existing gap when a duplicate is detected.
        
        Args:
            gap_id: ID of the gap to boost
            boost_priority: The priority of the rejected duplicate to consider
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current priority
                cursor.execute('SELECT priority FROM knowledge_gaps WHERE id = ?', (gap_id,))
                result = cursor.fetchone()
                
                if result:
                    current_priority = result[0]
                    # Increase priority slightly when duplicates are detected
                    # This indicates the topic is coming up repeatedly
                    new_priority = min(1.0, max(current_priority, boost_priority) + 0.05)
                    
                    cursor.execute('''
                        UPDATE knowledge_gaps SET priority = ? WHERE id = ?
                    ''', (new_priority, gap_id))
                    conn.commit()
                    
                    logging.debug(f"Boosted gap {gap_id} priority from {current_priority:.2f} "
                                f"to {new_priority:.2f} due to duplicate detection")
                                
        except Exception as e:
            logging.error(f"Error boosting gap priority: {e}")
            
    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 5 cognitive cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Single-item dequeue method abandoned in favor of batch processing via get_gaps_by_status('pending').
    # The autonomous_cognition._fill_knowledge_gaps activity processes all pending gaps in a single
    # batch rather than dequeuing one at a time, so the queue semantics this method provided are
    # no longer used. The attempt_count and last_attempt_at columns it manages still exist on the
    # row but aren't being incremented by the batch flow.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_get_next_gap(self) -> Optional[Tuple[int, str, str]]:
        """
        Get the highest priority unfulfilled knowledge gap that hasn't been recently attempted.
        Increments attempt_count each time a gap is dequeued so the filler can enforce
        the one-retry limit (attempt_count >= 2 → mark_failed on second all-stages failure).
        
        Returns:
            Tuple[int, str, str]: (gap_id, topic, description) or None if no gaps
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get highest-priority pending gap not attempted in the last hour
                cursor.execute('''
                    SELECT id, topic, description FROM knowledge_gaps
                    WHERE status = 'pending' 
                    AND (last_attempt_at IS NULL 
                         OR datetime(last_attempt_at) < datetime('now', '-1 hour'))
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                ''')
                
                result = cursor.fetchone()
                
                if result:
                    gap_id = result[0]
                    
                    # Stamp the attempt time AND increment the counter in one update.
                    # attempt_count is used downstream to enforce the one-retry limit.
                    cursor.execute('''
                        UPDATE knowledge_gaps 
                        SET last_attempt_at = CURRENT_TIMESTAMP,
                            attempt_count = attempt_count + 1
                        WHERE id = ?
                    ''', (gap_id,))
                    conn.commit()
                    
                    # Log current attempt number for visibility in logs
                    cursor.execute('SELECT attempt_count FROM knowledge_gaps WHERE id = ?', (gap_id,))
                    count_row = cursor.fetchone()
                    attempt_num = count_row[0] if count_row else '?'
                    
                    logging.info(f"Selected knowledge gap for filling: ID {gap_id}, "
                                 f"topic '{result[1]}' (attempt #{attempt_num})")
                
                return result
                
        except Exception as e:
            logging.error(f"Error getting next knowledge gap: {e}")
            return None
            
    def mark_fulfilled(self, gap_id: int, items_acquired: int = 0) -> bool:
        """
        Mark a knowledge gap as fulfilled and remove its embedding from the vector store.
        
        Args:
            gap_id: ID of the gap to mark fulfilled
            items_acquired: Number of knowledge items acquired for this gap
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get vector_id before updating status
                cursor.execute('SELECT vector_id FROM knowledge_gaps WHERE id = ?', (gap_id,))
                result = cursor.fetchone()
                vector_id = result[0] if result else None
                
                # Update status in SQL
                cursor.execute('''
                    UPDATE knowledge_gaps
                    SET status = 'fulfilled', 
                        fulfilled_at = CURRENT_TIMESTAMP,
                        items_acquired = ?
                    WHERE id = ?
                ''', (items_acquired, gap_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    # Remove embedding from Qdrant to allow similar future gaps
                    if vector_id:
                        self._remove_gap_embedding(vector_id)
                    
                    logging.info(f"✅ Marked knowledge gap {gap_id} as fulfilled "
                               f"with {items_acquired} items acquired")
                    return True
                else:
                    logging.warning(f"⚠️ Knowledge gap {gap_id} not found for marking fulfilled")
                    return False
                    
        except Exception as e:
            logging.error(f"❌ Error marking knowledge gap {gap_id} as fulfilled: {e}")
            return False

    # SLEEPING INFRASTRUCTURE 2026-05-19 (batch 5 cognitive cleanup review):
    # Currently 0 callers — NOT quarantined. Kept intentionally for future wiring.
    # Purpose: marks a gap as un-fillable so it stops being retried. Current architecture
    # leaves un-fillable gaps as 'pending' forever, so they retry on every fill cycle.
    # Recommended wiring: call from autonomous_cognition._fill_knowledge_gaps after N
    # consecutive failed fill attempts on the same gap_id (attempt_count column already
    # exists on the row, just isn't being incremented by the batch flow today).
    def mark_failed(self, gap_id: int, reason: str = "") -> bool:
        """
        Mark a knowledge gap as failed (couldn't be filled).
        Note: Does NOT remove from vector store - keeps blocking similar gaps.
        
        Args:
            gap_id: ID of the gap to mark failed
            reason: Optional reason for failure
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE knowledge_gaps
                    SET status = 'failed',
                        last_attempt_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (gap_id,))
                
                conn.commit()
                if cursor.rowcount > 0:
                    logging.info(f"Marked knowledge gap {gap_id} as failed: {reason}")
                    return True
                else:
                    logging.warning(f"Knowledge gap {gap_id} not found for marking failed")
                    return False
        except Exception as e:
            logging.error(f"Error marking knowledge gap {gap_id} as failed: {e}")
            return False
    
    def get_gaps_by_status(self, status: str = 'pending', limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get knowledge gaps by status.
        
        Args:
            status: Status to filter by ('pending', 'fulfilled', 'failed')
            limit: Maximum number of gaps to return
            
        Returns:
            List[Dict]: List of knowledge gaps with their details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, topic, description, priority, status, 
                           created_at, fulfilled_at, items_acquired, last_attempt_at, vector_id
                    FROM knowledge_gaps
                    WHERE status = ?
                    ORDER BY priority DESC, created_at DESC
                    LIMIT ?
                ''', (status, limit))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logging.error(f"Error getting knowledge gaps by status '{status}': {e}")
            return []

    
    # SLEEPING INFRASTRUCTURE 2026-05-19 (batch 5 cognitive cleanup review):
    # Currently 0 callers — NOT quarantined. Kept intentionally for future wiring.
    # Purpose: periodic cleanup of fulfilled gaps older than days_old to prevent SQLite + Qdrant
    # bloat. As QWEN runs long-term, fulfilled gaps accumulate indefinitely without this.
    # Recommended wiring: add as a maintenance task in admin.py (likely inside
    # _UNUSED_display_knowledge_management_tab once that admin UI is wired up — see batch 1)
    # OR add as a scheduled cognitive activity in autonomous_cognition's cognitive_activities
    # registry (e.g., weekly cadence).
    def cleanup_old_gaps(self, days_old: int = 30) -> int:
        """
        Clean up old fulfilled gaps to prevent database bloat.
        Also removes any orphaned embeddings from the vector store.
        
        M2 FIX: SQL queries now use parameterized datetime strings instead of
        .format() string interpolation.
        
        Args:
            days_old: Remove fulfilled gaps older than this many days
            
        Returns:
            int: Number of gaps removed
        """
        try:
            # Build the interval string once — used in both queries
            interval_str = f'-{int(days_old)} days'  # int() cast prevents injection if type loosened
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Collect vector_ids of gaps that will be deleted so we can
                # clean up their Qdrant embeddings after the SQL delete
                cursor.execute('''
                    SELECT vector_id FROM knowledge_gaps
                    WHERE status = 'fulfilled'
                    AND fulfilled_at < datetime('now', ?)
                    AND vector_id IS NOT NULL
                ''', (interval_str,))
                
                vector_ids = [row[0] for row in cursor.fetchall() if row[0]]
                
                # Delete the fulfilled gaps from SQL
                cursor.execute('''
                    DELETE FROM knowledge_gaps
                    WHERE status = 'fulfilled'
                    AND fulfilled_at < datetime('now', ?)
                ''', (interval_str,))
                
                removed = cursor.rowcount
                conn.commit()
                
            # Remove corresponding Qdrant embeddings to keep vector store clean
            for vector_id in vector_ids:
                self._remove_gap_embedding(vector_id)
            
            if removed > 0:
                logging.info(f"Cleaned up {removed} old fulfilled knowledge gaps "
                           f"and {len(vector_ids)} embeddings")
            
            return removed
            
        except Exception as e:
            logging.error(f"Error cleaning up old gaps: {e}")
            return 0
 
 

    # SLEEPING INFRASTRUCTURE 2026-05-19 (batch 5 cognitive cleanup review):
    # Currently 0 callers — NOT quarantined. Kept intentionally for future wiring.
    # Purpose: returns counts by status, success rate, recent additions, and Qdrant
    # collection size — diagnostic snapshot of knowledge gap system health.
    # Recommended wiring: admin.py Knowledge Management tab. The admin module already has a
    # quarantined _UNUSED_display_knowledge_management_tab (from batch 1) that was likely
    # intended to surface exactly this data. Reviving that tab + calling this method would
    # give the admin dashboard a proper knowledge gap monitor.
    def get_gap_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge gaps."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count by status
                cursor.execute('''
                    SELECT status, COUNT(*) as count
                    FROM knowledge_gaps
                    GROUP BY status
                ''')
                status_counts = dict(cursor.fetchall())
                stats['by_status'] = status_counts
                
                # Total across all statuses
                stats['total'] = sum(status_counts.values())
                
                # Success rate among attempted gaps
                fulfilled = status_counts.get('fulfilled', 0)
                total_attempted = fulfilled + status_counts.get('failed', 0)
                stats['success_rate'] = fulfilled / total_attempted if total_attempted > 0 else 0
                
                # Recent additions (last 7 days)
                cursor.execute('''
                    SELECT COUNT(*) FROM knowledge_gaps
                    WHERE created_at > datetime('now', '-7 days')
                ''')
                stats['recent_additions'] = cursor.fetchone()[0]
                
                # Total knowledge items acquired from fulfilled gaps
                cursor.execute('''
                    SELECT SUM(items_acquired) FROM knowledge_gaps
                    WHERE status = 'fulfilled'
                ''')
                result = cursor.fetchone()[0]
                stats['total_items_acquired'] = result if result else 0
                
                # Vector store gap count
                try:
                    if self._collection_initialized or self._ensure_gaps_collection_exists():
                        collection_info = self.qdrant_client.get_collection(self.gaps_collection_name)
                        stats['vector_store_count'] = collection_info.points_count
                    else:
                        stats['vector_store_count'] = 'N/A'
                except Exception:
                    # M3 FIX: was bare except: which catches SystemExit/KeyboardInterrupt
                    # Now only catching Exception subclasses as intended
                    stats['vector_store_count'] = 'N/A'
                
                return stats
                
        except Exception as e:
            logging.error(f"Error getting gap statistics: {e}")
            return {}

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 5 cognitive cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Manual SQL↔Qdrant resync utility. Never wired to admin or scheduled task. The add_gap()
    # path already keeps the two stores in sync at insert time, and mark_fulfilled() +
    # _remove_gap_embedding handle cleanup on the fulfillment path. A standalone resync is not
    # needed in the current flow.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_sync_vector_store(self) -> Dict[str, int]:
        """
        Synchronize the vector store with the SQL database.
        Removes orphaned embeddings and adds missing ones for pending gaps.
        
        Returns:
            Dict with counts of actions taken
        """
        try:
            results = {'removed': 0, 'added': 0, 'errors': 0}
            
            if not self._ensure_gaps_collection_exists():
                logging.warning("Could not ensure gaps collection for sync")
                return results
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all pending gaps
                cursor.execute('''
                    SELECT id, topic, description, vector_id 
                    FROM knowledge_gaps 
                    WHERE status = 'pending'
                ''')
                pending_gaps = cursor.fetchall()
                
                for gap_id, topic, description, vector_id in pending_gaps:
                    if not vector_id:
                        # Missing embedding - create it
                        new_vector_id = self._store_gap_embedding(gap_id, topic, description)
                        if new_vector_id:
                            cursor.execute('''
                                UPDATE knowledge_gaps SET vector_id = ? WHERE id = ?
                            ''', (new_vector_id, gap_id))
                            results['added'] += 1
                        else:
                            results['errors'] += 1
                
                conn.commit()
                
            logging.info(f"Vector store sync complete: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Error syncing vector store: {e}")
            return {'removed': 0, 'added': 0, 'errors': 1}