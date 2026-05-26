"""Vector database operations using Qdrant with enhanced search capabilities and verification.

TUNING HISTORY:
- 2026-02-05 (Step 2): Recalibrated thresholds for qwen3-embedding:8b (4096-dim) and disabled boosting
  * Increased similarity_threshold: 0.51 -> 0.68 (proper range for high-dim embeddings)
  * Increased comprehensive_threshold: 0.48 -> 0.62 (was in noise floor)
  * Increased selective_threshold: 0.62 -> 0.75 (stronger matches only)
  * Increased verification_threshold: 0.50 -> 0.65 (ensure retrieval accuracy)
  * DISABLED content-type boosting (commented out) - was corrupting similarity scores
  Purpose: Fix retrieval quality - previous thresholds were too low, causing irrelevant results
  Note: With 4096-dim embeddings, true matches score 0.70-0.95+, not 0.50-0.60

- 2026-02-02 (Step 1): Adjusted thresholds and content-type boosts for better proper noun search
  * Lowered similarity_threshold: 0.58 -> 0.52 (more forgiving for person names)
  * Lowered comprehensive_threshold: 0.54 -> 0.48
  * Increased conversation_summary boost: 0.08 -> 0.15 (overcomes embedding dilution)
  * Increased document_summary boost: 0.06 -> 0.12
  Purpose: Improve retrieval of conversation summaries containing specific person names
  Result: INCORRECT - lowering thresholds made retrieval worse, not better
"""

import time
# import threading  # DEAD CODE TEST 2026-05-17: unused per ruff F401
# import socket  # DEAD CODE TEST 2026-05-17: unused per ruff F401 + vulture
import os
import logging
import uuid
# import numpy as np  # DEAD CODE TEST 2026-05-17: unused per ruff F401 (watch for NameError on np. — numpy is heavy and sometimes used in subtle ways)
# import tempfile  # DEAD CODE TEST 2026-05-17: unused per ruff F401
# import subprocess  # DEAD CODE TEST 2026-05-17: unused at module level, re-imported locally at L126 inside _ensure_qdrant_running (ruff F401/F811)
# import requests  # DEAD CODE TEST 2026-05-17: unused at module level, re-imported locally at L127 inside _ensure_qdrant_running (ruff F401/F811)
# from pathlib import Path  # DEAD CODE TEST 2026-05-17: unused per ruff F401
from typing import List, Dict, Union, Any  # DEAD CODE TEST 2026-05-17: was 'List, Dict, Union, Optional, Any' — Optional unused per ruff F401
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from utils import RateLimiter  # Import from the shared utilities module
from qdrant_client.http.exceptions import UnexpectedResponse
from config import (
    QDRANT_LOCAL_PATH, 
    QDRANT_COLLECTION_NAME, 
    QDRANT_USE_LOCAL,
    QDRANT_URL,
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL,
    EMBEDDING_MODEL
)

class VectorDB:
    """Handles vector storage and similarity search using Qdrant."""
    
   
    def __init__(self):
        """Initialize the Vector DB with Qdrant."""
        logging.info("Initializing VectorDB with Qdrant")
        try:
       
            logging.info(f"Using {EMBEDDING_MODEL} for embeddings and {OLLAMA_MODEL} for chat")
            
            # Initialize Ollama embeddings with a dedicated embedding model
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            
            # Calibrated for qwen3-embedding:8b (4096-dimensional embeddings)
            # Initialize threshold values for different search modes
            # Updated 2026-02-05: Recalibrated for high-dimensional embedding space
            # With 4096-dim embeddings: true matches = 0.70-0.95+, marginal = 0.60-0.70, noise = 0.40-0.60
            self.similarity_threshold = 0.63    # Higher threshold for better precision - proper noun matches should be 0.70+ with 4096-dim
            self.comprehensive_threshold = 0.58   # Lower for edge cases
            self.selective_threshold = 0.72       # Selective: only strong 0.72+ (was 0.75)
            self.verification_threshold = 0.62    # Verification: ensures 0.62+ retrieval (was 0.65)
            # Lower threshold specifically for conversation_summary semantic+filter searches.
            # Summaries are multi-topic chunked text — proper noun mentions dilute the embedding
            # vector, causing scores of 0.50-0.62 that would otherwise be cut. The metadata
            # type filter already narrows the candidate pool, so a lower bar is safe here.
            self.conversation_summary_threshold = 0.45  # Only used when type=conversation_summary + query
            # Other default settings
            self.default_k = 10       # Default number of results to return
            self.max_k = 50         # Maximum number of results to ever return
            self.testing = False     # Flag for test mode
            
            # Initialize the Qdrant client before using it
            if QDRANT_USE_LOCAL:
                logging.info(f"Using local Qdrant at: {QDRANT_LOCAL_PATH}")
                # Only create the directory if we're actually using local storage
                os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
                self.client = QdrantClient(path=QDRANT_LOCAL_PATH)
            else:
                # ================================================================
                # DOCKER SELF-HEALING: Ensure Qdrant container is running
                # ================================================================
                logging.info(f"Using remote Qdrant server at: {QDRANT_URL}")
                
                if not self._ensure_qdrant_running():
                    raise RuntimeError(
                        "Failed to start Qdrant container. "
                        "Check Docker Desktop is running and try: "
                        "docker run -d --name qdrant -p 6333:6333 -p 6334:6334 "
                        "-v qdrant_storage:/qdrant/storage qdrant/qdrant"
                    )
                
                self.client = QdrantClient(url=QDRANT_URL, timeout=30.0)
            
            # Now initialize the store
            self.vector_store = self._initialize_store()
            
        except Exception as e:
            logging.error(f"VectorDB initialization error: {e}")
            raise

    def _ensure_qdrant_running(self, container_name: str = "qdrant", timeout: int = 30) -> bool:
        """
        Check if Qdrant Docker container is running, start it if stopped.
        NEVER auto-creates containers - exits with error if container doesn't exist.
        
        Args:
            container_name: Name of the Qdrant Docker container
            timeout: Seconds to wait for Qdrant to become healthy
        
        Returns:
            bool: True if Qdrant is running and healthy, False otherwise
        """
        import subprocess
        import requests
        
        # ================================================================
        # STEP 1: Check if Qdrant is already responding
        # ================================================================
        try:
            response = requests.get(f"{QDRANT_URL}/collections", timeout=3)
            if response.status_code == 200:
                logging.info("✅ Qdrant already running and healthy")
                return True
        except requests.exceptions.ConnectionError:
            logging.warning("⚠️ Qdrant not responding, checking container status...")
        except Exception as e:
            logging.warning(f"⚠️ Qdrant health check failed: {e}")
        
        # ================================================================
        # STEP 2: Check if container exists (running or stopped)
        # ================================================================
        try:
            # Check container status
            inspect_result = subprocess.run(
                ["docker", "inspect", "--format={{.State.Status}}", container_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if inspect_result.returncode == 0:
                container_status = inspect_result.stdout.strip()
                logging.info(f"📦 Container '{container_name}' exists with status: {container_status}")
                
                # ================================================================
                # STEP 3: Verify volume is mounted
                # ================================================================
                volume_check = subprocess.run(
                    ["docker", "inspect", "--format={{json .Mounts}}", container_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if volume_check.returncode == 0:
                    import json
                    try:
                        mounts = json.loads(volume_check.stdout)
                        has_volume = any(
                            mount.get('Type') == 'volume' and 
                            mount.get('Name') == 'qdrant_storage' and
                            mount.get('Destination') == '/qdrant/storage'
                            for mount in mounts
                        )
                        
                        if not has_volume:
                            logging.error(
                                f"❌ CRITICAL: Container '{container_name}' exists but "
                                f"does NOT have 'qdrant_storage' volume mounted!\n"
                                f"This container will not persist data. Please:\n"
                                f"  1. Stop this container: docker stop {container_name}\n"
                                f"  2. Remove it: docker rm {container_name}\n"
                                f"  3. Create properly with: docker run -d --name {container_name} "
                                f"--restart unless-stopped -p 6333:6333 -p 6334:6334 "
                                f"-v qdrant_storage:/qdrant/storage qdrant/qdrant"
                            )
                            return False
                        else:
                            logging.info("✅ Volume 'qdrant_storage' is properly mounted")
                            
                    except json.JSONDecodeError:
                        logging.warning("⚠️ Could not parse volume mounts, proceeding anyway")
                
                # ================================================================
                # STEP 4: Check and update restart policy
                # ================================================================
                restart_check = subprocess.run(
                    ["docker", "inspect", "--format={{.HostConfig.RestartPolicy.Name}}", container_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if restart_check.returncode == 0:
                    restart_policy = restart_check.stdout.strip()
                    if restart_policy == "no" or restart_policy == "":
                        logging.warning(f"⚠️ Container has no restart policy, updating to 'unless-stopped'...")
                        subprocess.run(
                            ["docker", "update", "--restart", "unless-stopped", container_name],
                            capture_output=True,
                            timeout=10
                        )
                        logging.info("✅ Updated restart policy to 'unless-stopped'")
                    else:
                        logging.info(f"✅ Restart policy is set to: {restart_policy}")
                
                # ================================================================
                # STEP 5: Start container if stopped
                # ================================================================
                if container_status in ["exited", "stopped", "created"]:
                    logging.info(f"🚀 Starting stopped container '{container_name}'...")
                    start_result = subprocess.run(
                        ["docker", "start", container_name],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if start_result.returncode != 0:
                        logging.error(f"❌ Failed to start container: {start_result.stderr}")
                        return False
                    
                    logging.info(f"✅ Started container '{container_name}'")
                    
                elif container_status == "running":
                    # Container is running but Qdrant not responding - wait for it
                    logging.info(f"🔄 Container is running, waiting for Qdrant to respond...")
                
                else:
                    logging.error(f"❌ Container in unexpected state: {container_status}")
                    return False
            
            else:
                # ================================================================
                # STEP 6: Container doesn't exist - ERROR and exit
                # ================================================================
                logging.error(
                    f"❌ CRITICAL: Container '{container_name}' does not exist!\n"
                    f"Refusing to auto-create container with production data.\n"
                    f"Please create the container manually:\n\n"
                    f"  docker run -d \\\n"
                    f"    --name {container_name} \\\n"
                    f"    --restart unless-stopped \\\n"
                    f"    -p 6333:6333 \\\n"
                    f"    -p 6334:6334 \\\n"
                    f"    -v qdrant_storage:/qdrant/storage \\\n"
                    f"    qdrant/qdrant\n\n"
                    f"If you had a container with data, check 'docker ps -a' and restart it."
                )
                return False
                
        except subprocess.TimeoutExpired:
            logging.error("❌ Docker command timed out - is Docker Desktop running?")
            return False
        except FileNotFoundError:
            logging.error("❌ Docker not found in PATH - is Docker Desktop installed?")
            return False
        except Exception as e:
            logging.error(f"❌ Docker error: {e}")
            return False
        
        # ================================================================
        # STEP 7: Wait for Qdrant to become healthy
        # ================================================================
        logging.info(f"⏳ Waiting up to {timeout}s for Qdrant to initialize...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{QDRANT_URL}/collections", timeout=3)
                if response.status_code == 200:
                    logging.info("✅ Qdrant is now healthy and ready")
                    
                    # Final verification: check collections exist
                    try:
                        collections_response = response.json()
                        collection_names = [c.get('name') for c in collections_response.get('result', {}).get('collections', [])]
                        if collection_names:
                            logging.info(f"📚 Found {len(collection_names)} collection(s): {', '.join(collection_names)}")
                        else:
                            logging.info("📭 No collections found yet (this is normal for new setups)")
                    except:
                        pass  # Non-critical
                    
                    return True
            except requests.exceptions.ConnectionError:
                pass  # Still starting up
            except Exception:
                pass  # Ignore transient errors during startup
            time.sleep(1)
        
        logging.error(f"❌ Qdrant failed to become healthy within {timeout}s")
        return False
    
    def _initialize_store(self):
        """Initialize or load the Qdrant vector store with improved error handling."""
        try:
            # First get sample embedding to determine actual dimension
            retry_count = 3
            sample_embedding = None
        
            # Retry embedding generation with exponential backoff
            for attempt in range(retry_count):
                try:
                    sample_embedding = self.embeddings.embed_documents(["INITIALIZATION"])
                    break
                except Exception as e:
                    if attempt < retry_count - 1:
                        backoff = (2 ** attempt) * 2  # 2, 4, 8 seconds
                        logging.warning(f"Embedding generation failed (attempt {attempt+1}/{retry_count}): {e}")
                        logging.info(f"Retrying in {backoff} seconds...")
                        time.sleep(backoff)
                    else:
                        logging.error(f"Failed to generate embeddings after {retry_count} attempts: {e}")
                        raise
        
            if not sample_embedding:
                raise ValueError("Failed to generate sample embedding")
            
            actual_dimension = len(sample_embedding[0])

            # Update the instance attribute to match actual dimension
            self.embedding_dimension = actual_dimension
            logging.info(f"Detected embedding dimension: {actual_dimension}")

            # Collection management with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Check if collection exists
                    collection_info = self.client.get_collection(QDRANT_COLLECTION_NAME)
                    existing_dimension = collection_info.config.params.vectors.size
            
                    # SAFETY CHECK: Fail on dimension mismatch - never auto-delete
                    if existing_dimension != actual_dimension:
                        error_msg = (
                            f"CRITICAL: Embedding dimension mismatch!\n"
                            f"  Collection '{QDRANT_COLLECTION_NAME}' has dimension: {existing_dimension}\n"
                            f"  Current embedding model produces dimension: {actual_dimension}\n"
                            f"  This usually means the embedding model changed.\n"
                            f"  To fix:\n"
                            f"    1. Verify correct embedding model is configured\n"
                            f"    2. If intentional model change, run migrate_embeddings.py\n"
                            f"    3. Never auto-delete - your memories are irreplaceable"
                        )
                        logging.error(error_msg)
                        raise RuntimeError(error_msg)

                    logging.info(f"Found existing Qdrant collection: {QDRANT_COLLECTION_NAME} with correct dimension {existing_dimension}")
                    # Successfully validated collection, break the retry loop
                    break
                
                except UnexpectedResponse:
                    # Collection doesn't exist - this is OK, create it
                    logging.info(f"Creating new Qdrant collection: {QDRANT_COLLECTION_NAME} with dimension {actual_dimension}")
                    self.client.create_collection(
                        collection_name=QDRANT_COLLECTION_NAME,
                        vectors_config=qdrant_models.VectorParams(
                            size=actual_dimension,
                            distance=qdrant_models.Distance.COSINE
                        )
                    )
            
                    # Add initialization point to ensure collection is properly set up
                    point_id = str(uuid.uuid4())
                    self.client.upsert(
                        collection_name=QDRANT_COLLECTION_NAME,
                        points=[
                            qdrant_models.PointStruct(
                                id=point_id,
                                payload={"text": "INITIALIZATION", "source": "system"},
                                vector=sample_embedding[0]
                            )
                        ]
                    )
                    logging.info("Added initialization point to Qdrant collection")
                    # Successfully created collection, break the retry loop
                    break
                    
                except Exception as collection_error:
                    if attempt < max_retries - 1:
                        backoff = (2 ** attempt) * 3  # 3, 6, 12 seconds
                        logging.warning(f"Collection operation failed (attempt {attempt+1}/{max_retries}): {collection_error}")
                        logging.info(f"Retrying in {backoff} seconds...")
                        time.sleep(backoff)
                    else:
                        logging.error(f"Failed to manage collection after {max_retries} attempts: {collection_error}")
                        raise

            # ================================================================
            # PAYLOAD INDEXES — required for reliable metadata filter searches
            # ================================================================
            # Without these indexes, every [SEARCH: | type=X] call performs a
            # full collection scan. On small collections this works but is slow;
            # on larger collections Qdrant can silently time out the scan,
            # producing intermittent "not found" results even when data exists.
            #
            # create_payload_index is idempotent — safe to call on every startup.
            # It only does real work if the index doesn't already exist.
            # ================================================================
            index_fields = [
                ("metadata.type",        qdrant_models.PayloadSchemaType.KEYWORD),
                ("metadata.source",      qdrant_models.PayloadSchemaType.KEYWORD),
                ("metadata.date",        qdrant_models.PayloadSchemaType.KEYWORD),
                ("metadata.due_date",    qdrant_models.PayloadSchemaType.KEYWORD),
                ("metadata.tracking_id", qdrant_models.PayloadSchemaType.KEYWORD),
            ]
            for field_name, field_schema in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=QDRANT_COLLECTION_NAME,
                        field_name=field_name,
                        field_schema=field_schema
                    )
                    logging.info(f"Payload index ensured for field: {field_name}")
                except Exception as index_error:
                    # Non-fatal — log and continue. Index creation fails if it
                    # already exists in some client versions; that's fine.
                    logging.warning(
                        f"Could not create payload index for '{field_name}' "
                        f"(may already exist): {index_error}"
                    )

            # Use Qdrant class
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=QDRANT_COLLECTION_NAME,
                embedding=self.embeddings  # Changed from 'embeddings' to 'embedding'
            )

            # Return the initialized vector store
            return vector_store

        except Exception as e:
            logging.error(f"Error initializing Qdrant store: {e}", exc_info=True)
            raise

    def batch_add_texts(self, texts: List[str], metadatas: List[dict] = None, 
                       batch_size: int = 10) -> bool:
        """
        Add multiple texts in batches to avoid overwhelming the database.
    
        Args:
            texts: List of text strings to add
            metadatas: List of metadata dictionaries
            batch_size: Number of texts to add in each batch
        
        Returns:
            bool: True if all texts were successfully added
        """
        if not texts:
            return True
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        rate_limiter = RateLimiter(operations_per_second=2)  # Limit to 2 batches per second
    
        success_count = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
        
            # Generate IDs for this batch
            batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_texts))]
        
            try:
                # Add batch to vector store
                self.vector_store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                success_count += len(batch_texts)
            
                # Respect rate limits
                rate_limiter.wait_if_needed()
            
                # Log progress for long operations
                if len(texts) > batch_size * 2:
                    logging.info(f"Batch processing progress: {min(i+batch_size, len(texts))}/{len(texts)}")
                
            except Exception as e:
                logging.error(f"Error in batch {i//batch_size + 1}: {e}")
            
        # Return True only if all texts were added successfully
        return success_count == len(texts)
    
    def _chunk_summary_text(self, text: str, metadata: dict,
                             chunk_size_words: int = 200,
                             overlap_words: int = 30) -> list:
        """
        Split a conversation summary into overlapping word-based chunks for focused vector storage.

        Solves the "embedding dilution" problem where a long multi-topic summary
        produces a blurry averaged vector that matches nothing well. Each chunk
        produces a sharp, topic-focused embedding that QWEN's [SEARCH:] commands
        can actually match against.

        A brief header is prepended to each chunk so QWEN sees date/part context
        immediately in raw search results without needing to parse metadata.

        Args:
            text: Full summary text to split
            metadata: Parent metadata dict — copied and extended per chunk
            chunk_size_words: Target words per chunk (default 200 ≈ ~260 tokens)
            overlap_words: Words of overlap between chunks to preserve context
                           at boundaries (default 30 words)

        Returns:
            List of (chunk_text, chunk_metadata) tuples — one per chunk.
            Returns [(text, metadata)] as fallback if chunking fails, so
            storage never silently drops a summary.
        """
        try:
            # Split on whitespace into word list
            words = text.split()

            if not words:
                logging.warning("CHUNK_SUMMARY: Empty text passed to chunker — returning empty list")
                return []

            # ----------------------------------------------------------------
            # Short summary guard: if summary is less than 1.5x chunk size,
            # splitting adds no value — store as a single focused vector.
            # ----------------------------------------------------------------
            if len(words) <= int(chunk_size_words * 1.5):
                logging.info(
                    f"CHUNK_SUMMARY: Summary is {len(words)} words "
                    f"(threshold {int(chunk_size_words * 1.5)}) — storing as single vector, no split needed"
                )
                # Still apply chunk metadata fields for schema consistency
                single_metadata = metadata.copy()
                single_metadata['chunk_index'] = 0
                single_metadata['total_chunks'] = 1
                single_metadata['parent_summary_id'] = metadata.get('summary_id', 'unknown')
                single_metadata['is_chunk'] = True
                # No header added for single-chunk — it would be redundant
                return [(text, single_metadata)]

            # ----------------------------------------------------------------
            # Build chunk start positions using sliding window with overlap.
            # Step = chunk_size - overlap so adjacent chunks share overlap_words
            # words at their boundary, preserving topic continuity.
            # ----------------------------------------------------------------
            step = max(1, chunk_size_words - overlap_words)  # Guard against zero step
            chunk_starts = list(range(0, len(words), step))
            total_chunks = len(chunk_starts)

            # Extract best available date for the header label
            summary_date = (
                metadata.get('date') or
                metadata.get('summary_date') or
                str(metadata.get('created_at', ''))[:10] or
                'Unknown date'
            )

            chunks = []
            for idx, start in enumerate(chunk_starts):
                end = start + chunk_size_words
                chunk_words = words[start:end]
                chunk_body = ' '.join(chunk_words)

                # Header gives QWEN immediate temporal + positional context
                # when this chunk surfaces in search results
                header = (
                    f"[Conversation Summary {summary_date} | "
                    f"Part {idx + 1} of {total_chunks}]"
                )
                chunk_text = f"{header}\n{chunk_body}"

                # Each chunk inherits all parent metadata fields so existing
                # filters (type, date, source, etc.) continue to work unchanged
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = idx
                chunk_metadata['total_chunks'] = total_chunks
                chunk_metadata['parent_summary_id'] = metadata.get('summary_id', 'unknown')
                chunk_metadata['is_chunk'] = True
                # type remains 'conversation_summary' — no filter changes needed

                chunks.append((chunk_text, chunk_metadata))

                logging.debug(
                    f"CHUNK_SUMMARY: Built chunk {idx + 1}/{total_chunks} — "
                    f"{len(chunk_words)} words, word offset {start}"
                )

            logging.info(
                f"CHUNK_SUMMARY: Split {len(words)}-word summary into "
                f"{total_chunks} chunks "
                f"(chunk_size={chunk_size_words} words, overlap={overlap_words} words)"
            )
            return chunks

        except Exception as e:
            logging.error(f"CHUNK_SUMMARY: Error in _chunk_summary_text: {e}", exc_info=True)
            # Safe fallback — return original text so storage never fails silently
            logging.warning("CHUNK_SUMMARY: Falling back to single-vector storage for this summary")
            return [(text, metadata)]

    def add_text(self, text: str, metadata: dict = None, memory_id: str = None,
                retry_count: int = 2, memory_db_rollback: callable = None,
                duplicate_threshold: float = 0.98) -> tuple[bool, str]:
        """
        Add text to the vector store with robust error handling and transaction coordination.

        For conversation_summary type: automatically chunks the summary into
        overlapping word-based segments before storage, so each vector covers
        one focused topic cluster rather than a diluted multi-topic average.
        This dramatically improves QWEN's autonomous [SEARCH:] relevance.

        All other memory types follow the original single-vector storage path.

        Args:
            text: The text content to store
            metadata: Optional metadata dictionary
            memory_id: Optional unique identifier for tracking
            retry_count: Number of retry attempts on failure
            memory_db_rollback: Optional callback function for rolling back SQL on failure
            duplicate_threshold: Similarity threshold for duplicate detection (default 0.98)
                                Higher values = stricter matching (fewer false duplicates)
                                Use 0.995+ for conversation summaries to allow similar content

        Returns:
            tuple[bool, str]: (success, reason)
                - (True, "stored")     — Successfully stored
                - (False, "duplicate") — Rejected as duplicate (not an error)
                - (False, "error")     — Failed due to actual error
        """

        # Guard: reject empty or whitespace-only text immediately
        if not text or not text.strip():
            logging.warning("Attempted to add empty text")
            return False, "error"

        # Initialize verify counter if it doesn't exist yet
        self.verify_count = getattr(self, 'verify_count', 0) + 1
        cleaned_text = text.strip()

        # Log the duplicate threshold being used for this call
        logging.debug(f"DUPLICATE_CHECK: Using threshold {duplicate_threshold} for duplicate detection")

        # ================================================================
        # DUPLICATE CHECK — Skip for conversation_summary type
        # ================================================================
        # Conversation summaries always bypass duplicate detection because:
        # 1. Each represents a unique temporal snapshot of a conversation
        # 2. They naturally share structural similarities ("I had a conversation with Ken...")
        # 3. Even "similar" summaries capture entirely different conversation content
        # 4. Duplicate detection was incorrectly rejecting legitimate new summaries
        # ================================================================

        # Determine memory type from metadata — used for both duplicate skip and chunk routing
        memory_type = metadata.get('type', '') if metadata else ''

        # Types that skip duplicate checking entirely
        skip_duplicate_types = ['conversation_summary']
        skip_duplicate_check = memory_type in skip_duplicate_types

        if skip_duplicate_check:
            logging.info(
                f"DUPLICATE_CHECK: Skipping for type '{memory_type}' — always store temporal snapshots"
            )
        else:
            # Run duplicate check for all non-summary memory types
            try:
                results = self.search(
                    query=cleaned_text,
                    k=5,
                    mode="selective",
                    skip_boost=True  # Raw scores only for accurate duplicate detection
                )

                for result in results:
                    result_content = result.get('content', '')
                    similarity = result.get('similarity_score', 0)

                    # Exact text match — always reject regardless of threshold
                    if result_content == cleaned_text:
                        logging.info(
                            f"Exact duplicate found in vector DB, skipping: {cleaned_text[:50]}..."
                        )
                        return False, "duplicate"

                    # Near-duplicate based on configurable similarity threshold
                    if similarity > duplicate_threshold and len(result_content) > 10:
                        logging.info(
                            f"Near-duplicate found in vector DB "
                            f"(similarity: {similarity:.3f} > threshold: {duplicate_threshold}), "
                            f"skipping: {cleaned_text[:50]}..."
                        )
                        return False, "duplicate"

            except Exception as search_error:
                # Non-fatal: log and continue — don't block storage on a failed duplicate check
                logging.warning(f"Error checking for duplicates in vector DB: {search_error}")

        # ================================================================
        # CONVERSATION SUMMARY CHUNKING
        # ================================================================
        # Route conversation_summary type through the chunked storage path.
        # Instead of one diluted multi-topic vector, we store N focused vectors
        # (one per 200-word chunk) so QWEN's [SEARCH:] commands match the
        # specific chunk that covers the relevant topic rather than missing
        # a blurry average that scores below threshold.
        #
        # All other memory types fall through to the normal single-vector path.
        # ================================================================
        if memory_type == 'conversation_summary':
            # ================================================================
            # CONVERSATION SUMMARY CHUNKING ONLY
            # ================================================================
            # document_summary is intentionally excluded here. Doc summaries are
            # capped at 500–1000 words by the LLM prompt so they never need
            # splitting, and the chunker was prepending a "[Conversation Summary]"
            # header — wrong type label — which also corrupted metadata searches.
            # document_summary falls through to the normal single-vector path below.
            # ================================================================
            logging.info(
                "CHUNK_SUMMARY: Detected conversation_summary type — routing to chunked storage"
            )

            # Build metadata_copy for chunks using same cleanup rules as normal path
            chunk_base_metadata = metadata.copy() if metadata else {}
            chunk_base_metadata.pop('page_content', None)  # Avoid LangChain field conflicts
            chunk_base_metadata.pop('text', None)
            chunk_base_metadata.pop('content', None)

            # Apply tracking IDs to chunk metadata base
            if memory_id is not None:
                chunk_base_metadata['memory_id'] = memory_id
                chunk_base_metadata['tracking_id'] = memory_id  # Both fields for compatibility

            # Ensure required fields exist
            if 'source' not in chunk_base_metadata:
                chunk_base_metadata['source'] = 'unknown'
            if 'type' not in chunk_base_metadata:
                chunk_base_metadata['type'] = 'conversation_summary'

            # Generate the list of (chunk_text, chunk_metadata) tuples
            chunks = self._chunk_summary_text(cleaned_text, chunk_base_metadata)

            if not chunks:
                # Chunker returned empty — log and fall through to normal storage path
                logging.error(
                    "CHUNK_SUMMARY: _chunk_summary_text returned empty list — "
                    "falling through to normal single-vector storage"
                )
                # NOTE: Do NOT return here — let execution continue to normal path below

            else:
                # Store each chunk as its own independent vector in Qdrant
                stored_count = 0
                failed_count = 0

                for chunk_text, chunk_metadata in chunks:
                    chunk_id = str(uuid.uuid4())  # Unique ID per chunk
                    try:
                        self.vector_store.add_texts(
                            texts=[chunk_text],
                            metadatas=[chunk_metadata],
                            ids=[chunk_id]
                        )
                        stored_count += 1
                        logging.debug(
                            f"CHUNK_SUMMARY: Stored chunk "
                            f"{chunk_metadata.get('chunk_index', '?') + 1}/"
                            f"{chunk_metadata.get('total_chunks', '?')} "
                            f"with Qdrant ID {chunk_id}"
                        )

                    except Exception as chunk_error:
                        failed_count += 1
                        logging.error(
                            f"CHUNK_SUMMARY: Failed to store chunk "
                            f"{chunk_metadata.get('chunk_index', '?')}: {chunk_error}"
                        )

                logging.info(
                    f"CHUNK_SUMMARY: Finished — {stored_count} chunks stored, "
                    f"{failed_count} failed | "
                    f"parent summary_id: {chunk_base_metadata.get('summary_id', 'unknown')}"
                )

                # Success if at least one chunk stored (partial storage beats total loss)
                if stored_count > 0:
                    return True, "stored"
                else:
                    # All chunks failed — attempt SQL rollback to keep DBs in sync
                    logging.error("CHUNK_SUMMARY: All chunks failed to store")
                    if memory_db_rollback is not None:
                        try:
                            memory_db_rollback()
                            logging.info("CHUNK_SUMMARY: Successfully rolled back MemoryDB entry")
                        except Exception as rollback_error:
                            logging.error(f"CHUNK_SUMMARY: Rollback error: {rollback_error}")
                    return False, "error"

        # ================================================================
        # NORMAL SINGLE-VECTOR STORAGE PATH
        # All memory types other than conversation_summary arrive here.
        # conversation_summary only reaches here if _chunk_summary_text
        # returned an empty list (edge case fallback).
        # ================================================================

        # Ensure metadata is a clean dictionary
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {"source": str(metadata)}

        # Copy metadata and remove fields that conflict with LangChain content storage
        metadata_copy = metadata.copy()
        metadata_copy.pop('page_content', None)
        metadata_copy.pop('text', None)
        metadata_copy.pop('content', None)

        # Apply tracking IDs
        if memory_id is not None:
            metadata_copy["memory_id"] = memory_id
            metadata_copy["tracking_id"] = memory_id  # Both fields for compatibility

        # Ensure required fields exist
        if "source" not in metadata:
            metadata["source"] = "unknown"
        if "type" not in metadata:
            metadata["type"] = "general"

        # Generate unique Qdrant point ID for this text
        text_id = str(uuid.uuid4())

        # Retry loop with exponential backoff for transient Qdrant errors
        for attempt in range(retry_count):
            try:
                # Store via LangChain Qdrant wrapper
                self.vector_store.add_texts(
                    texts=[cleaned_text],
                    metadatas=[metadata_copy],
                    ids=[text_id]
                )

                logging.info(f"[Attempt {attempt+1}] Text added to Qdrant with ID {text_id}")

                # Periodic spot-check verification (every 20 stores)
                if self.verify_count % 20 == 0:
                    try:
                        verification_results = self.search(
                            query=cleaned_text,
                            k=2,
                            mode="default"
                        )

                        if verification_results and len(verification_results) > 0:
                            if verification_results[0]['similarity_score'] >= self.verification_threshold:
                                logging.info(
                                    f"Verification successful "
                                    f"(score: {verification_results[0]['similarity_score']:.2f})"
                                )
                            else:
                                logging.warning(
                                    f"Verification found match but below threshold: "
                                    f"{verification_results[0]['similarity_score']:.2f}"
                                )
                        else:
                            logging.warning("Verification found no matches")

                    except Exception as verify_error:
                        # Non-fatal — storage already succeeded
                        logging.warning(f"Verification error (non-critical): {verify_error}")

                return True, "stored"

            except Exception as e:
                # Determine if this error type is worth retrying
                retryable = (
                    "already accessed" in str(e) or
                    "connection" in str(e).lower() or
                    "timed out" in str(e).lower()
                )

                if retryable and attempt < retry_count - 1:
                    # Exponential backoff: 1s, 2s, 4s...
                    backoff = 1.0 * (2 ** attempt)
                    logging.warning(
                        f"Qdrant error on attempt {attempt+1}: {e}. Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                else:
                    logging.error(
                        f"Error adding text to Qdrant after {attempt+1} attempts: {e}"
                    )

                    # Attempt SQL rollback to keep both databases in sync
                    if memory_db_rollback is not None:
                        try:
                            memory_db_rollback()
                            logging.info("Successfully rolled back MemoryDB entry")
                        except Exception as rollback_error:
                            logging.error(f"Error rolling back MemoryDB: {rollback_error}")

                    return False, "error"

        return False, "error"
    
    def delete_text(self, text: str) -> bool:
        """
        Delete a specific text entry from the Qdrant vector store.

        Uses search_with_ids() to locate the target memory via the fixed
        client.query_points() path — bypasses the LangChain wrapper bug
        where similarity_search_with_score() with score_threshold silently
        returns zero results, which previously caused all deletions to fail
        silently.

        Args:
            text (str): The text to delete.

        Returns:
            bool: True if at least one matching entry was successfully deleted,
                False if no match was found or all deletions failed.
        """
        # Guard against None or empty text
        if text is None or not text.strip():
            logging.warning("delete_text: Attempted to delete None or empty text")
            return False

        try:
            # ================================================================
            # FIXED: Use search_with_ids() — client.query_points() path
            # ================================================================
            # Replaces the old similarity_search_with_score() call which used
            # the LangChain wrapper. That wrapper has a known bug where passing
            # score_threshold silently returns 0 results, causing all deletions
            # to fail without any error being raised.
            #
            # search_with_ids() returns dicts with a guaranteed 'id' key set to
            # the actual Qdrant point ID (from point.id), which client.delete()
            # needs directly. The old path relied on doc.metadata['id'] which
            # LangChain does not reliably populate.
            #
            # mode="selective" (threshold 0.72) keeps deletion precise —
            # we only delete entries we're highly confident are the right match.
            # ================================================================
            results = self.search_with_ids(query=text, k=5, mode="selective")

            # No match found above the selective threshold
            if not results:
                logging.warning(
                    f"delete_text: No matching entry found for: '{text[:50]}...'"
                )
                return False

            # Iterate results and delete each matching Qdrant point by its ID
            deleted = False
            for result in results:
                # 'id' is always present — set from point.id in search_with_ids()
                point_id = result.get('id')
                score = result.get('similarity_score', 0.0)

                if not point_id:
                    # Defensive guard — should not happen with search_with_ids()
                    logging.warning(
                        f"delete_text: Result missing 'id' field "
                        f"(score={score:.3f}) — skipping"
                    )
                    continue

                try:
                    self.client.delete(
                        collection_name=QDRANT_COLLECTION_NAME,
                        points_selector=qdrant_models.PointIdsList(
                            points=[point_id]
                        )
                    )
                    deleted = True
                    logging.info(
                        f"delete_text: Deleted point ID {point_id} "
                        f"(similarity={score:.3f})"
                    )

                except Exception as delete_error:
                    # Log individual point failure but continue attempting others
                    logging.error(
                        f"delete_text: Failed to delete point {point_id}: {delete_error}"
                    )

            if not deleted:
                logging.warning(
                    f"delete_text: Search found results but all deletions failed "
                    f"for: '{text[:50]}...'"
                )

            return deleted

        except Exception as e:
            logging.error(f"delete_text: Unexpected error: {e}", exc_info=True)
            return False
        
    def delete_by_id(self, vector_id):
        """Delete a vector by its ID from the vector database.
        
        Args:
            vector_id (str): The ID of the vector to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Delete the vector from Qdrant
            self.client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=qdrant_models.PointIdsList(  # Fixed: use qdrant_models instead of models
                    points=[vector_id]
                )
            )
            logging.info(f"Vector with ID {vector_id} deleted successfully")
            return True
        except Exception as e:
            logging.error(f"Error deleting vector with ID {vector_id}: {e}", exc_info=True)
            return False
        
    def delete_by_memory_id(self, memory_id: str) -> tuple[bool, int]:
        """
        Delete all Qdrant points associated with a given memory_id (tracking_id).
        
        Used by chatbot.delete_memory_by_id() for ID-based FORGET operations.
        Handles chunked memories correctly — a single logical memory may span
        multiple Qdrant points (one per chunk), all sharing the same memory_id
        in their metadata. This method scrolls for ALL matching points and
        deletes each by its point_id.
        
        Filters on metadata.tracking_id (KEYWORD-indexed at collection init,
        vector_db.py line 419 — fast exact match). The 'memory_id' and
        'tracking_id' payload fields hold identical UUID values per the
        add_text() storage pattern, but tracking_id is the indexed one.
        
        Args:
            memory_id (str): The UUID memory_id / tracking_id to delete.
        
        Returns:
            tuple[bool, int]: (success, count_deleted)
                - success=True if deletion completed cleanly (even if 0 points found)
                - success=False on Qdrant API error
                - count_deleted = number of Qdrant points actually deleted
        """
        # Guard against None or empty memory_id — no DB call needed
        if not memory_id or not str(memory_id).strip():
            logging.warning("delete_by_memory_id: Called with None or empty memory_id")
            return False, 0
        
        memory_id = str(memory_id).strip()
        
        try:
            # ============================================================
            # Build metadata filter targeting the indexed tracking_id field
            # ============================================================
            # metadata.tracking_id is registered as a KEYWORD payload index
            # at collection init time (see line 419) for fast exact match.
            metadata_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.tracking_id",
                        match=qdrant_models.MatchValue(value=memory_id)
                    )
                ]
            )
            
            # ============================================================
            # Scroll for ALL matching points (handles chunked memories)
            # ============================================================
            # Use client.scroll() rather than query_points() because no
            # semantic ranking is needed — pure metadata filter, no embedding.
            # Paginate up to a safety cap to handle pathologically long
            # chunked memories without unbounded looping.
            SCROLL_BATCH_SIZE = 100       # Per-call limit
            MAX_SCROLL_ITERATIONS = 10    # Safety cap = 1000 chunks max
            
            point_ids_to_delete = []
            next_offset = None
            iteration = 0
            
            while iteration < MAX_SCROLL_ITERATIONS:
                iteration += 1
                
                try:
                    scroll_result, next_offset = self.client.scroll(
                        collection_name=QDRANT_COLLECTION_NAME,
                        scroll_filter=metadata_filter,
                        limit=SCROLL_BATCH_SIZE,
                        offset=next_offset,
                        with_payload=False,   # We only need point IDs
                        with_vectors=False
                    )
                except Exception as scroll_err:
                    logging.error(
                        f"delete_by_memory_id: Scroll error on iteration {iteration} "
                        f"for memory_id={memory_id}: {scroll_err}",
                        exc_info=True
                    )
                    return False, 0
                
                # Collect point IDs from this batch
                for point in scroll_result:
                    if point.id is not None:
                        point_ids_to_delete.append(point.id)
                
                # Stop when there are no more pages
                if next_offset is None:
                    break
            
            # Warn if we hit the iteration cap — unusual and worth flagging
            if iteration >= MAX_SCROLL_ITERATIONS and next_offset is not None:
                logging.warning(
                    f"delete_by_memory_id: Hit scroll iteration cap "
                    f"({MAX_SCROLL_ITERATIONS}) for memory_id={memory_id} — "
                    f"some points may remain undeleted"
                )
            
            # ============================================================
            # No matching points — clean success (nothing to do)
            # ============================================================
            # Not treated as an error. The caller (chatbot.delete_memory_by_id)
            # may have already cleaned SQL; missing vector entries just mean
            # the memory was already orphaned or never indexed.
            if not point_ids_to_delete:
                logging.info(
                    f"delete_by_memory_id: No vector points found for "
                    f"memory_id={memory_id} (clean exit, count=0)"
                )
                return True, 0
            
            logging.info(
                f"delete_by_memory_id: Found {len(point_ids_to_delete)} point(s) "
                f"for memory_id={memory_id} — deleting"
            )
            
            # ============================================================
            # Delete all matching points in a single Qdrant call
            # ============================================================
            try:
                self.client.delete(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points_selector=qdrant_models.PointIdsList(
                        points=point_ids_to_delete
                    )
                )
                logging.info(
                    f"delete_by_memory_id: Successfully deleted "
                    f"{len(point_ids_to_delete)} point(s) for memory_id={memory_id}"
                )
                return True, len(point_ids_to_delete)
            
            except Exception as delete_err:
                logging.error(
                    f"delete_by_memory_id: Delete operation failed for "
                    f"memory_id={memory_id}: {delete_err}",
                    exc_info=True
                )
                return False, 0
        
        except Exception as e:
            logging.error(
                f"delete_by_memory_id: Unexpected error for memory_id={memory_id}: {e}",
                exc_info=True
            )
            return False, 0
        
    def search(self, query: str = None, k: int = None, mode: str = "default", 
                metadata_filters: Dict[str, Any] = None, skip_boost: bool = False) -> List[Dict[str, Union[str, float, dict]]]:
        """
        Enhanced search with metadata filtering capabilities and robust retry logic.

        Args:
            query (str, optional): Text query for semantic search (can be None for pure metadata search)
            k (int, optional): Number of results to return
            mode (str): Search mode ("default", "comprehensive", or "selective")
            metadata_filters (Dict[str, Any], optional): Dictionary of metadata filters
                - 'type': Filter by memory type (e.g., 'self', 'general', etc.)
                - 'tags': Filter by tags (string or list of strings)
                - 'min_confidence': Minimum confidence value (float between 0-1)
                - 'max_age_days': Maximum age in days (int)
                - 'source': Filter by source (string)
            skip_boost (bool): If True, skip the conversation_summary score boost.
                            Use this for duplicate detection to get raw similarity scores.

        Returns:
            List[Dict]: Search results with metadata and similarity scores
        """
        # Validate inputs with consistent guards
        if query is None and (metadata_filters is None or not metadata_filters):
            logging.warning("Both query and metadata_filters are empty or None, cannot perform search")
            return []

        if query is not None and not isinstance(query, str):
            logging.warning(f"Invalid query type: {type(query)}, expected string")
            query = str(query)

        # Normalize parameters    
        k = max(1, k or self.default_k)
        mode = mode.lower() if mode else "default"

        # Validate mode
        if mode not in ["default", "comprehensive", "selective"]:
            logging.warning(f"Unknown search mode: {mode}, defaulting to 'default'")
            mode = "default"

        try:
            # Set threshold based on mode
            if mode == "comprehensive":
                threshold = self.comprehensive_threshold
                k = min(k, self.max_k)  # Respect max_k while honoring user request
            elif mode == "selective":
                threshold = self.selective_threshold
            else:  # default
                threshold = self.similarity_threshold

            # ================================================================
            # CONVERSATION SUMMARY THRESHOLD OVERRIDE
            # When a text query AND type=conversation_summary filter are both
            # present, substitute the dedicated lower threshold so chunked
            # summaries containing proper nouns are not silently cut.
            # Fires ONLY when both conditions are true — all other paths unchanged.
            #
            # Rationale: chunked summaries are multi-topic. A proper noun like
            # "Kiro" mentioned briefly in a 200-word chunk produces a diluted
            # embedding that legitimately scores 0.50-0.62 — below the standard
            # 0.63 threshold. The metadata filter already constrains the pool
            # to conversation_summary type, so a lower similarity bar is safe.
            # ================================================================
            if (query and
                    metadata_filters and
                    isinstance(metadata_filters, dict) and
                    metadata_filters.get('type') == 'conversation_summary'):
                original_threshold = threshold
                threshold = self.conversation_summary_threshold
                logging.info(
                    f"CONV_SUMMARY_OVERRIDE: type=conversation_summary + query detected — "
                    f"threshold lowered from {original_threshold:.2f} to {threshold:.2f}"
                )

            # Log the search attempt with metadata filters
            filter_str = str(metadata_filters) if metadata_filters else "None"
            query_str = query[:50] + "..." if query and len(query) > 50 else query
            logging.info(f"Executing search: query='{query_str}', mode={mode}, k={k}, threshold={threshold:.2f}, metadata_filters={filter_str}")

            # Convert metadata filters to Qdrant filter format if provided
            qdrant_filter = None
            if metadata_filters and isinstance(metadata_filters, dict):
                # ✅ NORMALIZE: Handle both flat and nested metadata formats
                # Case 1: Nested format like {'metadata': {'type': 'X', 'date': 'Y'}}
                if 'metadata' in metadata_filters and isinstance(metadata_filters['metadata'], dict):
                    normalized_filters = metadata_filters['metadata']
                    logging.debug(f"Unpacked nested metadata format: {normalized_filters}")
                else:
                    # Case 2: Flat format - strip "metadata." prefix from keys if present
                    normalized_filters = {}
                    for key, value in metadata_filters.items():
                        clean_key = key.replace('metadata.', '') if key.startswith('metadata.') else key
                        normalized_filters[clean_key] = value
                    logging.debug(f"Normalized flat metadata format: {normalized_filters}")
                
                # ✅ Convert tags string to array if needed
                if 'tags' in normalized_filters and isinstance(normalized_filters['tags'], str):
                    tags_str = normalized_filters['tags']
                    normalized_filters['tags'] = [tag.strip() for tag in tags_str.split(',')]
                    logging.debug(f"Converted tags from string to array: {normalized_filters['tags']}")
                
                # Use normalized filters for processing
                metadata_filters = normalized_filters
                
                filter_conditions = []
        
                # Process type filter
                if 'type' in metadata_filters and metadata_filters['type']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.type",
                            match=qdrant_models.MatchValue(value=metadata_filters['type'])
                        )
                    )
        
                # Process tags filter (can be a single tag or list of tags)
                if 'tags' in metadata_filters and metadata_filters['tags']:
                    tags = metadata_filters['tags']
                    if isinstance(tags, str):
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key="metadata.tags",
                                match=qdrant_models.MatchText(text=tags)
                            )
                        )
                    elif isinstance(tags, list):
                        tag_conditions = [
                            qdrant_models.FieldCondition(
                                key="metadata.tags",
                                match=qdrant_models.MatchText(text=tag)
                            ) for tag in tags if tag
                        ]
                        if tag_conditions:
                            filter_conditions.append(
                                qdrant_models.Filter(should=tag_conditions)
                            )
        
                # Process min_confidence filter with safe conversion
                if 'min_confidence' in metadata_filters:
                    try:
                        min_confidence = float(metadata_filters['min_confidence'])
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key="metadata.confidence",
                                range=qdrant_models.Range(gte=min_confidence)
                            )
                        )
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Invalid min_confidence value: {metadata_filters['min_confidence']}, error: {e}")
        
                # Process max_age_days filter with safe conversion
                if 'max_age_days' in metadata_filters:
                    try:
                        from datetime import datetime, timedelta
                        max_age = int(metadata_filters['max_age_days'])
                        cutoff_date = (datetime.now() - timedelta(days=max_age)).isoformat()
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key="metadata.created_at",
                                range=qdrant_models.Range(gte=cutoff_date)
                            )
                        )
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Invalid max_age_days value: {metadata_filters['max_age_days']}, error: {e}")
            
                # Process source filter
                if 'source' in metadata_filters and metadata_filters['source']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.source",
                            match=qdrant_models.MatchValue(value=metadata_filters['source'])
                        )
                    )
            
                # Process date filter
                if 'date' in metadata_filters and metadata_filters['date']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.date",
                            match=qdrant_models.MatchValue(value=metadata_filters['date'])
                        )
                    )

                # Legacy field name support for backward compatibility
                if 'summary_date' in metadata_filters and metadata_filters['summary_date']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.summary_date",
                            match=qdrant_models.MatchValue(value=metadata_filters['summary_date'])
                        )
                    )
                    
                # Process due_date filter (for reminders)
                if 'due_date' in metadata_filters and metadata_filters['due_date']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.due_date",
                            match=qdrant_models.MatchValue(value=metadata_filters['due_date'])
                        )
                    )

                # Process memory_id filter
                if 'memory_id' in metadata_filters and metadata_filters['memory_id']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.memory_id",
                            match=qdrant_models.MatchValue(value=metadata_filters['memory_id'])
                        )
                    )
        
                # Combine all conditions with AND
                if filter_conditions:
                    qdrant_filter = qdrant_models.Filter(must=filter_conditions)
                else:
                    logging.info("No valid metadata filters were created")

            # ================================================================
            # SEARCH EXECUTION with retry logic
            # ================================================================
            docs_and_scores = []
            max_retries = 3
        
            for attempt in range(max_retries):
                try:
                    if query:
                        # ================================================================
                        # SEMANTIC + OPTIONAL FILTER PATH
                        # ================================================================
                        # CRITICAL FIX: Bypass LangChain's similarity_search_with_score()
                        # wrapper when a metadata filter is present. The LangChain wrapper
                        # has a known issue where passing both score_threshold AND filter
                        # together silently returns 0 results even for verbatim text matches.
                        #
                        # Fix: Use self.client.query_points() directly (same approach as
                        # search_with_ids). Apply score_threshold in Python after retrieval
                        # instead of delegating it to Qdrant. This path now handles BOTH
                        # filtered and unfiltered semantic searches for consistency.
                        # ================================================================
                        query_embedding = self.embeddings.embed_query(query)

                        # Fetch candidates — no score_threshold here to avoid LangChain bug.
                        # We apply the threshold ourselves below after getting raw results.
                        raw_points = self.client.query_points(
                            collection_name=QDRANT_COLLECTION_NAME,
                            query=query_embedding,
                            query_filter=qdrant_filter,   # None = no filter, works either way
                            limit=min(k * 3, self.max_k), # Fetch extra to allow threshold filtering
                            with_payload=True,
                            with_vectors=False
                        ).points

                        # Apply score threshold in Python — avoids LangChain wrapper bug
                        from langchain_core.documents import Document as LCDocument
                        docs_and_scores = []
                        for point in raw_points:
                            if point.score >= threshold:
                                payload = point.payload or {}
                                page_content = payload.get('page_content', payload.get('text', ''))
                                # Unwrap LangChain nested metadata structure
                                if 'metadata' in payload and isinstance(payload['metadata'], dict):
                                    meta = payload['metadata']
                                else:
                                    # Legacy flat payload fallback
                                    meta = {k: v for k, v in payload.items()
                                            if k not in ('page_content', 'text')}
                                doc = LCDocument(page_content=page_content, metadata=meta)
                                docs_and_scores.append((doc, point.score))

                        logging.info(
                            f"Direct query_points search: {len(raw_points)} raw candidates, "
                            f"{len(docs_and_scores)} passed threshold {threshold:.2f} "
                            f"on attempt {attempt+1}"
                        )
                        break

                    elif qdrant_filter:
                        # ================================================================
                        # FILTER-ONLY PATH: Use client.scroll()
                        # ================================================================
                        # similarity_search("") embeds an empty string and runs cosine
                        # similarity against it, producing garbage scores (0.1-0.4) that
                        # fail the downstream threshold check even when real matches exist.
                        # client.scroll() is the correct Qdrant API for metadata-only search.
                        # We assign score=1.0 to all scroll results — exact metadata matches.
                        # ================================================================
                        try:
                            scroll_results, _ = self.client.scroll(
                                collection_name=QDRANT_COLLECTION_NAME,
                                scroll_filter=qdrant_filter,
                                limit=min(k, self.max_k),
                                with_payload=True,
                                with_vectors=False
                            )

                            from langchain_core.documents import Document as LCDocument
                            docs_and_scores = []
                            for point in scroll_results:
                                payload = point.payload or {}
                                page_content = payload.get('page_content', payload.get('text', ''))
                                # Unwrap LangChain nested metadata structure
                                if 'metadata' in payload and isinstance(payload['metadata'], dict):
                                    meta = payload['metadata']
                                else:
                                    meta = {k: v for k, v in payload.items()
                                            if k not in ('page_content', 'text')}
                                    logging.debug(
                                        f"SCROLL: Point has no 'metadata' key — "
                                        f"using flat extraction fallback. Keys: {list(payload.keys())}"
                                    )
                                doc = LCDocument(page_content=page_content, metadata=meta)
                                docs_and_scores.append((doc, 1.0))

                            logging.info(
                                f"Metadata-only scroll successful on attempt {attempt+1}: "
                                f"{len(docs_and_scores)} results"
                            )
                            break

                        except Exception as scroll_error:
                            logging.error(f"Error in filter-only scroll (attempt {attempt+1}): {scroll_error}")
                            raise scroll_error
                
                except Exception as search_error:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds exponential backoff
                        logging.warning(f"Vector search attempt {attempt+1}/{max_retries} failed: {str(search_error)}")
                        logging.info(f"Retrying search in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"All {max_retries} vector search attempts failed: {str(search_error)}")
                        return []

            # Validate result before formatting
            if docs_and_scores is None:
                logging.warning(f"Search returned None for query: {query}")
                return []

            # =========================================================================
            # FORMAT RESULTS
            # Boosting is disabled — raw similarity scores only.
            # See comments in original code for re-enable guidance.
            # =========================================================================
            formatted_results = []
            
            for doc, score in docs_and_scores:
                try:
                    page_content = getattr(doc, 'page_content', 'Unknown content') 
                    doc_metadata = getattr(doc, 'metadata', {}) or {}
            
                    try:
                        score_value = float(score)
                    except (ValueError, TypeError):
                        score_value = 0.0

                    formatted_results.append({
                        'content': page_content,
                        'similarity_score': score_value,
                        'metadata': doc_metadata,
                        'above_threshold': score_value >= threshold
                    })
                except Exception as format_error:
                    logging.error(f"Error formatting search result: {format_error}")

            # Sort by similarity score descending
            formatted_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

            logging.info(
                f"Search found {len(formatted_results)} results | "
                f"mode={mode}, threshold={threshold:.2f}, metadata_filters={filter_str}"
            )
            return formatted_results

        except Exception as e:
            logging.error(f"Error in search with metadata filters: {e}", exc_info=True)
            return []

    def search_with_ids(self, query: str = None, k: int = None, mode: str = "default", 
                   metadata_filters: Dict[str, Any] = None) -> List[Dict[str, Union[str, float, dict]]]:
        """
        Enhanced search that returns actual Qdrant point IDs for deletion operations.
        Applies the same conversation_summary threshold override as search() so that
        chunked summaries containing proper nouns are not silently filtered out.
        
        Args:
            query (str, optional): Text query for semantic search
            k (int, optional): Number of results to return
            mode (str): Search mode ("default", "comprehensive", or "selective")
            metadata_filters (Dict[str, Any], optional): Dictionary of metadata filters
        
        Returns:
            List[Dict]: Search results with metadata, similarity scores, and point IDs
        """
        # Validate inputs
        if query is None and (metadata_filters is None or not metadata_filters):
            logging.warning("Both query and metadata_filters are empty or None, cannot perform search")
            return []
        
        if query is not None and not isinstance(query, str):
            logging.warning(f"Invalid query type: {type(query)}, expected string")
            query = str(query)
        
        # Normalize parameters    
        k = max(1, k or self.default_k)
        mode = mode.lower() if mode else "default"
        
        # Set threshold based on mode
        if mode == "comprehensive":
            threshold = self.comprehensive_threshold
            k = min(k, self.max_k)
        elif mode == "selective":
            threshold = self.selective_threshold
        else:  # default
            threshold = self.similarity_threshold

        # ================================================================
        # CONVERSATION SUMMARY THRESHOLD OVERRIDE
        # Mirror of the same override in search(). When a text query AND
        # type=conversation_summary filter are both present, substitute the
        # dedicated lower threshold so chunked summaries containing proper
        # nouns (e.g. "Kiro", "Ollie", "Lucian") are not cut by the standard
        # 0.63 threshold due to embedding dilution from multi-topic chunks.
        # Only fires when both conditions are true — all other paths unchanged.
        # ================================================================
        if (query and
                metadata_filters and
                isinstance(metadata_filters, dict) and
                metadata_filters.get('type') == 'conversation_summary'):
            original_threshold = threshold
            threshold = self.conversation_summary_threshold
            logging.info(
                f"CONV_SUMMARY_OVERRIDE (search_with_ids): type=conversation_summary + query detected — "
                f"threshold lowered from {original_threshold:.2f} to {threshold:.2f}"
            )
        
        try:
            # Convert metadata filters to Qdrant filter format if provided
            qdrant_filter = None
            if metadata_filters and isinstance(metadata_filters, dict):
                filter_conditions = []
                
                # Process type filter
                if 'type' in metadata_filters and metadata_filters['type']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.type",
                            match=qdrant_models.MatchValue(value=metadata_filters['type'])
                        )
                    )
                
                # Process source filter
                if 'source' in metadata_filters and metadata_filters['source']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.source",
                            match=qdrant_models.MatchValue(value=metadata_filters['source'])
                        )
                    )

                # Process date filter
                if 'date' in metadata_filters and metadata_filters['date']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.date",
                            match=qdrant_models.MatchValue(value=metadata_filters['date'])
                        )
                    )

                # Process memory_id filter
                if 'memory_id' in metadata_filters and metadata_filters['memory_id']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.memory_id",
                            match=qdrant_models.MatchValue(value=metadata_filters['memory_id'])
                        )
                    )
                
                # Combine all conditions with AND only if we have valid conditions
                if filter_conditions:
                    qdrant_filter = qdrant_models.Filter(must=filter_conditions)
                else:
                    logging.info("search_with_ids: No valid metadata filter conditions built")
            
            # Use direct Qdrant client for search with IDs
            if query:
                # Generate embedding for the query
                query_embedding = self.embeddings.embed_query(query)
                
                # Search using Qdrant client directly — returns ScoredPoint objects
                # which include the actual Qdrant point ID needed for deletion
                search_result = self.client.query_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query=query_embedding,
                    query_filter=qdrant_filter,
                    limit=k,
                    score_threshold=threshold,
                    with_payload=True,
                    with_vectors=False
                ).points  # query_points returns QueryResponse — extract .points list
                    
            else:
                # Filter-only path: scroll returns exact metadata matches, no vector math
                search_result, _ = self.client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    scroll_filter=qdrant_filter,
                    limit=k,
                    with_payload=True,
                    with_vectors=False
                )
            
            # Format results preserving the Qdrant point ID for callers that
            # need to perform deletion (the primary purpose of this method)
            formatted_results = []
            for point in search_result:
                try:
                    payload = point.payload or {}

                    # Extract text content from payload
                    content = payload.get('page_content', payload.get('text', 'Unknown content'))
                    
                    # ----------------------------------------------------------------
                    # Unwrap LangChain's nested metadata structure.
                    # QdrantVectorStore.add_texts() wraps caller metadata under a
                    # 'metadata' key: { "page_content": "...", "metadata": {...} }
                    # Using flat extraction here would double-nest everything and
                    # cause metadata fields like 'type' to be unreachable by callers.
                    # ----------------------------------------------------------------
                    if 'metadata' in payload and isinstance(payload['metadata'], dict):
                        # Standard LangChain-stored point — unwrap nested dict
                        metadata = payload['metadata']
                    else:
                        # Legacy or manually-upserted point — flat extraction fallback
                        metadata = {k: v for k, v in payload.items() 
                                    if k not in ('page_content', 'text')}
                        logging.debug(
                            f"search_with_ids: Point has no 'metadata' key — "
                            f"using flat extraction fallback. Keys: {list(payload.keys())}"
                        )
                    
                    # Get similarity score — scroll results have no score so default 1.0
                    score = float(getattr(point, 'score', 1.0))
                    
                    formatted_results.append({
                        'id': str(point.id),  # Actual Qdrant point ID — required for deletion
                        'content': content,
                        'similarity_score': score,
                        'metadata': metadata,
                        'above_threshold': score >= threshold
                    })
                    
                except Exception as format_error:
                    logging.error(f"search_with_ids: Error formatting result point {getattr(point, 'id', 'unknown')}: {format_error}")
            
            # Sort by similarity score descending
            formatted_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            logging.info(
                f"search_with_ids: found {len(formatted_results)} results | "
                f"mode={mode}, threshold={threshold:.2f}, k={k}"
            )
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error in search_with_ids: {e}", exc_info=True)
            return []
      
    def search_with_pagination(self, query: str, page_size: int = 10, page: int = 1, threshold: float = None) -> Dict[str, Union[List, int]]:
        """
        Paginated search using Qdrant's native offset capability for efficiency.
        
        Args:
            query (str): Search query
            page_size (int): Results per page (default: 10)
            page (int): Page number (1-indexed)
            threshold (float): Optional similarity threshold override
        
        Returns:
            Dict: Paginated results with metadata
        """
        try:
            # Validate inputs
            if not query or not query.strip():
                return {
                    'results': [],
                    'total_results': 0,
                    'current_page': page,
                    'total_pages': 0,
                    'page_size': page_size
                }
            
            # Ensure page is at least 1
            page = max(1, page)
            
            # Use provided threshold or default
            search_threshold = threshold if threshold is not None else self.similarity_threshold
            
            # Calculate offset for this page
            offset = (page - 1) * page_size
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Get total count first (to calculate total pages)
            # Use query_points - search() was removed in qdrant-client 1.12+
            initial_search = self.client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=query_embedding,  # Changed from query_vector to query
                limit=self.max_k,  # Get max results to count total matches
                score_threshold=search_threshold
            ).points  # Returns QueryResponse, need .points for the list
            
            total_above_threshold = len([r for r in initial_search if r.score >= search_threshold])
            
            # Now get the specific page of results we need
            search_result = self.client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=query_embedding,  # Changed from query_vector to query
                limit=page_size,  # Fixed: was 'k' (undefined)
                offset=offset,    # Added: for pagination
                score_threshold=search_threshold,  # Fixed: was 'threshold'
                with_payload=True,
                with_vectors=False
                # Removed: query_filter=qdrant_filter (was undefined)
            ).points  # Returns QueryResponse, need .points for the list
            
            # Format results
            formatted_results = []
            for point in search_result:
                try:
                    # Extract content from payload
                    payload = point.payload or {}
                    content = payload.get('page_content', payload.get('text', 'Unknown content'))
                    
                    # Get metadata (remove the 'page_content' key if it exists)
                    metadata = {k: v for k, v in payload.items() if k != 'page_content'}
                    
                    # Get similarity score
                    score = float(point.score)
                    
                    formatted_results.append({
                        'id': str(point.id),
                        'content': content,
                        'similarity_score': score,
                        'metadata': metadata,
                        'above_threshold': score >= search_threshold
                    })
                    
                except Exception as format_error:
                    logging.error(f"Error formatting search result: {format_error}")
            
            # Calculate total pages
            total_pages = (total_above_threshold + page_size - 1) // page_size if total_above_threshold > 0 else 0
            
            logging.info(f"Paginated search: page {page}/{total_pages}, returned {len(formatted_results)} results")
            
            return {
                'results': formatted_results,
                'total_results': total_above_threshold,
                'current_page': page,
                'total_pages': total_pages,
                'page_size': page_size
            }
        
        except Exception as e:
            logging.error(f"Error in paginated search: {e}", exc_info=True)
            return {
                'results': [],
                'total_results': 0,
                'current_page': page,
                'total_pages': 0,
                'page_size': page_size
            }
                
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the Qdrant vector store.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Check if the collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections)
            
            if not collection_exists:
                return {
                    "status": "error",
                    "message": f"Collection {QDRANT_COLLECTION_NAME} does not exist",
                    "collection_count": len(collections)
                }
                
            # Get collection info
            collection_info = self.client.get_collection(QDRANT_COLLECTION_NAME)
            
            # Get point count
            count_result = self.client.count(
                collection_name=QDRANT_COLLECTION_NAME,
                count_filter=None  # Count all points
            )
            
            return {
                "status": "healthy",
                "collection_name": QDRANT_COLLECTION_NAME,
                "vectors_count": count_result.count,
                "vector_dimension": collection_info.config.params.vectors.size,
                "storage_type": "local" if QDRANT_USE_LOCAL else "remote"
            }
            
        except Exception as e:
            logging.error(f"Error checking Qdrant health: {e}")
            return {
                "status": "error",
                "message": str(e)
            }