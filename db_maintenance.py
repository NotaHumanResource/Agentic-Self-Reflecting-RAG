#!/usr/bin/env python3
"""
Database maintenance utility for AI memory systems.
"""
import os
import sys
import logging
import json
import time
import argparse
import uuid
import sqlite3
from datetime import datetime
import inspect

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Keep only console output
    ]
)

# ============================================================
# CHUNKED MEMORY TYPES
# ============================================================
# Memory types whose add_text() path splits one SQLite row into
# multiple Qdrant points (conversation summary chunking added
# 2026-03 in vector_db.py). The health check and sync logic must
# account for this so they don't mistake extra points for
# duplicates, and don't try to re-add "missing" entries that are
# actually already present as chunks.
#
# Extend this list if new chunked types are added in vector_db.py.
# ============================================================
CHUNKED_MEMORY_TYPES = {
    'conversation_summary',  # chunked in _chunk_summary_text()
}

# Define DatabaseMaintenance class outside of main function
class DatabaseMaintenance:
    """Database maintenance utility for DeepSeek memory systems."""
    
    def __init__(self, vector_db, memory_db):
        """Initialize with database instances."""
        self.vector_db = vector_db
        self.memory_db = memory_db

    def get_vector_count(self):
        """Get the count of vectors in the Qdrant collection using improved methods."""
        try:
            if not hasattr(self.vector_db, 'client'):
                return 0
                
            collection_name = getattr(self.vector_db, 'collection_name', 'deepseek_memory_optimized')
            
            # Method 1: Try to get collection info with better error handling
            try:
                collection_info = self.vector_db.client.get_collection(collection_name)
                
                # Try different ways to access points_count
                if hasattr(collection_info, 'points_count'):
                    return collection_info.points_count
                elif hasattr(collection_info, 'result') and hasattr(collection_info.result, 'points_count'):
                    return collection_info.result.points_count
                elif isinstance(collection_info, dict) and 'result' in collection_info:
                    result = collection_info['result']
                    if isinstance(result, dict) and 'points_count' in result:
                        return result['points_count']
                        
            except Exception as e:
                logging.warning(f"Method 1 failed: {e}")
            
            # Method 2: Try using scroll to count (slower but more reliable)
            try:
                scroll_result = self.vector_db.client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_payload=False,
                    with_vectors=False
                )
                
                # If scroll works, we can estimate count or get it from response
                if hasattr(scroll_result, 'next_page_offset') and scroll_result.next_page_offset is None:
                    # Small collection, try counting all
                    all_results = self.vector_db.client.scroll(
                        collection_name=collection_name,
                        limit=10000,
                        with_payload=False,
                        with_vectors=False
                    )
                    if hasattr(all_results, 'points'):
                        return len(all_results.points)
                
            except Exception as e:
                logging.warning(f"Method 2 failed: {e}")
            
            # Method 3: Use a search query to estimate
            try:
                # Use your VectorDB's search method with a broad query
                search_results = self.vector_db.search(
                    query="the",  # Very common word
                    mode="comprehensive",
                    k=10000  # Large number to get estimate
                )
                
                if search_results:
                    # This gives us a lower bound
                    estimated_count = len(search_results)
                    logging.info(f"Estimated vector count using search: {estimated_count}+")
                    return estimated_count
                    
            except Exception as e:
                logging.warning(f"Method 3 failed: {e}")
            
            logging.warning("All vector count methods failed")
            return 0
            
        except Exception as e:
            logging.error(f"Error getting vector count: {e}")
            return 0
        
    def is_autonomous_thought(self, content, memory_type=None, metadata=None):
        """Enhanced autonomous thought detection with guaranteed header patterns."""
        
        # Primary check: memory type
        if memory_type == "autonomous_thought":
            return True
        
        # Secondary check: metadata
        if metadata and isinstance(metadata, dict):
            source = metadata.get("source", "")
            if source == "autonomous_cognition":
                return True
            
            thought_type = metadata.get("thought_type", "")
            if thought_type in ["memory_optimization", "capabilities_reflection", 
                            "knowledge_analysis", "command_effectiveness", 
                            "user_categorization", "knowledge_reflection"]:
                return True
            
            # Check metadata type field
            meta_type = metadata.get("type", "")
            if meta_type in ["autonomous_thought", "memory_optimization", 
                            "capabilities_reflection", "knowledge_analysis"]:
                return True
        
        # Tertiary check: ALL header patterns (your complete list + guaranteed ones)
        autonomous_patterns = [
            "# Memory Organization Optimization",    # Guaranteed from our prompts
            "# Capabilities Reflection",             # Guaranteed from our prompts  
            "# Knowledge Gap Analysis",              # From your original list + our prompts
            "# Memory Usage Analysis",               # From your original list
            "# Command Effectiveness Analysis",      # From your original list
            "# User Information Categorization"      # From your original list
            "# Memory Consolidation Synthesis",      # Protects consolidation_pulse outputs
        ]
        
        # Check if content starts with any header (most reliable)
        content_lines = content.strip().split('\n')
        if content_lines and content_lines[0].strip() in autonomous_patterns:
            return True
        
        # Also check for headers anywhere in the first few lines (backup)
        first_lines = '\n'.join(content_lines[:3])
        for pattern in autonomous_patterns:
            if pattern in first_lines:
                return True
        
        # Additional content indicators (your original patterns)
        additional_indicators = [
            "autonomous_cognition",
            "thought_type:",
            "cognitive_state:",
            "AUTONOMOUS THOUGHT:",  # From reflection files
            "## Identified Knowledge Gaps:",
            "## Improvement Strategies:",
            "## Key User Attributes:"
        ]
        
        content_lower = content.lower()
        for indicator in additional_indicators:
            if indicator.lower() in content_lower:
                return True
        
        return False

    def get_all_memories_from_db(self, exclude_reminders=True):
        """Get all memories from the SQLite database directly with schema detection.
        
        Args:
            exclude_reminders (bool): Whether to exclude reminder entries
            
        Returns:
            List[Dict]: List of memory dictionaries
        """
        try:
            db_path = getattr(self.memory_db, 'db_path', None)
            if not db_path:
                logging.error("No db_path found in memory_db")
                return []
                
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # First, get the table schema to determine available columns
            cursor.execute("PRAGMA table_info(memories)")
            columns_info = cursor.fetchall()
            columns = [column[1] for column in columns_info]
            logging.info(f"Available columns in memories table: {columns}")
            
            # Build query dynamically based on available columns
            select_columns = ["id", "content"]
            if "memory_type" in columns:
                select_columns.append("memory_type")
            if "source" in columns:
                select_columns.append("source")
            if "tags" in columns:
                select_columns.append("tags")
            if "created_at" in columns:
                select_columns.append("created_at")
            if "updated_at" in columns:
                select_columns.append("updated_at")
            if "metadata" in columns:
                select_columns.append("metadata")
            if "confidence" in columns:
                select_columns.append("confidence")
            if "weight" in columns:  
                select_columns.append("weight")
            
            # Create query
            query = f"SELECT {', '.join(select_columns)} FROM memories"
            
            # Add condition to exclude reminders if requested
            if exclude_reminders and "memory_type" in columns:
                query += " WHERE memory_type != 'reminder'"
                
            logging.info(f"Using query: {query}")
            
            cursor.execute(query)
            rows = cursor.fetchall()
            memories = []
            
            for row in rows:
                # Convert row to dict
                memory = {}
                for column in select_columns:
                    memory[column] = row[column]
                
                # Parse metadata if present
                try:
                    if "metadata" in memory and memory["metadata"]:
                        metadata = json.loads(memory["metadata"])
                        memory["metadata"] = metadata
                    else:
                        memory["metadata"] = {}
                except Exception as e:
                    logging.warning(f"Error parsing metadata: {e}")
                    memory["metadata"] = {}
                
                # Handle confidence/weight mapping
                if "confidence" not in memory and "weight" in memory:
                    memory["confidence"] = memory["weight"]
                elif "confidence" not in memory:
                    # Default confidence if neither exists
                    memory["confidence"] = 0.5
                
                memories.append(memory)
            
            conn.close()
            num_reminders_filtered = 0
            if exclude_reminders:
                num_reminders_filtered = " (excluding reminders)"
            logging.info(f"Retrieved {len(memories)} memories from database{num_reminders_filtered}")
            return memories
        except Exception as e:
            logging.error(f"Error getting all memories: {e}", exc_info=True)
            return []
        
    def check_database_stats(self, include_reminders=False):
        """Get statistics about both memory databases.
        
        Args:
            include_reminders (bool): Whether to include reminders in the stats
            
        Returns:
            dict: Database statistics
        """
        stats = {
            "memory_db_total": 0,
            "memory_db_non_reminders": 0,
            "memory_db_reminders": 0,
            "memory_db_autonomous_thoughts": 0,
            "memory_db_syncable": 0,  # NEW: Count of entries that should be in VectorDB
            "vector_db_total": 0,
            "discrepancy": 0,
            "discrepancy_percentage": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Get memory DB counts with detailed breakdown
            try:
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get total count
                    cursor.execute("SELECT COUNT(*) FROM memories")
                    stats["memory_db_total"] = cursor.fetchone()[0]
                    
                    # Count reminders separately
                    cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = 'reminder'")
                    stats["memory_db_reminders"] = cursor.fetchone()[0]
                    
                    # Count autonomous thoughts separately
                    cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = 'autonomous_thought'")
                    stats["memory_db_autonomous_thoughts"] = cursor.fetchone()[0]
                    
                    # Calculate non-reminders
                    stats["memory_db_non_reminders"] = stats["memory_db_total"] - stats["memory_db_reminders"]
                    
                    # Calculate syncable entries (should be in both DBs)
                    # This excludes both reminders AND autonomous thoughts
                    stats["memory_db_syncable"] = (stats["memory_db_total"] - 
                                                stats["memory_db_reminders"] - 
                                                stats["memory_db_autonomous_thoughts"])
                    
                    # Check if reminders table exists
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name='reminders'
                    """)
                    
                    if cursor.fetchone():
                        # Count entries in dedicated reminders table
                        cursor.execute("SELECT COUNT(*) FROM reminders")
                        stats["dedicated_reminders_count"] = cursor.fetchone()[0]
                        
            except Exception as e:
                logging.error(f"Error counting memory DB: {e}")
            
            # Get vector DB count
            vector_count = 0
            try:
                if hasattr(self.vector_db, 'client'):
                    vector_count = self.get_vector_count()
                    logging.info(f"Used alternative method to get vector count: {vector_count}")
            except Exception as e:
                logging.error(f"Error counting vector DB: {e}")
            
            # Ensure vector_count is never None
            stats["vector_db_total"] = 0 if vector_count is None else vector_count

            # ============================================================
            # DISCREPANCY CALCULATION — chunked-type aware
            # ============================================================
            # conversation_summary rows are split into multiple Qdrant
            # points by the chunker in vector_db.add_text(). Comparing
            # raw SQLite row count against raw Qdrant point count will
            # always show a false "excess" in the vector DB.
            #
            # Instead we count chunked-type SQLite rows separately and
            # exclude them from the simple 1:1 comparison. The check for
            # those types is: do they have at least ONE Qdrant point?
            # That is handled by check_bidirectional_sync_issues() which
            # already reports "missing_in_vector_db: 0" when healthy.
            # ============================================================
            try:
                # Count chunked-type rows so we can exclude from 1:1 comparison
                chunked_type_list = list(CHUNKED_MEMORY_TYPES)
                placeholders = ','.join(['?' for _ in chunked_type_list])
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        f"SELECT COUNT(*) FROM memories WHERE memory_type IN ({placeholders})",
                        chunked_type_list
                    )
                    chunked_row_count = cursor.fetchone()[0]
                stats["chunked_type_rows"] = chunked_row_count
            except Exception as chunk_count_err:
                logging.warning(f"Could not count chunked-type rows: {chunk_count_err}")
                chunked_row_count = 0
                stats["chunked_type_rows"] = 0

            # Non-chunked syncable rows should have exactly 1 vector point each
            non_chunked_syncable = stats["memory_db_syncable"] - chunked_row_count

            # Qdrant points attributable to chunked types (anything above non-chunked count)
            vector_count_final = stats["vector_db_total"] or 0
            non_chunked_vector_points = max(0, vector_count_final - chunked_row_count)

            # Discrepancy = difference in the non-chunked 1:1 portion only
            stats["discrepancy"] = abs(non_chunked_syncable - non_chunked_vector_points)
            if non_chunked_syncable > 0:
                stats["discrepancy_percentage"] = (
                    stats["discrepancy"] / non_chunked_syncable
                ) * 100
            else:
                stats["discrepancy_percentage"] = 0

            logging.info(
                f"Database stats: {stats} | "
                f"chunked_rows={chunked_row_count}, "
                f"non_chunked_syncable={non_chunked_syncable}, "
                f"non_chunked_vector_points={non_chunked_vector_points}"
            )
            return stats
        
        except Exception as e:
            logging.error(f"Error getting database stats: {e}", exc_info=True)
            # Return a valid stats dict even on error
            return {
                "memory_db_total": 0,
                "memory_db_non_reminders": 0,
                "memory_db_reminders": 0,
                "memory_db_autonomous_thoughts": 0,
                "memory_db_syncable": 0,
                "vector_db_total": 0,
                "discrepancy": 0,
                "discrepancy_percentage": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def cleanup_orphaned_vectors(self):
        """Find and remove vector entries that don't exist in memory DB using direct deletion.
        EXTRA PROTECTIVE: Never deletes autonomous thought content or reminders, even if they appear orphaned.
        
        FIXES APPLIED:
        - Improved content comparison with whitespace normalization
        - Better handling of encoding differences that could cause false orphan detection
        """
        logging.info("Starting orphaned vector cleanup with EXTRA protection for autonomous thoughts...")
        
        stats = {
            "total_checked": 0,
            "orphans_found": 0,
            "cleanup_success": 0,
            "cleanup_failed": 0,
            "reminders_protected": 0,
            "autonomous_thoughts_protected": 0,
            "potential_autonomous_thoughts_protected": 0  # Content that might be autonomous thoughts
        }
        
        try:
            # Get all syncable content strings from memory DB (ONLY entries that should be in both DBs)
            syncable_contents = set()
            syncable_contents_normalized = set()  # FIX: Normalized versions for comparison
            protected_contents = set()  # Content to never delete
            
            def normalize_content(content):
                """Normalize content for comparison - handles whitespace and encoding differences."""
                if not content:
                    return ""
                # Strip, normalize whitespace, and lowercase for comparison
                return ' '.join(content.strip().lower().split())
            
            try:
                db_path = getattr(self.memory_db, 'db_path', None)
                if db_path:
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        
                        # Get ONLY syncable content (excludes SQLite-only data)
                        cursor.execute("""
                            SELECT content FROM memories 
                            WHERE memory_type != 'reminder' 
                            AND memory_type != 'autonomous_thought'
                        """)
                        rows = cursor.fetchall()
                        for row in rows:
                            if row[0]:
                                syncable_contents.add(row[0])
                                syncable_contents_normalized.add(normalize_content(row[0]))
                    
                    # Get ALL autonomous thought content to protect (even if somehow in VectorDB)
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT content FROM memories WHERE memory_type = 'autonomous_thought'")
                        rows = cursor.fetchall()
                        for row in rows:
                            if row[0]:
                                protected_contents.add(row[0])
                    
                    # Get reminder content to protect
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT content FROM memories WHERE memory_type = 'reminder'")
                        rows = cursor.fetchall()
                        for row in rows:
                            if row[0]:
                                protected_contents.add(row[0])
                    
                    # Also check dedicated reminders table
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name='reminders'
                        """)
                        
                        if cursor.fetchone():
                            cursor.execute("SELECT content FROM reminders")
                            rows = cursor.fetchall()
                            for row in rows:
                                if row[0]:
                                    protected_contents.add(row[0])
                    
                    logging.info(f"Loaded {len(syncable_contents)} syncable contents")
                    logging.info(f"Loaded {len(protected_contents)} protected contents (autonomous thoughts + reminders)")
                    
            except Exception as e:
                logging.error(f"Error loading memory contents: {e}")
                return stats
            
            # Define autonomous thought patterns for extra protection
            autonomous_patterns = [
                "# Memory Organization Optimization",
                "# Capabilities Reflection", 
                "# Knowledge Gap Analysis",
                "# Memory Usage Analysis",
                "# Command Effectiveness Analysis",
                "# User Information Categorization",
                "# Knowledge Acquisition Reflection",
                "# Claude Knowledge Integration",
                "Beginning analysis of knowledge gaps",
                "autonomous_cognition",
                "thought_type:",
                "cognitive_state:",
                "AUTONOMOUS THOUGHT:",
                "## Identified Knowledge Gaps:",
                "## Improvement Strategies:",
                "## Key User Attributes:"
            ]
            
            # Search for vectors using various queries
            search_queries = [
                "memory information storage",
                "document text data content", 
                "conversation summary chat",
                "reflection learning thinking",
                "important critical essential"
            ]
            
            processed_contents = set()
            
            for query in search_queries:
                logging.info(f"Searching for vectors with query: '{query}'")
                
                for k_value in [200, 500, 1000]:
                    try:
                        search_results = self.vector_db.search(
                            query=query,
                            mode="comprehensive",
                            k=k_value
                        )
                        
                        if search_results:
                            logging.info(f"Retrieved {len(search_results)} results for query '{query}' with k={k_value}")
                            
                            for result in search_results:
                                content = result.get('content', '')
                                metadata = result.get('metadata', {})
                                
                                # Skip if empty or already processed
                                if not content or content in processed_contents:
                                    continue
                                    
                                processed_contents.add(content)
                                stats["total_checked"] += 1
                                
                                # CRITICAL: Always protect content that's in protected_contents
                                if content in protected_contents:
                                    stats["autonomous_thoughts_protected"] += 1
                                    logging.debug(f"PROTECTING known autonomous thought/reminder: {content[:50]}...")
                                    continue
                                
                                # EXTRA PROTECTION: Check for autonomous thought patterns in content
                                is_likely_autonomous = False
                                content_lower = content.lower()
                                
                                for pattern in autonomous_patterns:
                                    if pattern.lower() in content_lower:
                                        is_likely_autonomous = True
                                        break
                                
                                if is_likely_autonomous:
                                    stats["potential_autonomous_thoughts_protected"] += 1
                                    logging.debug(f"PROTECTING potential autonomous thought pattern: {content[:50]}...")
                                    continue
                                
                                # EXTRA PROTECTION: Check metadata for autonomous indicators
                                if metadata:
                                    metadata_source = metadata.get('source', '')
                                    metadata_type = metadata.get('type', '')
                                    
                                    if ('autonomous' in str(metadata_source).lower() or 
                                        'autonomous' in str(metadata_type).lower() or
                                        metadata_type == 'autonomous_thought'):
                                        stats["autonomous_thoughts_protected"] += 1
                                        logging.debug(f"PROTECTING by metadata autonomous indicator: {content[:50]}...")
                                        continue
                                
                                # PROTECTION: Skip obvious reminders
                                if ("remind" in content_lower or 
                                    "[REMINDER:" in content or
                                    "reminder:" in content_lower):
                                    stats["reminders_protected"] += 1
                                    logging.debug(f"PROTECTING reminder pattern: {content[:50]}...")
                                    continue
                                
                                # =========================================================
                                # FIX 5: Improved content comparison
                                # Check both exact match AND normalized match to handle
                                # whitespace/encoding differences
                                # =========================================================
                                content_normalized = normalize_content(content)
                                is_in_memory_db = (content in syncable_contents or 
                                                   content_normalized in syncable_contents_normalized)
                                
                                if not is_in_memory_db:
                                    stats["orphans_found"] += 1
                                    logging.info(f"Found orphaned vector: {content[:50]}...")
                                    
                                    try:
                                        # Final safety check before deletion
                                        if (len(content) > 200 and 
                                            any(word in content_lower for word in ['analysis', 'reflection', 'cognition', 'autonomous', 'memory'])):
                                            logging.warning(f"SAFETY: Skipping deletion of complex content that might be autonomous: {content[:50]}...")
                                            stats["potential_autonomous_thoughts_protected"] += 1
                                            continue
                                        
                                        # Safe to delete
                                        success = self.vector_db.delete_text(content)
                                        if success:
                                            stats["cleanup_success"] += 1
                                            logging.info(f"Successfully deleted orphaned vector")
                                        else:
                                            stats["cleanup_failed"] += 1
                                            logging.warning(f"Failed to delete orphaned vector")
                                            
                                    except Exception as e:
                                        stats["cleanup_failed"] += 1
                                        logging.error(f"Error deleting orphaned vector: {e}")
                            
                            if len(search_results) >= k_value * 0.5:
                                break
                        
                    except Exception as e:
                        logging.error(f"Error processing query '{query}' with k={k_value}: {e}")
                
                # Log progress
                logging.info(f"Processed {stats['total_checked']} vectors, " +
                            f"found {stats['orphans_found']} orphans, " +
                            f"protected {stats['autonomous_thoughts_protected']} autonomous thoughts, " +
                            f"protected {stats['potential_autonomous_thoughts_protected']} potential autonomous thoughts, " +
                            f"protected {stats['reminders_protected']} reminders")
            
            logging.info(f"EXTRA PROTECTIVE orphaned vector cleanup completed: {stats}")
            logging.info(f"Total autonomous content protected: {stats['autonomous_thoughts_protected'] + stats['potential_autonomous_thoughts_protected']}")
            return stats
            
        except Exception as e:
            logging.error(f"Error during EXTRA PROTECTIVE orphaned vector cleanup: {e}", exc_info=True)
            return stats
        
    def enhanced_health_check(self):
        """Perform a comprehensive health check of both databases."""
        health_report = {
            "stats": self.check_database_stats(),
            "duplicates": {
                "memory_db": self.find_memory_db_duplicates(),
                "vector_db": self.find_vector_db_duplicates()
            },
            "sync_issues": self.check_bidirectional_sync_issues(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate overall health score
        memory_health = 100 - min(100, health_report["duplicates"]["memory_db"]["percentage"] * 100)
        vector_health = 100 - min(100, health_report["duplicates"]["vector_db"]["percentage"] * 100)
        sync_health = 100 - min(100, health_report["stats"]["discrepancy_percentage"])
        
        health_report["health_score"] = {
            "memory_db": round(memory_health, 1),
            "vector_db": round(vector_health, 1),
            "sync": round(sync_health, 1),
            "overall": round((memory_health + vector_health + sync_health) / 3, 1)
        }
        
        return health_report    
    
    def find_memory_db_duplicates(self):
        """Find duplicate entries in the memory database, excluding SQLite-only data."""
        logging.info("Checking for duplicate entries in memory database (excluding SQLite-only data)...")
        
        stats = {
            "total_entries": 0,
            "syncable_entries": 0,  # NEW: Entries that should be in VectorDB
            "unique_contents": 0,
            "duplicates_found": 0,
            "duplicate_groups": 0,
            "percentage": 0,
            "reminders_excluded": 0,
            "autonomous_thoughts_excluded": 0
        }
        
        try:
            db_path = getattr(self.memory_db, 'db_path', None)
            if not db_path:
                logging.error("No db_path found in memory_db")
                return stats
                
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute("SELECT COUNT(*) FROM memories")
                stats["total_entries"] = cursor.fetchone()[0]
                
                # Count excluded entries
                cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = 'reminder'")
                stats["reminders_excluded"] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = 'autonomous_thought'")
                stats["autonomous_thoughts_excluded"] = cursor.fetchone()[0]
                
                # Calculate syncable entries
                stats["syncable_entries"] = (stats["total_entries"] - 
                                        stats["reminders_excluded"] - 
                                        stats["autonomous_thoughts_excluded"])
                
                # Find duplicates ONLY among syncable entries
                cursor.execute("""
                    SELECT content, COUNT(*) as count, GROUP_CONCAT(id) as ids
                    FROM memories
                    WHERE memory_type != 'reminder' 
                    AND memory_type != 'autonomous_thought'
                    GROUP BY content
                    HAVING COUNT(*) > 1
                    ORDER BY COUNT(*) DESC
                """)
                
                duplicate_groups = cursor.fetchall()
                stats["duplicate_groups"] = len(duplicate_groups)
                
                # Count total duplicates
                for _, count, _ in duplicate_groups:
                    stats["duplicates_found"] += (count - 1)  # Subtract 1 to count only extras
                
                # Get count of unique contents among syncable entries
                cursor.execute("""
                    SELECT COUNT(DISTINCT content) FROM memories
                    WHERE memory_type != 'reminder' 
                    AND memory_type != 'autonomous_thought'
                """)
                stats["unique_contents"] = cursor.fetchone()[0]
                
                # Calculate percentage based on syncable entries only
                if stats["syncable_entries"] > 0:
                    stats["percentage"] = stats["duplicates_found"] / stats["syncable_entries"]
            
            logging.info(f"Found {stats['duplicates_found']} duplicate entries in memory DB " +
                        f"(excluding {stats['reminders_excluded']} reminders and " +
                        f"{stats['autonomous_thoughts_excluded']} autonomous thoughts)")
            return stats
        
        except Exception as e:
            logging.error(f"Error finding memory DB duplicates: {e}", exc_info=True)
            return stats

    def remove_memory_db_duplicates(self):
        """Remove duplicate entries from memory database, keeping the newest or highest weight entry.
        Excludes SQLite-only data (reminders and autonomous thoughts) from removal."""
        logging.info("Removing duplicate entries from memory database (excluding SQLite-only data)...")
        
        stats = {
            "duplicate_groups": 0,
            "duplicates_removed": 0,
            "errors": 0,
            "autonomous_thoughts_protected": 0,
            "reminders_protected": 0
        }
        
        try:
            db_path = getattr(self.memory_db, 'db_path', None)
            if not db_path:
                logging.error("No db_path found in memory_db")
                return stats
                
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Find duplicates ONLY among syncable entries (exclude SQLite-only data)
                cursor.execute("""
                    SELECT content, COUNT(*) as count, GROUP_CONCAT(id) as ids 
                    FROM memories
                    WHERE memory_type != 'reminder' 
                    AND memory_type != 'autonomous_thought'
                    GROUP BY content
                    HAVING COUNT(*) > 1
                """)
                
                duplicate_groups = cursor.fetchall()
                stats["duplicate_groups"] = len(duplicate_groups)
                
                # Process each group of duplicates
                for content, count, ids_str in duplicate_groups:
                    try:
                        ids = [int(id_str) for id_str in ids_str.split(',')]
                        
                        # Find the best record to keep (newest or highest weight)
                        # Check if confidence column exists, fall back to weight
                        try:
                            cursor.execute("""
                                SELECT id FROM memories
                                WHERE id IN ({})
                                ORDER BY confidence DESC, created_at DESC
                                LIMIT 1
                            """.format(','.join(['?' for _ in ids])), ids)
                        except:
                            # Fall back to weight if confidence doesn't exist
                            cursor.execute("""
                                SELECT id FROM memories
                                WHERE id IN ({})
                                ORDER BY weight DESC, created_at DESC
                                LIMIT 1
                            """.format(','.join(['?' for _ in ids])), ids)
                        
                        keep_id = cursor.fetchone()[0]
                        
                        # Delete all duplicates except the one to keep
                        delete_ids = [id for id in ids if id != keep_id]
                        for id_to_delete in delete_ids:
                            cursor.execute("DELETE FROM memories WHERE id = ?", (id_to_delete,))
                            stats["duplicates_removed"] += 1
                            
                    except Exception as e:
                        logging.error(f"Error processing duplicate group for content '{content[:50]}...': {e}")
                        stats["errors"] += 1
                
                conn.commit()
                
            logging.info(f"Removed {stats['duplicates_removed']} duplicate entries from memory DB " +
                        f"(protected SQLite-only autonomous thoughts and reminders)")
            return stats
        
        except Exception as e:
            logging.error(f"Error removing memory DB duplicates: {e}", exc_info=True)
            return stats
        
    def find_vector_db_duplicates(self):
        """Find duplicate entries in the vector database by content similarity."""
        logging.info("Checking for duplicate entries in vector database...")
        
        stats = {
            "total_entries": 0,
            "duplicates_found": 0,
            "duplicate_groups": 0,
            "percentage": 0
        }
        
        try:
            # Get total count
            vector_count = self.get_vector_count()
            stats["total_entries"] = vector_count
            
            if vector_count == 0:
                return stats
                
            # Use search queries to find potential duplicates
            search_queries = [
                "memory information storage",
                "document text data content",
                "reminder message note", 
                "conversation summary chat",
                "reflection learning thinking",
                "important critical essential"
            ]
            
            processed_contents = set()
            content_id_map = {}  # Maps content to vector ids
            
            # Process each search query
            for query in search_queries:
                logging.info(f"Searching for potential duplicates with query: '{query}'")
                
                try:
                    search_results = self.vector_db.search(
                        query=query,
                        mode="comprehensive",
                        k=500
                    )
                    
                    if not search_results:
                        continue
                        
                    # Check for duplicates in this batch
                    for result in search_results:
                        content = result.get('content', '')
                        vector_id = result.get('id', None)
                        
                        if not content or content in processed_contents:
                            continue
                            
                        processed_contents.add(content)
                        
                        # Search for this exact content to find duplicates
                        exact_matches = self.vector_db.search(
                            query=content,
                            mode="default",
                            k=10
                        )
                        
                        if exact_matches and len(exact_matches) > 1:
                            # Found duplicates
                            duplicate_ids = [match.get('id') for match in exact_matches 
                                            if match.get('id') and match.get('similarity_score', 0) > 0.98]
                            
                            if len(duplicate_ids) > 1:  # Only count if we have actual duplicates
                                content_id_map[content] = duplicate_ids
                                stats["duplicates_found"] += (len(duplicate_ids) - 1)
                                stats["duplicate_groups"] += 1
                    
                except Exception as e:
                    logging.error(f"Error processing query '{query}': {e}")
            
            # Calculate percentage
            if stats["total_entries"] > 0:
                stats["percentage"] = stats["duplicates_found"] / stats["total_entries"]
            
            logging.info(f"Found {stats['duplicates_found']} duplicate entries in vector DB")
            
            # Store the content-id map for use in removal
            self._vector_duplicate_map = content_id_map
            
            return stats
        
        except Exception as e:
            logging.error(f"Error finding vector DB duplicates: {e}", exc_info=True)
            return stats

    def remove_vector_db_duplicates(self):
        """Remove duplicate entries from vector database, keeping one entry per content."""
        logging.info("Removing duplicate entries from vector database...")
        
        stats = {
            "duplicate_groups": 0,
            "duplicates_removed": 0,
            "errors": 0
        }
        
        try:
            # Check if we have duplicate data from previous find operation
            if not hasattr(self, '_vector_duplicate_map') or not self._vector_duplicate_map:
                # Run find operation first
                find_stats = self.find_vector_db_duplicates()
                stats["duplicate_groups"] = find_stats["duplicate_groups"]
                if stats["duplicate_groups"] == 0:
                    logging.info("No duplicates found in vector DB")
                    return stats
            else:
                stats["duplicate_groups"] = len(self._vector_duplicate_map)
            
            # Process each group of duplicates
            for content, vector_ids in self._vector_duplicate_map.items():
                try:
                    if len(vector_ids) <= 1:
                        continue
                    
                    # Keep the first ID, delete the rest
                    keep_id = vector_ids[0]
                    delete_ids = vector_ids[1:]
                    
                    for id_to_delete in delete_ids:
                        # Use the vector_db's delete_by_id method if available
                        if hasattr(self.vector_db, 'delete_by_id'):
                            success = self.vector_db.delete_by_id(id_to_delete)
                        else:
                            # Fall back to using the client directly
                            collection_name = getattr(self.vector_db, 'collection_name', 'deepseek_memory_optimized')
                            success = self.vector_db.client.delete(
                                collection_name=collection_name,
                                points_selector=id_to_delete
                            )
                        
                        if success:
                            stats["duplicates_removed"] += 1
                        else:
                            stats["errors"] += 1
                            
                except Exception as e:
                    logging.error(f"Error removing vector duplicate for content '{content[:50]}...': {e}")
                    stats["errors"] += 1
            
            logging.info(f"Removed {stats['duplicates_removed']} duplicate entries from vector DB")
            return stats
        
        except Exception as e:
            logging.error(f"Error removing vector DB duplicates: {e}", exc_info=True)
            return stats
        
    def check_bidirectional_sync_issues(self):
        """Check for bidirectional sync issues between memory DB and vector DB.
        Excludes SQLite-only data (reminders and autonomous thoughts) from analysis."""
        logging.info("Checking for bidirectional sync issues (excluding SQLite-only data)...")
        
        stats = {
            "missing_in_vector_db": 0,
            "missing_in_memory_db": 0,
            "reminders_excluded": 0,
            "autonomous_thoughts_excluded": 0,
            "mismatched_metadata": 0,
            "checked_memories": 0,
            "checked_vectors": 0
        }
        
        try:
            # Step 1: Get all syncable memories from memory DB
            db_path = getattr(self.memory_db, 'db_path', None)
            if not db_path:
                logging.error("No db_path found in memory_db")
                return stats
                
            syncable_memories = []
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get all non-reminder, non-autonomous-thought entries
                cursor.execute("""
                    SELECT id, content, memory_type, metadata, source
                    FROM memories
                    WHERE memory_type != 'reminder' 
                    AND memory_type != 'autonomous_thought'
                """)
                
                rows = cursor.fetchall()
                for row in rows:
                    memory = dict(row)
                    # Parse metadata if present
                    try:
                        if memory.get("metadata"):
                            memory["metadata"] = json.loads(memory["metadata"])
                        else:
                            memory["metadata"] = {}
                    except:
                        memory["metadata"] = {}
                    syncable_memories.append(memory)
                
                # Count excluded entries for reporting
                cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = 'reminder'")
                stats["reminders_excluded"] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = 'autonomous_thought'")
                stats["autonomous_thoughts_excluded"] = cursor.fetchone()[0]
            
            memory_content_set = {m.get("content", "") for m in syncable_memories if m.get("content")}
            stats["checked_memories"] = len(syncable_memories)
            
            # Step 2: Check each syncable memory against vector DB
            for memory in syncable_memories:
                content = memory.get("content", "")
                if not content:
                    continue
                            
                # Check if memory exists in vector DB
                vector_results = self.vector_db.search(
                query=content,
                mode="default",  
                k=1
            )
                
                if not vector_results or len(vector_results) == 0:
                    stats["missing_in_vector_db"] += 1
            
            # Step 3: Sample vector DB to check for entries missing from memory DB
            search_queries = [
                "memory information", "document content", "conversation chat",
                "reflection thinking", "important essential"
            ]
            
            processed_vectors = set()
            
            for query in search_queries:
                try:
                    search_results = self.vector_db.search(
                        query=query,
                        mode="comprehensive",
                        k=100
                    )
                    
                    if not search_results:
                        continue
                        
                    for result in search_results:
                        vector_content = result.get('content', '')
                        
                        if not vector_content or vector_content in processed_vectors:
                            continue
                            
                        # Skip obvious SQLite-only content patterns
                        if ("remind" in vector_content.lower() or 
                            "[REMINDER:" in vector_content or
                            "# Memory Organization Optimization" in vector_content or
                            "# Capabilities Reflection" in vector_content or
                            "# Knowledge Gap Analysis" in vector_content):
                            continue
                        
                        processed_vectors.add(vector_content)
                        stats["checked_vectors"] += 1
                        
                        # Check if vector exists in memory DB
                        if vector_content not in memory_content_set:
                            stats["missing_in_memory_db"] += 1
                            
                except Exception as e:
                    logging.error(f"Error checking vector sync with query '{query}': {e}")
            
            logging.info(f"Sync check completed: {stats['missing_in_vector_db']} missing in vector DB, " + 
                        f"{stats['missing_in_memory_db']} missing in memory DB " +
                        f"(excluded {stats['reminders_excluded']} reminders, " +
                        f"{stats['autonomous_thoughts_excluded']} autonomous thoughts)")
            return stats
        
        except Exception as e:
            logging.error(f"Error checking bidirectional sync: {e}", exc_info=True)
            return stats
            
    def fix_all_database_issues(self):
        """Fix all database issues: duplicates, synchronization, and orphans.
        Protects SQLite-only data (reminders and autonomous thoughts) from modification."""
        logging.info("Starting comprehensive database repair (protecting SQLite-only data)...")
        
        results = {
            "initial_health": self.enhanced_health_check(),
            "steps": {},
            "final_health": None,
            "sqlite_only_protection": {
                "autonomous_thoughts_protected": True,
                "reminders_protected": True,
                "explanation": "SQLite-only data excluded from all repair operations"
            }
        }
        
        # Step 1: Remove duplicates from memory DB (excluding SQLite-only)
        logging.info("Step 1: Removing memory DB duplicates (protecting SQLite-only data)...")
        results["steps"]["memory_duplicates"] = self.remove_memory_db_duplicates()
        
        # Step 2: Remove duplicates from vector DB
        logging.info("Step 2: Removing vector DB duplicates...")
        results["steps"]["vector_duplicates"] = self.remove_vector_db_duplicates()
        
        # Step 3: Sync missing entries from memory DB to vector DB (excluding SQLite-only)
        logging.info("Step 3: Syncing syncable entries from memory DB to vector DB...")
        results["steps"]["memory_to_vector_sync"] = self.full_database_synchronization()
        
        # Step 4: Clean up orphaned vectors (protecting SQLite-only content)
        logging.info("Step 4: Cleaning up orphaned vectors (protecting SQLite-only data)...")
        results["steps"]["cleanup_orphans"] = self.cleanup_orphaned_vectors()
        
        # Step 5: Check final health
        logging.info("Step 5: Checking final database health...")
        results["final_health"] = self.enhanced_health_check()
        
        # Calculate improvement
        initial_score = results["initial_health"]["health_score"]["overall"]
        final_score = results["final_health"]["health_score"]["overall"]
        results["improvement"] = final_score - initial_score
        
        logging.info(f"Database repair completed. Health score improved from {initial_score} to {final_score}")
        logging.info("All SQLite-only data (autonomous thoughts, reminders) was protected during repair")
        return results

    def reset_and_resync_vector_db(self):
        """Reset the vector database and resync all entries from memory DB.
        
        FIXES APPLIED:
        - Updated to QWEN3 Vector embeddings model)
        - Added verification step after resync to check sync success
        - Improved error handling and logging
        """
        try:
            # Import collection name from config (it's not an instance attribute)
            from config import QDRANT_COLLECTION_NAME
            collection_name = QDRANT_COLLECTION_NAME
            
            logging.info(f"Starting vector database reset for collection: {collection_name}")
            
            # Step 1: Delete the collection
            logging.info(f"Deleting collection {collection_name}")
            try:
                delete_result = self.vector_db.client.delete_collection(collection_name=collection_name)
                logging.info(f"Collection deletion result: {delete_result}")
                logging.info(f"Successfully deleted collection {collection_name}")
            except Exception as e:
                logging.error(f"Error deleting collection: {e}")
                # Continue anyway - collection might not exist
                
            # Step 2: Recreate the collection with the same schema
            try:
                # =========================================================
                # FIX 3: Corrected embedding dimension fallback
                # QWEN3 Embeddings uses 4096
                # =========================================================
                dimension = getattr(self.vector_db, 'embedding_dimension', 4096)  # FIXED: was 5120
                
                # Double-check by trying to get from config
                try:
                    from config import QDRANT_CONFIG
                    config_dimension = QDRANT_CONFIG.get('vector_size', 4096)
                    if dimension != config_dimension:
                        logging.warning(f"Dimension mismatch: instance={dimension}, config={config_dimension}. Using config value.")
                        dimension = config_dimension
                except ImportError:
                    pass
                # =========================================================
                
                logging.info(f"Recreating collection {collection_name} with dimension {dimension}")
                
                # Import qdrant_models for proper type references
                from qdrant_client.http import models as qdrant_models
                
                # Create collection with proper configuration
                create_result = self.vector_db.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=dimension,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logging.info(f"Collection creation result: {create_result}")
                logging.info(f"Successfully recreated collection {collection_name}")
                
            except Exception as e:
                logging.error(f"Error recreating collection: {e}")
                return {
                    "error": f"Failed to recreate collection: {str(e)}",
                    "total_checked": 0,
                    "sync_successes": 0,
                    "sync_failures": 1,
                    "reset_successful": False
                }
                    
            # Step 3: Sync all entries from memory DB
            logging.info("Starting full resync from memory DB")
            sync_stats = self.full_database_synchronization()
            
            # =========================================================
            # FIX 4: Added verification step after resync
            # Checks that the sync was successful by comparing counts
            # =========================================================
            logging.info("Verifying resync results...")
            
            # Get counts for verification
            verification_stats = self.check_database_stats()
            
            sync_stats["reset_successful"] = True
            sync_stats["collection_recreated"] = True
            sync_stats["verification"] = {
                "vector_db_count": verification_stats.get("vector_db_total", 0),
                "memory_db_syncable": verification_stats.get("memory_db_syncable", 0),
                "discrepancy": verification_stats.get("discrepancy", 0),
                "discrepancy_percentage": verification_stats.get("discrepancy_percentage", 0)
            }
            
            # Warn if significant discrepancy remains
            if verification_stats.get("discrepancy_percentage", 0) > 5:
                logging.warning(f"⚠️ Resync completed but {verification_stats['discrepancy_percentage']:.1f}% discrepancy remains")
                sync_stats["verification"]["warning"] = "Significant discrepancy remains after resync"
            else:
                logging.info(f"✅ Resync verified: {verification_stats['discrepancy_percentage']:.1f}% discrepancy")
            # =========================================================
            
            logging.info(f"Vector database reset completed successfully with stats: {sync_stats}")
            return sync_stats
            
        except Exception as e:
            logging.error(f"Critical error in reset_and_resync_vector_db: {e}", exc_info=True)
            return {
                "error": f"Critical error during reset: {str(e)}",
                "reset_successful": False,
                "total_checked": 0,
                "sync_successes": 0,
                "sync_failures": 1
            }

        
    def full_database_synchronization(self):
        """Synchronize all syncable entries between memory DB and vector DB.
        Excludes SQLite-only data (reminders and autonomous thoughts)."""
        logging.info("Starting full database synchronization (excluding SQLite-only data)")
        
        # Get only syncable memories from memory DB (exclude SQLite-only)
        db_path = getattr(self.memory_db, 'db_path', None)
        if not db_path:
            logging.error("No db_path found in memory_db")
            return {"error": "No database path found"}
        
        syncable_memories = []
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all syncable entries (exclude SQLite-only data)
            cursor.execute("""
                SELECT id, content, memory_type, metadata, source, confidence, tags, created_at
                FROM memories
                WHERE memory_type != 'reminder' 
                AND memory_type != 'autonomous_thought'
            """)
            
            rows = cursor.fetchall()
            for row in rows:
                memory = dict(row)
                # Parse metadata if present
                try:
                    if memory.get("metadata"):
                        memory["metadata"] = json.loads(memory["metadata"])
                    else:
                        memory["metadata"] = {}
                except:
                    memory["metadata"] = {}
                syncable_memories.append(memory)
        
        logging.info(f"Retrieved {len(syncable_memories)} syncable memories from memory DB")
        
        # Process memories in batches
        sync_stats = {
            "total_checked": 0,
            "missing_in_vector": 0,
            "sync_attempts": 0,
            "sync_failures": 0,
            "sync_successes": 0,
            "autonomous_thoughts_protected": 0,
            "reminders_protected": 0,
            "batch_size": 50
        }
        
        batch_size = sync_stats["batch_size"]
        total_batches = (len(syncable_memories) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(syncable_memories))
            batch = syncable_memories[start_idx:end_idx]
            
            logging.info(f"Processing batch {batch_idx+1}/{total_batches} (memories {start_idx+1}-{end_idx})")
            
            for memory in batch:
                sync_stats["total_checked"] += 1
                content = memory.get("content", "")
                memory_id = memory.get("id", "")
                memory_type = memory.get("memory_type", "general")
                
                if not content:
                    logging.warning(f"Skipping empty content memory with ID {memory_id}")
                    continue

                # ============================================================
                # CHUNKED TYPE GUARD
                # ============================================================
                # conversation_summary (and any future chunked types) store one
                # SQLite row as N overlapping Qdrant points. Searching for the
                # full original text against those chunks will never score 0.95+
                # because each chunk is only a 200-word excerpt. The old code
                # therefore treated every chunked row as "missing" and re-added
                # it, stacking duplicate chunk sets on each repair run.
                #
                # Fix: for chunked types use a metadata filter on memory_id
                # (written to every chunk's metadata). If at least one point
                # exists with this memory_id, the row is already present.
                # ============================================================
                if memory_type in CHUNKED_MEMORY_TYPES:
                    try:
                        chunk_results = self.vector_db.search(
                            query="",   # scroll path — metadata filter only
                            mode="default",
                            k=1,
                            metadata_filters={"memory_id": str(memory_id)}
                        )
                        time.sleep(0.05)
                        if chunk_results:
                            # At least one chunk present — row is synced
                            logging.debug(
                                f"CHUNKED TYPE '{memory_type}': memory_id={memory_id} "
                                f"has {len(chunk_results)} chunk(s) — skipping re-sync"
                            )
                            continue
                        else:
                            # No chunks found at all — genuinely missing, fall through to add
                            logging.info(
                                f"CHUNKED TYPE '{memory_type}': memory_id={memory_id} "
                                f"has no chunks in vector DB — will re-sync"
                            )
                    except Exception as chunk_search_err:
                        logging.warning(
                            f"Could not check chunks for memory_id={memory_id}: "
                            f"{chunk_search_err} — falling back to content search"
                        )
                        # Fall through to normal content search below on error

                # Normal content-similarity existence check for non-chunked types
                # (also used as fallback if chunk metadata search fails above)
                try:
                    vector_results = self.vector_db.search(
                        query=content,
                        mode="selective",  # high-precision matching
                        k=1
                    )
                    # Rate limit: small delay to prevent overwhelming Ollama embedding service
                    time.sleep(0.05)  # 50ms delay between searches
                except Exception as search_error:
                    logging.error(f"Error searching vector DB for memory {memory_id}: {search_error}")
                    vector_results = []
                    # Longer delay on error to let Ollama recover
                    time.sleep(0.5)

                # Calculate match score
                match_score = 0
                if vector_results and len(vector_results) > 0:
                    match_score = vector_results[0].get("similarity_score", 0)

                if not vector_results or len(vector_results) == 0 or match_score < 0.95:
                    # Missing in vector DB, need to sync
                    sync_stats["missing_in_vector"] += 1
                    sync_stats["sync_attempts"] += 1
                    
                    try:
                        # Extract metadata properly
                        metadata = memory.get("metadata", {})
                        source = memory.get("source", metadata.get("source", ""))
                        confidence = memory.get("confidence", metadata.get("confidence", 0.5))
                        tags = memory.get("tags", metadata.get("tags", ""))
                        
                        # Create proper metadata for vector DB
                        vector_metadata = {
                            "memory_id": str(memory_id),  # Ensure it's a string
                            "type": memory_type,
                            "source": source,
                            "confidence": float(confidence) if confidence else 0.5,
                            "tags": tags
                        }
                        
                        # Add any other metadata that might be present
                        for key, value in metadata.items():
                            if key not in vector_metadata:
                                vector_metadata[key] = value
                        
                        # FIXED: Add to vector DB and handle the tuple return
                        success, reason = self.vector_db.add_text(
                            text=content,
                            metadata=vector_metadata,
                            memory_id=str(memory_id)
                        )
                        
                        if success:
                            sync_stats["sync_successes"] += 1
                            logging.info(f"Successfully synced memory ID {memory_id} to vector DB")
                        else:
                            sync_stats["sync_failures"] += 1
                            logging.error(f"Failed to sync memory ID {memory_id} to vector DB (reason: {reason})")
                        
                        # Rate limit: delay after each sync to prevent overwhelming Ollama
                        time.sleep(0.1)  # 100ms delay after each write
                            
                    except Exception as e:
                        sync_stats["sync_failures"] += 1
                        logging.error(f"Error syncing memory ID {memory_id}: {e}")
            
            # Log batch completion
            logging.info(f"Completed batch {batch_idx+1}/{total_batches}. " +
                        f"Progress: {sync_stats['total_checked']}/{len(syncable_memories)} " +
                        f"({(sync_stats['total_checked']/len(syncable_memories)*100):.1f}%)")
            
            # Rate limit: pause between batches to let Ollama recover
            if batch_idx < total_batches - 1:  # Don't delay after the last batch
                time.sleep(1.0)  # 1 second pause between batches
                logging.info("Pausing 1 second between batches for Ollama stability...")
        
        # Final statistics
        logging.info(f"Synchronization complete. Statistics: {sync_stats}")
        
        return sync_stats

def main():
    parser = argparse.ArgumentParser(description='DeepSeek memory database maintenance utility')
    parser.add_argument('--check', action='store_true', help='Check database statistics')
    parser.add_argument('--sync', action='store_true', help='Perform full database synchronization')
    parser.add_argument('--health', action='store_true', help='Run health check')
    parser.add_argument('--cleanup-orphans', action='store_true', help='Clean up orphaned vector entries')
    parser.add_argument('--reset-vector-db', action='store_true', help='Reset vector DB and resync from memory DB')
    
    # Add new arguments
    parser.add_argument('--enhanced-health', action='store_true', help='Run enhanced health check with duplicate detection')
    parser.add_argument('--find-duplicates', action='store_true', help='Find duplicate entries in both databases')
    parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate entries from both databases')
    parser.add_argument('--fix-all', action='store_true', help='Fix all database issues (duplicates, sync, orphans)')
    parser.add_argument('--output', type=str, default='', help='Output file for results (JSON)')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    # Add new reminder-specific options
    parser.add_argument('--include-reminders', action='store_true', help='Include reminders in operations (not recommended)')
    parser.add_argument('--check-reminders', action='store_true', help='Check dedicated reminders table')
    parser.add_argument('--migrate-reminders', action='store_true', help='Migrate reminders from memories to dedicated table')
    
    args = parser.parse_args()
    
    # Fix the typo in the argument check and add new arguments
    if not (args.check or args.sync or args.health or args.cleanup_orphans or args.reset_vector_db or 
            args.enhanced_health or args.find_duplicates or args.remove_duplicates or args.fix_all or
            args.check_reminders or args.migrate_reminders):
        parser.print_help()
        return
    
    # Import your chatbot
    try:
        # Add the project directory to path if needed
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # First let's examine your config file
        import config
        
        if args.debug:
            print("Available configuration variables:")
            for name, value in inspect.getmembers(config):
                if not name.startswith('__'):
                    print(f"  {name} = {value}")
        
        # Now import your chatbot and required modules
        from chatbot import Chatbot
        
        # Create an instance of your chatbot 
        chatbot = Chatbot()
        
        # Access the databases directly from the chatbot
        vector_db = chatbot.vector_db
        memory_db = chatbot.memory_db
        
        # Import ReminderManager if available
        try:
            from reminders import ReminderManager
            reminder_manager = ReminderManager(memory_db.db_path)
            has_reminder_manager = True
        except ImportError:
            has_reminder_manager = False
            logging.warning("ReminderManager module not found, reminder-specific operations will be limited")
        
        # Print some debug info about the databases if requested
        if args.debug:
            print("\nVector DB Info:")
            for attr_name in dir(vector_db):
                if not attr_name.startswith('__'):
                    attr_value = getattr(vector_db, attr_name)
                    if not callable(attr_value):
                        print(f"  {attr_name} = {attr_value}")
                        
            print("\nMemory DB Info:")
            for attr_name in dir(memory_db):
                if not attr_name.startswith('__'):
                    attr_value = getattr(memory_db, attr_name)
                    if not callable(attr_value) and not isinstance(attr_value, dict):
                        print(f"  {attr_name} = {attr_value}")
        
        # Create maintenance instance
        maintenance = DatabaseMaintenance(vector_db, memory_db)
        
        results = {}
        
        # Check reminders
        if args.check_reminders and has_reminder_manager:
            logging.info("Checking dedicated reminders table...")
            start_time = time.time()
            
            # Count reminders in main memories table
            with sqlite3.connect(memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = 'reminder'")
                memory_reminders_count = cursor.fetchone()[0]
                
                # Count reminders in dedicated table
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='reminders'
                """)
                
                if cursor.fetchone():
                    cursor.execute("SELECT COUNT(*) FROM reminders")
                    dedicated_reminders_count = cursor.fetchone()[0]
                else:
                    dedicated_reminders_count = 0
            
            duration = time.time() - start_time
            results['reminders_check'] = {
                'memories_table_reminders': memory_reminders_count,
                'dedicated_table_reminders': dedicated_reminders_count,
                'duration_seconds': duration
            }
            
            # Print summary
            print(f"\nReminders Check Results:")
            print(f"Reminders in memories table: {memory_reminders_count}")
            print(f"Reminders in dedicated table: {dedicated_reminders_count}")
            print(f"Check completed in {duration:.2f} seconds\n")
        
        # Migrate reminders
        if args.migrate_reminders and has_reminder_manager:
            logging.info("Migrating reminders to dedicated table...")
            start_time = time.time()
            found, migrated = reminder_manager.migrate_from_memories_table()
            duration = time.time() - start_time
            
            results['reminders_migration'] = {
                'found': found,
                'migrated': migrated,
                'duration_seconds': duration
            }
            
            # Print summary
            print(f"\nReminders Migration Results:")
            print(f"Found {found} reminders in memories table")
            print(f"Successfully migrated {migrated} reminders to dedicated table")
            print(f"Migration completed in {duration:.2f} seconds\n")
        
        # Run enhanced health check
        if args.enhanced_health:
            logging.info("Running enhanced health check...")
            start_time = time.time()
            health_report = maintenance.enhanced_health_check()
            duration = time.time() - start_time
            results['enhanced_health'] = health_report
            results['enhanced_health']['duration_seconds'] = duration
            
            # Print summary
            print(f"\nEnhanced Health Check Results:")
            print(f"Overall health score: {health_report['health_score']['overall']}/100")
            print(f"Memory DB health: {health_report['health_score']['memory_db']}/100")
            print(f"Vector DB health: {health_report['health_score']['vector_db']}/100")
            print(f"Sync health: {health_report['health_score']['sync']}/100")
            print(f"Memory DB duplicates: {health_report['duplicates']['memory_db']['duplicates_found']}")
            print(f"Vector DB duplicates: {health_report['duplicates']['vector_db']['duplicates_found']}")
            print(f"Missing in vector DB: {health_report['sync_issues']['missing_in_vector_db']}")
            print(f"Missing in memory DB: {health_report['sync_issues']['missing_in_memory_db']}")
            print(f"Health check completed in {duration:.2f} seconds\n")
        
        # Find duplicates
        if args.find_duplicates:
            logging.info("Finding duplicate entries...")
            start_time = time.time()
            memory_dups = maintenance.find_memory_db_duplicates()
            vector_dups = maintenance.find_vector_db_duplicates()
            duration = time.time() - start_time
            results['duplicates'] = {
                'memory_db': memory_dups,
                'vector_db': vector_dups,
                'duration_seconds': duration
            }
            
            # Print summary
            print(f"\nDuplicate Detection Results:")
            print(f"Memory DB: {memory_dups['duplicates_found']} duplicates in {memory_dups['duplicate_groups']} groups")
            print(f"Vector DB: {vector_dups['duplicates_found']} duplicates in {vector_dups['duplicate_groups']} groups")
            print(f"Duplicate detection completed in {duration:.2f} seconds\n")
        
        # Remove duplicates
        if args.remove_duplicates:
            logging.info("Removing duplicate entries...")
            start_time = time.time()
            memory_removal = maintenance.remove_memory_db_duplicates()
            vector_removal = maintenance.remove_vector_db_duplicates()
            duration = time.time() - start_time
            results['duplicate_removal'] = {
                'memory_db': memory_removal,
                'vector_db': vector_removal,
                'duration_seconds': duration
            }
            
            # Print summary
            print(f"\nDuplicate Removal Results:")
            print(f"Memory DB: {memory_removal['duplicates_removed']} duplicates removed")
            print(f"Vector DB: {vector_removal['duplicates_removed']} duplicates removed")
            print(f"Duplicate removal completed in {duration:.2f} seconds\n")
        
        # Fix all issues
        if args.fix_all:
            logging.info("Fixing all database issues...")
            start_time = time.time()
            fix_results = maintenance.fix_all_database_issues()
            duration = time.time() - start_time
            fix_results['duration_seconds'] = duration
            results['fix_all'] = fix_results
            
            # Print summary
            print(f"\nDatabase Repair Results:")
            print(f"Initial health score: {fix_results['initial_health']['health_score']['overall']}/100")
            print(f"Final health score: {fix_results['final_health']['health_score']['overall']}/100")
            print(f"Improvement: {fix_results['improvement']:.1f} points")
            print(f"Memory duplicates removed: {fix_results['steps']['memory_duplicates']['duplicates_removed']}")
            print(f"Vector duplicates removed: {fix_results['steps']['vector_duplicates']['duplicates_removed']}")
            print(f"Entries synchronized: {fix_results['steps']['memory_to_vector_sync']['sync_successes']}")
            print(f"Orphans cleaned: {fix_results['steps']['cleanup_orphans']['cleanup_success']}")
            print(f"Reminders skipped: {fix_results['steps']['cleanup_orphans'].get('reminders_skipped', 0)}")
            print(f"Database repair completed in {duration:.2f} seconds\n")
        
        # Run vector DB reset and resync
        if args.reset_vector_db:
            logging.info("Starting vector DB reset and resync...")
            start_time = time.time()
            reset_result = maintenance.reset_and_resync_vector_db()
            duration = time.time() - start_time
            
            if reset_result and isinstance(reset_result, dict):
                reset_result['duration_seconds'] = duration
                results['reset_vector_db'] = reset_result
                
                # Print summary
                print(f"\nVector DB Reset and Resync Results:")
                print(f"Total memories synced: {reset_result.get('total_checked', 0)}")
                print(f"Successfully synced: {reset_result.get('sync_successes', 0)}")
                print(f"Failed to sync: {reset_result.get('sync_failures', 0)}")
                print(f"Reminders skipped: {reset_result.get('reminders_skipped', 0)}")
                print(f"Reset and resync completed in {duration:.2f} seconds\n")
            else:
                print(f"\nError during vector DB reset and resync")
        
        # Run database statistics check
        if args.check:
            logging.info("Checking database statistics...")
            start_time = time.time()
            stats = maintenance.check_database_stats(include_reminders=args.include_reminders)
            duration = time.time() - start_time
            results['stats'] = stats
            results['stats']['duration_seconds'] = duration
            
            # Print summary
            print(f"\nDatabase Statistics:")
            print(f"Memory DB entries (total): {stats['memory_db_total']}")
            print(f"Memory DB entries (non-reminders): {stats.get('memory_db_non_reminders', 'unknown')}")
            print(f"Memory DB entries (reminders): {stats.get('memory_db_reminders', 'unknown')}")
            if 'dedicated_reminders_count' in stats:
                print(f"Dedicated reminders table entries: {stats['dedicated_reminders_count']}")
            print(f"Vector DB entries: {stats['vector_db_total']}")
            print(f"Discrepancy: {stats['discrepancy']} entries ({stats['discrepancy_percentage']:.1f}%)")
            print(f"Check completed in {duration:.2f} seconds\n")
        
        # Run synchronization
        if args.sync:
            logging.info("Starting database synchronization...")
            start_time = time.time()
            sync_stats = maintenance.full_database_synchronization()
            duration = time.time() - start_time
            if isinstance(sync_stats, dict):
                sync_stats['duration_seconds'] = duration
            results['synchronization'] = sync_stats
            
            # Print summary
            if 'error' in sync_stats:
                print(f"\nSynchronization Error: {sync_stats['error']}")
            else:
                print(f"\nSynchronization Results:")
                print(f"Total memories checked: {sync_stats['total_checked']}")
                print(f"Missing in vector DB: {sync_stats['missing_in_vector']}")
                print(f"Successfully synced: {sync_stats['sync_successes']}")
                print(f"Failed to sync: {sync_stats['sync_failures']}")
                print(f"Reminders skipped: {sync_stats.get('reminders_skipped', 0)}")
                print(f"Synchronization completed in {duration:.2f} seconds\n")
            
            # Run a final stats check after sync
            if args.check:
                logging.info("Checking final database statistics after sync...")
                final_stats = maintenance.check_database_stats(include_reminders=args.include_reminders)
                results['final_stats'] = final_stats
                
                print(f"Final Database Statistics:")
                print(f"Memory DB entries (total): {final_stats['memory_db_total']}")
                print(f"Memory DB entries (non-reminders): {final_stats.get('memory_db_non_reminders', 'unknown')}")
                print(f"Vector DB entries: {final_stats['vector_db_total']}")
                print(f"Remaining discrepancy: {final_stats['discrepancy']} entries ({final_stats['discrepancy_percentage']:.1f}%)")
        
        # Clean up orphaned vectors
        if args.cleanup_orphans:
            logging.info("Starting orphaned vector cleanup...")
            start_time = time.time()
            orphan_stats = maintenance.cleanup_orphaned_vectors()
            duration = time.time() - start_time
            if isinstance(orphan_stats, dict):
                orphan_stats['duration_seconds'] = duration
            results['orphan_cleanup'] = orphan_stats
            
            # Print summary
            print(f"\nOrphaned Vector Cleanup Results:")
            print(f"Total vectors checked: {orphan_stats['total_checked']}")
            print(f"Orphans found: {orphan_stats['orphans_found']}")
            print(f"Successfully cleaned: {orphan_stats['cleanup_success']}")
            print(f"Failed to clean: {orphan_stats['cleanup_failed']}")
            print(f"Reminders skipped: {orphan_stats.get('reminders_skipped', 0)}")
            print(f"Cleanup completed in {duration:.2f} seconds\n")
            
            # Run a final stats check after orphan cleanup
            if args.check:
                logging.info("Checking final database statistics after orphan cleanup...")
                final_stats = maintenance.check_database_stats(include_reminders=args.include_reminders)
                results['final_stats_after_cleanup'] = final_stats
                
                print(f"Final Database Statistics After Cleanup:")
                print(f"Memory DB entries (total): {final_stats['memory_db_total']}")
                print(f"Memory DB entries (non-reminders): {final_stats.get('memory_db_non_reminders', 'unknown')}")
                print(f"Vector DB entries: {final_stats['vector_db_total']}")
                print(f"Remaining discrepancy: {final_stats['discrepancy']} entries ({final_stats['discrepancy_percentage']:.1f}%)")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        
    except Exception as e:
        logging.error(f"Error during maintenance: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())