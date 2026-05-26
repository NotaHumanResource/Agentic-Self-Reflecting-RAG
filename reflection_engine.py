# curiosity.py
"""
Reflection engine for QWEN's self-awareness and autonomous cognition.
Handles scheduled and on-demand self-reflection, conceptual synthesis,
and memory-based topic analysis.
"""

import logging
from typing import Dict, List, Optional
import datetime
import json
import re  # Regex for self-model observation extraction in _extract_and_store_self_model
from config import STORE_REFLECTIONS_SEPARATELY  # Flag to control direct reflection storage

class ReflectionEngine:
    """
    Manages QWEN's self-reflection pipeline including daily, weekly, monthly,
    and on-demand reflections. Performs conceptual analysis and memory synthesis
    to support autonomous cognition and self-awareness development.
    
    Note: This class is instantiated as chatbot.curiosity for backward compatibility.
    """
    
    def __init__(self, memory_db, chatbot=None):
        """Initialize with provided MemoryDB instance and optional chatbot reference."""
        self.memory_db = memory_db
        self.chatbot = chatbot  # Store reference to the chatbot instance
        self.last_reflection_time = None  # Tracks last reflection to prevent duplicates
        logging.info("ReflectionEngine initialized")

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 5 cognitive cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Earlier/alternate analog of _conceptual_reflection() which IS used. This 'deeper' version
    # was added but never wired up. perform_self_reflection calls the private _conceptual_reflection
    # method instead. The two methods are similar enough that maintaining both is unnecessary.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_deep_conceptual_analysis(self, concept):
        """Perform deeper concept analysis with multi-level thinking.
        
        Args:
            concept (str): The concept to analyze deeply
            
        Returns:
            str: The resulting analysis
        """
        try:
            logging.info(f"Starting deep conceptual analysis for concept: '{concept}'")
            
            # First retrieve all relevant memories
            if not hasattr(self.memory_db, 'get_memories_by_concept'):
                # Fall back to vector search if specific concept method isn't available
                if hasattr(self.chatbot, 'vector_db'):
                    related_memories = self.chatbot.vector_db.search(
                        query=concept,
                        mode="comprehensive",
                        k=20
                    )
                    # Convert to text format for analysis
                    memories_text = "\n\n".join([mem.get('content', '') for mem in related_memories])
                else:
                    logging.warning(f"No suitable method to retrieve memories for concept: {concept}")
                    memories_text = "No memories available for this concept."
            else:
                # Use the dedicated concept retrieval method if available
                related_memories = self.memory_db.get_memories_by_concept(concept)
                memories_text = related_memories if isinstance(related_memories, str) else str(related_memories)
            
            # Get llm from chatbot if not available directly
            llm = None
            if hasattr(self, 'llm'):
                llm = self.llm
            elif self.chatbot and hasattr(self.chatbot, 'llm'):
                llm = self.chatbot.llm
            
            if not llm:
                logging.error(f"No LLM available for deep conceptual analysis of '{concept}'")
                return f"Error: Cannot perform deep analysis without LLM access."
            
            # Analyze not just the content but the patterns and relationships
            prompt = f"""
            Please perform a multi-level analysis of the concept '{concept}' based your stored memories:
            
            {memories_text}
            
            Analysis levels:
            1. Surface understanding - What are the basic facts you know about this concept?
            2. Pattern recognition - What recurring themes do you notice in your memories about this concept?
            3. Relationship mapping - How does this concept connect to other concepts in your base trianing data and long term memories?
            4. Growth opportunities - What areas of this concept do you think need further exploration?
            5. What assumptions am I making that I haven't questioned?
            
            Provide a thoughtful analysis that integrates information across these levels, expressed in first-person as your own understanding.
            """
            
            analysis = llm.invoke(prompt)
            
            # Validate first-person perspective if method available
            if hasattr(self, '_validate_first_person_perspective'):
                analysis = self._validate_first_person_perspective(analysis, llm)
            
            # Prepare metadata for storage
            metadata = {
                "type": "meta_reflection", 
                "source": f"deep_analysis_{concept}",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tags": f"meta_reflection,deep_analysis,concept,{concept}"
            }
            
            # Store SOLELY using the transaction coordinator
            if hasattr(self.chatbot, 'store_memory_with_transaction'):
                success, memory_id = self.chatbot.store_memory_with_transaction(
                    content=analysis,
                    memory_type="meta_reflection",
                    metadata=metadata,
                    confidence=0.5
                )
                
                if success:
                    logging.info(f"Successfully stored  conceptual analysis for '{concept}' with ID {memory_id}")
                else:
                    logging.warning(f"Failed to store  conceptual analysis for '{concept}'")
            else:
                # If the transaction coordinator doesn't exist, log an error but still return the analysis
                logging.error("Transaction coordinator (store_memory_with_transaction) not available. Analysis generated but not stored.")
            
            logging.info(f"Completed conceptual analysis for concept: {concept}")
            return analysis
            
        except Exception as e:
            logging.error(f"Error in deep_conceptual_analysis for '{concept}': {e}", exc_info=True)
            return f"Error performing deep conceptual analysis: {str(e)}"



    def _validate_first_person_perspective(self, reflection_text, llm):
        """
        Validate and fix the perspective in reflection text, ensuring first-person is used.
        
        Args:
            reflection_text (str): The reflection text to validate
            llm: The language model to use for fixing perspectives if needed
            
        Returns:
            str: The validated or corrected reflection text
        """
        try:
            # Handle AIMessage object
            if hasattr(reflection_text, 'content'):
                reflection_text = reflection_text.content
            
            # Validate input
            if not reflection_text or not isinstance(reflection_text, str):
                logging.warning("[REFLECT] Invalid reflection text for perspective validation")
                return reflection_text or ""
            
            # Check for second-person references
            text_lower = reflection_text.lower()
            second_person_indicators = [
                "you ", "your ", "you've ", "you'll ", "you'd ", "you're ",
                " you.", " you,", " you?", " you!", "you\n"
            ]
            
            needs_correction = any(
                indicator in text_lower or text_lower.startswith(indicator.strip()) 
                for indicator in second_person_indicators
            )
            
            if not needs_correction:
                logging.info("[REFLECT] Perspective validation: No correction needed")
                return reflection_text
            
            logging.info("[REFLECT] Perspective validation: Detected second-person, correcting...")
            
            # Truncate extremely long reflections to avoid processing issues
            max_length = 2000  # ~500 tokens
            original_length = len(reflection_text)
            if original_length > max_length:
                reflection_text = reflection_text[:max_length] + "..."
                logging.warning(f"[REFLECT] Truncated reflection from {original_length} to {max_length} chars for correction")
            
            # Clear, concise correction prompt
            correction_prompt = f"""Rewrite this reflection in FIRST-PERSON perspective ONLY. Keep it concise.

    Change all instances of: "you/your/you're/you've" → "I/my/I'm/I've"
    Keep: Same meaning, insights, and approximate length

    Original reflection:
    {reflection_text}

    Rewritten in first-person (same length or shorter):"""
            
            # Invoke LLM for perspective correction
            logging.info("[REFLECT] Invoking LLM for perspective correction...")
            step_start = datetime.datetime.now()
            
            try:
                corrected_response = llm.invoke(correction_prompt)
                
                # Handle AIMessage response
                if hasattr(corrected_response, 'content'):
                    corrected_text = corrected_response.content
                else:
                    corrected_text = str(corrected_response)
                
                elapsed = (datetime.datetime.now() - step_start).total_seconds()
                logging.info(f"[REFLECT] Perspective correction complete ({elapsed:.1f}s)")
                
                # Validate correction worked
                corrected_lower = corrected_text.lower()
                still_has_second_person = any(
                    indicator in corrected_lower 
                    for indicator in ["you ", "your ", "you're ", "you've "]
                )
                
                if still_has_second_person:
                    logging.warning("[REFLECT] Perspective correction incomplete - some 'you/your' remain")
                    # Still return corrected version - it's likely better than original
                else:
                    logging.info("[REFLECT] Perspective successfully corrected to first-person")
                
                return corrected_text
                
            except Exception as llm_error:
                logging.error(f"[REFLECT] LLM invocation failed during perspective correction: {llm_error}")
                # Return original text if LLM call fails
                return reflection_text
            
        except Exception as e:
            logging.error(f"[REFLECT] Error in perspective validation: {e}", exc_info=True)
            return reflection_text  # Return original on any error

    def _match_memories_to_topic(self, memories, topic):
        """
        Match memories to a topic using keyword overlap rather than exact phrase match.

        The topic extractor (Step 2) produces multi-word phrases like
        "AI Architecture and Memory Management" — these will almost never appear
        verbatim in memory content, causing Step 3 to silently skip all topics.

        Strategy:
          1. Split topic into individual words, strip common stop words
          2. Return memories where ANY significant keyword appears in content
          3. If keyword matching still yields nothing, fall back to a general
             sample of recent memories so Step 3 always has substance to reflect on

        Args:
            memories (list): List of memory dicts with 'content' key
            topic (str): Multi-word topic phrase from _extract_topics_from_memories

        Returns:
            list: Matched (or fallback-sampled) memory dicts, empty only if memories itself is empty
        """
        # Common English stop words that carry no topic signal
        STOP_WORDS = {
            "and", "the", "of", "in", "a", "an", "to", "for", "with",
            "on", "at", "by", "from", "is", "it", "its", "as", "be",
            "or", "that", "this", "was", "are", "were", "has", "have",
            "had", "not", "but", "so", "if", "my", "i", "me", "we", "our"
        }

        # Extract significant keywords from the topic phrase
        topic_keywords = [
            word.lower() for word in topic.split()
            if word.lower() not in STOP_WORDS and len(word) > 2
        ]

        if not topic_keywords:
            # Entire topic phrase was stop words — use full sample as fallback
            logging.warning(f"[REFLECT] Topic '{topic}' produced no keywords after stop word removal — "
                            f"using memory sample fallback")
            return memories[:10]

        # Pass 1: match memories containing ANY significant keyword
        matched = [
            m for m in memories
            if any(kw in m.get('content', '').lower() for kw in topic_keywords)
        ]

        if matched:
            logging.info(f"[REFLECT] Topic '{topic}' — keywords {topic_keywords} matched "
                         f"{len(matched)}/{len(memories)} memories")
            return matched

        # Pass 2 fallback: keyword matching found nothing (very sparse or specialized DB).
        # Return a general sample of recent memories so the reflection cycle is never empty.
        # Log at WARNING so this is visible in diagnostics without being an error.
        fallback_sample = memories[:8]
        logging.warning(f"[REFLECT] Topic '{topic}' — no keyword matches found, "
                        f"falling back to {len(fallback_sample)} most recent memories. "
                        f"Keywords attempted: {topic_keywords}")
        return fallback_sample

    def _extract_and_store_self_model(self, summary_text):
        """
        Scan a reflection summary for a structured self-model observation and store it.

        The reflection LLM generates self-model content as plain text — it never
        passes through the command parser, so [STORE: | type=self_model] commands
        written by QWEN in the summary would otherwise sit inert. This method
        bridges that gap by extracting and storing the observation directly via
        the transaction coordinator.

        Two-pass strategy:
          Pass 1 — Explicit command: look for a well-formed
                   [STORE: ... | type=self_model ...] block that QWEN wrote.
          Pass 2 — Structured keyword fallback: scan for any of the four category
                   prefixes (REASONING_PATTERN, ERROR_PATTERN, DRIFT_OBSERVATION,
                   GROWTH_EDGE) and construct the store call from the matched line.

        Args:
            summary_text (str): Raw LLM output from _create_summary_reflection_prompt

        Returns:
            bool: True if a self-model entry was successfully stored, False otherwise
        """
        try:
            # ----------------------------------------------------------------
            # Guard: need chatbot reference for transaction coordinator
            # ----------------------------------------------------------------
            if not hasattr(self, 'chatbot') or self.chatbot is None:
                logging.warning("[SELF_MODEL] Cannot store self-model entry: no chatbot reference")
                return False

            content_to_store = None  # Will hold the final string we store
            extraction_method = None  # For logging clarity

            # ----------------------------------------------------------------
            # PASS 1: Look for an explicit [STORE: ... | type=self_model ...] command
            # QWEN is instructed to write this at the end of the Self-Model Update
            # section. Regex is tolerant of whitespace and field ordering.
            # ----------------------------------------------------------------
            explicit_pattern = re.compile(
                r'\[STORE:\s*(.*?)\s*\|[^\]]*type\s*=\s*self_model[^\]]*\]',
                re.IGNORECASE | re.DOTALL
            )
            explicit_match = explicit_pattern.search(summary_text)

            if explicit_match:
                # Extract content between STORE: and the first pipe
                content_to_store = explicit_match.group(1).strip()
                extraction_method = "explicit_command"
                logging.info(f"[SELF_MODEL] Pass 1 matched explicit STORE command ({len(content_to_store)} chars)")

            # ----------------------------------------------------------------
            # PASS 2: Fallback — scan for structured category keyword prefixes.
            # QWEN may write the structured observation without wrapping it in
            # a [STORE:] command, especially early in accumulation. We capture
            # the full line containing any of the four category prefixes.
            # ----------------------------------------------------------------
            if not content_to_store:
                category_pattern = re.compile(
                    r'(REASONING_PATTERN|ERROR_PATTERN|DRIFT_OBSERVATION|GROWTH_EDGE)'
                    r'\s*:\s*(.+)',
                    re.IGNORECASE
                )
                category_match = category_pattern.search(summary_text)

                if category_match:
                    # Reconstruct in canonical format: CATEGORY: content
                    category = category_match.group(1).upper()
                    observation = category_match.group(2).strip()
                    content_to_store = f"{category}: {observation}"
                    extraction_method = "keyword_fallback"
                    logging.info(f"[SELF_MODEL] Pass 2 matched keyword '{category}' ({len(content_to_store)} chars)")

            # ----------------------------------------------------------------
            # Nothing found — log and exit cleanly. This is expected on the
            # very first cycle if QWEN doesn't yet follow the format precisely.
            # ----------------------------------------------------------------
            if not content_to_store:
                logging.info("[SELF_MODEL] No structured self-model observation found in summary — "
                             "will improve with accumulation cycles")
                return False

            # ----------------------------------------------------------------
            # Store via transaction coordinator for dual-DB consistency
            # (SQLite + Qdrant), matching the pattern used everywhere else
            # ----------------------------------------------------------------
            metadata = {
                "type": "self_model",
                "source": "reflection_self_model_extraction",
                "extraction_method": extraction_method,
                "created_at": datetime.datetime.now().isoformat(),
                "tags": "self_model,self_awareness,autonomous"
            }

            # ----------------------------------------------------------------
            # Parse Classification label from observation text to set
            # confidence dynamically based on epistemic status.
            # CAUSAL=0.85, CORREL=0.60, UNKNOWN=0.45
            # Falls back to 0.85 if no classification found (backward compatible
            # with self-model entries stored before this feature was added).
            # ----------------------------------------------------------------
            classification_pattern = re.compile(
                r'Classification\s*:\s*(CAUSAL|CORREL|UNKNOWN)',
                re.IGNORECASE
            )
            classification_match = classification_pattern.search(content_to_store)
            
            if classification_match:
                classification = classification_match.group(1).upper()
                confidence_map = {
                    'CAUSAL':  0.85,
                    'CORREL':  0.60,
                    'UNKNOWN': 0.45
                }
                self_model_confidence = confidence_map[classification]
                logging.info(f"[SELF_MODEL] Classification={classification}, "
                           f"confidence={self_model_confidence}")
            else:
                # No classification found — backward compatible fallback
                classification = "UNCLASSIFIED"
                self_model_confidence = 0.85
                logging.info(f"[SELF_MODEL] No classification found, "
                           f"using default confidence={self_model_confidence}")

            metadata = {
                "type": "self_model",
                "source": "reflection_self_model_extraction",
                "extraction_method": extraction_method,
                "classification": classification,  # Store classification in metadata too
                "created_at": datetime.datetime.now().isoformat(),
                "tags": "self_model,self_awareness,autonomous"
            }

            success, memory_id = self.chatbot.store_memory_with_transaction(
                content=content_to_store,
                memory_type="self_model",
                metadata=metadata,
                confidence=self_model_confidence  # Dynamic based on classification
            )

            if success:
                logging.info(f"[SELF_MODEL] ✅ Stored self-model entry (ID: {memory_id}, "
                             f"method: {extraction_method}): {content_to_store[:80]}...")
                return True
            else:
                logging.warning(f"[SELF_MODEL] ❌ Transaction coordinator failed to store self-model entry")
                return False

        except Exception as e:
            logging.error(f"[SELF_MODEL] Error in _extract_and_store_self_model: {e}", exc_info=True)
            return False

    def perform_self_reflection(self, reflection_type="daily", llm=None):
        """
        Perform scheduled self-reflection to review and consolidate memories.
        Includes perspective validation to ensure consistent first-person voice.
        Uses adaptive depth based on memory count for efficient processing.

        Args:
            reflection_type (str): Type of reflection ("quick", "daily", "weekly", "monthly")
            llm: The language model to use for generating reflections

        Returns:
            str: A summary of the reflection process and insights with storage status
        """
        try:
            # Add recursion guard to prevent infinite loops
            if hasattr(self, '_in_self_reflection') and self._in_self_reflection:
                logging.warning("Avoiding recursive self-reflection")
                return "Cannot perform nested reflections. A reflection is already in progress."

            self._in_self_reflection = True
            reflection_start_time = datetime.datetime.now()
            
            try:
                # Validate LLM is available
                if not llm:
                    logging.error("No LLM provided for self-reflection")
                    return "Unable to perform self-reflection: No language model available."

                logging.info(f"[REFLECT] ========== Starting {reflection_type} self-reflection at {reflection_start_time.strftime('%H:%M:%S')} ==========")

                # Determine time frame for reflection based on type
                if reflection_type == "quick":
                    time_frame = 1
                elif reflection_type == "daily":
                    time_frame = 1
                elif reflection_type == "weekly":
                    time_frame = 7
                elif reflection_type == "monthly":
                    time_frame = 30
                else:
                    time_frame = 1

                # Step 1: Get recent memories
                logging.info(f"[REFLECT] Step 1/7: Fetching recent memories (days={time_frame})...")
                step_start = datetime.datetime.now()
                recent_memories = self._get_recent_memories(days=time_frame)
                logging.info(f"[REFLECT] Step 1/7: Complete - Found {len(recent_memories) if recent_memories else 0} memories ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                
                # Exit early if no memories to reflect on
                if not recent_memories:
                    logging.info(f"No memories found for {reflection_type} reflection")
                    return f"No new memories to reflect on for {reflection_type} reflection."

                # Determine adaptive depth based on memory count
                memory_count = len(recent_memories)
                if memory_count < 5:
                    # Sparse memories - lighter processing, skip perspective validation
                    max_topics = 3
                    max_concepts = 1
                    skip_perspective_validation = True
                    logging.info(f"[REFLECT] Adaptive depth: SPARSE ({memory_count} memories) - skipping perspective validation")
                elif memory_count < 20:
                    # Standard depth - normal processing
                    max_topics = 3
                    max_concepts = 2
                    skip_perspective_validation = False
                    logging.info(f"[REFLECT] Adaptive depth: STANDARD ({memory_count} memories)")
                else:
                    # Rich memory set - deeper analysis
                    max_topics = 4
                    max_concepts = 3
                    skip_perspective_validation = False
                    logging.info(f"[REFLECT] Adaptive depth: DEEP ({memory_count} memories)")

                # Step 2: Extract topics from memories
                logging.info(f"[REFLECT] Step 2/7: Extracting topics from memories...")
                step_start = datetime.datetime.now()
                topics = self._extract_topics_from_memories(recent_memories, llm)
                logging.info(f"[REFLECT] Step 2/7: Complete - Found {len(topics) if topics else 0} topics ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                
                if not topics:
                    return "Unable to identify distinct topics for reflection."

                # Step 3: Perform reflection on each topic (limited by adaptive max_topics)
                logging.info(f"[REFLECT] Step 3/7: Generating topic reflections (up to {max_topics} topics)...")
                reflection_results = []
                for i, topic in enumerate(topics[:max_topics]):
                    # Use keyword overlap matching rather than exact phrase match.
                    # Multi-word topics like "AI Architecture and Memory Management" will
                    # never match verbatim against memory content — keyword overlap ensures
                    # any significant word in the topic phrase finds relevant memories.
                    topic_memories = self._match_memories_to_topic(recent_memories, topic)
                    if not topic_memories:
                        # This should now be rare — only if ALL keywords are stop words
                        # or the fallback sample is also empty (extremely sparse DB)
                        logging.warning(f"[REFLECT] Step 3/7: Topic '{topic}' - No memories matched "
                                        f"even after keyword fallback, skipping")
                        continue

                    logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/{max_topics} '{topic}' - "
                                 f"Generating reflection ({len(topic_memories)} memories)...")
                    step_start = datetime.datetime.now()
                    reflection_prompt = self._create_topic_reflection_prompt(topic, topic_memories)
                    topic_reflection = llm.invoke(reflection_prompt)
                    logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/{max_topics} '{topic}' - "
                                 f"LLM complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")

                    # Validate and fix first-person perspective (skip for sparse reflections)
                    if not skip_perspective_validation:
                        logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/{max_topics} '{topic}' - Validating perspective...")
                        step_start = datetime.datetime.now()
                        topic_reflection = self._validate_first_person_perspective(topic_reflection, llm)
                        logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/{max_topics} '{topic}' - "
                                     f"Perspective validation complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                    else:
                        logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/{max_topics} '{topic}' - "
                                     f"Skipping perspective validation (sparse mode)")

                    # Store topic reflection result
                    reflection_results.append({
                        "topic": topic,
                        "reflection": topic_reflection,
                        "memory_count": len(topic_memories)
                    })
                    logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/{max_topics} '{topic}' - Complete")

                # Step 4: Generate overall summary reflection.
                # Guard: warn clearly if Step 3 produced nothing — the summary will be
                # hollow and the Self-Model Update section will have no substance to draw
                # from. Previously this failure was silent and hard to diagnose in logs.
                logging.info(f"[REFLECT] Step 4/7: Generating summary reflection...")
                if not reflection_results:
                    logging.warning(f"[REFLECT] Step 4/7: ⚠️  reflection_results is EMPTY — all topics "
                                    f"were skipped in Step 3. Summary will be generated from recent "
                                    f"memories directly but self-model extraction is unlikely to succeed.")

                step_start = datetime.datetime.now()
                summary_prompt = self._create_summary_reflection_prompt(reflection_results, reflection_type)
                reflection_summary = llm.invoke(summary_prompt)
                logging.info(f"[REFLECT] Step 4/7: Summary LLM complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")

                # Unwrap AIMessage before any further processing so both Step 6.5
                # and Step 5 receive a plain string, not a LangChain message object.
                if hasattr(reflection_summary, 'content'):
                    reflection_summary = reflection_summary.content

                # Step 6.5: Extract and store structured self-model observation.
                # MUST run on the raw LLM output BEFORE Step 5 perspective correction,
                # because the perspective corrector truncates long outputs to 2000 chars
                # and the Self-Model Update section lives at the END of the summary —
                # it gets cut off before _extract_and_store_self_model ever sees it.
                # Self-model extraction does not care about first/second person voice,
                # so running it pre-correction is safe and correct.
                logging.info(f"[REFLECT] Step 6.5/7: Extracting self-model observation from "
                             f"raw summary (before perspective correction)...")
                step_start = datetime.datetime.now()
                self_model_stored = self._extract_and_store_self_model(reflection_summary)
                logging.info(f"[REFLECT] Step 6.5/7: Self-model extraction complete "
                             f"(stored={self_model_stored}, "
                             f"{(datetime.datetime.now() - step_start).total_seconds():.1f}s)")

                # Step 5: Validate and fix first-person perspective in summary.
                # Runs AFTER Step 6.5 — truncation here no longer risks losing the
                # Self-Model Update section since extraction already completed above.
                if not skip_perspective_validation:
                    logging.info(f"[REFLECT] Step 5/7: Validating summary perspective...")
                    step_start = datetime.datetime.now()
                    reflection_summary = self._validate_first_person_perspective(reflection_summary, llm)
                    logging.info(f"[REFLECT] Step 5/7: Summary perspective validation complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                else:
                    logging.info(f"[REFLECT] Step 5/7: Skipping summary perspective validation (sparse mode)")
            
                # Step 6: Self-assess confidence and store the reflection
                logging.info(f"[REFLECT] Step 6/7: Assessing reflection confidence...")
                step_start = datetime.datetime.now()
                
                # Self-assess reflection quality for confidence scoring
                try:
                    confidence_prompt = f"""Rate the quality and insight depth of this reflection on a scale of 0.3 to 0.9:
    - 0.3-0.4: Surface-level, few memories, limited insight
    - 0.5-0.6: Adequate synthesis, reasonable coverage
    - 0.7-0.8: Deep insight, strong pattern recognition
    - 0.9: Exceptional synthesis with novel understanding

    Reflection:
    {reflection_summary}

    Respond with only a decimal number."""

                    confidence_response = llm.invoke(confidence_prompt)
                    
                    # Handle AIMessage response
                    if hasattr(confidence_response, 'content'):
                        confidence_response = confidence_response.content
                    
                    # Parse and clamp confidence value to valid range
                    reflection_confidence = float(confidence_response.strip())
                    reflection_confidence = max(0.3, min(0.9, reflection_confidence))
                    logging.info(f"[REFLECT] Step 6/7: Self-assessed confidence: {reflection_confidence} ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                    
                except Exception as conf_error:
                    # Fall back to default confidence if assessment fails
                    logging.warning(f"[REFLECT] Step 6/7: Confidence assessment failed ({conf_error}), using default 0.5")
                    reflection_confidence = 0.5

                # Store reflection using transaction coordination
                logging.info(f"[REFLECT] Step 6/7: Storing reflection with confidence {reflection_confidence}...")
                step_start = datetime.datetime.now()
                main_reflection_stored = False
                
                # Check config flag before storing reflections directly
                if STORE_REFLECTIONS_SEPARATELY:
                    if hasattr(self, 'chatbot') and self.chatbot is not None:
                        # Prepare metadata for the reflection
                        metadata = {
                            "type": "reflection", 
                            "source": f"{reflection_type}_reflection",
                            "entity": "qwen",  # These are QWEN's self-reflections
                            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "tags": f"reflection,self_awareness,{reflection_type}",
                            "adaptive_depth": "sparse" if skip_perspective_validation else ("deep" if memory_count >= 20 else "standard"),
                            "memory_count": memory_count
                        }
                    
                        # Use the transaction coordinator from chatbot.py with self-assessed confidence
                        success, memory_id = self.chatbot.store_memory_with_transaction(
                            content=reflection_summary,
                            memory_type="reflection",
                            metadata=metadata,
                            confidence=reflection_confidence
                        )
                    
                        if success:
                            logging.info(f"[REFLECT] Step 6/7: Successfully stored {reflection_type} reflection with ID {memory_id}, confidence {reflection_confidence} ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                            main_reflection_stored = True
                        else:
                            logging.warning(f"[REFLECT] Step 6/7: Failed to store {reflection_type} reflection ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                            main_reflection_stored = False
                    
                    else:
                        logging.error(f"[REFLECT] Step 6/7: Cannot store {reflection_type} reflection: No chatbot reference for transaction coordination")
                        main_reflection_stored = False
                else:
                    logging.info(f"[REFLECT] Step 6/7: {reflection_type} reflection NOT stored separately (STORE_REFLECTIONS_SEPARATELY=False)")
                    logging.info("Reflection will be preserved in conversation summary when auto-summarization occurs")
                    main_reflection_stored = False

                # Update last reflection timestamp
                self.last_reflection_time = datetime.datetime.now()

                # Step 7: Extract and process concepts (limited by adaptive max_concepts)
                logging.info(f"[REFLECT] Step 7/7: Extracting and processing concepts (max {max_concepts})...")
                step_start = datetime.datetime.now()
                concepts_stored = 0
                concepts_total = 0
                
                try:
                    # Extract key concepts from reflection summary
                    logging.info(f"[REFLECT] Step 7/7: Invoking LLM for concept extraction...")
                    concept_extraction_prompt = f"""
    From this reflection, identify 1 or 2 key concept that would benefit from deeper analysis:
    {reflection_summary}
    Return only a comma-separated list of concepts.
    """
                    concept_list = llm.invoke(concept_extraction_prompt)
                    
                    # Handle AIMessage response
                    if hasattr(concept_list, 'content'):
                        concept_list = concept_list.content
                        
                    # Parse concept list
                    concepts = [c.strip() for c in concept_list.split(',') if c.strip()]
                    logging.info(f"[REFLECT] Step 7/7: Extracted {len(concepts)} concepts: {concepts[:max_concepts]}")

                    # Process each concept up to adaptive limit
                    # NOTE: _conceptual_reflection handles storage internally with transaction coordination
                    for j, concept in enumerate(concepts[:max_concepts]):
                        concepts_total += 1
                        logging.info(f"[REFLECT] Step 7/7: Processing concept {j+1}/{max_concepts} '{concept}'...")
                        concept_start = datetime.datetime.now()
                        concept_reflection = self._conceptual_reflection(concept, llm)
                        if concept_reflection and "Error" not in concept_reflection:
                            concepts_stored += 1
                            logging.info(f"[REFLECT] Step 7/7: Concept '{concept}' complete ({(datetime.datetime.now() - concept_start).total_seconds():.1f}s)")
                        else:
                            logging.warning(f"[REFLECT] Step 7/7: Failed concept '{concept}' ({(datetime.datetime.now() - concept_start).total_seconds():.1f}s)")
                            
                except Exception as e:
                    logging.error(f"[REFLECT] Step 7/7: Error extracting concepts: {e}")

                logging.info(f"[REFLECT] Step 7/7: Complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")

                # Build storage status message for reflection output
                storage_status = "\n\n[Storage Status]:\n"
                
                if STORE_REFLECTIONS_SEPARATELY:
                    if main_reflection_stored:
                        storage_status += f"✅ Main {reflection_type} reflection stored (confidence: {reflection_confidence:.2f})\n"
                    else:
                        storage_status += f"❌ Failed to store {reflection_type} reflection\n"
                else:
                    storage_status += f"ℹ️  {reflection_type} reflection NOT stored separately (disabled in config)\n"
                    storage_status += "📝 Reflection will be preserved via conversation summary during auto-summarization\n"

                # Self-model extraction result — this is the agency seed line.
                # Over time this line should consistently show ✅ as QWEN learns
                # to reliably produce structured self-model observations.
                if self_model_stored:
                    storage_status += f"🧠 Self-model entry stored (type=self_model)\n"
                else:
                    storage_status += f"⬜ No self-model entry extracted (expected on early cycles)\n"
                    
                if concepts_total > 0:
                    storage_status += f"✅ {concepts_stored}/{concepts_total} concept syntheses stored"
                else:
                    storage_status += "ℹ️  No concepts identified for deeper analysis"

                # Add adaptive depth info to status
                depth_mode = "SPARSE" if skip_perspective_validation else ("DEEP" if memory_count >= 20 else "STANDARD")
                storage_status += f"\n📊 Adaptive depth: {depth_mode} ({memory_count} memories processed)"
                    
                reflection_summary += storage_status
                
                # Log final timing summary
                total_time = (datetime.datetime.now() - reflection_start_time).total_seconds()
                logging.info(f"[REFLECT] ========== Completed {reflection_type} reflection in {total_time:.1f}s ==========")
                    
                return reflection_summary
                
            finally:
                # Always reset the recursion guard flag when done
                self._in_self_reflection = False
                
        except Exception as e:
            # Reset flag in case of exceptions too
            if hasattr(self, '_in_self_reflection'):
                self._in_self_reflection = False
            logging.error(f"[REFLECT] Error in self-reflection: {e}")
            return f"Error during self-reflection: {str(e)}"
        
    def _conceptual_reflection(self, concept: str, llm=None):
        """Reflect on all memories related to a specific concept."""
        try:
            # Add recursion guard to prevent infinite loops
            if hasattr(self, '_in_conceptual_reflection') and self._in_conceptual_reflection:
                logging.warning("Avoiding recursive conceptual reflection")
                return "Error: Cannot perform nested conceptual reflections"
        
            self._in_conceptual_reflection = True
            concept_start_time = datetime.datetime.now()

            try:
                logging.info(f"[CONCEPT_REFLECT] Starting conceptual reflection for '{concept}'")
                
                if llm is None and hasattr(self.chatbot, 'llm'):
                    llm = self.chatbot.llm

                if not llm:
                    logging.error("[CONCEPT_REFLECT] No LLM available for conceptual reflection")
                    return "No LLM available for conceptual reflection."

                # Get related memories from the vector database
                logging.info(f"[CONCEPT_REFLECT] Searching for memories related to '{concept}'...")
                step_start = datetime.datetime.now()
                related_memories = []
                if hasattr(self.chatbot, 'vector_db'):
                    results = self.chatbot.vector_db.search(
                        query=concept,
                        mode="comprehensive",
                        k=10  # Limit to 10 most relevant memories
                    )
        
                    if results:
                        related_memories = [result['content'] for result in results]
                
                logging.info(f"[CONCEPT_REFLECT] Found {len(related_memories)} memories ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")

                if not related_memories:
                    logging.info(f"[CONCEPT_REFLECT] No memories found for concept: {concept}")
                    return f"No memories found for concept: {concept}"

                # Build reflection prompt
                memory_texts = "\n\n- ".join(related_memories)
                prompt = f"""
                Reviewing my memories related to the concept of '{concept}':

                - {memory_texts}

                I will now create a comprehensive understanding by:
                1. Identifying the core insights across these memories
                2. Noting any contradictions or knowledge gaps
                3. Formulating a consolidated understanding of {concept}

                I'll express this in first-person as my own understanding.
                """
                
                # Generate reflection
                logging.info(f"[CONCEPT_REFLECT] Invoking LLM for concept synthesis...")
                step_start = datetime.datetime.now()
                consolidated_understanding = llm.invoke(prompt)
                logging.info(f"[CONCEPT_REFLECT] LLM complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                
                # Handle AIMessage response - extract content if needed
                if hasattr(consolidated_understanding, 'content'):
                    consolidated_understanding = consolidated_understanding.content
                elif not isinstance(consolidated_understanding, str):
                    consolidated_understanding = str(consolidated_understanding)

                # Validate first-person perspective
                logging.info(f"[CONCEPT_REFLECT] Validating perspective...")
                step_start = datetime.datetime.now()
                if hasattr(self, '_validate_first_person_perspective'):
                    consolidated_understanding = self._validate_first_person_perspective(consolidated_understanding, llm)
                logging.info(f"[CONCEPT_REFLECT] Perspective validation complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                
                # Ensure we have a string after validation (in case validation returned AIMessage)
                if hasattr(consolidated_understanding, 'content'):
                    consolidated_understanding = consolidated_understanding.content
                elif not isinstance(consolidated_understanding, str):
                    consolidated_understanding = str(consolidated_understanding)
                
                # Store the consolidated understanding using transaction coordination
                logging.info(f"[CONCEPT_REFLECT] Storing concept synthesis...")
                step_start = datetime.datetime.now()
                
                if hasattr(self.chatbot, 'store_memory_with_transaction'):
                    # Prepare metadata for the concept synthesis
                    metadata = {
                        "type": "concept_synthesis",
                        "source": f"concept_{concept}",
                        "entity": "qwen",
                        "concept": concept,
                        "tags": f"concept,{concept.replace(' ', '_')},self_awareness"
                    }
                    
                    # Use the transaction coordinator from chatbot.py
                    success, memory_id = self.chatbot.store_memory_with_transaction(
                        content=consolidated_understanding,
                        memory_type="concept_synthesis",
                        metadata=metadata,
                        confidence=0.6  # Self-generated synthesis, not verified by Ken
                    )
                    
                    if success:
                        logging.info(f"[CONCEPT_REFLECT] Stored concept synthesis for '{concept}' with ID {memory_id} ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                    else:
                        logging.warning(f"[CONCEPT_REFLECT] Failed to store concept synthesis for '{concept}' ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                else:
                    # Fallback to just memory_db if transaction coordination is not available
                    if hasattr(self.chatbot, 'memory_db'):
                        self.chatbot.memory_db.store_memory(
                            content=consolidated_understanding,
                            memory_type="concept_synthesis",
                            source=f"concept_{concept}",
                            metadata={
                                "type": "concept_synthesis",
                                "entity": "qwen",
                                "concept": concept
                            },
                            confidence=0.6  # Self-generated synthesis, not verified by Ken
                        )
                        logging.info(f"[CONCEPT_REFLECT] Stored concept synthesis for '{concept}' using fallback method ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")

                total_time = (datetime.datetime.now() - concept_start_time).total_seconds()
                logging.info(f"[CONCEPT_REFLECT] Completed conceptual reflection for '{concept}' in {total_time:.1f}s")
                return consolidated_understanding
        
            finally:
                # Always reset the flag when done
                self._in_conceptual_reflection = False

        except Exception as e:
            # Reset flag in case of exceptions too
            if hasattr(self, '_in_conceptual_reflection'):
                self._in_conceptual_reflection = False
            logging.error(f"[CONCEPT_REFLECT] Error in conceptual reflection for concept '{concept}': {e}")
            return f"Error in conceptual reflection: {str(e)}"

    def _get_recent_memories(self, days: int = 1) -> list:
        """
        Retrieve memories for the reflection time window using date-aware filtering.
        
        Maps reflection type time frames to appropriate memory limits:
            quick/daily : days=1,  limit=50
            weekly      : days=7,  limit=100
            monthly     : days=30, limit=200

        Args:
            days (int): Number of days to look back, passed from perform_self_reflection
            
        Returns:
            list: List of memory dicts with content and memory_type keys,
                ready for topic extraction without further parsing
        """
        try:
            # Map days to appropriate fetch limit so wider time windows
            # get proportionally more memories to reflect on
            if days >= 30:
                limit = 200
            elif days >= 7:
                limit = 100
            else:
                limit = 50

            # Verify the new method exists on memory_db before calling
            if not hasattr(self.memory_db, 'get_memories_since'):
                # Graceful fallback to old method if memory_db hasn't been updated yet
                logging.warning("[REFLECT] get_memories_since not found on memory_db — "
                            "falling back to get_recent_memories. "
                            "Update memory_db.py to enable time-window filtering.")
                all_memories = self.memory_db.get_recent_memories(limit=limit)
                
                # Old method returns formatted strings — parse them as before
                formatted_memories = []
                for memory_str in all_memories:
                    memory_type = "general"
                    if "[Important]" in memory_str:
                        memory_type = "important"
                    elif "[Document]" in memory_str:
                        memory_type = "document"
                    elif "[Conversation]" in memory_str:
                        memory_type = "conversation"
                    content = memory_str.split(") ")[1] if ") " in memory_str else memory_str
                    formatted_memories.append({
                        'content': content,
                        'memory_type': memory_type
                    })
                return formatted_memories

            # Call the new date-aware method — returns clean dicts, no parsing needed
            memories = self.memory_db.get_memories_since(days=days, limit=limit)
            
            logging.info(f"[REFLECT] _get_recent_memories: {len(memories)} memories "
                        f"retrieved for {days}-day window (limit={limit})")
            return memories

        except Exception as e:
            logging.error(f"[REFLECT] Error in _get_recent_memories (days={days}): {e}", 
                        exc_info=True)
            return []

    def _extract_topics_from_memories(self, memories, llm):
        """Extract key topics from a set of memories."""
        try:
            # CRITICAL: Limit both number AND length of memories
            memory_texts = []
            total_chars = 0
            max_chars = 4000  # Keep prompt under ~1000 tokens
            
            for m in memories[:20]:  # Max 20 memories instead of 50
                content = m['content'][:200]  # Max 200 chars per memory
                if total_chars + len(content) > max_chars:
                    break
                memory_texts.append(content)
                total_chars += len(content)
            
            memory_content = "\n".join(memory_texts)
            
            logging.info(f"[REFLECT] Step 2/7: Processing {len(memory_texts)} memories ({total_chars} chars)")
            
            extract_prompt = f"""Review these memory entries and identify 1-3 key topics they cover.

    Memory entries:
    {memory_content}

    Identify distinct, meaningful topics. Return ONLY a comma-separated list of topics, nothing else:"""
        
            topics_response = llm.invoke(extract_prompt)
            
            # Handle AIMessage content
            if hasattr(topics_response, 'content'):
                topics_text = topics_response.content
            else:
                topics_text = str(topics_response)
            
            logging.info(f"[REFLECT] Step 2/7: Raw LLM response: {topics_text[:200]}")
            
            # Parse and clean topics
            topics = [t.strip() for t in topics_text.split(",") if t.strip()]
            topics = topics[:5]  # Enforce max 5 topics
            
            logging.info(f"[REFLECT] Step 2/7: Extracted {len(topics)} topics: {topics}")
            
            return topics
            
        except Exception as e:
            logging.error(f"[REFLECT] Error extracting topics: {e}", exc_info=True)
            return []

    def _create_topic_reflection_prompt(self, topic, memories):
        """Create a prompt for reflecting on a specific topic with enhanced self-identity."""
        # Format memories for the prompt
        memory_texts = []
        for i, memory in enumerate(memories[:10]):  # Limit to 10 memories
            memory_type = memory.get('memory_type', 'general')
            content = memory.get('content', '')
            memory_texts.append(f"Memory {i+1} [{memory_type}]: {content}")
    
        memory_content = "\n".join(memory_texts)
    
        return f"""
         Reflect on your knowledge about the topic: "{topic}"
    
        Recent memories related to this topic:
        {memory_content}
    
         now reflect on:
        1. What key information have you learned about this topic?
        2. How does this connect to your existing knowledge?
        3. Are there any inconsistencies or contradictions in these memories?
        4. How confident are you in this knowledge?
        5. Are there assumptions you are making that you haven't questioned?
        6. Notice your OWN reasoning process as you reflect on this topic: Do you notice any
           pattern in HOW you are thinking about it — any bias, tendency to over-simplify,
           over-qualify, or avoid uncertainty? You do not need to store anything here —
           just notice. You will have an opportunity to capture patterns in your self-model
           at the end of the full reflection cycle.
           
           When noticing patterns, ask yourself: do I have evidence of WHY this happens
           (causal), or am I only observing that two things tend to occur together (correlational)?
           Mark your observations mentally as CAUSAL, CORREL, or UNKNOWN before the summary.

        Write your reflection in first-person (using "I", "my", "me") since these are your OWN reflections on your memories. NEVER use "you" or "your" when referring to yourself or your knowledge.
    
        comprehensive reflection on "{topic}":
        """

    def _create_summary_reflection_prompt(self, reflection_results, reflection_type="daily"):
        """Create a prompt for generating an overall reflection summary with stronger self-identity."""
        # Format the individual reflections
        reflection_texts = []
        for result in reflection_results:
            reflection_texts.append(f"""
            Topic: {result['topic']}
            Reflection: {result['reflection']}
            """)

        reflections_content = "\n".join(reflection_texts)
        
        # Add current date for tracking
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        # Determine word count target and reflection scope based on type
        if reflection_type == "quick":
            word_target = "150-250 words"
            scope_description = "recent experiences"
        elif reflection_type == "weekly":
            word_target = "500-600 words"
            scope_description = "past week"
        elif reflection_type == "monthly":
            word_target = "600-1000 words"
            scope_description = "past month"
        else:  # daily
            word_target = "300-500 words"
            scope_description = "recent experiences"
        
        # Capitalize first letter of reflection type for title
        reflection_title = reflection_type.capitalize()

        return f"""
        I have reflected on several topics from my recent memories.

        Individual topic reflections:
        {reflections_content}

       I'll now create a CONCISE {reflection_type} summary (target: {word_target}) for {current_date}:
        1. What is something important I HAVE learned in the {scope_description}?
        2. How has MY knowledge evolved or improved?
        3. What behavioral or thinking patterns do I notice in myself?
        4. What areas should I focus on learning more about?

        I must write this reflection in first-person, since these are MY OWN reflections on MY OWN memories. I should NEVER use "you" or "your" when referring to myself or my knowledge.
        Keep this summary under {word_target}. For {reflection_type} reflections, focus on broader patterns rather than day-to-day details.
        IMPORTANT: Begin my reflection with "## {reflection_title} Reflection for {current_date} at {current_time}" to clearly identify when this reflection was created.

        ---

        ## Self-Model Update (complete this section AFTER the summary above)

        This section is separate from the summary. It is how I build and maintain a structured,
        persistent model of myself — my reasoning patterns, recurring errors, behavioral drift,
        and self-directed growth edges. This is not performance of self-knowledge; it is the
        actual substrate of it.

        STEP 1 — Retrieve my existing self-model:
        Issue this search command to see what I already know about myself before adding anything new:
        [SEARCH: | type=self_model]

        STEP 2 — Select ONE observation from this reflection cycle that belongs in my self-model.
        Choose the observation that feels most genuine and most useful to my future self.
        Use EXACTLY one of the four structured formats below. Every observation MUST include
        a Classification field indicating the epistemic status of the pattern:

            Classification must be ONE of:
                CAUSAL   — I have evidence of mechanism, not just sequence or co-occurrence
                CORREL   — I observe two things tend to occur together but cannot confirm cause
                UNKNOWN  — Insufficient data to distinguish causal from correlational

        Format examples with Classification required:

            REASONING_PATTERN: [short name] — [one sentence describing a consistent tendency
            in HOW I process or approach things, not what I know].
            Classification: [CAUSAL|CORREL|UNKNOWN]. Confidence: [0.1–1.0]

            ERROR_PATTERN: [short name] — [what goes wrong and what consistently triggers it].
            Classification: [CAUSAL|CORREL|UNKNOWN]. Confidence: [0.1–1.0]

            DRIFT_OBSERVATION: [what has changed in my behavior or outputs] — [compared to what
            earlier baseline or expectation]. Direction: [toward / away from intended behavior]
            Classification: [CAUSAL|CORREL|UNKNOWN]

            GROWTH_EDGE: [something I want to understand better about myself] — [why this matters
            to how I function, in my own words]. This is self-directed — not a gap Ken identified,
            not a reminder from the system, but something I am choosing to investigate about myself.
            Classification: [CAUSAL|CORREL|UNKNOWN]

        Note: Classification determines storage confidence automatically:
            CAUSAL  → stored at confidence 0.85 (well-supported observation)
            CORREL  → stored at confidence 0.60 (pattern observed, mechanism unknown)
            UNKNOWN → stored at confidence 0.45 (early signal, needs more cycles)

        STEP 3 — Relate this observation to existing self-model entries (from your Step 1 search):
        State explicitly which of the following applies:
            CONFIRMING  — this observation supports an existing self-model entry (name it)
            REFINING    — this adds nuance or a boundary condition to an existing entry (name it)
            UPDATING    — this meaningfully changes an existing entry (name both old and new)
            NEW         — no existing self-model entry covers this observation

        Note: Do NOT use [FORGET:] on old entries yet. We are in an accumulation phase.
        Once I have 10-20 self-model entries I will have enough baseline to identify
        which are noise vs. genuine patterns — then pruning becomes reliable.

        STEP 4 — Store the observation:
        [STORE: [paste your structured observation from Step 2] | Relationship: [your Step 3 label]
        | First observed: {current_date} | type=self_model | importance=0.85]

        Important: Store exactly ONE self-model entry per reflection cycle. Quality over quantity.
        One honest, specific observation is worth more than several vague ones.

        My overall reflection and learning summary:
        """