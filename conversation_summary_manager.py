"""Consolidated conversation summarization management."""
import logging
import datetime
import sys
import re
import math
import time
import uuid
from typing import List, Dict, Tuple, Any, Optional
from config import MODEL_PARAMS
from utils import calculate_tokens

class ConversationSummaryManager:
    """Manages conversation summarization, storage, and context window management."""
    
    def __init__(self, chatbot):
        """Initialize with references to required components."""
        self.chatbot = chatbot
        self.memory_db = chatbot.memory_db
        self.vector_db = chatbot.vector_db
        self.llm = chatbot.llm
        self.summarization_threshold = 0.90  # Default: summarize at 90% context utilization
        self._in_summary_generation = False  # Guard against recursive summarization
        self.summary_counter = 0  # Track number of summaries generated in this session
        
        # Set up logging
        logging.info("ConversationSummaryManager initialized")

    def _clean_summary_content(self, content: str) -> str:
        """
        Clean summary content before storage/embedding.
        Removes <think> blocks that pollute embeddings and waste vector space.
        
        Args:
            content: Raw summary text potentially containing think tags
            
        Returns:
            str: Cleaned content with think blocks removed
        """
        if not content:
            return content
        
        # Remove <think>...</think> blocks (including empty ones)
        # Handles multiline content with DOTALL flag
        cleaned = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        
        # Also handle edge cases: unclosed tags or malformed variants
        cleaned = re.sub(r'</?think>\s*', '', cleaned)
        
        # Strip leading/trailing whitespace that may result
        cleaned = cleaned.strip()
        
        if cleaned != content:
            original_len = len(content)
            cleaned_len = len(cleaned)
            logging.info(f"SUMMARY_CLEAN: Removed think tags, {original_len} -> {cleaned_len} chars")
        
        return cleaned
        
    def _estimate_tokens(self, text: str) -> int:
        """Use unified token estimation from utils."""
        from utils import calculate_tokens
        return calculate_tokens(text)

    # SLEEPING INFRASTRUCTURE 2026-05-19 (batch 6 utility cleanup review):
    # Currently 0 callers — NOT quarantined. Part of the auto-summarization-at-token-threshold
    # chain. Checks current token usage against summarization_threshold (default 90%) and
    # signals whether summarization is needed. Uses chatbot.get_unified_token_count() as the
    # primary source (same data the UI shows).
    def check_token_usage(self, conversation: List[Dict], max_tokens: int) -> Tuple[bool, int]:
        """Check if summarization is needed using the SAME token estimation as the UI."""
        
        # CRITICAL: Skip if document processing is in progress
        try:
            import streamlit as st
            if hasattr(st, 'session_state'):
                if st.session_state.get('file_processing_in_progress', False):
                    logging.info("TOKEN CHECK: Skipping during document processing")
                    return False, 0
                if st.session_state.get('skip_conversation_reload', False):
                    logging.info("TOKEN CHECK: Skipping due to skip flag")
                    return False, 0
        except:
            pass
        
        # USE THE UNIFIED TOKEN COUNTING METHOD - same as UI
        if hasattr(self.chatbot, 'get_unified_token_count'):
            current_tokens, max_tokens_unified, percentage = self.chatbot.get_unified_token_count()
            # Use the passed max_tokens if provided, otherwise use the unified one
            if max_tokens and max_tokens != max_tokens_unified:
                percentage = (current_tokens / max_tokens) * 100
            else:
                max_tokens = max_tokens_unified
        else:
            # Fallback - should never happen now
            logging.warning("TOKEN CHECK: get_unified_token_count not available, using fallback")
            current_tokens = sum(self._estimate_tokens(msg.get('content', '')) 
                            for msg in conversation if msg.get('content'))
            percentage = (current_tokens / max_tokens) * 100
        
        needs_summary = percentage >= (self.summarization_threshold * 100)  # 90%
        
        logging.info(f"UNIFIED TOKEN CHECK - Messages: {len(conversation)}")
        logging.info(f"UNIFIED TOKEN CHECK - Tokens: {current_tokens:,}/{max_tokens:,} ({percentage:.1f}%)")
        logging.info(f"UNIFIED TOKEN CHECK - Needs summary: {needs_summary}")
        
        return needs_summary, current_tokens
            
    
    # SLEEPING INFRASTRUCTURE 2026-05-19 (batch 6 utility cleanup review):
    # Currently 0 callers — NOT quarantined. Entry point of the auto-summarization-at-token-threshold
    # subsystem. The active flow uses generate_summary() directly (see chatbot.py L3285) and the
    # caller handles storage manually via ConversationStateManager.update_summary(). This method
    # represents an alternative architecture where token usage is monitored continuously and
    # summarization auto-triggers when the threshold is reached, then resets the conversation.
    # Recommended wiring: call from chatbot's main response loop after each turn. Would also
    # need the bug in store_summary_with_metadata fixed (see note there).
    def summarize_if_needed(self, conversation: List[Dict], max_tokens: int = None) -> Tuple[bool, Optional[str], List[Dict]]:
        """
        Check and summarize conversation if needed based on token usage.
        """
        # If max_tokens wasn't provided, use the default from config
        if max_tokens is None:
            max_tokens = MODEL_PARAMS.get("num_ctx", 32768)
            
        # Check if summarization is needed
        needs_summary, token_count = self.check_token_usage(conversation, max_tokens)
        
        # Rest of the method remains the same
        if not needs_summary:
            return False, None, conversation
            
        # Generate summary
        logging.info("Token threshold exceeded - generating conversation summary")
        summary = self.generate_summary(conversation)
        
        if not summary:
            logging.error("Failed to generate conversation summary")
            return False, None, conversation
            
        # Store the summary with date and time metadata
        # This is where we need to make our changes to ensure date/time are included
        timestamp = datetime.datetime.now()
        formatted_date = timestamp.strftime("%Y-%m-%d")
        formatted_time = timestamp.strftime("%H:%M:%S")
        
        # Prepare metadata with standardized format including date and time
        metadata = {
        "type": "conversation_summary",
        "source": "auto_summarization",
        "created_at": timestamp.isoformat(),
        "summary_id": f"summary_{timestamp.strftime('%Y%m%d%H%M%S')}",
        "is_latest": True,
        "date": formatted_date,  # Add standardized field
        "time": formatted_time,  # Add standardized field
        "summary_date": formatted_date,  # Keep for backward compatibility
        "summary_time": formatted_time,  # Keep for backward compatibility
        "tags": ["conversation_summary", f"date={formatted_date}"],  # ✅ CORRECT - ARRAY
        "tracking_id": str(uuid.uuid4())  # Add unique tracking ID
    }
        
        # Call store_summary with updated metadata
        success = self.store_summary_with_metadata(summary, metadata)
        
        if not success:
            logging.error("Failed to store conversation summary")
            return False, None, conversation
            
        # Reset conversation and add summary context
        updated_conversation = self._reset_conversation(conversation, summary, formatted_date, formatted_time)
        
        return True, summary, updated_conversation
    
    # SLEEPING INFRASTRUCTURE 2026-05-19 (batch 6 utility cleanup review):
    # Currently 0 callers — NOT quarantined. Part of the auto-summarization chain. Storage path
    # with transaction coordination across SQL + Qdrant via store_memory_with_transaction.
    #
    # BUG TO FIX BEFORE REVIVING: Internally calls self._update_previous_summaries_status(...)
    # at L210 — a method that is NOT DEFINED anywhere in this file. If this method runs, the
    # AttributeError gets caught by the surrounding try/except and reported as "Error storing
    # conversation summary", making the function return False even though the summary IS already
    # stored. To revive, either:
    #   (a) implement _update_previous_summaries_status to query memory_db for previous
    #       conversation_summary entries and update their metadata is_latest=False, OR
    #   (b) remove the call entirely — is_latest is set in 2 places (deepseek head/tail) but
    #       never READ anywhere in the codebase, so it's a no-op field functionally.
    # Option (b) is the lower-risk fix.
    def store_summary_with_metadata(self, summary: str, metadata: Dict[str, Any]) -> bool:
        """Store the summary using transaction coordination with specific metadata."""
        if not summary or not summary.strip():
            logging.warning("Attempted to store empty or None summary")
            return False

        try:
            # Clean think tags before storage to improve embedding quality
            summary = self._clean_summary_content(summary)
            
            self.summary_counter += 1
            logging.info(f"SUMMARY_LIFECYCLE: Storing summary #{self.summary_counter} of session")
            logging.info(f"SUMMARY_CONTENT: {summary[:100]}...")
            
            # Ensure metadata has minimum required fields
            if "type" not in metadata:
                metadata["type"] = "conversation_summary"
                
            if "created_at" not in metadata:
                metadata["created_at"] = datetime.datetime.now().isoformat()
                
            if "summary_id" not in metadata:
                timestamp = datetime.datetime.now()
                metadata["summary_id"] = f"summary_{timestamp.strftime('%Y%m%d%H%M%S')}"
        
            # Use transaction coordinator to ensure consistency across both databases
            success, memory_id = self.chatbot.store_memory_with_transaction(
                content=summary,
                memory_type="conversation_summary",  # Standardized type
                metadata=metadata,
                confidence=0.5  # Consistent confidence value
            )
        
            if success:
                logging.info(f"Stored conversation summary with ID {memory_id} and summary_id {metadata.get('summary_id')}")
                
                # Update previous summaries to mark them as not latest
                self._update_previous_summaries_status(metadata.get('summary_id'))
                
                return True
            else:
                logging.error(f"Failed to store conversation summary")
                return False

        except Exception as e:
            logging.error(f"Error storing conversation summary: {e}", exc_info=True)
            return False
    
    # SLEEPING INFRASTRUCTURE 2026-05-19 (batch 6 utility cleanup review):
    # Currently 0 callers — NOT quarantined. NOT part of the auto-summarization chain (despite
    # being in this file). Standalone utility that formats conversation messages as
    # "Role: content" lines and reports token usage via unified token counting. Note the
    # docstring's "no artificial limits" framing — intentional architecture (Ollama handles
    # the 128K window natively). Could be useful for any module needing to flatten conversation
    # history into a single string for prompts.
    def format_conversation_context(self, conversation: List[Dict], max_tokens: int = None) -> str:
        """
        Format conversation messages - let Ollama handle the 128K context window natively.
        No artificial limits - use unified token counting only for monitoring.
        """
        try:
            formatted = []
            
            # Simple formatting - NO token limiting
            for msg in conversation:
                role = msg.get('role', '').capitalize()
                content = msg.get('content', '')
                
                if not content:
                    continue
                    
                formatted.append(f"{role}: {content.strip()}")
                    
            result = "\n".join(formatted)
            
            # Use unified token counting for logging only (no truncation)
            if hasattr(self.chatbot, 'get_unified_token_count'):
                current_tokens, max_tokens_unified, percentage = self.chatbot.get_unified_token_count()
                logging.info(f"CONTEXT_BUILD: Formatted {len(formatted)} messages, {current_tokens:,} tokens ({percentage:.1f}%) - NO ARTIFICIAL LIMITS")
            
            return result
            
        except Exception as e:
            logging.error(f"Error formatting conversation context: {e}", exc_info=True)
            return ""
    
    def generate_summary(self, conversation: List[Dict]) -> Optional[str]:
        """
        Generate a conversation summary with smart chunking and error handling.
        
        Args:
            conversation: The conversation history
            
        Returns:
            Optional[str]: Generated summary or None if failed
        """
        # Add recursion guard to prevent infinite loops
        if self._in_summary_generation:
            logging.warning("Avoiding recursive conversation summary generation")
            return None
        
        self._in_summary_generation = True
        
        try:
            # Guard against None or empty conversation
            if not conversation:
                logging.warning("Attempted to summarize None or empty conversation")
                return None
                
            # Check if conversation is too large for direct processing
            total_tokens = sum(self._estimate_tokens(msg.get('content', '')) 
                              for msg in conversation if msg.get('content'))
            max_context = MODEL_PARAMS.get("num_ctx", 1000000) * 0.8  # 80% of context
            
            # Choose appropriate summarization method based on size
            if total_tokens > max_context:
                logging.info(f"Large conversation detected ({total_tokens} tokens), using chunked summarization")
                return self._generate_chunked_summary(conversation)
            else:
                logging.info(f"Regular summarization for {total_tokens} tokens")
                return self._generate_direct_summary(conversation)
                
        except Exception as e:
            logging.error(f"Error in summary generation: {e}", exc_info=True)
            return None
        finally:
            # Always reset the flag when done
            self._in_summary_generation = False
    
    def _generate_direct_summary(self, conversation: List[Dict]) -> Optional[str]:
        """
        Generate a summary directly using the LLM with an enhanced prompt.
        
        Args:
            conversation: The conversation history
            
        Returns:
            Optional[str]: Generated summary or None if failed
        """
        try:
            # Format the conversation for summarization
            messages_text = []
            
            for msg in conversation:
                if not isinstance(msg, dict):
                    continue
                
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if not content:
                    continue
                
                if role == 'user':
                    messages_text.append(f"User: {content}")
                elif role == 'assistant':
                    messages_text.append(f"Assistant: {content.strip()}")
                # Skip 'system' messages entirely - they include old summaries

            if not messages_text:
                return None

            # Get current date and time
            now = datetime.datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%H:%M:%S")
            
            # Build conversation text
            conversation_text = '\n'.join(messages_text)
            
            # Create enhanced prompt with explicit first-person instructions
            # /no_think MUST be at the very start to suppress <think> tags
            summary_prompt = f"""/no_think
I am reviewing my conversation history to create a summary for my future self.

CONVERSATION HISTORY:
{conversation_text}

TASK: Create a concise summary of this conversation using FIRST-PERSON language.

CRITICAL INSTRUCTION: 
- Write as if YOU are the AI remembering this conversation
- Use "I", "me", "my" when referring to yourself the AI.
- Use "Ken" when referring to the human user Ken. 
- Write naturally, as if writing in your own journal

EXAMPLE FORMAT (follow this style):
"I had a conversation with Ken about [topic]. I learned that Ken prefers [preference]. 
I helped him with [task] by [method]. We discussed [concepts], and I discovered that 
[insight]. Ken mentioned [important fact] which I should remember for future conversations.
I suggested [recommendation] and explained [concept]. The conversation revealed that 
Ken is working on [project], and I should keep this context for our next discussion."

INCLUDE IN YOUR SUMMARY:
- Key facts and context about Ken and our conversation
- Command patterns that worked well 
- User preferences you've discovered
- Topics that remain open for future discussion
- Important insights or breakthroughs 
- anything you want to remember for next time

FORMATTING GUIDELINES:
1. Keep the summary under approximately one to two pages
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
                summary = self.llm.invoke(summary_prompt)
            except Exception as e:
                logging.error(f"LLM error during summarization: {e}", exc_info=True)
                # Create a fallback basic summary for critical failures
                summary = self._generate_fallback_summary(conversation)
                
            return summary.strip() if summary else None
            
        except Exception as e:
            logging.error(f"Error generating direct summary: {e}", exc_info=True)
            return self._generate_fallback_summary(conversation)
    
    def _generate_chunked_summary(self, conversation: List[Dict]) -> Optional[str]:
        """
        Generate a summary for large conversations by processing in chunks.
        
        Args:
            conversation: The full conversation history
            
        Returns:
            Optional[str]: A summary of the key points from the conversation
        """
        try:
            logging.info("Using chunked summarization for large conversation")
            
            # Configuration
            chunk_size = 10  # Initial number of messages per chunk
            
            # If conversation is too short for chunking, just return a direct summary
            if len(conversation) <= chunk_size * 2:
                return self._generate_direct_summary(conversation[-chunk_size:])
            
            # Split conversation into older and recent parts
            older_msgs = conversation[:-chunk_size]
            recent_msgs = conversation[-chunk_size:]
            
            # First, summarize the older part of the conversation
            logging.info(f"Summarizing older portion ({len(older_msgs)} messages)")
            older_summary = self._generate_direct_summary(older_msgs)
            
            if not older_summary:
                # If older summary failed, try with fewer messages
                logging.warning("Failed to summarize older messages, reducing chunk size")
                half_size = max(3, len(older_msgs) // 2)
                return self._generate_chunked_summary(conversation[:half_size] + recent_msgs)
            
            # Create a new conversation with the older summary + recent messages
            synthetic_conversation = [
                {'role': 'system', 'content': 'Previous conversation summary: ' + older_summary}
            ] + recent_msgs
            
            # Generate final summary
            logging.info("Generating final summary combining older summary and recent messages")
            final_summary = self._generate_direct_summary(synthetic_conversation)
            
            if not final_summary:
                # If final summary failed but we have the older summary, use it
                logging.warning("Failed to generate final summary, using older summary")
                return "Summary of earlier conversation: " + older_summary
                
            return final_summary
            
        except Exception as e:
            logging.error(f"Error in chunked summarization: {e}", exc_info=True)
            # Fallback to just the most recent messages if everything fails
            try:
                most_recent = conversation[-5:]  # Just the last 5 messages as ultimate fallback
                return self._generate_direct_summary(most_recent)
            except:
                return "Failed to summarize conversation due to its size."
    
    def _generate_fallback_summary(self, conversation: List[Dict]) -> str:
        """
        Generate a basic summary for when LLM summarization fails.
        
        Args:
            conversation: The conversation history
            
        Returns:
            str: A basic summary derived directly from message content
        """
        logging.info("Using fallback summary generation")
        
        # Get the last few messages
        last_messages = conversation[-5:] if len(conversation) > 5 else conversation
        basic_summary = "Conversation included: "
        
        for msg in last_messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if len(content) > 50:
                content = content[:50] + "..."
            
            if role == 'user':
                basic_summary += f"User asked about {content}. "
            elif role == 'assistant':
                basic_summary += f"Assistant discussed {content}. "
        
        logging.info(f"Generated fallback summary: {basic_summary}")
        return basic_summary
        
              
    # Preserve the full conversation history while injecting a fresh summary
    # SLEEPING INFRASTRUCTURE 2026-05-19 (batch 6 utility cleanup review):
    # Currently 0 callers — NOT quarantined. Part of the auto-summarization chain. Inserts the
    # new summary message at the top of the conversation while preserving ALL original messages.
    # Removes any prior summary system messages to avoid duplication. Place where the
    # "summary + full history" design lives.
    def _reset_conversation(self, conversation: List[Dict], summary: str, date: str = None, time: str = None) -> List[Dict]:
        """
        At token limit: Add summary and preserve ALL conversation history.
        Summary is for LLM context, not for truncation.
        """
        try:
            # Format date/time string for display if provided
            datetime_str = ""
            if date:
                datetime_str = f" ({date}"
                if time:
                    datetime_str += f" at {time}"
                datetime_str += ")"
            
            # IMPORTANT: Keep ALL original messages - no truncation
            new_conversation = []
            
            # Remove any existing old summaries to avoid duplication
            for msg in conversation:
                content = msg.get('content', '')
                # Skip old summary messages
                if (msg.get('role') == 'system' and 
                    any(phrase in content for phrase in [
                        'Previous conversation summary',
                        'Conversation summarized to save',
                        '📋 Conversation Summary',
                        'conversation summary'
                    ])):
                    logging.info("Skipping old summary message during reset")
                    continue
                
                # Keep all other messages
                new_conversation.append(msg)
            
            # Add the NEW summary at the beginning for LLM context
            summary_message = {
                'role': 'system',
                'content': f"📋 Recent Conversation Summary{datetime_str}: {summary}\n\n[This summary provides context. Full conversation history follows below.]"
            }
            
            # Insert summary at the beginning (after any system prompts)
            # Find the right position - after initial system prompts but before conversation
            insert_position = 0
            for i, msg in enumerate(new_conversation):
                if msg.get('role') == 'system' and 'system prompt' in msg.get('content', '').lower():
                    insert_position = i + 1
                elif msg.get('role') in ['user', 'assistant']:
                    break
                    
            new_conversation.insert(insert_position, summary_message)
            
            logging.info(f"Conversation reset: {len(conversation)} -> {len(new_conversation)} messages (FULL HISTORY PRESERVED with summary)")
            return new_conversation
            
        except Exception as e:
            logging.error(f"Error resetting conversation: {e}", exc_info=True)
            return conversation
            
