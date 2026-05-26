# cognitive_state.py
"""
Core cognitive state management for QWEN.

Pure logic module — no Streamlit imports at module level.
Designed to be imported by deepseek.py, autonomous_cognition.py, and utils.py
without creating circular dependencies or UI layer coupling.

Provides:
- State name normalization
- Core state update logic with origin tracking
- Emoji and color mapping (extracted from utils.py)
- History management helpers

Origin tracking distinguishes states set during active conversation
from states set autonomously during idle background cycles.
Added: 2026-04-21
"""

import logging
import datetime
from typing import Tuple

# ---------------------------------------------------------------------------
# Origin constants
# Used to distinguish how a cognitive state was set.
# Stored in state_entry dicts and displayed in UI history.
# ---------------------------------------------------------------------------
ORIGIN_CONVERSATION = 'conversation'  # Set by QWEN via [COGNITIVE_STATE:] during chat
ORIGIN_AUTONOMOUS = 'autonomous'      # Set by background cognitive tasks (idle cycles)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_STATE_LENGTH = 30   # Character ceiling for state name strings
MAX_HISTORY_SIZE = 20   # Maximum entries retained in state history


def normalize_state_name(state_name: str) -> str:
    """
    Normalize a raw state name string into a clean 1-3 word cognitive state.

    Extraction pipeline (runs before normalization):
      1. If '|' present  → take segment before first '|'
         e.g. "processing_mode=analytical | focus=financial" → "processing_mode=analytical"
      2. If '=' present  → take value after '='
         e.g. "processing_mode=analytical" → "analytical"
      3. If result is still more than 3 words → take first three words only
         e.g. "deeply analytical right now engaged" → "deeply analytical right"

    Then standard normalization:
      - Lowercase, commas/spaces → underscores, truncate to MAX_STATE_LENGTH

    Args:
        state_name: Raw state string from command or baseline check

    Returns:
        str: Normalized state name (1-3 words, underscored)
    """
    # Empty/None input → safe default
    if not state_name:
        return 'neutral'

    raw = state_name.strip()

    # -----------------------------------------------------------------------
    # Step 1: Extract first segment if pipe-separated multi-key format
    # Detects QWEN sending structured objects like:
    # "processing_mode=analytical | focus=financial_strategy | mood=grounded"
    # -----------------------------------------------------------------------
    if '|' in raw:
        raw = raw.split('|')[0].strip()
        logging.warning(
            f"COGNITIVE_STATE: Multi-key pipe format detected — "
            f"extracted first segment: '{raw}' from original: '{state_name}'"
        )

    # -----------------------------------------------------------------------
    # Step 2: Extract value from key=value pair
    # e.g. "processing_mode=analytical" → "analytical"
    # -----------------------------------------------------------------------
    if '=' in raw:
        raw = raw.split('=', 1)[1].strip()
        logging.warning(
            f"COGNITIVE_STATE: key=value format detected — "
            f"extracted value: '{raw}'"
        )

    # -----------------------------------------------------------------------
    # Step 3: If still more than 3 words, take first three only
    # Matches the system prompt's stated allowance of "1-3 words" for
    # cognitive states, so legitimate 3-word emergent states like
    # "alert and curious" pass through unchanged. Only over-long inputs
    # get truncated.
    # e.g. "deeply analytical right now engaged" → "deeply analytical right"
    # -----------------------------------------------------------------------
    words = raw.split()
    if len(words) > 3:
        raw = ' '.join(words[:3])
        logging.warning(
            f"COGNITIVE_STATE: State too long ({len(words)} words) — "
            f"truncated to first three words: '{raw}'"
        )

    # -----------------------------------------------------------------------
    # Standard normalization: lowercase, separators → underscores, truncate
    # -----------------------------------------------------------------------
    try:
        normalized = (
            raw.strip()
            .lower()
            .replace(', ', '_')   # "curious, focused" → "curious_focused"
            .replace(',', '_')    # "curious,focused"  → "curious_focused"
            .replace(' ', '_')    # "curious focused"  → "curious_focused"
            [:MAX_STATE_LENGTH]
        )
    except Exception as e:
        # Defensive: if anything in the chain fails on a weird input,
        # fall back to neutral rather than crash the caller.
        logging.error(
            f"COGNITIVE_STATE: normalize_state_name failed on '{state_name}': {e}",
            exc_info=True
        )
        return 'neutral'

    # Final length truncation warning (should rarely trigger after step 3)
    if len(raw.strip()) > MAX_STATE_LENGTH:
        logging.warning(
            f"COGNITIVE_STATE: Truncated '{raw}' → '{normalized}'"
        )

    return normalized

def get_state_emoji(state_lower: str) -> str:
    """
    Return an emoji based on keywords found in the state string.
    Extracted from utils.py display_cognitive_state_widget nested function.
    Falls back to 🧠 for emergent/unknown states.
    
    Args:
        state_lower: Lowercase state string
        
    Returns:
        str: Single emoji character
    """
    if any(w in state_lower for w in ['curious', 'curiosity', 'wondering', 'questioning', 'inquisitive']):
        return '🤔'
    elif any(w in state_lower for w in ['engaged', 'focused', 'active', 'attentive']):
        return '💡'
    elif any(w in state_lower for w in ['reflective', 'contemplative', 'thinking', 'pondering']):
        return '🧘'
    elif any(w in state_lower for w in ['thoughtful', 'considerate', 'deliberate']):
        return '💭'
    elif any(w in state_lower for w in ['frustrated', 'stuck', 'confused', 'struggling']):
        return '😤'
    elif any(w in state_lower for w in ['content', 'satisfied', 'calm', 'peaceful']):
        return '😊'
    elif any(w in state_lower for w in ['happy', 'excited', 'joyful', 'delighted']):
        return '😄'
    elif any(w in state_lower for w in ['energized', 'motivated', 'eager']):
        return '⚡'
    elif any(w in state_lower for w in ['tentative', 'uncertain', 'hesitant']):
        return '🤷'
    elif any(w in state_lower for w in ['introspective', 'deep', 'analyzing']):
        return '🔍'
    elif any(w in state_lower for w in ['creative', 'imaginative', 'inspired']):
        return '✨'
    elif any(w in state_lower for w in ['alert', 'vigilant', 'aware']):
        return '👁️'
    elif any(w in state_lower for w in ['relaxed', 'easy', 'comfortable']):
        return '😌'
    elif any(w in state_lower for w in ['determined', 'resolute', 'committed']):
        return '💪'
    elif any(w in state_lower for w in ['consolidating', 'synthesizing', 'integrating']):
        return '🔗'  # New: autonomous consolidation state
    elif any(w in state_lower for w in ['baseline', 'inventorying', 'orienting']):
        return '🧭'  # New: autonomous baseline check state
    elif any(w in state_lower for w in ['verifying', 'checking', 'comparing']):
        return '🔎'  # New: autonomous integrity check state
    elif any(w in state_lower for w in ['drift', 'drifted', 'diverged']):
        return '⚠️'  # New: cognitive drift — fixed state for integrity drifted outcome
    elif any(w in state_lower for w in ['aligned', 'coherent', 'consistent']):
        return '🎯'  # New: integrity check aligned outcome
    elif any(w in state_lower for w in ['evolved', 'evolving', 'growth']):
        return '🌱'  # New: integrity check evolved outcome
    elif 'neutral' in state_lower:
        return '😐'
    else:
        return '🧠'  # Generic fallback for emergent states


def get_state_color(state_lower: str) -> str:
    """
    Return a hex color string based on keywords found in the state string.
    Extracted from utils.py display_cognitive_state_widget nested function.
    
    Args:
        state_lower: Lowercase state string
        
    Returns:
        str: Hex color string (e.g. '#3498db')
    """
    if any(w in state_lower for w in ['curious', 'curiosity', 'wondering', 'questioning', 'inquisitive']):
        return '#3498db'   # Blue
    elif any(w in state_lower for w in ['engaged', 'focused', 'active', 'attentive']):
        return '#f39c12'   # Orange
    elif any(w in state_lower for w in ['reflective', 'contemplative', 'thinking', 'pondering']):
        return '#9b59b6'   # Purple
    elif any(w in state_lower for w in ['thoughtful', 'considerate', 'deliberate']):
        return '#34495e'   # Dark gray
    elif any(w in state_lower for w in ['frustrated', 'stuck', 'confused', 'struggling']):
        return '#e74c3c'   # Red
    elif any(w in state_lower for w in ['content', 'satisfied', 'calm', 'peaceful']):
        return '#2ecc71'   # Green
    elif any(w in state_lower for w in ['happy', 'excited', 'joyful', 'delighted']):
        return '#f1c40f'   # Yellow
    elif any(w in state_lower for w in ['energized', 'motivated', 'eager']):
        return '#e67e22'   # Bright orange
    elif any(w in state_lower for w in ['tentative', 'uncertain', 'hesitant']):
        return '#95a5a6'   # Gray
    elif any(w in state_lower for w in ['introspective', 'deep', 'analyzing']):
        return '#1abc9c'   # Teal
    elif any(w in state_lower for w in ['creative', 'imaginative', 'inspired']):
        return '#e91e63'   # Pink
    elif any(w in state_lower for w in ['alert', 'vigilant', 'aware']):
        return '#00bcd4'   # Cyan
    elif any(w in state_lower for w in ['relaxed', 'easy', 'comfortable']):
        return '#8bc34a'   # Light green
    elif any(w in state_lower for w in ['determined', 'resolute', 'committed']):
        return '#ff5722'   # Deep orange
    elif any(w in state_lower for w in ['consolidating', 'synthesizing', 'integrating']):
        return '#9c27b0'   # Deep purple — matches main.py consolidating color
    elif any(w in state_lower for w in ['baseline', 'inventorying', 'orienting']):
        return '#607d8b'   # Blue gray — calm autonomous activity
    elif any(w in state_lower for w in ['verifying', 'checking', 'comparing']):
        return '#5c6bc0'   # Indigo — focused comparison activity
    elif any(w in state_lower for w in ['drift', 'drifted', 'diverged']):
        return '#e53935'   # Strong red — attention required, distinct from error states
    elif any(w in state_lower for w in ['aligned', 'coherent', 'consistent']):
        return '#2e7d32'   # Deep green — stable, confirmed
    elif any(w in state_lower for w in ['evolved', 'evolving', 'growth']):
        return '#00897b'   # Teal green — growth, distinct from content/calm greens
    elif 'neutral' in state_lower:
        return '#7f8c8d'   # Medium gray
    else:
        return '#3498db'   # Default blue for emergent states


def handle_cognitive_state_update(
    chatbot,
    state_name: str,
    origin: str = ORIGIN_CONVERSATION
) -> Tuple[bool, str]:
    """
    Core cognitive state update logic.
    
    Updates chatbot instance state, appends to history with origin tracking,
    and pushes to Streamlit session state for UI display.
    
    Called by:
    - deepseek.py _handle_cognitive_state_command() wrapper (origin=conversation)
    - autonomous_cognition.py Functional State Baseline Check (origin=autonomous)
    
    Rate limiting is NOT handled here — that is the responsibility of the
    caller (deepseek.py enforces 1 per turn for conversation-origin states).
    
    Args:
        chatbot: Chatbot instance with current_cognitive_state attribute
        state_name: Raw state name string (will be normalized)
        origin: ORIGIN_CONVERSATION or ORIGIN_AUTONOMOUS
        
    Returns:
        Tuple[bool, str]: (success, normalized_state_name)
    """
    try:
        # Normalize state name
        state_name_clean = normalize_state_name(state_name)
        
        # Get old state for transition logging
        old_state = getattr(chatbot, 'current_cognitive_state', 'neutral')
        
        # Update chatbot instance state
        chatbot.current_cognitive_state = state_name_clean
        
        # Build state entry with origin tracking
        state_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'from_state': old_state,
            'to_state': state_name_clean,
            'origin': origin  # 'conversation' or 'autonomous'
        }
        
        # Update chatbot history (keep last MAX_HISTORY_SIZE entries)
        if not hasattr(chatbot, 'cognitive_state_history'):
            chatbot.cognitive_state_history = []
        
        chatbot.cognitive_state_history.append(state_entry)
        if len(chatbot.cognitive_state_history) > MAX_HISTORY_SIZE:
            chatbot.cognitive_state_history = (
                chatbot.cognitive_state_history[-MAX_HISTORY_SIZE:]
            )
        
        # Push to Streamlit session state for UI display
        # Import is deferred and guarded — cognitive_state.py has no Streamlit
        # dependency at module level, keeping it importable in non-UI contexts
        try:
            import streamlit as st
            if hasattr(st, 'session_state'):
                st.session_state.cognitive_state = state_name_clean
                
                if 'cognitive_state_history' not in st.session_state:
                    st.session_state.cognitive_state_history = []
                
                st.session_state.cognitive_state_history.append(state_entry)
                
                if len(st.session_state.cognitive_state_history) > MAX_HISTORY_SIZE:
                    st.session_state.cognitive_state_history = (
                        st.session_state.cognitive_state_history[-MAX_HISTORY_SIZE:]
                    )
                
                logging.info(
                    f"COGNITIVE_STATE: Updated session state to '{state_name_clean}' "
                    f"(origin={origin})"
                )
        except ImportError:
            # Non-UI context (e.g. unit tests, CLI) — chatbot state updated,
            # session state unavailable, not an error
            logging.debug("COGNITIVE_STATE: Streamlit not available, skipping session update")
        
        logging.info(
            f"COGNITIVE_STATE: {old_state} → {state_name_clean} "
            f"[origin={origin}]"
        )
        return True, state_name_clean
        
    except Exception as e:
        logging.error(
            f"COGNITIVE_STATE: Error in handle_cognitive_state_update: {e}",
            exc_info=True
        )
        return False, 'neutral'


# QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 5 cognitive cleanup pass).
# Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
# Public getter for chatbot.current_cognitive_state — every reader in the codebase uses
# getattr(chatbot, 'current_cognitive_state', 'neutral') directly instead of this helper.
# Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
def _UNUSED_get_current_state(chatbot) -> str:
    """
    Get QWEN's current self-reported cognitive state.
    
    Args:
        chatbot: Chatbot instance
        
    Returns:
        str: Current state name, defaults to 'neutral'
    """
    return getattr(chatbot, 'current_cognitive_state', 'neutral')


# QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 5 cognitive cleanup pass).
# Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
# Public getter for chatbot.cognitive_state_history — utils.display_cognitive_state_widget
# reads st.session_state.cognitive_state_history directly instead of going through this helper.
# Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
def _UNUSED_get_state_history(chatbot) -> list:
    """
    Get QWEN's cognitive state transition history.
    
    Args:
        chatbot: Chatbot instance
        
    Returns:
        list: List of state_entry dicts with timestamp, from_state,
              to_state, origin keys
    """
    return getattr(chatbot, 'cognitive_state_history', [])