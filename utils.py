# Modified utils.py - File Import Feature with Memory Command Testing and Speech Settings
"""Utility functions and logging setup for the AI Assistant."""
import time
import logging
import os
import json
import datetime 
from typing import Dict
import streamlit as st
from config import DOCS_PATH, SUPPORTED_EXTENSIONS

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        filename='AI.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class RateLimiter:
    """Limits the rate of operations to avoid overwhelming resources.
    
    This utility class is used to control the frequency of operations,
    particularly for database operations in MemoryDB and VectorDB.
    
    Args:
        operations_per_second (float): Maximum operations per second
    """
    
    def __init__(self, operations_per_second=5):
        """Initialize with specified operations per second rate.
        
        Args:
            operations_per_second (float): Maximum operations per second
        """
        self.operations_per_second = operations_per_second
        self.last_operation_time = time.time()
        logging.debug(f"RateLimiter initialized with {operations_per_second} ops/second")

    def wait_if_needed(self):
        """Wait if we're exceeding our rate limit.
        
        Calculates the time since the last operation and waits if
        necessary to maintain the desired operations per second rate.
        """
        current_time = time.time()
        elapsed = current_time - self.last_operation_time
        min_interval = 1.0 / self.operations_per_second
    
        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            time.sleep(wait_time)
            logging.debug(f"Rate limited: waited {wait_time:.3f}s")
        
        self.last_operation_time = time.time()

def create_status_indicators() -> Dict:
    """Create status indicators for the Streamlit interface (disabled)."""
    return {
        'FAISS': None,
        'DB': None,
        'Docs': None,
        'Model': None
    }

def ensure_directories():
    """Ensure all required directories exist."""
    from config import DOCS_PATH  # QDRANT_LOCAL_PATH removed - using Docker at localhost:6333
    
    directories = [DOCS_PATH]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")

def calculate_tokens(text: str) -> int:
    """
    Standardized token estimation for LLM context tracking using both
    character and word-based methods. Returns a conservative estimate
    by taking the maximum of the two methods.
    
    The 4 chars/token ratio is a reasonable average for transformer
    tokenizers (Qwen, GPT, Claude, etc.) for English text. Word-based
    estimate provides a fallback that's less sensitive to whitespace
    and punctuation patterns.
    
    Args:
        text (str): The text to estimate tokens for
        
    Returns:
        int: Estimated token count (0 for empty/invalid input)
    """
    # Guard: handle empty or non-string input
    if not text or not isinstance(text, str):
        return 0
    
    # Method 1: Character-based estimate (~4 chars/token average)
    char_estimate = len(text) / 4
    
    # Method 2: Word-based fallback (~1.3 tokens/word average)
    word_estimate = len(text.split()) * 1.3
    
    # Conservative: take the larger estimate to avoid undercounting
    return int(max(char_estimate, word_estimate))


def save_reflection_schedule(schedule_dict):
    """Save reflection schedule to a JSON file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reflection_config.json")
        with open(config_path, 'w') as f:
            json.dump(schedule_dict, f)
        logging.info(f"Saved reflection schedule to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving reflection schedule: {e}")
        return False

def load_reflection_schedule():
    """Load reflection schedule from a JSON file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reflection_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                schedule = json.load(f)
            logging.info(f"Loaded reflection schedule from {config_path}")
            return schedule
        else:
            # Return default schedule if file doesn't exist
            return {
                'daily': False,
                'weekly': False,
                'monthly': False
            }
    except Exception as e:
        logging.error(f"Error loading reflection schedule: {e}")
        # Return default schedule on error
        return {
            'daily': False,
            'weekly': False,
            'monthly': False
        }

def is_autonomous_thinking_disabled():
    """Check if memory management is explicitly disabled in config file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Only return True if explicitly set to disabled
                return config.get("memory_management_disabled", False)
        return False  # Default to enabled (not disabled)
    except Exception as e:
        logging.error(f"Error checking memory management config: {e}")
        return False  # Default to enabled on error
    
def set_autonomous_thinking_disabled(disabled=False):
    """Set whether memory management is disabled in config file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_config.json")
        
        # Load existing config or create new one
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Update config - use the new key name
        config["memory_management_disabled"] = disabled
        
        # Keep backward compatibility with old key
        config["autonomous_thinking_disabled"] = disabled
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        return True
    except Exception as e:
        logging.error(f"Error setting memory management config: {e}")
        return False
    
def get_disabled_cognitive_activities() -> list:
    """
    Get list of cognitive activities that are disabled from automatic scheduling.
    
    Returns:
        list: List of activity names that should be skipped by the scheduler.
              Returns empty list if none disabled or on error.
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("disabled_cognitive_activities", [])
        return []  # Default: all activities enabled
    except Exception as e:
        logging.error(f"Error getting disabled cognitive activities: {e}")
        return []  # Default to all enabled on error


def set_cognitive_activity_enabled(activity_name: str, enabled: bool) -> bool:
    """
    Enable or disable a specific cognitive activity for automatic scheduling.
    
    Args:
        activity_name (str): The internal name of the cognitive activity
        enabled (bool): True to enable in scheduler, False to disable
        
    Returns:
        bool: True if successfully saved, False on error
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_config.json")
        
        # Load existing config or create new one
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Get or initialize the disabled activities list
        disabled = config.get("disabled_cognitive_activities", [])
        
        # Update the list based on enabled state
        if enabled and activity_name in disabled:
            # Re-enabling: remove from disabled list
            disabled.remove(activity_name)
            logging.info(f"Cognitive activity '{activity_name}' enabled for scheduling")
        elif not enabled and activity_name not in disabled:
            # Disabling: add to disabled list
            disabled.append(activity_name)
            logging.info(f"Cognitive activity '{activity_name}' disabled from scheduling")
        
        # Save back to config
        config["disabled_cognitive_activities"] = disabled
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception as e:
        logging.error(f"Error setting cognitive activity state for '{activity_name}': {e}")
        return False

# QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (utils.py cleanup pass).
# Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
def _UNUSED_update_session_memory_counter(command_type: str):
    """Update session memory counter for a specific command type."""
    try:
        # Initialize session counters if they don't exist
        if 'memory_command_counts' not in st.session_state:
            st.session_state.memory_command_counts = {
                'store': 0,
                'retrieve': 0,
                'reflect': 0,
                'forget': 0,
                'reminder': 0,
                'reminder_complete': 0,  
                'summarize': 0,  # This tracks summarize_conversation commands
                'discuss_with_claude': 0,
                'help': 0,
                'show_system_prompt': 0,
                'modify_system_prompt': 0,
                'self_dialogue': 0,
                'web_search': 0,
                'cognitive_state': 0
            }
        
        # Map command variations to counter keys
        command_mapping = {
            'summarize_conversation': 'summarize',  # ADD THIS MAPPING
            'summarize': 'summarize'
        }
        
        # Use mapped command or original
        counter_key = command_mapping.get(command_type, command_type)
                
        # Increment the specific counter
        if counter_key in st.session_state.memory_command_counts:
            st.session_state.memory_command_counts[counter_key] += 1
            logging.info(f"Updated session counter for {command_type} -> {counter_key}: {st.session_state.memory_command_counts[counter_key]}")
        else:
            logging.warning(f"⚠️ Unknown command type for session counter: {command_type}")
            
    except Exception as e:
        logging.error(f"❌ Error updating session counter for {command_type}: {e}")

def display_self_reflection_widget():
    """Display self-reflection interactive widget in the sidebar for scheduling reflections."""
    # Display help information WITHOUT using expanders
    st.markdown(display_self_reflection_help())
    
    # Add horizontal line for visual separation
    st.markdown("---")
    
    # Immediate reflection section
    st.markdown("#### Run Reflection Now")
    
    # Create three columns for the reflection buttons
    col1, col2, col3 = st.columns(3)
    
    # Add reflection buttons to each column with unique keys
    if col1.button("Daily", key="daily_reflection"):
        # Run immediate daily reflection
        with st.spinner("Performing daily reflection..."):
            if 'chatbot' in st.session_state:
                indicators = {'Model': st.empty(), 'DB': st.empty()}
                indicators['Model'].info("🤖")
                indicators['DB'].info("💾")
                
                reflection = st.session_state.chatbot.reflection_engine.perform_self_reflection(
                    reflection_type="daily",
                    llm=st.session_state.chatbot.llm
                )
                
                # Add to chat history
                if 'messages' in st.session_state:
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "reflect"
                    })
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Self-Reflection (daily):\n\n{reflection}"
                    })
                st.success("Daily reflection complete!")
    
    if col2.button("Week", key="weekly_reflection"):
        # Run immediate weekly reflection
        with st.spinner("Performing weekly reflection..."):
            if 'chatbot' in st.session_state:
                indicators = {'Model': st.empty(), 'DB': st.empty()}
                indicators['Model'].info("🤖")
                indicators['DB'].info("💾")
                
                reflection = st.session_state.chatbot.reflection_engine.perform_self_reflection(
                    reflection_type="weekly",
                    llm=st.session_state.chatbot.llm
                )
                
                # Add to chat history
                if 'messages' in st.session_state:
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "reflect weekly"
                    })
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Self-Reflection (weekly):\n\n{reflection}"
                    })
                st.success("Weekly reflection complete!")
    
    if col3.button("Month", key="monthly_reflection"):
        # Run immediate monthly reflection
        with st.spinner("Performing monthly reflection..."):
            if 'chatbot' in st.session_state:
                indicators = {'Model': st.empty(), 'DB': st.empty()}
                indicators['Model'].info("🤖")
                indicators['DB'].info("💾")
                
                reflection = st.session_state.chatbot.reflection_engine.perform_self_reflection(
                    reflection_type="monthly",
                    llm=st.session_state.chatbot.llm
                )
                
                # Add to chat history
                if 'messages' in st.session_state:
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "reflect monthly"
                    })
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Self-Reflection (monthly):\n\n{reflection}"
                    })
                st.success("Monthly reflection complete!")
    
    # Add horizontal line for visual separation
    st.markdown("---")
    
    # Scheduled reflection section
    st.markdown("#### Scheduled Reflections")
    
    # Load saved schedule 
    if 'scheduled_reflections' not in st.session_state:
        st.session_state.scheduled_reflections = load_reflection_schedule()
    
    # ADD DAILY REFLECTION SCHEDULING OPTION
    daily_scheduled = st.toggle("Daily Reflection",
                               value=st.session_state.scheduled_reflections['daily'],
                               help="When enabled, AI will perform daily reflections automatically (6:15 AM)")
    
    # Weekly and monthly reflection scheduling options
    weekly_scheduled = st.toggle("Week Reflection",
                               value=st.session_state.scheduled_reflections['weekly'],
                               help="When enabled, AI will perform weekly reflections automatically (Sunday, 9:15 AM)")
    
    monthly_scheduled = st.toggle("Month Reflection",
                                value=st.session_state.scheduled_reflections['monthly'],
                                help="When enabled, AI will perform monthly reflections automatically (1st day of month, 12:20 PM)")
    
    # Update session state based on toggles
    # Check if daily scheduling changed
    if daily_scheduled != st.session_state.scheduled_reflections['daily']:
        st.session_state.scheduled_reflections['daily'] = daily_scheduled
        # Save the updated schedule
        save_reflection_schedule(st.session_state.scheduled_reflections)
        if daily_scheduled:
            st.info("Daily reflection scheduled (6:15 AM)")
        else:
            st.info("Daily reflection unscheduled")
    
    # Check if weekly scheduling changed
    if weekly_scheduled != st.session_state.scheduled_reflections['weekly']:
        st.session_state.scheduled_reflections['weekly'] = weekly_scheduled
        # Save the updated schedule
        save_reflection_schedule(st.session_state.scheduled_reflections)
        if weekly_scheduled:
            st.info("Weekly reflection scheduled (Sunday, 9:15 AM)")
        else:
            st.info("Weekly reflection unscheduled")
    
    # Check if monthly scheduling changed
    if monthly_scheduled != st.session_state.scheduled_reflections['monthly']:
        st.session_state.scheduled_reflections['monthly'] = monthly_scheduled
        # Save the updated schedule
        save_reflection_schedule(st.session_state.scheduled_reflections)
        if monthly_scheduled:
            st.info("Monthly reflection scheduled (1st day of month, 12:20 PM)")
        else:
            st.info("Monthly reflection unscheduled")
    
    return ""  # Return empty string since the widget handles its own display

def display_cognitive_state_widget():
    """Display cognitive state widget - standalone, not in an expander."""
    
    # Import pure logic from cognitive_state module
    # (emoji/color functions extracted there to avoid duplication)
    from cognitive_state import get_state_emoji, get_state_color
    
    # CRITICAL: Always initialize cognitive state if it doesn't exist
    if 'cognitive_state' not in st.session_state:
        st.session_state.cognitive_state = 'neutral'
        logging.info("Initialized cognitive_state to 'neutral' in session state")
    
    if 'cognitive_state_history' not in st.session_state:
        st.session_state.cognitive_state_history = []
        logging.info("Initialized cognitive_state_history in session state")
    
    current_state = st.session_state.cognitive_state
    display_state = current_state.replace('_', ' ').title() if current_state else 'Neutral'
    
    emoji = get_state_emoji(current_state.lower())
    color = get_state_color(current_state.lower())
    
    # Display current state
    st.markdown(
        f"<div style='font-size:1.1em; font-weight:bold; color:{color};'>"
        f"{emoji} Current: {display_state}"
        f"</div>",
        unsafe_allow_html=True
    )
    
    state_count = st.session_state.memory_command_counts.get('cognitive_state', 0)
    if state_count > 0:
        st.caption(f"✅ Updated {state_count} time{'s' if state_count != 1 else ''} this session")
    else:
        st.caption("⏳ Awaiting first state update from model")
    
    if st.session_state.cognitive_state_history:
        with st.expander(
            f"📊 State History ({len(st.session_state.cognitive_state_history)} changes)",
            expanded=False
        ):
            recent_history = st.session_state.cognitive_state_history[-5:]
            
            for entry in reversed(recent_history):
                try:
                    timestamp = datetime.datetime.fromisoformat(entry['timestamp'])
                    time_str = timestamp.strftime("%H:%M:%S")
                    
                    from_state = entry['from_state']
                    to_state = entry['to_state']
                    
                    # Origin badge — shows whether state was set during conversation
                    # or by an autonomous background task (new field, default gracefully)
                    origin = entry.get('origin', 'conversation')
                    origin_badge = "🤖" if origin == 'autonomous' else "💬"
                    
                    from_display = from_state.replace('_', ' ').title() if from_state else 'Unknown'
                    to_display = to_state.replace('_', ' ').title() if to_state else 'Unknown'
                    from_emoji = get_state_emoji(from_state.lower()) if from_state else '❓'
                    to_emoji = get_state_emoji(to_state.lower()) if to_state else '❓'
                    
                    st.markdown(
                        f"<div style='font-size:0.85em; margin-bottom:4px;'>"
                        f"⏱️ {time_str} {origin_badge}: "
                        f"{from_emoji} {from_display} → {to_emoji} {to_display}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    logging.error(f"Error displaying state history entry: {e}")
                    st.text(f"Entry: {entry}")
    
def display_settings_widget():
    """Display settings for memory command visibility and counters."""
    st.markdown("### Memory Command Visibility")
     
    # ALWAYS ON: Initialize and set training_mode to always be True
    if 'memory_command_training' not in st.session_state:
        st.session_state.memory_command_training = True

    # REMOVED TOGGLE: Always set training mode to True
    training_mode = True
    st.session_state.memory_command_training = True
    
    # Always update the enhancer's training mode to True
    if 'chatbot' in st.session_state and hasattr(st.session_state.chatbot, 'deepseek_enhancer'):
        st.session_state.chatbot.deepseek_enhancer.training_mode = True
        
    # Show permanent status (no toggle needed)
    st.info("✅ Memory commands show success/failure indicators")
    st.markdown("*Insight into the AI's memory operations.*")
    
    # Add Memory Command Counter section
    st.markdown("### Memory Command Counts")

    # Initialize counters in session state if they don't exist
    if 'memory_command_counts' not in st.session_state:
        st.session_state.memory_command_counts = {
            'store': 0,
            'search': 0,
            'retrieve': 0,
            'reflect': 0,
            'forget': 0,
            'reminder': 0,
            'summarize': 0,
            'discuss_with_claude': 0,
            'help': 0,
            'show_system_prompt': 0,
            'modify_system_prompt': 0,
            'self_dialogue': 0,
            'web_search': 0,
            'cognitive_state': 0
        }

    # Create a smaller, compact display with labels
    st.markdown("<p style='font-size:0.9em; margin-bottom:5px;'>Session Counters:</p>", unsafe_allow_html=True)

    # Use a 2-column layout with stacked sections
    col1, col2 = st.columns(2)
    
    with col1:
        # Core Operations section
        st.markdown("<div style='font-size:0.75em; color:#888; margin-bottom:3px;'>Core Operations</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Store:</b> {st.session_state.memory_command_counts['store']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Forget:</b> {st.session_state.memory_command_counts['forget']}</div>", unsafe_allow_html=True)
        
        # Reflect counter — reflect_concept removed (command had no handler, counter always 0)
        st.markdown(f"<div style='font-size:0.8em;'><b>Reflect:</b> {st.session_state.memory_command_counts['reflect']}</div>", unsafe_allow_html=True)
        
        # Utilities section (stacked below Core Operations)
        st.markdown("<div style='font-size:0.75em; color:#888; margin-bottom:3px; margin-top:10px;'>Utilities</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Web Search:</b> {st.session_state.memory_command_counts.get('web_search', 0)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Help:</b> {st.session_state.memory_command_counts.get('help', 0)}</div>", unsafe_allow_html=True)
        
        # Calculate and show total
        total = sum(st.session_state.memory_command_counts.values())
        st.markdown(f"<div style='font-size:0.8em; font-weight:bold; color:#0066cc; margin-top:5px;'><b>Total:</b> {total}</div>", unsafe_allow_html=True)

    with col2:
        # Management section
        st.markdown("<div style='font-size:0.75em; color:#888; margin-bottom:3px;'>Management</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Search:</b> {st.session_state.memory_command_counts.get('search', 0)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Reminder:</b> {st.session_state.memory_command_counts['reminder']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Summarize:</b> {st.session_state.memory_command_counts.get('summarize_conversation', 0)}</div>", unsafe_allow_html=True)
        
        # Dialogue/System section (stacked below Management)
        st.markdown("<div style='font-size:0.75em; color:#888; margin-bottom:3px; margin-top:10px;'>Dialogue/System</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Discuss with Claude:</b> {st.session_state.memory_command_counts.get('discuss_with_claude', 0)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Self Dialogue:</b> {st.session_state.memory_command_counts.get('self_dialogue', 0)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Cognitive State:</b> {st.session_state.memory_command_counts.get('cognitive_state', 0)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Show Prompt:</b> {st.session_state.memory_command_counts.get('show_system_prompt', 0)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8em;'><b>Modify Prompt:</b> {st.session_state.memory_command_counts.get('modify_system_prompt', 0)}</div>", unsafe_allow_html=True)
               
              
    # Add Lifetime Counters display - REORGANIZED VERSION
    st.markdown("<p style='font-size:0.9em; margin-top:15px; margin-bottom:5px;'>Lifetime Counters:</p>", unsafe_allow_html=True)

    # Get lifetime counters if available
    try:
        if ('chatbot' in st.session_state and 
            hasattr(st.session_state.chatbot, 'deepseek_enhancer') and
            hasattr(st.session_state.chatbot.deepseek_enhancer, 'lifetime_counters')):
            
            lifetime_counters = st.session_state.chatbot.deepseek_enhancer.lifetime_counters.get_counters()

            # Use same 2-column layout for consistency
            col1, col2 = st.columns(2)

            with col1:
                # Core Operations section
                st.markdown("<div style='font-size:0.75em; color:#888; margin-bottom:3px;'>Core Operations</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Store:</b> {lifetime_counters.get('store', 0)}</div>", 
                            unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Forget:</b> {lifetime_counters.get('forget', 0)}</div>", 
                            unsafe_allow_html=True)
                
                # Reflect counter — reflect_concept removed (command had no handler, counter always 0)
                lifetime_total_reflect = lifetime_counters.get('reflect', 0)
                st.markdown(f"<div style='font-size:0.8em;'><b>Reflect:</b> {lifetime_total_reflect}</div>", 
                        unsafe_allow_html=True)
                
                # Utilities section (stacked below Core Operations)
                st.markdown("<div style='font-size:0.75em; color:#888; margin-bottom:3px; margin-top:10px;'>Utilities</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Web Search:</b> {lifetime_counters.get('web_search', 0)}</div>", 
                            unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Help:</b> {lifetime_counters.get('help', 0)}</div>", 
                            unsafe_allow_html=True)
                
                # Calculate and show lifetime total
                relevant_counters = ['store', 'search', 'retrieve', 'reflect', 'forget', 'reminder',
                                'summarize', 'discuss_with_claude',
                                'help', 'show_system_prompt', 'modify_system_prompt',
                                'self_dialogue', 'web_search']
                lifetime_total = sum(lifetime_counters.get(key, 0) for key in relevant_counters)
                st.markdown(f"<div style='font-size:0.8em; margin-top:5px; font-weight:bold; color:#0066cc;'><b>Total:</b> {lifetime_total}</div>", 
                            unsafe_allow_html=True)

            with col2:
                # Management section
                st.markdown("<div style='font-size:0.75em; color:#888; margin-bottom:3px;'>Management</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Search:</b> {lifetime_counters.get('search', 0)}</div>", 
                            unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Reminder:</b> {lifetime_counters.get('reminder', 0)}</div>", 
                            unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Summarize:</b> {lifetime_counters.get('summarize', 0)}</div>", 
                            unsafe_allow_html=True)
                
                # Dialogue/System section (stacked below Management)
                st.markdown("<div style='font-size:0.75em; color:#888; margin-bottom:3px; margin-top:10px;'>Dialogue/System</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Discuss with Claude:</b> {lifetime_counters.get('discuss_with_claude', 0)}</div>", 
                            unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Self Dialogue:</b> {lifetime_counters.get('self_dialogue', 0)}</div>", 
                            unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Cognitive State:</b> {lifetime_counters.get('cognitive_state', 0)}</div>",
                             unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Show Prompt:</b> {lifetime_counters.get('show_system_prompt', 0)}</div>", 
                            unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;'><b>Modify Prompt:</b> {lifetime_counters.get('modify_system_prompt', 0)}</div>", 
                            unsafe_allow_html=True)

        else:
            st.markdown("<div style='font-size:0.8em; color:gray;'>Lifetime counters not available</div>", 
                        unsafe_allow_html=True)
            
    except Exception as e:
        st.markdown("<div style='font-size:0.8em; color:red;'>Error loading lifetime counters</div>", 
                    unsafe_allow_html=True)
        logging.error(f"Error accessing lifetime counters: {e}")

# QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (utils.py cleanup pass).
# Also imports nonexistent 'local_speech' module — would crash at import time if reached.
# Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
def _UNUSED_display_speech_settings_widget():
    """Display speech settings widget in the sidebar for STT and TTS toggles."""
    st.markdown("### Speech Settings")
    
    # Import required modules
    try:
        from local_speech import local_speech_utils
        from speech_utils import speech_utils
    except ImportError as e:
        st.error(f"Speech modules not available: {e}")
        logging.error(f"Failed to import speech modules: {e}")
        return
    
    # Ensure session state variables exist
    if 'speech_to_text_enabled' not in st.session_state:
        st.session_state.speech_to_text_enabled = False
    if 'text_to_speech_enabled' not in st.session_state:
        st.session_state.text_to_speech_enabled = False
    if 'continuous_listening_active' not in st.session_state:
        st.session_state.continuous_listening_active = False
    
    # Get previous state to detect changes
    previous_stt_enabled = st.session_state.speech_to_text_enabled
    
    # Speech-to-Text toggle
    new_stt_enabled = st.toggle(
        "Speech-to-Text",
        value=st.session_state.speech_to_text_enabled,
        help="When enabled, speak your message instead of typing"
    )
    
    # Detect changes in the toggle state
    if new_stt_enabled != previous_stt_enabled:
        st.session_state.speech_to_text_enabled = new_stt_enabled
        
        # If enabling, start continuous listening
        if new_stt_enabled:
            # Define the speech callback that will update a session variable
            def speech_callback(text):
                """Callback function for speech recognition results.
                
                Args:
                    text: The recognized speech text
                """
                try:
                    # Store the recognized text in session state
                    st.session_state.last_recognized_speech = text
                    # Set a flag to trigger processing in the main loop
                    st.session_state.speech_input_received = True
                    logging.info(f"Speech callback received: '{text}'")
                except Exception as e:
                    logging.error(f"Error in speech callback: {e}", exc_info=True)
            
            # Start continuous listening
            try:
                success = speech_utils.start_continuous_listening(callback=speech_callback)
                if success:
                    st.session_state.continuous_listening_active = True
                    st.success("Continuous listening activated")
                else:
                    st.error("Failed to start continuous listening")
                    st.session_state.speech_to_text_enabled = False
            except Exception as e:
                logging.error(f"Error starting continuous listening: {e}", exc_info=True)
                st.error(f"Error: {str(e)}")
                st.session_state.speech_to_text_enabled = False
        else:
            # If disabling, stop continuous listening
            try:
                speech_utils.stop_continuous_listening()
                st.session_state.continuous_listening_active = False
            except Exception as e:
                logging.error(f"Error stopping continuous listening: {e}", exc_info=True)
    
    # Text-to-Speech toggle
    st.session_state.text_to_speech_enabled = st.toggle(
        "Text-to-Speech",
        value=st.session_state.text_to_speech_enabled,
        help="When enabled, AI will speak responses aloud"
    )
    
    # Add visual indicator for continuous listening mode
    if st.session_state.speech_to_text_enabled and st.session_state.continuous_listening_active:
        st.info("Speech recognition is active")

def save_speech_settings(settings_dict):
    """Save speech settings to a JSON file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_settings.json")
        with open(config_path, 'w') as f:
            json.dump(settings_dict, f)
        logging.info(f"Saved speech settings to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving speech settings: {e}")
        return False

def load_speech_settings():
    """Load speech settings from a JSON file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_settings.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                settings = json.load(f)
            logging.info(f"Loaded speech settings from {config_path}")
            return settings
        else:
            # Return default settings if file doesn't exist
            return {
                'speech_to_text_enabled': False,
                'text_to_speech_enabled': False,
                'wake_word_enabled': False,
                'wake_word_phrase': 'i have a question'
            }
    except Exception as e:
        logging.error(f"Error loading speech settings: {e}")
        # Return default settings on error
        return {
            'speech_to_text_enabled': False,
            'text_to_speech_enabled': False,
            'wake_word_enabled': False,
            'wake_word_phrase': 'i have a question'
        }
        
# QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (utils.py cleanup pass).
# Targets memory_management_config.json — actual settings flow uses system_config.json instead.
# Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
def _UNUSED_save_autonomous_cognition_settings(settings_dict):
    """Save memory management settings to a JSON file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_management_config.json")
        with open(config_path, 'w') as f:
            json.dump(settings_dict, f)
        logging.info(f"Saved memory management settings to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving memory management settings: {e}")
        return False

# QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (utils.py cleanup pass).
# Companion to _UNUSED_save_autonomous_cognition_settings — same orphan config flow.
# Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
def _UNUSED_load_autonomous_cognition_settings():
    """Load memory management settings from a JSON file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_management_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                settings = json.load(f)
            logging.info(f"Loaded memory management settings from {config_path}")
            return settings
        else:
            # Return default settings if file doesn't exist
            return {
                'enabled': False,
                'parameters': {
                    'cycle_interval': 300,  # Default 5 minutes
                    'max_thought_history': 20
                }
            }
    except Exception as e:
        logging.error(f"Error loading memory management settings: {e}")
        # Return default settings on error
        return {
            'enabled': False,
            'parameters': {
                'cycle_interval': 300,
                'max_thought_history': 20
            }
        }

def display_file_import_widget():
    """Display file import widget with proper duplicate prevention and error handling."""
    st.markdown("### 📁 File Import")
    
    # Get supported extensions from config and format for display
    extensions_str = ", ".join([ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS])
    
    # Create file uploader with supported extensions
    uploaded_file = st.file_uploader(
        f"Upload documents ({extensions_str})",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
        key="document_uploader"
    )
    
    if uploaded_file is not None:
        # Create unique file identifier
        file_hash = hash(uploaded_file.getvalue())
        file_key = f"processed_{uploaded_file.name}_{uploaded_file.size}_{file_hash}"
        processing_key = f"processing_{uploaded_file.name}_{file_hash}"
        
        # Check if already processed
        if file_key in st.session_state:
            st.success(f"✅ '{uploaded_file.name}' already processed this session")
            return
            
        # Check if currently processing
        if processing_key in st.session_state:
            st.warning(f"⏳ '{uploaded_file.name}' is currently being processed...")
            return
        
        # Set processing flags to prevent interference
        st.session_state[processing_key] = True
        st.session_state.file_processing_in_progress = True
        st.session_state.skip_conversation_reload = True
        
        try:
            # Show processing status
            status_placeholder = st.empty()
            status_placeholder.info("📝 Processing document...")
            
            # Save file to LocalDocs directory
            file_path = os.path.join(DOCS_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logging.info(f"File saved to: {file_path}")
            
            # Verify chatbot and document reader are available
            if not ('chatbot' in st.session_state and 
                    hasattr(st.session_state.chatbot, 'doc_reader')):
                status_placeholder.error("❌ System not ready")
                st.error("Document processing system not available")
                return
            
            # Process the document - CORRECT METHOD CALL
            logging.info(f"Starting document processing for: {uploaded_file.name}")
            result = st.session_state.chatbot.doc_reader.process_uploaded_document(uploaded_file.name)
            logging.info(f"Document processing result: {result[:200]}...")
            
            # Handle the result
            if result and isinstance(result, str):
                if ("successfully" in result.lower() or 
                    "✅" in result or 
                    "processed successfully" in result.lower()):
                    
                    status_placeholder.success("✅ Document processed!")
                    
                    # Extract summary for chat if available
                    if "**Summary:**" in result:
                        try:
                            summary_start = result.find("**Summary:**") + len("**Summary:**")
                            summary_end = result.find("**To retrieve")
                            if summary_end == -1:
                                summary_end = len(result)
                            
                            document_summary = result[summary_start:summary_end].strip()
                            
                            # ENHANCED: Store for chat injection WITH full summary
                            st.session_state.pending_document_message = {
                                "role": "assistant",
                                "content": (
                                    f"📄 **Document '{uploaded_file.name}' processed successfully!**\n\n"
                                    f"**Summary:**\n{document_summary}\n\n"
                                    f"*I can now answer questions about this document.*"
                                )
                            }
                            
                            # IMMEDIATE FIX: Also add directly to chatbot conversation
                            if 'chatbot' in st.session_state:
                                summary_message = {
                                    "role": "assistant",
                                    "content": (
                                        f"📄 **Document '{uploaded_file.name}' processed successfully!**\n\n"
                                        f"**Summary:**\n{document_summary}\n\n"
                                        f"*I can now answer questions about this document.*"
                                    )
                                }
                                st.session_state.chatbot.current_conversation.append(summary_message)
                                logging.info("DOCUMENT_PROCESSING: Added document summary directly to chatbot conversation")
                            
                            logging.info("Document summary prepared for chat injection")
                        except Exception as extract_error:
                            logging.error(f"Error extracting summary: {extract_error}")
                                
                    # Show success message
                    search_command = f"[SEARCH: {uploaded_file.name} | type=document_summary]"
                    st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                    st.code(search_command, language="text")
                    st.caption("↑ Copy this command to search for the document summary")
                    
                elif "already" in result.lower():
                    status_placeholder.info("📄 Already processed")
                    st.info(f"Document '{uploaded_file.name}' was already processed")
                    
                else:
                    status_placeholder.error("❌ Processing failed")
                    st.error(f"Processing failed: {result}")
                    
                # Mark as processed regardless of outcome
                st.session_state[file_key] = True
                
            else:
                status_placeholder.error("❌ Invalid response")
                st.error("Document processing returned invalid response")
                
        except Exception as e:
            logging.error(f"File import error: {e}", exc_info=True)
            status_placeholder.error("❌ Processing error")
            st.error(f"Error processing document: {str(e)}")
            
        finally:
            # Always clean up processing flags
            try:
                if processing_key in st.session_state:
                    del st.session_state[processing_key]
                if 'file_processing_in_progress' in st.session_state:
                    del st.session_state.file_processing_in_progress
                if 'skip_conversation_reload' in st.session_state:
                    del st.session_state.skip_conversation_reload
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up processing flags: {cleanup_error}")

# QUARANTINED 2026-05-18: No callers found in 38-file codebase scan (utils.py cleanup pass).
# Wrapper widget only — underlying save_command_guide_html in command_guide_generator.py IS live.
# Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
def _UNUSED_display_command_help_widget():
    """Display clean command help button that opens comprehensive guide."""
    st.markdown("### 📖 Command Reference")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("Access the complete command guide with examples, filters, and interactive search.")
    
    with col2:
        if st.button("📖 Open Guide", help="Opens comprehensive command reference in new tab"):
            try:
                from command_guide_generator import save_command_guide_html
                import webbrowser
                
                # Generate and save the guide
                file_path = save_command_guide_html()
                
                # Open in browser
                webbrowser.open(f'file://{os.path.abspath(file_path)}')
                
                st.success("✅ Command guide opened!")
                st.info("💡 Bookmark the guide for quick reference")
                
            except Exception as e:
                st.error(f"Error opening command guide: {str(e)}")
                logging.error(f"Command guide error: {e}", exc_info=True)

def display_self_reflection_help():
    """Display help information about self-reflection capabilities."""
    return """
    ### 🧩 Self-Reflection 
    
  YOU can perform regular self-reflection to improve reasoning. 
   
    **Benefits:**
    - Surfaces important patterns and insights
    - Strengthens reasoning abilities
  
    """

def display_sidebar_commands():
    
    # Create expandable sections for different command categories Other sidebar drop downs in main.py
    
    with st.sidebar.expander("📁 File Import", expanded=False):
        display_file_import_widget()

    with st.sidebar.expander("🌐 Web Learning", expanded=False):
        from main import display_web_learning_section
        display_web_learning_section(add_header=False)
        
    with st.sidebar.expander("🧩 Self-Reflection", expanded=False):
        display_self_reflection_widget()

    with st.sidebar.expander("🧠 Autonomous Thinking", expanded=False):
        from main import display_autonomous_cognition_section
        display_autonomous_cognition_section()
        
    # Database Maintenance section
    with st.sidebar.expander("⚙️ System Maintenance", expanded=False):
            st.markdown("### Database Maintenance")
            
            # Enhanced health check button
            if st.button("Run Enhanced Health Check", key="enhanced_health_check"):
                with st.spinner("Running enhanced health check..."):
                    try:
                        from db_maintenance import DatabaseMaintenance
                        
                        maintenance = DatabaseMaintenance(
                            vector_db=st.session_state.chatbot.vector_db,
                            memory_db=st.session_state.chatbot.memory_db
                        )
                        
                        health_report = maintenance.enhanced_health_check()
                        
                        # Display results
                        st.subheader("Health Report")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            overall_score = health_report['health_score']['overall']
                            color = "green" if overall_score > 80 else "orange" if overall_score > 60 else "red"
                            st.markdown(f"**Overall:** <span style='color:{color};'>{overall_score:.1f}/100</span>", unsafe_allow_html=True)
                        
                        with col2:
                            memory_score = health_report['health_score']['memory_db']
                            color = "green" if memory_score > 80 else "orange" if memory_score > 60 else "red"
                            st.markdown(f"**Memory DB:** <span style='color:{color};'>{memory_score:.1f}/100</span>", unsafe_allow_html=True)
                        
                        with col3:
                            vector_score = health_report['health_score']['vector_db']
                            color = "green" if vector_score > 80 else "orange" if vector_score > 60 else "red"
                            st.markdown(f"**Vector DB:** <span style='color:{color};'>{vector_score:.1f}/100</span>", unsafe_allow_html=True)
                        
                        # Enhanced database statistics display
                        st.subheader("Database Statistics")
                        stats = health_report.get('stats', {})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Memory Database Breakdown")
                            st.markdown(f"* **Total entries:** {stats.get('memory_db_total', 'N/A')}")
                            st.markdown(f"* **Syncable entries:** {stats.get('memory_db_syncable', 'N/A')} *(should be in both DBs)*")
                            st.markdown(f"* **Reminders:** {stats.get('memory_db_reminders', 'N/A')} *(SQLite-only)*")
                            st.markdown(f"* **Autonomous thoughts:** {stats.get('memory_db_autonomous_thoughts', 'N/A')} *(SQLite-only)*")
                            
                        with col2:
                            st.markdown("### Synchronization Status")
                            st.markdown(f"* **Vector DB entries:** {stats.get('vector_db_total', 'N/A')}")
                            discrepancy = stats.get('discrepancy', 0)
                            discrepancy_pct = stats.get('discrepancy_percentage', 0)
                            color = "green" if discrepancy == 0 else "orange" if discrepancy_pct < 5 else "red"
                            st.markdown(f"* **Sync discrepancy:** <span style='color:{color};'>{discrepancy} entries ({discrepancy_pct:.1f}%)</span>", unsafe_allow_html=True)
                        
                        # Display issues found - with better context
                        st.subheader("Issues Found (Excluding SQLite-Only Data)")
                        
                        sync_issues = health_report.get('sync_issues', {})
                        duplicates = health_report.get('duplicates', {})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Duplicate Issues")
                            memory_dups = duplicates.get('memory_db', {}).get('duplicates_found', 0)
                            vector_dups = duplicates.get('vector_db', {}).get('duplicates_found', 0)
                            st.markdown(f"* **Memory DB duplicates:** {memory_dups}")
                            st.markdown(f"* **Vector DB duplicates:** {vector_dups}")
                            
                            # Show exclusion counts if available
                            if 'reminders_excluded' in duplicates.get('memory_db', {}):
                                rem_excl = duplicates['memory_db']['reminders_excluded']
                                auto_excl = duplicates['memory_db'].get('autonomous_thoughts_excluded', 0)
                                st.markdown(f"* *Excluded from analysis: {rem_excl} reminders, {auto_excl} autonomous thoughts*")
                        
                        with col2:
                            st.markdown("### Synchronization Issues")
                            missing_vector = sync_issues.get('missing_in_vector_db', 0)
                            missing_memory = sync_issues.get('missing_in_memory_db', 0)
                            st.markdown(f"* **Missing in vector DB:** {missing_vector}")
                            st.markdown(f"* **Missing in memory DB:** {missing_memory}")
                            
                            # Show exclusion counts if available
                            if 'reminders_excluded' in sync_issues:
                                rem_excl = sync_issues['reminders_excluded']
                                auto_excl = sync_issues.get('autonomous_thoughts_excluded', 0)
                                st.markdown(f"* *Excluded from analysis: {rem_excl} reminders, {auto_excl} autonomous thoughts*")
                        
                        st.success("Enhanced health check completed!")
                    except Exception as e:
                        st.error(f"Error running enhanced health check: {str(e)}")
                        logging.error(f"Enhanced health check error: {e}", exc_info=True)
            
            
            # Find duplicates button - ENHANCED VERSION ignores SQL only entries, reminders, autonomous logs
            if st.button("Find Database Duplicates", key="find_duplicates"):
                with st.spinner("Scanning for duplicate entries (excluding SQLite-only data)..."):
                    try:
                        from db_maintenance import DatabaseMaintenance
                        
                        maintenance = DatabaseMaintenance(
                            vector_db=st.session_state.chatbot.vector_db,
                            memory_db=st.session_state.chatbot.memory_db
                        )
                        
                        memory_dups = maintenance.find_memory_db_duplicates()
                        vector_dups = maintenance.find_vector_db_duplicates()
                        
                        # Display results with better breakdown
                        st.subheader("Duplicate Detection Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Memory Database")
                            st.markdown(f"* **Total entries:** {memory_dups['total_entries']}")
                            st.markdown(f"* **Syncable entries:** {memory_dups.get('syncable_entries', 'N/A')} *(analyzed for duplicates)*")
                            st.markdown(f"* **Unique contents:** {memory_dups['unique_contents']}")
                            st.markdown(f"* **Duplicates found:** {memory_dups['duplicates_found']}")
                            st.markdown(f"* **Duplicate groups:** {memory_dups['duplicate_groups']}")
                            st.markdown(f"* **Percentage:** {memory_dups['percentage']*100:.1f}%")
                            
                            # Show what was excluded
                            rem_excl = memory_dups.get('reminders_excluded', 0)
                            auto_excl = memory_dups.get('autonomous_thoughts_excluded', 0)
                            if rem_excl > 0 or auto_excl > 0:
                                st.markdown("---")
                                st.markdown("**Excluded from analysis:**")
                                if rem_excl > 0:
                                    st.markdown(f"* Reminders: {rem_excl}")
                                if auto_excl > 0:
                                    st.markdown(f"* Autonomous thoughts: {auto_excl}")
                        
                        with col2:
                            st.markdown("### Vector Database")
                            st.markdown(f"* **Total entries:** {vector_dups['total_entries']}")
                            st.markdown(f"* **Duplicates found:** {vector_dups['duplicates_found']}")
                            st.markdown(f"* **Duplicate groups:** {vector_dups['duplicate_groups']}")
                            st.markdown(f"* **Percentage:** {vector_dups['percentage']*100:.1f}%")
                        
                        # Summary assessment
                        total_duplicates = memory_dups['duplicates_found'] + vector_dups['duplicates_found']
                        if total_duplicates == 0:
                            st.success("🎉 No duplicates found in either database!")
                        elif total_duplicates < 10:
                            st.info(f"ℹ️ Found {total_duplicates} total duplicates - relatively clean databases")
                        else:
                            st.warning(f"⚠️ Found {total_duplicates} total duplicates - consider running duplicate removal")
                        
                        st.success("Duplicate detection completed!")
                    except Exception as e:
                        st.error(f"Error finding duplicates: {str(e)}")
                        logging.error(f"Find duplicates error: {e}", exc_info=True)
            
            # Add a divider for the advanced options
            st.markdown("---")
            st.markdown("### Advanced Repair Options")
            st.warning("⚠️ These operations modify the database and may take some time to complete.")
            
            # Remove duplicates button
            if st.button("Remove Duplicates", key="remove_duplicates"):
                with st.spinner("Removing duplicate entries..."):
                    try:
                        from db_maintenance import DatabaseMaintenance
                        
                        maintenance = DatabaseMaintenance(
                            vector_db=st.session_state.chatbot.vector_db,
                            memory_db=st.session_state.chatbot.memory_db
                        )
                        
                        memory_removal = maintenance.remove_memory_db_duplicates()
                        vector_removal = maintenance.remove_vector_db_duplicates()
                        
                        # Display results
                        st.markdown("### Duplicate Removal Results")
                        st.markdown(f"* Memory DB: {memory_removal['duplicates_removed']} duplicates removed from {memory_removal['duplicate_groups']} groups")
                        st.markdown(f"* Vector DB: {vector_removal['duplicates_removed']} duplicates removed from {vector_removal['duplicate_groups']} groups")
                        
                        if memory_removal['errors'] > 0 or vector_removal['errors'] > 0:
                            st.warning(f"There were some errors: Memory DB: {memory_removal['errors']}, Vector DB: {vector_removal['errors']}")
                        
                        st.success("Duplicate removal completed!")
                    except Exception as e:
                        st.error(f"Error removing duplicates: {str(e)}")
                        logging.error(f"Remove duplicates error: {e}", exc_info=True)
            
            # Fix all database issues button
            if st.button("Fix All Database Issues", key="fix_all"):
                with st.spinner("Running comprehensive database repair..."):
                    try:
                        from db_maintenance import DatabaseMaintenance
                        
                        maintenance = DatabaseMaintenance(
                            vector_db=st.session_state.chatbot.vector_db,
                            memory_db=st.session_state.chatbot.memory_db
                        )
                        
                        fix_results = maintenance.fix_all_database_issues()
                        
                        # Display results
                        st.subheader("Database Repair Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            initial_score = fix_results['initial_health']['health_score']['overall']
                            final_score = fix_results['final_health']['health_score']['overall']
                            improvement = fix_results['improvement']
                            
                            st.markdown("### Health Score")
                            st.markdown(f"* Initial: {initial_score:.1f}/100")
                            st.markdown(f"* Final: {final_score:.1f}/100")
                            
                            color = "green" if improvement > 0 else "red" if improvement < 0 else "gray"
                            st.markdown(f"* Improvement: <span style='color:{color};'>{improvement:.1f} points</span>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### Issues Fixed")
                            st.markdown(f"* Memory duplicates: {fix_results['steps']['memory_duplicates']['duplicates_removed']}")
                            st.markdown(f"* Vector duplicates: {fix_results['steps']['vector_duplicates']['duplicates_removed']}")
                            st.markdown(f"* Synced entries: {fix_results['steps']['memory_to_vector_sync']['sync_successes']}")
                            st.markdown(f"* Orphans cleaned: {fix_results['steps']['cleanup_orphans']['cleanup_success']}")
                        
                        with col3:
                            st.markdown("### Remaining Issues")
                            remaining_dups_memory = fix_results['final_health']['duplicates']['memory_db']['duplicates_found']
                            remaining_dups_vector = fix_results['final_health']['duplicates']['vector_db']['duplicates_found']
                            remaining_sync = fix_results['final_health']['sync_issues']['missing_in_vector_db'] + fix_results['final_health']['sync_issues']['missing_in_memory_db']
                            
                            st.markdown(f"* Memory duplicates: {remaining_dups_memory}")
                            st.markdown(f"* Vector duplicates: {remaining_dups_vector}")
                            st.markdown(f"* Sync issues: {remaining_sync}")
                        
                        st.success("Database repair completed successfully!")
                    except Exception as e:
                        st.error(f"Error fixing database issues: {str(e)}")
                        logging.error(f"Fix all issues error: {e}", exc_info=True)
            
            # Enhanced Reset and Resync Vector DB - SINGLE IMPLEMENTATION
            st.markdown("---")
            st.markdown("### 🔄 Reset and Resync Vector Database")
            st.warning("⚠️ **This is a destructive operation that will completely rebuild your vector database!**")
            
            # Initialize session state for reset confirmation
            if 'reset_vector_db_confirmed' not in st.session_state:
                st.session_state.reset_vector_db_confirmed = False
            
            # Display information about the process
            st.markdown("**This process will:**")
            st.markdown("- 🗑️ Delete the entire vector database collection")
            st.markdown("- 🔧 Recreate it with the proper schema")
            st.markdown("- 🔄 Resync all non-reminder memories from SQL database")
            st.markdown("- ⏱️ Take several minutes depending on your data size")
            
            # Confirmation checkbox
            confirm_reset = st.checkbox(
                "✅ I understand this will completely rebuild the vector database",
                key="confirm_vector_reset",
                value=st.session_state.reset_vector_db_confirmed
            )
            
            # Update session state
            st.session_state.reset_vector_db_confirmed = confirm_reset
            
            # Reset button - only enabled when confirmed
            if st.button(
                "🚀 Reset and Resync Vector Database",
                disabled=not confirm_reset,
                key="execute_vector_reset"
            ):
                # Execute the reset operation
                with st.spinner("🔄 Resetting and resyncing vector database..."):
                    try:
                        # Force fresh reload to pick up any code changes without requiring restart
                        import importlib
                        import db_maintenance as _db_maintenance_module
                        importlib.reload(_db_maintenance_module)
                        from db_maintenance import DatabaseMaintenance

                        maintenance = DatabaseMaintenance(
                            vector_db=st.session_state.chatbot.vector_db,
                            memory_db=st.session_state.chatbot.memory_db
                        )
                        
                        # Create progress tracking
                        progress_container = st.container()
                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("🗑️ Step 1/4: Deleting existing vector database...")
                            progress_bar.progress(25)
                            
                            # Execute reset with detailed logging
                            logging.info("Starting vector database reset and resync operation")
                            reset_result = maintenance.reset_and_resync_vector_db()
                            logging.info(f"Reset operation completed with result: {reset_result}")
                            
                            status_text.text("🔧 Step 2/4: Recreating vector database schema...")
                            progress_bar.progress(50)
                            
                            status_text.text("🔄 Step 3/4: Syncing memories from SQL database...")
                            progress_bar.progress(75)
                            
                            status_text.text("✅ Step 4/4: Finalizing and cleanup...")
                            progress_bar.progress(100)
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                        
                        # Display results
                        if reset_result and isinstance(reset_result, dict):
                            st.success("🎉 Vector database reset and resync completed successfully!")
                            
                            # Metrics display
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Total Processed", reset_result.get('total_checked', 0))
                            with col2:
                                st.metric("✅ Successfully Synced", reset_result.get('sync_successes', 0))
                            with col3:
                                st.metric("❌ Failed", reset_result.get('sync_failures', 0))
                            
                            if reset_result.get('reminders_skipped', 0) > 0:
                                st.info(f"ℹ️ {reset_result['reminders_skipped']} reminders were properly excluded")
                            
                            # Reset the confirmation checkbox
                            st.session_state.reset_vector_db_confirmed = False
                            
                            # DISABLED: st.rerun() - this was breaking 128K context window
                            # Original issue: UI refresh was breaking conversation memory during document processing
                            logging.info("Database reset completed - UI refresh disabled to preserve conversation context")
                                
                        else:
                            st.error("❌ Reset process failed - no valid result returned")
                            logging.error(f"Reset returned invalid result: {reset_result}")
                            
                            # Display diagnostic information
                            with st.expander("🔍 Diagnostic Information"):
                                st.write(f"Result type: {type(reset_result)}")
                                st.write(f"Result value: {reset_result}")
                                
                    except Exception as e:
                        st.error(f"❌ Error during reset: {str(e)}")
                        logging.error(f"Vector database reset error: {e}", exc_info=True)
                        
                        with st.expander("🔍 Error Details"):
                            st.exception(e)
                            st.write("**Troubleshooting Steps:**")
                            st.write("1. Check that the vector database is accessible")
                            st.write("2. Verify that the memory database contains data")
                            st.write("3. Check the application logs for more details")
                            st.write("4. Try running a health check first")
            
            # Add spacing and final section
            st.markdown("---")