"""Session management for tracking user sessions and command usage."""
# Thin wrapper used during chatbot initialization to:
#   1. Generate the session UUID that propagates to chatbot.session_id
#      and deepseek_enhancer.session_id
#   2. Surface lifetime counter stats for startup logging via get_session_summary()
# Note: end_session() was removed — it had no callsite in chatbot.py or main.py.

import uuid
import datetime
import logging
from typing import Optional


class SessionManager:
    """Generates and tracks the current session ID, surfaces lifetime counter stats for logging."""

    def __init__(self, lifetime_counters):
        """Initialize session manager with lifetime counters reference."""
        self.lifetime_counters = lifetime_counters
        self.current_session_id = None
        self.session_start_time = None

    def start_new_session(self) -> str:
        """
        Start a new session, generate a UUID, and return the session ID.
        The returned ID is assigned to chatbot.session_id and
        deepseek_enhancer.session_id by the caller.
        """
        self.current_session_id = str(uuid.uuid4())
        self.session_start_time = datetime.datetime.now()
        logging.info(f"Started new session: {self.current_session_id}")
        return self.current_session_id

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 1 auth/admin cleanup pass).
    # Getter for self.current_session_id — neither the method nor the bare attribute
    # is referenced anywhere outside session_manager.py itself. SessionManager is only
    # instantiated in chatbot.py and only start_new_session() + get_session_summary() are used.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_get_current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.current_session_id

    def get_session_summary(self) -> dict:
        """
        Get a summary of the current session for startup logging.
        Returns command breakdown from lifetime_counters for the active session.
        """
        if not self.current_session_id:
            return {}

        try:
            session_counters = self.lifetime_counters.get_session_counters(self.current_session_id)
            total_commands = sum(session_counters.values())

            return {
                'session_id': self.current_session_id,
                'start_time': self.session_start_time.isoformat() if self.session_start_time else None,
                'duration_minutes': (
                    (datetime.datetime.now() - self.session_start_time).total_seconds() / 60
                    if self.session_start_time else 0
                ),
                'total_commands': total_commands,
                'command_breakdown': session_counters
            }
        except Exception as e:
            logging.error(f"Error getting session summary: {e}")
            return {}