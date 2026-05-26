import json
import logging
import datetime
import streamlit as st
from typing import Optional

def log_auth_activity(username: str, action: str, success: bool = True, details: str = ""):
    """Log authentication activities to file and console."""
    try:
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "username": username,
            "action": action,  # "login", "logout", "failed_login"
            "success": success,
            "details": details
        }
        
        # Log to auth activity file
        with open("auth_activity.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        # Also log to console
        status = "SUCCESS" if success else "FAILED"
        logging.info(f"AUTH {status}: {username} - {action} - {details}")
        
    except Exception as e:
        logging.error(f"Error logging auth activity: {e}")

def load_user_config():
    """Load user configuration from JSON file."""
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading user config: {e}")
        return None