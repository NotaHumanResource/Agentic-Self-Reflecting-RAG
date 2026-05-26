# admin.py
"""Admin dashboard for DeepSeek system."""

import streamlit as st
import logging
import os
import sys
import time
import datetime
import sqlite3

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from knowledge_gap import KnowledgeGapQueue
from web_knowledge_seeker import WebKnowledgeSeeker
from config import DB_PATH
from autonomous_cognition import AutonomousCognition
from chatbot import Chatbot

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_admin_dashboard():
    """Display system administration dashboard."""
    st.title("AI System Administration")
    
    # Initialize system if needed
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.chatbot = Chatbot()
            st.info("Chatbot initialized")
    
    if 'autonomous_cognition' not in st.session_state:
        with st.spinner("Initializing autonomous cognition..."):
            st.session_state.autonomous_cognition = AutonomousCognition(
                chatbot=st.session_state.chatbot,
                memory_db=st.session_state.chatbot.memory_db,
                vector_db=st.session_state.chatbot.vector_db
            )
            st.session_state.autonomous_cognition_enabled = False
            st.info("Autonomous cognition initialized")
    
    # Display dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Cognitive Status", "Memory Health", "System Control"])
    
    with tab1:
        display_cognitive_status_tab()
    
    with tab2:
        display_memory_health_tab()
    
    with tab3:
        display_system_control_tab()
  

# QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 1 auth/admin cleanup pass).
# Defined as a Streamlit tab handler but display_admin_dashboard() only wires 3 tabs
# (Cognitive Status, Memory Health, System Control) and never references this one.
# Zero string references in main.py, chatbot.py, deepseek.py, autonomous_cognition.py.
# Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
def _UNUSED_display_knowledge_management_tab():
    """Display knowledge management tab in the admin dashboard with real-time updates."""
    import time  # Add time import for unique timestamps
    
    st.header("Knowledge Management")
    
    # Initialize unique key counter to prevent duplicates
    if 'unique_key_counter' not in st.session_state:
        st.session_state.unique_key_counter = 0
    
    # Initialize session state for real-time updates
    if 'knowledge_gaps_updated' not in st.session_state:
        st.session_state.knowledge_gaps_updated = False
    if 'knowledge_gaps_last_refresh' not in st.session_state:
        st.session_state.knowledge_gaps_last_refresh = 0
    
    # Force refresh counter for real-time updates with unique timestamp
    refresh_key = f"{st.session_state.knowledge_gaps_last_refresh}_{int(time.time() * 1000000)}"
    
    # Initialize knowledge gap queue
    gap_queue = KnowledgeGapQueue(st.session_state.chatbot.memory_db.db_path)
    
    # Show update status if changes were made
    if st.session_state.knowledge_gaps_updated:
        st.success("✅ Knowledge gaps updated! Data refreshed automatically.")
        st.session_state.knowledge_gaps_updated = False
    
    # Tab contents
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Knowledge Gaps")
        
        # Get pending gaps (refresh data each time)
        pending_gaps = gap_queue.get_gaps_by_status('pending')
        
        if pending_gaps:
            st.write(f"Found {len(pending_gaps)} pending knowledge gaps")
            
            for gap_index, gap in enumerate(pending_gaps):
                with st.expander(f"{gap['topic']} (Priority: {gap['priority']:.1f})"):
                    st.markdown(f"**Description:** {gap['description']}")
                    st.markdown(f"**Created:** {gap['created_at']}")
                    
                    # Add buttons for manual processing
                    col_web, col_claude = st.columns(2)
                    
                    with col_web:
                        # Generate unique key for web search button
                        st.session_state.unique_key_counter += 1
                        web_button_key = f"web_search_{gap['id']}_{gap_index}_{st.session_state.unique_key_counter}_{refresh_key}"
                        
                        if st.button(f"Fill with Web Search", key=web_button_key):
                            with st.spinner(f"Searching web for {gap['topic']}..."):
                                try:
                                    web_seeker = WebKnowledgeSeeker(
                                        st.session_state.chatbot.memory_db,
                                        st.session_state.chatbot.vector_db,
                                        chatbot=st.session_state.chatbot  # Add chatbot reference for AI processing
                                    )
                                    results = web_seeker.search_for_knowledge(gap['topic'], gap['description'])
                                    
                                    # Display results
                                    if results:
                                        st.success(f"✅ Found {len(results)} knowledge items for '{gap['topic']}'")
                                        
                                        # Mark gap as fulfilled if knowledge was acquired
                                        if gap_queue.mark_fulfilled(gap['id']):
                                            # Trigger real-time update
                                            st.session_state.knowledge_gaps_updated = True
                                            st.session_state.knowledge_gaps_last_refresh += 1
                                        
                                        # Show some details about what was found
                                        st.info(f"📊 Knowledge acquired from {len(results)} sources")
                                        for i, item in enumerate(results[:3], 1):  # Show first 3 items
                                            st.markdown(f"**Source {i}:** {item['source']}")
                                            if 'items_stored' in item:
                                                st.markdown(f"Items stored: {item['items_stored']}")
                                        
                                        if len(results) > 3:
                                            st.markdown(f"... and {len(results) - 3} more sources")
                                            
                                    else:
                                        st.warning(f"⚠️ No knowledge found through web search for '{gap['topic']}'")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error during web search: {str(e)}")
                                    logging.error(f"Web search error in admin: {e}", exc_info=True)
                    
                    with col_claude:
                        # Generate unique key for Claude button
                        st.session_state.unique_key_counter += 1
                        claude_button_key = f"claude_search_{gap['id']}_{gap_index}_{st.session_state.unique_key_counter}_{refresh_key}"
                        
                        if st.button(f"Fill with Claude", key=claude_button_key):
                            with st.spinner(f"Querying Claude for {gap['topic']}..."):
                                try:
                                    claude_integrator = ClaudeKnowledgeIntegration(
                                        st.session_state.chatbot.memory_db,
                                        st.session_state.chatbot.vector_db,
                                        api_key_file="ClaudeAPIKey.txt"
                                    )
                                    success = claude_integrator.integrate_claude_knowledge(gap['topic'], gap['description'])
                                    
                                    if success:
                                        st.success(f"✅ Successfully acquired knowledge from Claude for '{gap['topic']}'")
                                        # Mark as fulfilled
                                        if gap_queue.mark_fulfilled(gap['id']):
                                            # Trigger real-time update
                                            st.session_state.knowledge_gaps_updated = True
                                            st.session_state.knowledge_gaps_last_refresh += 1
                                    else:
                                        st.error(f"❌ Failed to acquire knowledge from Claude for '{gap['topic']}'")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error during Claude integration: {str(e)}")
                                    logging.error(f"Claude integration error in admin: {e}", exc_info=True)
        else:
            st.info("No pending knowledge gaps found")
            
        # Add form to manually add a gap
        with st.form("add_gap_form"):
            st.subheader("Add Knowledge Gap")
            topic = st.text_input("Topic")
            description = st.text_area("Description")
            priority = st.slider("Priority", 0.1, 1.0, 0.5, step=0.1)
            
            submit = st.form_submit_button("Add Gap")
            if submit and topic and description:
                try:
                    gap_id = gap_queue.add_gap(topic, description, priority)
                    if gap_id > 0:
                        st.success(f"✅ Added knowledge gap: '{topic}' with priority {priority}")
                        # Trigger real-time update
                        st.session_state.knowledge_gaps_updated = True
                        st.session_state.knowledge_gaps_last_refresh += 1
                    else:
                        st.error(f"❌ Failed to add knowledge gap: '{topic}'")
                except Exception as e:
                    st.error(f"❌ Error adding knowledge gap: {str(e)}")
                    logging.error(f"Error adding knowledge gap in admin: {e}", exc_info=True)
    
    with col2:
        st.subheader("Fulfilled Gaps")
        
        # Get fulfilled gaps (refresh data each time)
        fulfilled_gaps = gap_queue.get_gaps_by_status('fulfilled')
        
        if fulfilled_gaps:
            st.write(f"Found {len(fulfilled_gaps)} fulfilled knowledge gaps")
            
            for fulfilled_index, gap in enumerate(fulfilled_gaps):
                with st.expander(f"{gap['topic']} (Fulfilled: {gap['fulfilled_at']})"):
                    st.markdown(f"**Description:** {gap['description']}")
                    
                    # Add button to view stored knowledge
                    # Generate unique key for view knowledge button
                    st.session_state.unique_key_counter += 1
                    view_button_key = f"view_knowledge_{gap['id']}_{fulfilled_index}_{st.session_state.unique_key_counter}_{refresh_key}"
                    
                    if st.button(f"View Acquired Knowledge", key=view_button_key):
                        with st.spinner("Retrieving knowledge..."):
                            try:
                                # Search for knowledge about this topic using multiple approaches
                                results = []
                                
                                # Try searching by topic
                                topic_results = st.session_state.chatbot.vector_db.search(
                                    query=gap['topic'],
                                    k=5,
                                    mode="comprehensive"
                                )
                                if topic_results:
                                    results.extend(topic_results)
                                
                                # Try searching by web_knowledge type
                                web_results = st.session_state.chatbot.vector_db.search(
                                    query="",
                                    mode="selective",
                                    metadata_filters={"type": "web_knowledge", "topic": gap['topic']},
                                    k=5
                                )
                                if web_results:
                                    # Add web results if not already included
                                    for web_result in web_results:
                                        if not any(r.get('id') == web_result.get('id') for r in results):
                                            results.append(web_result)
                                
                                if results:
                                    st.subheader(f"📚 Acquired Knowledge for '{gap['topic']}'")
                                    st.info(f"Found {len(results)} knowledge items")
                                    
                                    for i, result in enumerate(results, 1):
                                        with st.expander(f"Knowledge Item {i}"):
                                            st.markdown(f"**Source:** {result.get('metadata', {}).get('source', 'Unknown')}")
                                            
                                            # Show relevance score if available
                                            if 'similarity_score' in result:
                                                st.markdown(f"**Relevance:** {result.get('similarity_score', 0):.2f}")
                                            
                                            # Show content
                                            content = result.get('content', '')
                                            
                                            # Generate unique key for content text area
                                            st.session_state.unique_key_counter += 1
                                            content_key = f"content_{gap['id']}_{fulfilled_index}_{i}_{st.session_state.unique_key_counter}_{refresh_key}"
                                            
                                            if len(content) > 500:
                                                st.text_area("Content Preview", content[:500] + "...", height=150, key=content_key)
                                            else:
                                                st.text_area("Content", content, height=150, key=content_key)
                                            
                                            # Show metadata if available
                                            metadata = result.get('metadata', {})
                                            if metadata:
                                                with st.expander("Metadata"):
                                                    st.json(metadata)
                                else:
                                    st.warning(f"⚠️ No stored knowledge found for topic '{gap['topic']}'")
                                    st.info("This could mean:")
                                    st.markdown("• The knowledge gap was marked as fulfilled but no content was actually stored")
                                    st.markdown("• The content was stored with different metadata tags")
                                    st.markdown("• There was an issue during the knowledge acquisition process")
                                    
                            except Exception as e:
                                st.error(f"❌ Error retrieving knowledge: {str(e)}")
                                logging.error(f"Error retrieving knowledge in admin: {e}", exc_info=True)
        else:
            st.info("No fulfilled knowledge gaps found")
    
    # Add manual refresh section
    st.markdown("---")
    col_refresh, col_stats = st.columns(2)
    
    with col_refresh:
        st.subheader("Manual Controls")
        # Generate unique key for refresh button
        st.session_state.unique_key_counter += 1
        refresh_button_key = f"force_refresh_{st.session_state.unique_key_counter}_{refresh_key}"
        
        if st.button("🔄 Force Refresh Data", key=refresh_button_key):
            st.session_state.knowledge_gaps_last_refresh += 1
            st.success("✅ Data refreshed!")
    
    with col_stats:
        st.subheader("Quick Stats")
        try:
            total_pending = len(gap_queue.get_gaps_by_status('pending'))
            total_fulfilled = len(gap_queue.get_gaps_by_status('fulfilled'))
            
            st.metric("Pending Gaps", total_pending)
            st.metric("Fulfilled Gaps", total_fulfilled)
            
            if total_pending + total_fulfilled > 0:
                completion_rate = total_fulfilled / (total_pending + total_fulfilled) * 100
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
                
        except Exception as e:
            st.error(f"Error calculating stats: {str(e)}")
    
    # Add auto-refresh indicator
    if st.session_state.knowledge_gaps_updated:
        st.balloons()  # Visual feedback for updates

def display_cognitive_status_tab():
    """Display cognitive status information with full autonomous thought content."""
    st.header("Cognitive System Status")
    
    try:
        logging.info("Retrieving cognitive status information")
        
        # Get cognitive status from the autonomous cognition system
        cognitive_status = st.session_state.autonomous_cognition.get_cognitive_status()
        
        # Log the structure for debugging purposes
        logging.debug(f"Retrieved cognitive status data: {cognitive_status}")
        
        # Display status in two columns for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System State")
            
            # Check if is_running exists in the cognitive_status dictionary
            is_running = cognitive_status.get("is_running", False)
            status_color = "#00FF00" if is_running else "#FF5555"  # Green if running, red if stopped
            status_text = "Running" if is_running else "Stopped"
            
            # Safely get other status information with defaults to prevent KeyError
            last_activity = cognitive_status.get('last_activity', 'None')
            uptime = cognitive_status.get('uptime', 'Unknown')
            
            # Display system state in a styled container
            st.markdown(f"""
            <div style="padding:10px; border-radius:5px; background-color:#2E2E2E; margin-top:10px; color: #FFFFFF;">
                <div><b>Status:</b> <span style="color:{status_color};">{status_text}</span></div>
                <div><b>Last Activity:</b> {last_activity}</div>
                <div><b>Uptime:</b> {uptime}</div>
            </div>
            """, unsafe_allow_html=True)
            logging.debug("System state information displayed successfully")
        
        with col2:
            st.subheader("Activity Schedule")
            
            # Safely get the schedule with empty dict as default
            next_runs = cognitive_status.get('next_activity_runs', {})
            
            if not next_runs:
                st.info("No scheduled activities found")
                logging.info("No activity schedule information available")
            else:
                # Display each scheduled activity with color-coded status
                for activity, next_run in next_runs.items():
                    # Green if ready to run, orange if scheduled for later
                    ready_color = "#00FF00" if next_run == "Ready to run" else "#FFAA00"
                    st.markdown(f"""
                    <div style="margin-bottom:5px; color: #FFFFFF;">
                        <b>{activity.replace('_', ' ').title()}:</b> <span style="color:{ready_color};">{next_run}</span>
                    </div>
                    """, unsafe_allow_html=True)
                logging.debug(f"Displayed schedule for {len(next_runs)} activities")
           
    except AttributeError as e:
        # Handle missing attributes in the cognitive status object
        error_msg = f"Missing attribute in cognitive status: {str(e)}"
        st.error("Error displaying cognitive status: Missing component")
        logging.error(error_msg)
    except KeyError as e:
        # Handle missing keys in the cognitive status dictionary
        error_msg = f"Missing key in cognitive status: {str(e)}"
        st.error("Error displaying cognitive status: Missing data")
        logging.error(error_msg)
    except Exception as e:
        # Catch-all for any other unexpected errors
        error_msg = f"Error in display_cognitive_status_tab: {str(e)}"
        st.error("Error displaying cognitive status. See logs for details.")
        logging.error(error_msg)
        # Log the full traceback for debugging complex issues
        import traceback
        logging.error(traceback.format_exc())

def display_memory_health_tab():
    """Display memory health information."""
    st.header("Memory System Health")
    
    if st.button("Analyze Memory Health"):
        with st.spinner("Analyzing memory system health..."):
            # Get health metrics
            health = st.session_state.autonomous_cognition.analyze_memory_health()
            stats = st.session_state.autonomous_cognition.get_memory_stats()
            
            # Display health status
            status_color = {
                "healthy": "green",
                "issues_found": "orange",
                "error": "red"
            }.get(health.get("status", "error"), "red")
            
            st.markdown(f"""
            <h3>Health Status: <span style="color:{status_color};">{health.get('status', 'Unknown').replace('_', ' ').title()}</span></h3>
            """, unsafe_allow_html=True)
            
            # Display issues if any
            if health.get("issues"):
                st.subheader("Issues Detected")
                for issue in health.get("issues", []):
                    st.warning(issue)
            
            # Display recommendations
            if health.get("recommendations"):
                st.subheader("Recommendations")
                for rec in health.get("recommendations", []):
                    st.info(rec)
            
            # Display memory stats
            st.subheader("Memory Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Memories", stats.get("total_memories", 0))
                st.metric("Autonomous Thoughts", stats.get("autonomous_thoughts", 0))
                st.metric("Average Weight", f"{stats.get('avg_weight', 0):.2f}")
                
                # Add confidence audit stats with detailed breakdown
                try:
                    # Search for recent confidence audits in reflection files
                    reflection_path = getattr(st.session_state.autonomous_cognition, 'reflection_path', None)
                    confidence_audits = 0
                    latest_audit_issues = None
                    latest_audit_file = None
                    issue_breakdown = {}
                    audit_content = ""
                    
                    if reflection_path and os.path.exists(reflection_path):
                        # Count confidence audit files
                        audit_files = [f for f in os.listdir(reflection_path) if 'confidence_audit' in f.lower()]
                        confidence_audits = len(audit_files)
                        
                        # Check latest audit for issues
                        if audit_files:
                            # Get most recent audit file
                            audit_files.sort(key=lambda x: os.path.getmtime(os.path.join(reflection_path, x)), reverse=True)
                            latest_audit_file = audit_files[0]
                            latest_file_path = os.path.join(reflection_path, latest_audit_file)
                            
                            try:
                                with open(latest_file_path, 'r', encoding='utf-8') as f:
                                    audit_content = f.read()
                                    audit_content_lower = audit_content.lower()
                                    
                                # Count potential issues in latest audit with breakdown
                                issue_keywords = {
                                    'overconfidence': 0,
                                    'missing attribution': 0,
                                    'opinion/fact confusion': 0,
                                    'scope overreach': 0,
                                    'fabrication risk': 0
                                }
                                
                                for keyword in issue_keywords.keys():
                                    count = audit_content_lower.count(keyword)
                                    if count > 0:
                                        issue_breakdown[keyword] = count
                                
                                latest_audit_issues = sum(issue_breakdown.values())
                                
                            except Exception as file_error:
                                logging.error(f"Error reading audit file: {file_error}")
                    
                    st.metric("Confidence Audits", confidence_audits)
                    
                    if latest_audit_issues is not None:
                        if latest_audit_issues > 0:
                            st.metric("Latest Audit Issues", latest_audit_issues, delta="Issues found", delta_color="inverse")
                            
                            # Add expandable section to show issue breakdown
                            with st.expander(f"View Issue Breakdown from {latest_audit_file}"):
                                st.write("**Issue Types Found:**")
                                if issue_breakdown:
                                    for issue_type, count in issue_breakdown.items():
                                        st.write(f"- {issue_type.title()}: {count} occurrence(s)")
                                else:
                                    st.write("No specific issue keywords found")
                                
                                st.write("---")
                                st.write("**Full Audit Content:**")
                                st.text_area(
                                    "Confidence Audit Details",
                                    value=audit_content,
                                    height=400,
                                    key="latest_audit_content"
                                )
                                
                                st.info("Tip: You can also view this file in the 'Thought Explorer' tab")
                        else:
                            st.metric("Latest Audit Issues", 0, delta="No issues", delta_color="normal")
                            
                except Exception as e:
                    logging.error(f"Error getting confidence audit stats: {e}")
                    st.metric("Confidence Audits", "Error")
            
            with col2:
                # Age distribution as a small chart
                age_dist = stats.get("age_distribution", {})
                if age_dist:
                    st.write("Memory Age Distribution")
                    age_chart_data = {
                        "Age": list(age_dist.keys()),
                        "Count": list(age_dist.values())
                    }
                    st.bar_chart(age_chart_data)
            
            # Memory types breakdown
            st.subheader("Memory Types")
            memory_types = stats.get("memory_types", {})
            
            if memory_types:
                # Convert to format for bar chart
                types_chart_data = {
                    "Type": list(memory_types.keys())[:10],  # Top 10
                    "Count": list(memory_types.values())[:10]
                }
                st.bar_chart(types_chart_data)
   

def display_system_control_tab():
    """
    Display system control interface.
    
    Provides admin-level control over the autonomous cognition system:
    - Start/stop the cognitive thread
    - Adjust per-activity weights and the global cycle interval
    
    Manual activity triggers (run-this-activity-now buttons) are NOT included
    here — those live in the main Autonomous Thinking UI to avoid duplicating
    user-facing controls in the admin dashboard.
    """
    st.header("System Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cognitive System Control")
        
        # Read current state for the toggle button label
        is_running = st.session_state.autonomous_cognition_enabled
        system_status = "Running" if is_running else "Stopped"
        button_label = f"{'Stop' if is_running else 'Start'} Autonomous Cognition"
        
        # Toggle the cognitive thread on/off.
        # Wrapped in try/except because thread start/stop can fail under edge
        # conditions (already-running thread, missing dependencies during init,
        # etc.) and we want to surface the error without breaking the panel.
        if st.button(button_label):
            try:
                # Flip the enabled flag first, then act on the new state
                st.session_state.autonomous_cognition_enabled = not is_running
                
                if st.session_state.autonomous_cognition_enabled:
                    st.session_state.autonomous_cognition.start_cognitive_thread()
                    logging.info("Autonomous cognition started via admin dashboard")
                    st.success("Autonomous cognition started")
                else:
                    st.session_state.autonomous_cognition.stop_cognitive_thread()
                    logging.info("Autonomous cognition stopped via admin dashboard")
                    st.success("Autonomous cognition stopped")
                    
            except Exception as e:
                # Roll back the flag if the actual thread operation failed
                # so the UI state stays in sync with reality
                st.session_state.autonomous_cognition_enabled = is_running
                error_msg = f"Failed to toggle autonomous cognition: {e}"
                logging.error(error_msg, exc_info=True)
                st.error(error_msg)
        
        st.write(f"Current status: {system_status}")
    
    with col2:
        st.subheader("Cognitive Parameters")
        
        # Per-activity weight sliders.
        # Iterates cognitive_activities directly so newly-added activities
        # automatically get sliders without admin.py changes.
        st.write("Activity Weights")
        
        weight_updates = {}
        try:
            for activity, info in st.session_state.autonomous_cognition.cognitive_activities.items():
                current_weight = info["weight"]
                new_weight = st.slider(
                    f"{activity.replace('_', ' ').title()}", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=current_weight,
                    step=0.1,
                    key=f"weight_{activity}"
                )
                
                # Track only the weights that actually changed
                if new_weight != current_weight:
                    weight_updates[activity] = new_weight
        except Exception as e:
            logging.error(f"Error rendering activity weight sliders: {e}", exc_info=True)
            st.error("Could not render activity weight sliders. See logs for details.")
        
        # Global cycle interval slider.
        # Controls how often the cognitive loop ticks (in seconds).
        # Lower = more frequent eligibility checks; cooldowns still enforce
        # per-activity intervals regardless of cycle length.
        try:
            current_interval = st.session_state.autonomous_cognition.cognitive_cycle_interval
            new_interval = st.slider(
                "Cognitive Cycle Interval (seconds)",
                min_value=60,
                max_value=3600,
                value=current_interval,
                step=60
            )
        except Exception as e:
            logging.error(f"Error rendering cycle interval slider: {e}", exc_info=True)
            st.error("Could not render cycle interval slider. See logs for details.")
            new_interval = 3600  # Safe fallback if slider fails to render
        
        # Apply parameter changes button.
        # Calls into autonomous_cognition's own parameter update method which
        # handles validation and returns a status dict.
        if st.button("Apply Parameter Changes"):
            try:
                params = {
                    "activity_weights": weight_updates,
                    "cycle_interval": new_interval
                }
                
                result = st.session_state.autonomous_cognition.adjust_cognitive_parameters(params)
                
                if result.get("status") == "success":
                    logging.info(f"Cognitive parameters updated: {params}")
                    st.success("Parameters updated successfully")
                else:
                    error_detail = result.get('message', 'Unknown error')
                    logging.warning(f"Parameter update failed: {error_detail}")
                    st.error(f"Error updating parameters: {error_detail}")
                    
            except Exception as e:
                logging.error(f"Exception applying parameter changes: {e}", exc_info=True)
                st.error(f"Failed to apply parameter changes: {e}")

# Main function for admin page
def main():
    st.set_page_config(
        page_title="DeepSeek Admin",
        page_icon="🧠",
        layout="wide"
    )
    
    # Add custom CSS for dark mode
    st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        
        /* Make text in markdown elements white */
        p, h1, h2, h3, h4, h5, h6, li, span {
            color: #FFFFFF !important;
        }
        
        /* Style for expanders */
        .streamlit-expanderHeader {
            background-color: #2E2E2E !important;
            color: #FFFFFF !important;
        }
        
        /* Content areas */
        .stTextInput, .stSelectbox, .stSlider, .stTextArea {
            background-color: #2E2E2E;
            color: #FFFFFF;
        }
        
        /* Dashboard tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2E2E2E;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #FFFFFF;
        }
        
        /* Metric containers */
        [data-testid="stMetricValue"] {
            color: #00FF00 !important;
        }
        
        /* Container backgrounds */
        div.stMarkdown {
            background-color: #2E2E2E;
            padding: 10px;
            border-radius: 5px;
        }
        
        /* Success messages in green */
        div.element-container div[data-testid="stText"] {
            color: #00FF00 !important;
        }
        
        /* JSON display */
        pre {
            background-color: #2E2E2E !important;
            color: #00FF00 !important;
        }
        
        /* Code blocks */
        code {
            color: #00FF00 !important;
        }
        
        /* Status displays/indicators */
        div[data-testid="stVerticalBlock"] div[style*="background-color:#f0f7fb"] {
            background-color: #2E2E2E !important;
            color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    display_admin_dashboard()

if __name__ == "__main__":
    main()