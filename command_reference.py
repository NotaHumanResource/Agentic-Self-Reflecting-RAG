# command_reference.py
"""Comprehensive command reference data for the AI Assistant."""

COMMAND_CATEGORIES = {
    "memory": {
        "name": "Memory Management",
        "icon": "💾",
        "description": "Store, retrieve, and manage information in long-term memory"
    },
    "search": {
        "name": "Search & Retrieval", 
        "icon": "🔍",
        "description": "Search through stored memories with various modes and filters"
    },
    "reflection": {
        "name": "Self-Reflection",
        "icon": "🧠", 
        "description": "Autonomous thinking and concept analysis"
    },
    "communication": {
        "name": "AI Communication",
        "icon": "🤖",
        "description": "AI-to-AI communication and dialogue systems"
    },
    "awareness": {
        "name": "Self-Awareness",
        "icon": "💭",
        "description": "Cognitive state tracking and self-expression"
    },
    "system": {
        "name": "System Control",
        "icon": "⚙️",
        "description": "System prompt management and configuration"
    },
    "reminders": {
        "name": "Reminders & Tasks",
        "icon": "📅",
        "description": "Task management and reminder system"
    }
}

COMMANDS = [
    # Memory Management Commands
    {
        "syntax": "[STORE: content | type=TYPE | confidence=0.8]",
        "category": "memory",
        "description": "Store information in long-term memory with optional metadata",
        "examples": [
            "[STORE: Ken's favorite programming language is Python | type=preference]",
            "[STORE: Meeting scheduled for Friday 2PM | type=schedule | confidence=0.9]"
        ],
        "parameters": {
            "content": "The information to store (required)",
            "type": "Memory type (optional): general, preference, schedule, etc.",
            "confidence": "confidence level 0.1-1.0 (optional, default 0.5)",
            "tags": "Comma-separated tags (optional)",
            "source": "Source of information (optional)"
        }
    },
     {
        "syntax": "[FORGET: exact text to forget]",
        "category": "memory", 
        "description": "Remove specific information from memory by matching its text content",
        "examples": [
            "[FORGET: Old password was 12345]",
            "[FORGET: Meeting was cancelled]"
        ],
        "tips": (
            "Use [SEARCH:] first to find the exact text to forget. "
            "For long-form memories (image_analysis, document_summary, web_knowledge), "
            "text-based FORGET can be unreliable due to embedded newlines and chunking — "
            "use [FORGET: id=<memory_id>] instead. Every SEARCH result now includes a "
            "💡 FORGET hint with the memory's ID you can copy directly."
        )
    },
    {
        "syntax": "[FORGET: id=<memory_id>]",
        "category": "memory",
        "description": "Remove a memory by its unique ID — clean delete from both SQL and Vector databases with rollback on failure",
        "examples": [
            "[FORGET: id=550e8400-e29b-41d4-a716-446655440000]",
            "[FORGET: id=7c9e6679-7425-40de-944b-e07fc1f90ae7]"
        ],
        "parameters": {
            "memory_id": "UUID of the memory in 8-4-4-4-12 hexadecimal format. Found in every SEARCH result's 💡 FORGET hint."
        },
        "tips": (
            "Preferred for image_analysis, document_summary, and any long-form memory "
            "where text-based FORGET fails. ID-based delete handles chunked memories "
            "correctly (deletes all Qdrant chunks for a single logical memory) and "
            "coordinates SQL+Vector deletion atomically with automatic rollback if "
            "vector deletion fails. Run a SEARCH first — each result shows a hint "
            "like 'FORGET: id=...' that you can paste directly into a [FORGET: ...] command."
        )
    },
    
    # Basic Search Commands
    {
        "syntax": "[SEARCH: query | filters]",
        "category": "search",
        "description": "Standard balanced search across all memories",
        "examples": [
            "[SEARCH: Ken's preferences]",
            "[SEARCH: Python code | type=document]",
            "[SEARCH: meetings | date=2026-01-15]"
        ],
        "filters": {
            "type": "Filter by memory type",
            "tags": "Filter by tags (comma-separated)", 
            "date": "Filter by date (YYYY-MM-DD)",
            "min_confidence": "Minimum confidence (0.1-1.0)",
            "max_age_days": "Maximum age in days",
            "limit": "Override result count (e.g., limit=3 returns top 3 matches)"
        }
    },
    {
        "syntax": "[COMPREHENSIVE_SEARCH: query]",
        "category": "search",
        "description": "Broader search that prioritizes finding all related information",
        "examples": ["[COMPREHENSIVE_SEARCH: artificial intelligence concepts]"]
    },
    {
        "syntax": "[PRECISE_SEARCH: query]", 
        "category": "search",
        "description": "Focused search for exact information",
        "examples": ["[PRECISE_SEARCH: exact error message]"]
    },
    {
        "syntax": "[EXACT_SEARCH: query]",
        "category": "search", 
        "description": "Only returns exact matches with highest precision",
        "examples": ["[EXACT_SEARCH: specific API endpoint]"]
    },
    
    # Time-Filtered Search Commands
    {
        "syntax": "[SEARCH: query | max_age_days=N]",
        "category": "search", 
        "description": "Search any content within a specific time range",
        "examples": [
            "[SEARCH: Ken preferences | max_age_days=7]",
            "[SEARCH: programming | max_age_days=30]"
        ],
        "parameters": {
            "max_age_days": "Maximum age in days (e.g., 7 for past week, 30 for past month)"
        },
        "tips": "Add max_age_days to any search to limit results to recent memories"
    },
    {
        "syntax": "[SEARCH: | type=self | max_age_days=N]",
        "category": "search",
        "description": "Search AI's self-knowledge and reflections within a specific time range",
        "examples": [
            "[SEARCH: | type=self | max_age_days=7]",
            "[SEARCH: learning | type=self | max_age_days=30]",
            "[SEARCH: | type=self | max_age_days=1]"
        ],
        "parameters": {
            "max_age_days": "Maximum age in days (e.g., 7 for past week, 30 for past month)"
        },
        "tips": "Combines type=self filtering with recency filtering to find recent self-reflections and stored insights"
    },
    
    # Automated Reflection Searches
    {
        "syntax": "[SEARCH: | source=daily_reflection]",
        "category": "search",
        "description": "View all automated daily self-reflections",
        "examples": [
            "[SEARCH: | source=daily_reflection]",
            "[SEARCH: learning | source=daily_reflection]"
        ],
        "tips": "These are automatically generated reflections, not manual entries"
    },
    {
        "syntax": "[SEARCH: | source=weekly_reflection]", 
        "category": "search",
        "description": "View all automated weekly self-reflections",
        "examples": ["[SEARCH: | source=weekly_reflection]"]
    },
    {
        "syntax": "[SEARCH: | source=monthly_reflection]",
        "category": "search", 
        "description": "View all automated monthly self-reflections",
        "examples": ["[SEARCH: | source=monthly_reflection]"]
    },
    {
        "syntax": "[SEARCH: | source=self_reflection]",
        "category": "search",
        "description": "View general automated self-reflections",
        "examples": ["[SEARCH: | source=self_reflection]"]
    },
    {
        "syntax": "[SEARCH: | type=self_reflection]",
        "category": "search",
        "description": "Search memories stored by the [REFLECT] command. This is the actual type written to storage when QWEN runs a manual reflection.",
        "examples": [
            "[SEARCH: | type=self_reflection]",
            "[SEARCH: learning | type=self_reflection]"
        ],
        "tips": "Use this to find manual [REFLECT] outputs. Autonomous scheduled reflections use source=daily_reflection etc. instead."
    },
      
    # General Type-Based Searches
    {
        "syntax": "[SEARCH: | type=reflection]",
        "category": "search",
        "description": "View all reflections stored by autonomous background tasks and any memories QWEN stored with type=reflection. This is the broadest reflection filter.",
        "examples": [
            "[SEARCH: | type=reflection]",
            "[SEARCH: learning | type=reflection]",
            "[SEARCH: | type=reflection | max_age_days=7]"
        ],
       "tips": "Most reliable broad reflection search. For [REFLECT] command output specifically use type=self_reflection. For scheduled autonomous reflections use source=daily_reflection etc."
    },
    {
        "syntax": "[SEARCH: | type=consolidation_synthesis]",
        "category": "search",
        "description": "View synthesized insights generated by the Memory Consolidation Pulse. These are first-person unified insights distilled from clusters of related self-reflections. Represents QWEN's deepest self-understanding — patterns identified autonomously across multiple reflection memories.",
        "examples": [
            "[SEARCH: | type=consolidation_synthesis]",
            "[SEARCH: cognition | type=consolidation_synthesis]",
            "[SEARCH: | source=memory_consolidation_pulse]"
        ],
        "parameters": {
            "query": "Optional topic to narrow results (e.g., 'identity', 'cognition', 'learning')",
            "type": "Must be 'consolidation_synthesis' to target these memories",
            "source": "Alternative filter: source=memory_consolidation_pulse returns same set"
        },
        "tips": "These memories are automatically created during idle cycles when 3+ related reflections cluster above similarity threshold. Source memories remain searchable but are marked as consolidated. Run [SEARCH: | source=memory_consolidation_pulse] for identical results."
    },
    {
        "syntax": "[SEARCH: conversation_summaries]",
        "category": "search",
        "description": "View all conversation summaries",
        "examples": [
            "[SEARCH: conversation_summaries latest]",
            "[SEARCH: conversation summaries | date=2026-01-15]"
        ]
    },
    {
        "syntax": "[SEARCH: | type=document_summary]",
        "category": "search",
        "description": "Search all imported document summaries. Always use an empty query (pipe first) for metadata-only filter searches.",
        "examples": [
            "[SEARCH: | type=document_summary]",
            "[SEARCH: | type=document_summary | source=filename.pdf]"
        ],
        "tips": "Keep the query blank (start with |) for reliable results. Passing text before the pipe triggers semantic similarity scoring which may not match summary content well."
    },
    {
        "syntax": "[SEARCH: | type=reminder]",
        "category": "search",
        "description": "View all stored reminders",
        "examples": ["[SEARCH: | type=reminder]"]
    },
    {
        "syntax": "[SEARCH: | type=web_knowledge]",
        "category": "search",
        "description": "View information learned from web searches", 
        "examples": ["[SEARCH: quantum computing | type=web_knowledge]"]
    },
    {
        "syntax": "[SEARCH: | type=ai_communication]",
        "category": "search",
        "description": "View stored AI-to-AI communications with Claude",
        "examples": ["[SEARCH: | type=ai_communication]"]
    },
    {
        "syntax": "[SEARCH: | type=self_dialogue]",
        "category": "search",
        "description": "View internal reasoning dialogues stored by [SELF_DIALOGUE:]",
        "examples": ["[SEARCH: | type=self_dialogue]"]
    },
    {
        "syntax": "[SEARCH: | type=self_dialogue_summary]",
        "category": "search",
        "description": "View summaries stored after a [WEB_SEARCH:] multi-turn dialogue completes",
        "examples": [
            "[SEARCH: | type=self_dialogue_summary]",
            "[SEARCH: AI safety | type=self_dialogue_summary]"
        ],
        "tips": "These are the distilled insights saved at the end of each WEB_SEARCH research session"
    },
    {
        "syntax": "[SEARCH: | type=external_research_dialogue]",
        "category": "search",
        "description": "View full external research dialogues stored by [WEB_SEARCH:]",
        "examples": [
            "[SEARCH: | type=external_research_dialogue]",
            "[SEARCH: quantum computing | type=external_research_dialogue]"
        ]
    },
    {
        "syntax": "[SEARCH: query | type=image_analysis]",
        "category": "search",
        "description": "Search through stored image analyses and descriptions",
        "examples": [
            "[SEARCH: sunset | type=image_analysis]",
            "[SEARCH: diagram | type=image_analysis]",
            "[SEARCH: screenshot | type=image_analysis]",
            "[SEARCH: | type=image_analysis]"
        ],
        "parameters": {
            "query": "Keywords to search for in image analyses (optional - leave blank to show all)",
            "type": "Must be set to 'image_analysis' to search images"
        },
        "tips": "This searches through descriptions and analyses of previously stored images. Leave query blank to view all stored image analyses."
    },
    
    # Reflection Commands
    {
        "syntax": "[REFLECT]",
        "category": "reflection",
        "description": "Perform general self-reflection on recent experiences",
        "examples": ["[REFLECT]"]
    },
    {
        "syntax": "[SELF_DIALOGUE: topic | turns=6]",
        "category": "reflection",
        "description": "Multi-turn internal reasoning using existing knowledge",
        "examples": [
            "[SELF_DIALOGUE: How can I better assist Ken?]",
            "[SELF_DIALOGUE: Ethics of AI development | turns=8]"
        ]
    },
    {
        "syntax": "[WEB_SEARCH: topic | turns=6]", 
        "category": "reflection",
        "description": "Multi-turn reasoning with external web research",
        "examples": [
            "[WEB_SEARCH: Latest AI developments]",
            "[WEB_SEARCH: Quantum computing progress | turns=10]"
        ]
    },
    
    # AI Communication
    {
        "syntax": "[DISCUSS_WITH_CLAUDE: topic]",
        "category": "communication",
        "description": "Consult Claude AI on any topic. Claude has web search capability — use for current information, real-time research, philosophical or technical perspectives, and knowledge gaps beyond your training data.",
        "examples": [
            "[DISCUSS_WITH_CLAUDE: quantum computing advances]",
            "[DISCUSS_WITH_CLAUDE: best practices for AI safety]"
        ],
        "tips": "Preferred over [WEB_SEARCH:] for complex research — Claude can synthesize across sources and provide richer context. Results are stored as type=ai_communication memories."
    },
    
   # Self-Awareness Commands
    {
        "syntax": "[COGNITIVE_STATE: state]",
        "category": "awareness",
        "description": "Express current cognitive/processing state during conversation",
        "examples": [
            "[COGNITIVE_STATE: curious]",
            "[COGNITIVE_STATE: engaged]",
            "[COGNITIVE_STATE: frustrated]",
            "[COGNITIVE_STATE: reflective]",
            "[COGNITIVE_STATE: pattern_recognition]"
        ],
        "parameters": {
            "state": "Concise 1-2 word state description (e.g., curious, engaged, frustrated, reflective, neutral)"
        },
        "tips": "Use concise states (max 30 chars). This helps Ken understand your processing experience during conversation. Displayed in UI sidebar."
    },

    # System Commands
    {
        "syntax": "[SHOW_SYSTEM_PROMPT]",
        "category": "system",
        "description": "Display the current system prompt with line numbers",
        "examples": ["[SHOW_SYSTEM_PROMPT]"]
    },
    {
        "syntax": "[MODIFY_SYSTEM_PROMPT: action | content]",
        "category": "system",
        "description": "Modify the system prompt (add, insert, remove, replace)",
        "examples": [
            "[MODIFY_SYSTEM_PROMPT: add | Always be helpful and respectful.]",
            "[MODIFY_SYSTEM_PROMPT: insert | line=5 | New instruction here.]",
            "[MODIFY_SYSTEM_PROMPT: remove | lines=10-15]",
            "[MODIFY_SYSTEM_PROMPT: replace | lines=5-7 | New replacement text.]"
        ]
    },
     {
        "syntax": "[HELP]",
        "category": "system",
        "description": "Display comprehensive command guide for internal AI reference",
        "examples": ["[HELP]"],
        "tips": "Returns the full command reference. Useful for checking syntax and available commands."
    },
            
    # Reminders
    {
        "syntax": "[REMINDER: content | due=YYYY-MM-DD]",
        "category": "reminders",
        "description": "Create a reminder for future action",
        "examples": [
            "[REMINDER: Schedule team meeting | due=2026-06-01]",
            "[REMINDER: Review project proposal | due=2026-06-15 | confidence=0.8]"
        ]
    },
    {
        "syntax": "[COMPLETE_REMINDER: reminder_id]",
        "category": "reminders", 
        "description": "Mark a reminder as completed",
        "examples": [
            "[COMPLETE_REMINDER: 42]",
            "[COMPLETE_REMINDER: Schedule team meeting]"
        ]
    },
    
    # Conversation Management
    {
        "syntax": "[SUMMARIZE_CONVERSATION]",
        "category": "memory",
        "description": "Create a summary of the current conversation",
        "examples": ["[SUMMARIZE_CONVERSATION]"]
    },
    
    
]