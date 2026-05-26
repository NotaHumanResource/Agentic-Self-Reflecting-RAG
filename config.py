# config.py - CORRECTED for RTX 5090 32GB + 24-core i9 + 64GB RAM
"""Configuration settings optimized for QWEN3.6 with 128000 context preservation."""
import os
import sys 
import logging
from logging.handlers import TimedRotatingFileHandler
# Windows-safe rotating file handler with file-locking support.
# Replaces TimedRotatingFileHandler which fails on Windows at midnight rollover
# when log files are held open by multiple threads (PermissionError WinError 32).
from concurrent_log_handler import ConcurrentRotatingFileHandler  
from datetime import datetime  

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ollama_logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging with UTF-8 encoding AND daily rotation
log_file_path = os.path.join(logs_dir, 'ollama_context.log')

# Create Windows-safe rotating file handler with UTF-8 encoding.
# Replaces TimedRotatingFileHandler to eliminate the midnight WinError 32
# file-lock cascade that floods logs with PermissionError tracebacks when
# multiple threads (TTS, autonomous cognition, wake word listener, Streamlit
# script runner) hold file handles open during rollover. ConcurrentRotatingFileHandler
# uses proper file locking so rotation succeeds regardless of thread activity.
# Rotation is now size-based (25MB) instead of time-based (midnight).
try:
    file_handler = ConcurrentRotatingFileHandler(
        log_file_path,
        maxBytes=25 * 1024 * 1024,   # 25 MB per file (Notepad-friendly size)
        backupCount=20,               # Keep 20 rotated files (~500MB total retention)
        encoding='utf-8'
    )
except Exception as e:
    # Fail loudly and clearly if handler init breaks. Logging config errors
    # are notoriously hard to debug if they fall back silently to defaults.
    print(f"❌ Failed to initialize ConcurrentRotatingFileHandler for {log_file_path}: {e}")
    raise

# Create console handler with proper Windows encoding handling
console_handler = logging.StreamHandler(sys.stdout)

# WINDOWS-SPECIFIC FIX: Handle console encoding properly
if sys.platform.startswith('win'):
    try:
        # Try to reconfigure console to UTF-8
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        # Fallback for older Python versions or restricted consoles
        console_handler = logging.StreamHandler()
        console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)

# Configure formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Set handler levels
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)

# Configure root logger properly for rotating logs
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear any existing handlers to prevent duplicates
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add our configured handlers
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Create module logger
logger = logging.getLogger(__name__)

# ============================================================================
# STREAMLIT LOG HANDLER — Separate rotating log for crash diagnostics
# Writes to C:\Users\kenba\.streamlit\logs\streamlit_app.log
# Rotates daily at midnight, retains 7 days, UTF-8 encoded
# ============================================================================

# Resolve the Streamlit logs directory dynamically using the user's home folder
# This handles any machine/user without hardcoding the path
streamlit_logs_dir = os.path.join(os.path.expanduser("~"), ".streamlit", "logs")

try:
    # Create the directory if it doesn't exist yet
    os.makedirs(streamlit_logs_dir, exist_ok=True)

    streamlit_log_path = os.path.join(streamlit_logs_dir, "streamlit_app.log")

    # Create a separate Windows-safe rotating file handler for Streamlit logging.
    # Same fix as the main file_handler above — replaces TimedRotatingFileHandler
    # to eliminate the midnight WinError 32 cascade. This handler was the primary
    # source of the PermissionError flood seen on 2026-05-09 after midnight.
    streamlit_file_handler = ConcurrentRotatingFileHandler(
        streamlit_log_path,
        maxBytes=25 * 1024 * 1024,   # 25 MB per file (Notepad-friendly size)
        backupCount=20,               # Keep 20 rotated files (~500MB total retention)
        encoding='utf-8'
    )
    streamlit_file_handler.setLevel(logging.INFO)
    streamlit_file_handler.setFormatter(formatter)  # Reuse existing formatter

    # Add to root logger so all modules write to it automatically
    root_logger.addHandler(streamlit_file_handler)

    logger.info(f"✅ Streamlit log handler initialized: {streamlit_log_path}")
    logger.info("🔄 Streamlit log rotation: 25MB size-based, 20 file retention (Windows-safe locking)")

except Exception as e:
    # Don't let log setup failure crash the whole application
    logger.error(f"❌ Failed to initialize Streamlit log handler: {e}", exc_info=True)


# Test Unicode logging capability and confirm rotation is working
logger.info("✅ Unicode logging initialized successfully - AI can now track memory operations")
logger.info("🔄 Log rotation enabled - 25MB size-based rotation (Windows-safe locking)")

# Log file information for verification
logger.info(f"📁 Log file location: {log_file_path}")
logger.info(f"📅 Log rotation: Size-based at 25MB, suffix format: .1, .2, .3...")
logger.info(f"🗃️ Backup retention: {file_handler.backupCount} files (~500MB total)")

# Directory paths (unchanged)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "LocalDocs")
DB_PATH = os.path.join(BASE_DIR, "LongTermMemory_data.db")

# Qdrant settings (unchanged)
QDRANT_LOCAL_PATH = os.path.join(BASE_DIR, "qdrant_storage")
QDRANT_COLLECTION_NAME = "deepseek_memory_optimized"
QDRANT_USE_LOCAL = False
QDRANT_SERVER_HOST = "localhost"
QDRANT_SERVER_PORT = 6333
QDRANT_URL = "http://localhost:6333"

# Knowledge Gaps Vector Collection
# Used for semantic duplicate detection when creating knowledge gaps
QDRANT_GAPS_COLLECTION_NAME = "knowledge_gaps_embeddings"


QDRANT_CONFIG = {
    # ============= CONNECTION SETTINGS =============
    "host": QDRANT_SERVER_HOST,
    "port": QDRANT_SERVER_PORT,
    "timeout": 30.0,
    "prefer_grpc": True,
    "https": False,
    "api_key": None,
    
    # ============= COLLECTION SETTINGS =============
    "collection_name": QDRANT_COLLECTION_NAME,
    "local_path": QDRANT_LOCAL_PATH,
    "use_local": QDRANT_USE_LOCAL,
    
    # ============= VECTOR CONFIGURATION =============
    "vector_size": 4096,          # Matches QWEN embedding dimensions
    "distance": "Cosine",         # Best for semantic similarity (normalized vectors)
    
    # ============= HNSW INDEX CONFIGURATION =============
    # Optimized for 13K-150K vectors with balanced performance
    "hnsw_config": {
        "m": 24,                  # Links per node (16=default, 24=balanced, 32=high-quality)
        "ef_construct": 150,      # Index build quality (100=default, 150=balanced, 200=high)
        "full_scan_threshold": 20000,  # Use exact search below this point count
    },
    
    # ============= QUANTIZATION (75% MEMORY REDUCTION) =============
    # Reduces memory from ~40MB to ~10MB with <3% accuracy loss
    "quantization_config": {
        "scalar": {
            "type": "int8",       # Compress float32 (4 bytes) to int8 (1 byte)
            "quantile": 0.99,     # Clip outliers at 99th percentile
            "always_ram": True    # Keep quantized vectors in RAM for speed
        }
    },
    
    # ============= PAYLOAD INDEXING =============
    # Indexes nested fields inside metadata dict for fast filtering
    # Uses dot notation: "metadata.field_name"
    "payload_schema": {
        # Temporal filtering - both formats available
        "metadata.created_at": "keyword",      # ISO 8601 string (human-readable)
        "metadata.timestamp_unix": "integer",  # Unix timestamp (fast range queries)
                                               # Use this for "last week/month" filters
        
        # Categorical filtering
        "metadata.type": "keyword",       # Memory type (exact match)
        "metadata.source": "keyword",     # Memory origin (exact match)
        
        # Numerical filtering
        "metadata.confidence": "float",   # Confidence score (range queries)
        
        # Tag filtering
        "metadata.tags": "keyword",       # Tag string (exact match on full string)
    },
    
    # ============= STORAGE OPTIMIZATION =============
    "on_disk_payload": False,     # Keep all payloads in RAM (faster access)
                                  # Set True only if you exceed 500K vectors
    "memmap_threshold": 50000,    # Auto-enable memory mapping above 50K vectors
    
    # ============= SEARCH PARAMETERS =============
    # Used during query_points() operations
    "search_params": {
        "hnsw_ef": 96,            # Search accuracy (64=fast, 96=balanced, 128=high)
        "score_threshold": 0.35,  # Minimum similarity score (0.0-1.0)
        "exact": False,           # Use HNSW approximation (auto-switches below threshold)
    },
    
    # ============= BATCH OPERATIONS =============
    "batch_size": 100,            # Vectors per batch during bulk operations
    "parallel_requests": 4,       # Concurrent API requests for bulk ops
    
    # ============= RERANKING STRATEGY =============
    "use_reranking": True,        # Two-stage search: retrieve more, rerank top
    "rerank_multiplier": 3,       # Retrieve 3x candidates before reranking
                                  # Example: Want top 5? Retrieve 15, rerank to 5
}


# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
# OLLAMA_MODEL = "huihui_ai/Qwen3.6-abliterated:27b" 
# OLLAMA_MODEL = "huihui_ai/Qwen3.6-abliterated:35b" moe MODEL good at broad knowledge less reasoning and command consistency.
# OLLAMA_MODEL = "blackgrg26/WORMGPT-12:latest"
OLLAMA_MODEL = "tinyrick/gemma-4-31B-it-uncensored-heretic-llmfan46:Q4_K_M"  

# Embedding model used for vector operations (Qdrant inserts and similarity search)
# Slim variant uses num_ctx=2048 (vs upstream 40960) to allow dual-model resident loading
# on 32GB GPU. Built from qwen3-embedding:8b via custom Modelfile. Same 4096-dim output.
EMBEDDING_MODEL = "qwen3-embedding:slim"
# ============================================================================
# VISION MODELS — Image and Video Analysis
# ============================================================================
# Multimodal model used by the image analysis UI feature (image_processor.py).
# Defaults to the same model as OLLAMA_MODEL because huihui_ai/Qwen3.6-abliterated:27b
# is multimodal and handles vision natively. Kept as an independent constant so
# image analysis can be swapped without affecting the primary chat model
# (e.g., if a smaller/faster image-only model becomes preferable later).
IMAGE_MODEL = "huihui_ai/Qwen3.6-abliterated:27b"

# Multimodal model used by the video analysis UI feature (video_processor.py).
# Same default as IMAGE_MODEL — the dense 27B handles both image and video frame
# analysis. Decoupled so video processing can be swapped independently if needed.
VIDEO_MODEL = "huihui_ai/Qwen3.6-abliterated:27b"








# CORRECTED MODEL PARAMETERS - Aligned with hardware limits
MODEL_PARAMS = {
    # Context Window Configuration
    "num_ctx": 65536,           # Context window size in tokens (131K full capacity)
                                # Controls how much conversation history the model remembers
                                # Larger = better long-term memory, more VRAM usage
    
    # Response Creativity Control
    "temperature": 0.6,         # Controls randomness in responses (0.0-2.0 range)
                                # 0.7 = balanced creativity vs consistency
                                # Lower = more predictable, Higher = more creative/random
    
    # Response Length Limit
    "num_predict": 4608,        # Maximum tokens the model can generate in one response
                                # 4096 tokens ≈ 3000-6000 words depending on content
                                # Prevents runaway generation and controls response time
    
    # Token Selection Parameters
    "top_k": 20,               # Qwen3 is tuned for tighter sampling
                               # Smaller values = more focused responses
                               # Larger values = more diverse word choices
    
    "top_p": 0.95,             # Nucleus sampling - considers tokens totaling 95% probability
                               # Works with top_k to filter unlikely tokens
                               # 0.95 = good balance between quality and variety
    
    "presence_penalty": 0.3,    # Penalty for tokens already in context (0.0-2.0 range)
                                # OVERRIDE: Qwen3.6 base default is 1.5 — too aggressive for
                                # agentic RAG where command syntax ([SEARCH:], [STORE:]) and
                                # system prompt tokens MUST be re-emitted reliably across
                                # turns. 0.3 retains a small anti-loop nudge without fighting
                                # the architecture. See agentic-tuning discussion 2026-05-21.
    
    # HARDWARE OPTIMIZED SETTINGS
    "seed": -1,                 # Use a random seed each time, so responses will vary between runs
    "num_batch": 2048,          # Conservative for GPU memory changed 
    "num_thread": 24,           # CORRECTED: Use all 24 cores
    "num_gpu": 1,               # Singe GPU  
    "num_gqa": 8,           # optimal setting for Gemma models (matches their architecture)
    "split_mode": 0,        # # 0 = no splitting, keep layers together (optimal for single GPU)
    "num_gpu_layers": -1,   # Put ALL layers on GPU
    "main_gpu": 0,          # Use GPU 0 as primary
    
    # MEMORY OPTIMIZATION - Corrected for 32GB VRAM
    "use_mlock": False, # prevents the operating system from swapping memory pages to disk
    "use_mmap": True,   # only loads parts of the model into memory when actually needed
    "numa": False,      # Non-Uniform Memory Access) multi cpu systems where CPU times vary
    "low_vram": False,  # For systems with limited VRAM we have 32GB VRAM
    
       
    # QWEN-specific stop tokens
    "stop": [
    "<|endoftext|>",        # Qwen's primary end token
    "<|im_end|>",           # ChatML format end token
    "<|im_start|>",         # ChatML format start token (prevents continuation)
    "\n\nHuman:",           # Keep for compatibility
    "\n\nUser:",            # Keep for compatibility
    ]

}

# MINIMAL ATTENTION CONFIG - Full Ollama control
ATTENTION_CONFIG = {
    "max_conversation_length": 128000,    # Set the limit
    "context_compression_ratio": 0.0,     # No compression
    "system_message_preservation": True,  # Keep system prompts
    # Let Ollama handle everything else automatically
}

# CORRECTED HARDWARE CONFIG - Match current hardware
HARDWARE_CONFIG = {
    # GPU Memory Management
    "gpu_memory_fraction": 0.95,          # Allocates 90% of GPU VRAM (29GB of 32GB total)
                                          # Leaves 10% buffer for Windows, thermal safety, and dynamic operations
                                          # Higher values risk OOM errors, lower values waste performance
    
    # CPU Resource Allocation  
    "cpu_threads": 24,                    # Uses all 24 CPU cores of your Intel i9 processor
                                          # Handles preprocessing, tokenization, and non-GPU operations
                                          # Should match your physical core count for optimal performance
    
    # System RAM Configuration
    "memory_pool_size": "52GB",           # Allocates 48GB of your 64GB system RAM for model operations
                                          # Used for model loading, document processing, and CPU-side operations
                                          # Leaves 16GB for Windows and other applications
    
    # Attention Mechanism Optimization
    "flash_attention_enabled": True,      # Enables Flash Attention algorithm for memory-efficient processing
                                          # Reduces memory usage by ~40% during long context operations
                                          # Essential for 128K context window without memory overflow
    
    # Model Layer Distribution
    "gpu_layers": -1,                     # Loads ALL model layers onto GPU (-1 = unlimited/all layers)
                                          # Ensures 100% GPU utilization instead of CPU fallback
                                          # Alternative: positive number = specific layer count on GPU
    
    # Processing Optimization
    "parallel_processing": True,          # Enables parallel processing across multiple CPU threads
                                          # Speeds up document chunking, embeddings, and preprocessing
                                          # Works in conjunction with cpu_threads setting
    
    # Batch Processing Control
    "batch_size_multiplier": 1,           # Conservative multiplier for batch processing (1x = normal batch sizes)
                                          # Higher values = larger batches = more throughput but more VRAM usage
                                          # Set to 1 for stability, can increase to 2-4 if you have VRAM headroom
    
    # Key-Value Cache Memory
    "kv_cache_size": "16GB",              # Use more VRAM for longer context retention
                                          # Larger cache = longer conversation memory without recomputation
                                          # Optimal size balances context length vs available VRAM
                                          # 9GB supports ~90K token context with your hardware configuration
}

# CORRECTED DOCUMENT PROCESSING - Realistic limits
DOCUMENT_PROCESSING = {
    "max_concurrent_files": 4,            # FIXED: More conservative
    "chunk_cache_size": "8GB",            # FIXED: Realistic for system RAM
    "parallel_chunking": True,
    "max_chunk_workers": 12,              # FIXED: Half your cores
    "context_aware_chunking": True,
}

OODA_ENABLED_DEFAULT = False
OODA_MAX_CYCLES = 20

# CORRECTED OLLAMA ENVIRONMENT - Consistent with batch file
OLLAMA_ENV_CONFIG = {
    # Flash Attention Control
    "OLLAMA_FLASH_ATTENTION": "1",        # Enables Flash Attention algorithm (1=on, 0=off)
                                          # Must be string "1" not boolean True for environment variables
                                          # Reduces memory usage during attention computation by ~40%
                                          # Critical for handling 131K context windows efficiently
    
    # GPU Memory Management
    "OLLAMA_GPU_MEMORY_FRACTION": "0.95", # Tells Ollama to use 95% of available GPU memory
                                          # String format required for environment variable
                                          # Prevents Ollama from over-allocating and causing crashes
    
    # Model Loading Limits
    "OLLAMA_MAX_LOADED_MODELS": "2",      # Maximum number of models kept in VRAM simultaneously
                                          # Actually allows both QWEN + QWEN embeddings they're different model types
                                          # "1" ensures primary model (QWEN) gets maximum memory allocation
                                          # Higher values would split VRAM reducing individual model performance
    
    # Parallel Request Handling
    "OLLAMA_NUM_PARALLEL": "1",           # Number of parallel inference requests Ollama can handle
                                          # "1" = process one request at a time for maximum performance per request
                                          # Higher values = concurrent requests but slower individual responses
                                          # Conservative setting prevents memory contention and ensures stability
    
    # CPU Thread Allocation
    "OLLAMA_CPU_THREADS": "24",           # Number of CPU threads Ollama can use for preprocessing
                                          # Matches your Intel i9 24-core processor for optimal utilization
                                          # Used for tokenization, text processing, and non-GPU operations
                                          # String format required for environment variables
    
    # Key-Value Cache Size (CORRECTED)
    "OLLAMA_KV_CACHE_SIZE": "16384",      # 16GB - gives model more cache headroom
                                          # UPDATED: Was 11264 MB (11GB), now optimized for your hardware
                                          # 9GB supports ~90K active token context with efficient memory usage
                                          # Balances context length capability with VRAM availability
    
    # Batch Processing Size
    "OLLAMA_BATCH_SIZE": "2048",         # Larger batches = more VRAM usage but better performance
                                          # Larger batches = higher throughput but more VRAM usage
                                          # 4096 is optimal balance for RTX 5090 with 27B parameter model
                                          # Reduces to 2048 if experiencing memory pressure warnings
    
    # Context Window Size (CONFIRMED CORRECT)
    "OLLAMA_CONTEXT_SIZE": "65536",      # Maximum context window in tokens (131K tokens exactly)
                                          # VERIFIED: Matches ollama show output for gemma3:27b
                                          # Enables full context capability of the model
                                          # Critical for autonomous AI applications requiring long memory

    # Model Persistence in VRAM
    "OLLAMA_KEEP_ALIVE": "24h",           # How long to keep models loaded after last use
                                          # Default is 5m which causes premature eviction and
                                          # forces a fresh model load when chat resumes after
                                          # embedding-heavy work (the cause of the 23-minute hang
                                          # observed on 2026-05-05 Run #5). 24h keeps both QWEN
                                          # and the embedding model warm; Ollama still swaps
                                          # them on demand when both can't fit in VRAM, but it
                                          # will not proactively evict on a timer.
}

# SIMPLIFIED DEBUG CONFIG - Essential logging only
DEBUG_CONFIG = {
    "enable_context_logging": True,       # ENABLED: Help debug truncation
    "log_attention_weights": False,
    "track_token_usage": True,            # ENABLED: Monitor context usage
    "monitor_context_drift": True,        # ENABLED: Track gradual forgetting
    "context_diagnostic_interval": 100,   # Every 100 messages
}

# Document processing settings (unchanged)
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.md', '.rtf'}
DEFAULT_CHUNK_SIZE = 1000

# ============================================================================
# REFLECTION STORAGE SETTINGS
# ============================================================================
# Control whether daily/weekly/monthly reflections are stored directly in databases
# or preserved only through conversation summaries (recommended for efficiency)
STORE_REFLECTIONS_SEPARATELY = True  # Set to True to enable direct database storage
                                       # Set to False to store only via conversation summaries (default)
                                       # Reflections are injected into chat and preserved when auto-summarized

# ============================================================================
# MODEL PARAMETER VALIDATION
# ============================================================================

def validate_model_parameters():
    """Validate model parameters against hardware constraints."""
    logger.info("Validating model parameters...")
    
    # Validate context size
    expected_context = 65536  # Update validation check
    actual_context = MODEL_PARAMS.get("num_ctx", 0)
    
    if actual_context != expected_context:
        logger.error(f"Context size mismatch! Expected: {expected_context}, Got: {actual_context}")
        return False
    else:
        logger.info(f"✓ Context size validated: {actual_context} tokens (131K)")
    
    # UPDATED: Validate GPU memory settings for our optimized config
    gpu_fraction = HARDWARE_CONFIG.get("gpu_memory_fraction", 0)
    if gpu_fraction > 0.95:  # 0.95 is safe on dedicated 32GB card
        logger.warning(f"GPU memory fraction ({gpu_fraction}) may be too high for dual-model setup")
    else:
        logger.info(f"✓ GPU memory fraction: {gpu_fraction} (optimized for RTX 5090)")
    
    # UPDATED: Validate KV cache size matches our research
    kv_cache = HARDWARE_CONFIG.get("kv_cache_size", "0GB")
    kv_cache_gb = int(kv_cache.replace("GB", ""))
    if kv_cache_gb >= 9:  # anything 9GB+ is valid for your setup
        logger.info(f"✓ KV cache size: {kv_cache} (optimized for dual-model setup)")
    else:
        logger.warning(f"KV cache size ({kv_cache}) - below 9GB minimum recommendation")
    
    # Validate CPU threads
    cpu_threads = HARDWARE_CONFIG.get("cpu_threads", 0)
    if cpu_threads != 24:
        logger.warning(f"CPU threads ({cpu_threads}) doesn't match available cores (24)")
    else:
        logger.info(f"✓ CPU threads: {cpu_threads} (all cores)")
    
    # Log attention configuration
    max_conv_length = ATTENTION_CONFIG.get("max_conversation_length", 0)
    if max_conv_length != expected_context:
        logger.error(f"Conversation length mismatch! Expected: {expected_context}, Got: {max_conv_length}")
        return False
    else:
        logger.info(f"✓ Max conversation length: {max_conv_length} tokens")
    
    # NEW: Validate dual-model configuration
    max_models = int(OLLAMA_ENV_CONFIG.get("OLLAMA_MAX_LOADED_MODELS", "1"))
    if max_models >= 2:
        logger.info(f"✓ Dual-model setup: {max_models} models (QWEN + QWEN embeddings)")
    else:
        logger.warning(f"Single model setup - QWEN embeddings may need manual loading")
    
    logger.info("Model parameter validation completed successfully!")
    return True

# SLEEPING INFRASTRUCTURE 2026-05-19 (batch 6 utility cleanup review):
# Currently 0 callers — NOT quarantined. Module-level diagnostic logger that reports current
# token usage as percentage of max_tokens (default 128000), with WARNING at 90% and ERROR at
# 97%. Would be useful in chatbot's response loop to log context pressure before/after each
# turn, helping diagnose unexpected truncation or summarization triggers. Could be paired
# with the auto-summarization chain in conversation_summary_manager.py if that's revived.
def log_context_usage(current_tokens, max_tokens=128000):
    """Log current context usage for monitoring."""
    usage_percent = (current_tokens / max_tokens) * 100
    logger.info(f"Context usage: {current_tokens}/{max_tokens} tokens ({usage_percent:.1f}%)")
    
    if usage_percent > 90:
        logger.warning(f"High context usage: {usage_percent:.1f}%")
    elif usage_percent > 97:
        logger.error(f"Critical context usage: {usage_percent:.1f}% - Truncation risk!")

# SLEEPING INFRASTRUCTURE 2026-05-19 (batch 6 utility cleanup review):
# Currently 0 runtime callers — NOT quarantined. Only invoked when config.py is run as a
# standalone script via the __main__ block at the bottom of this file. Checks GPU memory
# via nvidia-smi and system RAM via psutil, returns True if either is under pressure (>90%
# GPU or >85% RAM). Useful for diagnosing OOM events or for adaptive context-size selection
# at runtime. Could be wired into autonomous cognition's scheduled tasks for proactive
# pressure monitoring.
def check_memory_pressure():
    """Check for memory pressure that might cause context truncation."""
    try:
        import psutil
        import subprocess
        
        # Check GPU memory if nvidia-smi available
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(','))
                gpu_usage = (used / total) * 100
                logger.info(f"GPU memory usage: {used}MB/{total}MB ({gpu_usage:.1f}%)")
                
                if gpu_usage > 90:
                    logger.warning(f"High GPU memory usage: {gpu_usage:.1f}%")
                    return True
        except Exception as e:
            logger.debug(f"Could not check GPU memory: {e}")
        
        # Check system RAM
        ram = psutil.virtual_memory()
        ram_usage = ram.percent
        logger.info(f"System RAM usage: {ram_usage:.1f}%")
        
        if ram_usage > 85:
            logger.warning(f"High system RAM usage: {ram_usage:.1f}%")
            return True
            
    except ImportError:
        logger.debug("psutil not available for memory monitoring")
    except Exception as e:
        logger.debug(f"Memory pressure check failed: {e}")
    
    return False

# ============================================================================
# INITIALIZATION
# ============================================================================

# Run validation on import
if __name__ == "__main__":
    validate_model_parameters()
    check_memory_pressure()
else:
    # Run validation when imported
    validate_model_parameters()