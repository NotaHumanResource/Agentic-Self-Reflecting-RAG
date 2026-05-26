# web_knowledge_seeker.py
"""
Enhanced web knowledge seeker integration and AI-driven content selection.

Active search engines: StartPage (primary), Wikipedia (fallback/API).
DuckDuckGo, Brave, and Bing were removed — binary response issues (DDG),
no active search method (Brave), and no API key setup (Bing).
"""
# import datetime  # DEAD CODE TEST 2026-05-17: shadowed by line 22 'from datetime import datetime' (ruff F811)
# from importlib.metadata import metadata  # DEAD CODE TEST 2026-05-17: duplicate, unused (ruff F401)
# from importlib.metadata import metadata  # DEAD CODE TEST 2026-05-17: duplicate, unused (ruff F401)
import logging
import requests
import time
import re
import random  
import uuid
import json
import os
import io                    # For wrapping PDF bytes in-memory without temp files
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta 
from bs4 import BeautifulSoup
from urllib.parse import urlparse  # DEAD CODE TEST 2026-05-17: was 'urljoin, urlparse' — urljoin unused per ruff F401

# PDF extraction — same library used by document_reader.py
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("⚠️ PyPDF2 not available — PDF URLs will be skipped")


# Custom exception for runtime PDF detection — raised by fetch strategies
# when _validate_content_type() returns the "PDF_CONTENT" signal, so the
# caller (_fetch_webpage_content_enhanced) can break out of the HTML
# strategy loop and route to _fetch_pdf_content() instead of retrying
# and eventually blacklisting the domain incorrectly.
class PDFContentDetected(Exception):
    """Raised when a fetch strategy detects PDF Content-Type at runtime."""
    pass


class WebKnowledgeSeeker:
    """Enhanced web knowledge seeker with multiple search engines and anti-blocking measures."""
    
    def __init__(self, memory_db, vector_db, chatbot=None):
        """
        Initialize the enhanced web knowledge seeker.
        
        Args:
            memory_db: Memory database instance for storing knowledge
            vector_db: Vector database instance for semantic search
            chatbot: Optional chatbot instance for AI-based content extraction
            
        Raises:
            ValueError: If required dependencies are None or invalid
        """
        try:
            # Validate required dependencies
            if memory_db is None:
                raise ValueError("memory_db cannot be None")
            if vector_db is None:
                raise ValueError("vector_db cannot be None")
                
            self.memory_db = memory_db
            self.vector_db = vector_db
            self.chatbot = chatbot
            
            # Initialize requests session with error handling
            try:
                self.session = requests.Session()
            except Exception as session_error:
                logging.error(f"Failed to create requests session: {session_error}")
                raise
            
            # Enhanced configuration
            self.min_request_interval = 3  # Increased base delay
            self.max_request_interval = 7  # Maximum delay range
            self.last_request_time = 0
            self.min_content_length = 50  # Minimum content length filter
            
            # Text quality thresholds for garbage detection
            self.max_replacement_char_ratio = 0.01  # Maximum 1% replacement characters allowed
            self.max_non_printable_ratio = 0.05  # Maximum 5% non-printable characters
            
            # Domain blacklist management
            self.blacklist_file = 'failed_domains.json'
            self.blacklist_duration = 24 * 60 * 60  # 24 hours in seconds
            
            # Load blacklist with error handling
            try:
                self.failed_domains = self._load_blacklist()
            except Exception as blacklist_error:
                logging.error(f"Error loading blacklist, starting with empty: {blacklist_error}")
                self.failed_domains = {}
            
            # Config-blocked domains — user-editable JSON, loaded once at startup.
            # These are structurally unscrapable domains (social media SPAs, etc.)
            # that should never enter the fetch pipeline regardless of blacklist state.
            self.blocked_domains_config_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'search_blocked_domains.json'
            )
            try:
                self.config_blocked_domains = self._load_blocked_domains_config()
            except Exception as config_error:
                logging.error(f"Error loading blocked domains config, using empty set: {config_error}")
                self.config_blocked_domains = set()
            
            # Session-level URL fetch cache for WEB_SEARCH runs.
            # Maps url -> sanitized content string so the same URL is never
            # fetched more than once per WebKnowledgeSeeker instance lifetime.
            # A new instance is created per WEB_SEARCH command, so this resets
            # automatically between top-level searches.
            self.session_fetch_cache: dict = {}
            
            # Searx instances (disabled)
            self.searx_instances = []
            self.current_searx_index = 0
            
            # Active search engines only — DDG removed (compressed binary responses),
            # Brave removed (no _search_brave method), Bing removed (no API key).
            self.search_engines = [
                'startpage',    # Primary — good quality results
                'wikipedia'     # Reliable fallback — direct API, no blocking
            ]
            
            logging.info("✅ Enhanced Web Knowledge Seeker initialized with engines: StartPage, Wikipedia")
            
        except Exception as init_error:
            logging.error(f"Critical error initializing WebKnowledgeSeeker: {init_error}")
            raise

    # ========================================================================
    # HEADER ROTATION AND DELAY UTILITIES
    # ========================================================================

    def _get_headers_rotation(self):
        """
        Enhanced header rotation with realistic browser fingerprints.
        
        Returns:
            dict: Random browser headers to avoid detection
        """
        try:
            headers_pool = [
                {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"'
                },
                {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"macOS"'
                },
                {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1'
                },
                {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                },
                {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Linux"'
                },
                {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Microsoft Edge";v="120"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"'
                }
            ]
            
            # Randomly select a header set
            selected_headers = random.choice(headers_pool).copy()
            
            # 30% chance to add a referer to simulate natural browsing
            if random.random() < 0.3:
                referers = [
                    'https://www.google.com/',
                    'https://www.bing.com/',
                    'https://github.com/',
                    'https://stackoverflow.com/'
                ]
                selected_headers['Referer'] = random.choice(referers)
                
            return selected_headers
            
        except Exception as e:
            logging.error(f"Error generating headers rotation: {e}")
            # Return basic fallback headers
            return {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }

    def _apply_enhanced_delays(self, last_request_time=None):
        """
        Apply enhanced delay strategy with randomization to avoid rate limiting.
        
        Args:
            last_request_time: Optional override for last request timestamp
        """
        try:
            current_time = time.time()
            
            if last_request_time is None:
                last_request_time = self.last_request_time
                
            time_since_last = current_time - last_request_time
            
            # Calculate base delay with randomization
            base_delay = random.uniform(self.min_request_interval, self.max_request_interval)
            
            # Add jitter for more natural timing
            jitter = random.uniform(0.5, 2.5)
            total_delay = base_delay + jitter
            
            # Check if we need to wait
            if time_since_last < total_delay:
                sleep_time = total_delay - time_since_last
                logging.info(f"⏳ Applying enhanced delay: {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            
        except Exception as e:
            logging.error(f"Error applying delays: {e}")
            # Apply minimal safe delay on error
            time.sleep(2)

    # ========================================================================
    # DOMAIN BLACKLIST MANAGEMENT
    # ========================================================================

    def _load_blacklist(self):
        """
        Load domain blacklist from persistent file.
        
        Returns:
            dict: Blacklisted domains with expiration info
        """
        try:
            if os.path.exists(self.blacklist_file):
                with open(self.blacklist_file, 'r') as f:
                    blacklist_data = json.load(f)
                    
                # Clean expired entries and handle format compatibility
                current_time = datetime.now()
                cleaned_blacklist = {}
                
                for domain, entry in blacklist_data.items():
                    try:
                        # Handle both old (string) and new (dict) formats
                        if isinstance(entry, str):
                            # Old format — convert to new format, skip if expired
                            blacklist_time = datetime.fromisoformat(entry)
                            if current_time - blacklist_time < timedelta(seconds=self.blacklist_duration):
                                expiry_time = blacklist_time + timedelta(seconds=self.blacklist_duration)
                                cleaned_blacklist[domain] = {
                                    'blacklisted_at': entry,
                                    'expires_at': expiry_time.isoformat(),
                                    'error_type': 'legacy',
                                    'duration_minutes': self.blacklist_duration // 60
                                }
                        else:
                            # New format — check if still valid
                            expires_at = datetime.fromisoformat(entry['expires_at'])
                            if current_time < expires_at:
                                cleaned_blacklist[domain] = entry
                                
                    except Exception as parse_error:
                        logging.warning(f"Skipping invalid blacklist entry for {domain}: {parse_error}")
                        continue
                
                # Save cleaned blacklist back to file
                self._save_blacklist(cleaned_blacklist)
                
                logging.info(f"📋 Loaded {len(cleaned_blacklist)} blacklisted domains")
                return cleaned_blacklist
                
        except FileNotFoundError:
            logging.info("No existing blacklist file found, starting fresh")
        except json.JSONDecodeError as json_error:
            logging.error(f"Invalid JSON in blacklist file: {json_error}")
        except Exception as e:
            logging.error(f"Error loading blacklist: {e}")
            
        return {}
    
    def _load_blocked_domains_config(self) -> set:
        """
        Load the user-editable blocked domains config file.
        If the file does not exist, create it with a sensible default set.

        Returns:
            set: Lowercase domain strings to block permanently.
        """
        try:
            if not os.path.exists(self.blocked_domains_config_file):
                default_config = {
                    "_comment": "User-editable list of structurally unscrapable domains.",
                    "_instructions": [
                        "Add domains that consistently return empty, garbled, or login-walled content.",
                        "Subdomains are blocked automatically.",
                        "No 'https://' or trailing slashes.",
                        "Changes take effect on next QWEN startup."
                    ],
                    "blocked_domains": [
                        "facebook.com", "reddit.com", "twitter.com", "x.com",
                        "instagram.com", "tiktok.com", "linkedin.com", "pinterest.com",
                        "threads.net", "snapchat.com", "tumblr.com"
                    ]
                }
                with open(self.blocked_domains_config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
                logging.info(
                    f"📝 Created default blocked domains config: {self.blocked_domains_config_file}"
                )
                return set(default_config["blocked_domains"])

            with open(self.blocked_domains_config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            raw_domains = config_data.get("blocked_domains", [])
            domain_set = set(d.lower().strip() for d in raw_domains if isinstance(d, str) and d.strip())

            logging.info(
                f"📋 Loaded {len(domain_set)} config-blocked domains from {self.blocked_domains_config_file}"
            )
            return domain_set

        except json.JSONDecodeError as json_error:
            logging.error(
                f"Invalid JSON in blocked domains config {self.blocked_domains_config_file}: {json_error}"
            )
            return set()
        except Exception as e:
            logging.error(f"Error loading blocked domains config: {e}")
            return set()

    def _is_domain_config_blocked(self, url: str) -> bool:
        """
        Check whether a URL's domain (or any parent domain) is in the
        user-editable config blocked list.

        Subdomain matching: blocking 'reddit.com' also blocks
        'old.reddit.com', 'www.reddit.com', etc.

        Args:
            url: Full URL to check

        Returns:
            bool: True if this URL should be skipped without fetching
        """
        try:
            if not self.config_blocked_domains:
                return False

            netloc = urlparse(url).netloc.lower()
            if ':' in netloc:
                netloc = netloc.split(':')[0]   # strip port number

            if not netloc:
                return False

            # Direct match first (fastest path)
            if netloc in self.config_blocked_domains:
                return True

            # Subdomain match: walk up the domain hierarchy
            parts = netloc.split('.')
            for i in range(1, len(parts)):
                parent = '.'.join(parts[i:])
                if parent in self.config_blocked_domains:
                    return True

            return False

        except Exception as e:
            logging.error(f"Error in _is_domain_config_blocked for {url}: {e}")
            return False  # Safe default — don't block on error

    def _blacklist_domain_graduated(self, domain, error_type='403'):
        """
        Implement graduated blacklisting based on error type.
        Repeat offenders get longer bans than first-time failures.
        
        Args:
            domain: Domain or URL to blacklist
            error_type: Type of error ('403', '429', 'timeout', etc.)
        """
        try:
            parsed_domain = urlparse(domain).netloc if domain.startswith('http') else domain
            current_time = datetime.now()
            
            # Repeat offenders get longer blacklist
            if parsed_domain in self.failed_domains:
                blacklist_duration = 4 * 60 * 60  # 4 hours for repeat failures
            else:
                # Duration varies by error type
                if error_type == '403':
                    blacklist_duration = 30 * 60  # 30 minutes (might be temporary)
                elif error_type == '429':
                    blacklist_duration = 2 * 60 * 60  # 2 hours for rate limiting
                elif error_type == 'timeout':
                    blacklist_duration = 15 * 60  # 15 minutes for timeouts
                elif error_type == 'login_wall':
                    blacklist_duration = 12 * 60 * 60  # 12 hours — login walls don't self-resolve
                else:
                    blacklist_duration = 60 * 60  # 1 hour for other errors
            
            # Store with expiration time
            expiry_time = current_time + timedelta(seconds=blacklist_duration)
            self.failed_domains[parsed_domain] = {
                'blacklisted_at': current_time.isoformat(),
                'expires_at': expiry_time.isoformat(),
                'error_type': error_type,
                'duration_minutes': blacklist_duration // 60
            }
            
            self._save_blacklist()
            
            logging.warning(f"🚫 Blacklisted {parsed_domain} for {blacklist_duration//60} minutes ({error_type} error)")
            
        except Exception as e:
            logging.error(f"Error in graduated blacklisting for {domain}: {e}")
    
    def _blacklist_domain(self, domain, error_type='403'):
        """
        Legacy wrapper that delegates to the graduated blacklisting system.
        
        Args:
            domain: Domain or URL to blacklist
            error_type: Type of error encountered
        """
        return self._blacklist_domain_graduated(domain, error_type)

    def _is_domain_blacklisted(self, url):
        """
        Check if a domain is currently blacklisted. Handles both old and new
        blacklist entry formats. Expired entries are removed on access.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if domain is currently blacklisted
        """
        try:
            domain = urlparse(url).netloc
            
            if domain in self.failed_domains:
                blacklist_info = self.failed_domains[domain]
                
                # Handle both old format (string) and new format (dict)
                if isinstance(blacklist_info, str):
                    # Old format — use 24 hour default
                    blacklist_time = datetime.fromisoformat(blacklist_info)
                    if datetime.now() - blacklist_time < timedelta(hours=24):
                        return True
                else:
                    # New format with expiry time
                    expires_at = datetime.fromisoformat(blacklist_info['expires_at'])
                    if datetime.now() < expires_at:
                        return True
                
                # Entry expired — remove it
                del self.failed_domains[domain]
                self._save_blacklist()
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking blacklist for {url}: {e}")
            return False

    def _save_blacklist(self, blacklist_data=None):
        """
        Save domain blacklist to persistent file.
        
        Args:
            blacklist_data: Optional data to save; uses self.failed_domains if None
        """
        try:
            data_to_save = blacklist_data if blacklist_data is not None else self.failed_domains
            with open(self.blacklist_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving blacklist: {e}")

    # ========================================================================
    # TEXT SANITIZATION AND QUALITY VALIDATION
    # ========================================================================

    def _sanitize_text_for_storage(self, text: str, url: str = "unknown") -> Optional[str]:
        """
        Sanitize text before storage to remove garbage characters.
        Implements strict quality checks to prevent corrupted text from
        being stored in the database.
        
        Args:
            text: Text to sanitize
            url: Source URL for logging
            
        Returns:
            Optional[str]: Sanitized text or None if quality is too low
        """
        try:
            if not text:
                return None
                
            # Check for Unicode replacement characters (â)
            replacement_char_count = text.count('â')
            replacement_ratio = replacement_char_count / len(text) if text else 1
            
            if replacement_ratio > self.max_replacement_char_ratio:
                logging.warning(
                    f"â ï¸ Text has {replacement_ratio:.2%} replacement characters "
                    f"(threshold: {self.max_replacement_char_ratio:.2%}) for {url}"
                )
                return None
            
            # Remove any replacement characters that slipped through
            if replacement_char_count > 0:
                text = text.replace('â', '')
                logging.info(f"ð§¹ Removed {replacement_char_count} replacement characters from {url}")
            
            # Check for excessive non-printable characters
            non_printable_count = sum(
                1 for c in text 
                if ord(c) < 32 and c not in '\n\r\t'
            )
            non_printable_ratio = non_printable_count / len(text) if text else 1
            
            if non_printable_ratio > self.max_non_printable_ratio:
                logging.warning(
                    f"⚠️ Text has {non_printable_ratio:.2%} non-printable characters "
                    f"(threshold: {self.max_non_printable_ratio:.2%}) for {url}"
                )
                return None
            
            # Remove control characters except newlines, tabs, carriage returns
            text = ''.join(
                char for char in text
                if ord(char) >= 32 or char in '\n\r\t'
            )
            
            # Validate clean UTF-8 round-trip
            try:
                test_encode = text.encode('utf-8', errors='strict')
                text = test_encode.decode('utf-8', errors='strict')
            except UnicodeEncodeError as encode_error:
                logging.error(f"Text contains invalid UTF-8 characters for {url}: {encode_error}")
                # Recover by stripping problem chars
                text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            
            # Minimum length check after sanitization
            if len(text.strip()) < self.min_content_length:
                logging.warning(f"Text too short after sanitization ({len(text)} chars) for {url}")
                return None
            
            # Character distribution check — text should have letters and spaces
            letter_count = sum(1 for c in text if c.isalpha())
            space_count = sum(1 for c in text if c.isspace())
            
            if len(text) > 100:  # Only check distribution for longer texts
                letter_ratio = letter_count / len(text)
                space_ratio = space_count / len(text)
                
                if letter_ratio < 0.4:  # Less than 40% letters
                    logging.warning(f"Text has unusual character distribution (only {letter_ratio:.1%} letters) for {url}")
                    return None
                    
                if space_ratio < 0.05:  # Less than 5% spaces
                    logging.warning(f"Text has unusual spacing (only {space_ratio:.1%} spaces) for {url}")
                    return None
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            logging.debug(f"✅ Text sanitization successful for {url}: {len(text)} chars")
            return text
            
        except Exception as e:
            logging.error(f"Error sanitizing text for {url}: {e}")
            return None

    # ========================================================================
    # CONTENT VALIDATION AND CLEANING METHODS
    # ========================================================================

    def _validate_content_type(self, response, url: str) -> tuple:
        """
        Validate that the response Content-Type is processable HTML/text.
        PDFs return a distinguishable reason string so the caller can route
        them to _fetch_pdf_content() instead of discarding them entirely.
        
        Args:
            response: The requests Response object
            url: The URL being fetched (for logging)
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
            Special reason value "PDF_CONTENT" signals a PDF that can be handled.
        """
        try:
            content_type = response.headers.get('Content-Type', '').lower()
            
            logging.debug(f"Content-Type for {url}: {content_type}")
            
            # Valid HTML/text content types
            valid_types = [
                'text/html',
                'text/plain',
                'application/xhtml+xml',
                'application/xml',
                'text/xml'
            ]
            
            # PDF is separated so the caller can route it to the PDF handler
            pdf_type = 'application/pdf'
            
            # All other binary/non-text types we skip entirely
            invalid_types = [
                'application/octet-stream',
                'image/',
                'video/',
                'audio/',
                'application/zip',
                'application/gzip',
                'application/x-tar',
                'application/x-rar',
                'application/msword',
                'application/vnd.ms-',
                'application/vnd.openxmlformats',
                'font/',
                'application/javascript',
                'application/json'
            ]
            
            # Check for PDF first — special return value so caller can route it
            if pdf_type in content_type:
                logging.info(f"📄 PDF content-type detected at runtime for {url}")
                return False, "PDF_CONTENT"
            
            # Check remaining invalid types
            for invalid_type in invalid_types:
                if invalid_type in content_type:
                    reason = f"Invalid content type: {content_type}"
                    logging.info(f"⚠️ Skipping {url}: {reason}")
                    return False, reason
            
            # Check for valid types
            for valid_type in valid_types:
                if valid_type in content_type:
                    return True, "Valid HTML/text content"
            
            # No Content-Type header — attempt to process anyway
            if not content_type or content_type == '':
                return True, "No Content-Type header, will validate content"
            
            # Unknown content type — cautious attempt
            logging.warning(f"⚠️ Unknown Content-Type '{content_type}' for {url}, attempting to process")
            return True, f"Unknown Content-Type: {content_type}"
            
        except Exception as e:
            logging.error(f"Error validating Content-Type for {url}: {e}")
            return True, "Content-Type validation error, attempting to process"

    def _validate_content_is_html(self, content: str, url: str) -> tuple:
        """
        Validate that content appears to be HTML/text, not binary data.
        
        Args:
            content: The decoded page content
            url: The URL (for logging)
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        try:
            if not content:
                return False, "Empty content"
            
            # Check content length
            if len(content) < self.min_content_length:
                return False, f"Content too short ({len(content)} chars)"
            
            # Sample from start of content for efficiency
            sample = content[:500]
            
            # Check for common HTML indicators
            html_indicators = ['<html', '<body', '<div', '<p>', '<!DOCTYPE', '<head']
            has_html = any(indicator.lower() in sample.lower() for indicator in html_indicators)
            
            if has_html:
                return True, "HTML content detected"
            
            # Check for common binary file signatures
            binary_signatures = [
                '\x00',           # Null bytes
                '\x1f\x8b',       # Gzip
                'PK\x03\x04',     # ZIP/DOCX/XLSX
            ]
            
            for sig in binary_signatures:
                if sig in sample[:20]:
                    logging.info(f"⚠️ Skipping {url}: Binary file signature detected")
                    return False, "Binary file signature detected"
            
            return True, "Content appears to be valid HTML/text"
            
        except Exception as e:
            logging.error(f"Error validating content for {url}: {e}")
            return False, f"Content validation error: {e}"

    def _detect_login_wall(self, content: str, url: str) -> tuple:
        """
        Detect if page content is behind a login wall or paywall.
        
        Args:
            content: The page content
            url: The URL (for logging)
            
        Returns:
            Tuple[bool, str]: (is_login_wall, reason)
        """
        try:
            content_lower = content.lower()
            
            # Login wall indicators
            login_indicators = [
                'sign in to view',
                'log in to view',
                'login to view',
                'sign in to continue',
                'log in to continue',
                'create an account',
                'sign up to view',
                'register to view',
                'join to view',
                'members only',
                'subscribe to read',
                'subscribe to view',
                'subscription required',
                'premium content',
                'unlock this content',
                'please sign in',
                'please log in',
                'authentication required',
                'you must be logged in',
                'login required'
            ]
            
            found_indicators = []
            for indicator in login_indicators:
                if indicator in content_lower:
                    found_indicators.append(indicator)
            
            # Two or more indicators = definite login wall
            if len(found_indicators) >= 2:
                reason = f"Login wall detected: {', '.join(found_indicators[:3])}"
                logging.info(f"⚠️ Skipping {url}: {reason}")
                return True, reason
            
            # Short content with even one login indicator is suspicious
            if len(content) < 2000 and found_indicators:
                reason = f"Short page with login indicator: {found_indicators[0]}"
                logging.info(f"⚠️ Skipping {url}: {reason}")
                return True, reason
            
            # LinkedIn special case — always behind login
            domain = urlparse(url).netloc.lower()
            if 'linkedin.com' in domain:
                if 'sign in' in content_lower or 'log in' in content_lower:
                    reason = "LinkedIn login wall detected"
                    logging.info(f"⚠️ Skipping {url}: {reason}")
                    return True, reason
            
            return False, "No login wall detected"
            
        except Exception as e:
            logging.error(f"Error detecting login wall for {url}: {e}")
            return False, "Login wall detection error"

    def _strip_ai_thinking(self, ai_response: str) -> str:
        """
        Strip chain-of-thought reasoning and internal thinking from AI responses.
        DeepSeek and similar models often include their reasoning process.
        This removes those patterns to get clean extracted knowledge.
        
        Args:
            ai_response: The raw AI response
            
        Returns:
            str: Cleaned response with thinking patterns removed
        """
        try:
            if not ai_response:
                return ""
            
            cleaned = ai_response
            
            # Pattern 1: Remove explicit thinking blocks
            thinking_block_patterns = [
                r'<think>.*?</think>',
                r'<thinking>.*?</thinking>',
                r'<reasoning>.*?</reasoning>',
                r'<internal>.*?</internal>',
                r'\[thinking\].*?\[/thinking\]',
                r'\[internal\].*?\[/internal\]'
            ]
            
            for pattern in thinking_block_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # Pattern 2: Remove common chain-of-thought openers line by line
            cot_openers = [
                r"^(?:Okay|Ok|Alright|Let me|Let's|I need to|I should|I'll|I will|First,? I)[\s,]",
                r"^(?:The user wants|The query asks|This question|To answer this)",
                r"^(?:Let me (?:think|analyze|consider|examine|look at|break down))",
                r"^(?:Looking at|Analyzing|Examining|Considering|Breaking down)",
                r"^(?:Step \d+:|First,|Second,|Third,|Finally,|Next,)"
            ]
            
            lines = cleaned.split('\n')
            filtered_lines = []
            skip_until_content = True
            
            for line in lines:
                line_stripped = line.strip()
                
                if skip_until_content and not line_stripped:
                    continue
                
                is_thinking_line = False
                for pattern in cot_openers:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        is_thinking_line = True
                        break
                
                # Meta-commentary patterns — AI talking about the content rather than extracting it
                meta_patterns = [
                    r"^(?:The (?:content|text|article|page|source) (?:mentions|discusses|talks about|provides|contains))",
                    r"^(?:From (?:the|this) (?:content|text|article|source|page))",
                    r"^(?:Based on (?:the|this|my) (?:analysis|reading|review))",
                    r"^(?:After (?:analyzing|reviewing|reading|examining))",
                    r"^(?:I (?:found|noticed|observed|see|can see) that)"
                ]
                
                for pattern in meta_patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        is_thinking_line = True
                        break
                
                if is_thinking_line:
                    continue
                else:
                    skip_until_content = False
                    filtered_lines.append(line)
            
            cleaned = '\n'.join(filtered_lines)
            
            # Pattern 3: Remove task-reference phrases inline
            task_references = [
                r"(?:The (?:relevant|extracted|key|important) (?:information|content|points|data) (?:is|are|includes?):?)",
                r"(?:Here (?:is|are) the (?:extracted|relevant|key) (?:information|points|content):?)",
                r"(?:(?:Key|Main|Important|Relevant) (?:points?|information|findings?|takeaways?):?)"
            ]
            
            for pattern in task_references:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
            # Clean up extra whitespace
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
            cleaned = re.sub(r' {2,}', ' ', cleaned)
            cleaned = cleaned.strip()
            
            # Safety check — if we stripped too much, return partial cleanup only
            if len(cleaned) < len(ai_response) * 0.2:
                logging.warning("AI thinking stripping removed too much content, using partial cleanup")
                partial_clean = ai_response
                for pattern in thinking_block_patterns:
                    partial_clean = re.sub(pattern, '', partial_clean, flags=re.DOTALL | re.IGNORECASE)
                return partial_clean.strip()
            
            return cleaned
            
        except Exception as e:
            logging.error(f"Error stripping AI thinking: {e}")
            return ai_response

    def _handle_response_encoding(self, response, url: str) -> Optional[str]:
        """
        Properly handle response encoding to avoid garbled text.
        Uses strict validation to prevent garbage characters from passing through.
        
        Args:
            response: The requests Response object
            url: The URL (for logging)
            
        Returns:
            Optional[str]: Decoded text content or None if decoding fails
        """
        try:
            # Try the apparent encoding from the response first
            if response.encoding:
                try:
                    text = response.text
                    # Strict: reject if any replacement characters present
                    if text.count('â') == 0:
                        return text
                    else:
                        logging.debug(f"Response encoding produced {text.count('â')} replacement characters for {url}")
                except Exception:
                    pass
            
            # Check for charset in Content-Type header
            content_type = response.headers.get('Content-Type', '')
            charset_match = re.search(r'charset=([^\s;]+)', content_type, re.IGNORECASE)
            if charset_match:
                detected_encoding = charset_match.group(1).strip('"\'')
                try:
                    text = response.content.decode(detected_encoding)
                    if text.count('â') == 0:
                        logging.debug(f"Successfully decoded {url} using header charset {detected_encoding}")
                        return text
                    else:
                        logging.debug(f"Header charset produced {text.count('â')} replacement characters for {url}")
                except Exception:
                    pass
            
            # Try common encodings in order
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
            
            for encoding in encodings_to_try:
                try:
                    text = response.content.decode(encoding)
                    if text.count('â') == 0:
                        logging.debug(f"Successfully decoded {url} with {encoding}")
                        return text
                except (UnicodeDecodeError, LookupError):
                    continue
            
            # Try chardet if available
            try:
                import chardet
                detected = chardet.detect(response.content)
                if detected and detected.get('encoding') and detected.get('confidence', 0) > 0.7:
                    text = response.content.decode(detected['encoding'])
                    if text.count('â') == 0:
                        logging.debug(f"Decoded {url} with chardet-detected {detected['encoding']} (confidence: {detected.get('confidence')})")
                        return text
            except ImportError:
                pass
            except Exception:
                pass
            
            # Last resort — decode with errors='ignore' to strip problem chars
            text = response.content.decode('utf-8', errors='ignore')
            
            if not text or len(text) < 50:
                logging.warning(f"⚠️ Decoding with errors='ignore' produced very short content for {url}")
                return None
            
            # Validate that we didn't strip too much content
            original_length = len(response.content)
            decoded_length = len(text.encode('utf-8'))
            
            if decoded_length < original_length * 0.8:
                logging.warning(f"⚠️ Decoding removed {((original_length - decoded_length) / original_length):.1%} of content for {url}")
                return None
            
            logging.info(f"✅ Decoded {url} with errors='ignore' (kept {(decoded_length / original_length):.1%} of content)")
            return text
            
        except Exception as e:
            logging.error(f"Error handling encoding for {url}: {e}")
            return None

    # ========================================================================
    # SEARCH ENGINE METHODS
    # ========================================================================

    def _search_startpage(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search using StartPage with enhanced scraping and error handling.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, str]]: List of search results with 'url' and 'title' keys
        """
        try:
            logging.info(f"🔍 Starting StartPage search for: '{query}'")
            
            # Check if StartPage is blacklisted
            startpage_url = "https://www.startpage.com"
            if self._is_domain_blacklisted(startpage_url):
                logging.info("⚠️ StartPage is blacklisted, skipping")
                return []
            
            # Apply enhanced delays to avoid rate limiting
            self._apply_enhanced_delays()
            
            # Prepare search URL
            search_url = "https://www.startpage.com/sp/search"
            params = {
                'query': query,
                'cat': 'web',
                'pl': '',
                'language': 'english',
                'rcount': max_results
            }
            
            headers = self._get_headers_rotation()
            
            # Make request with retries
            response = None
            for attempt in range(3):
                try:
                    response = requests.get(search_url, params=params, headers=headers, timeout=15)
                    if response.status_code == 200:
                        break
                    elif response.status_code == 429:
                        logging.warning(f"🚫 Rate limited by StartPage (attempt {attempt + 1})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    elif response.status_code in [403, 503]:
                        logging.warning(f"🚫 Blocked by StartPage: HTTP {response.status_code}")
                        self._blacklist_domain(startpage_url, str(response.status_code))
                        return []
                    else:
                        logging.warning(f"⚠️ StartPage returned HTTP {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    logging.warning(f"⏰ StartPage timeout (attempt {attempt + 1})")
                    if attempt == 2:  # Last attempt
                        self._blacklist_domain(startpage_url, 'timeout')
                        return []
                        
                except requests.exceptions.ConnectionError:
                    logging.warning(f"🔌 StartPage connection error")
                    self._blacklist_domain(startpage_url, 'connection_error')
                    return []
                    
                except Exception as req_error:
                    logging.error(f"Request error on attempt {attempt + 1}: {req_error}")
                    if attempt == 2:
                        return []
            
            if not response or response.status_code != 200:
                logging.warning("❌ StartPage search failed after retries")
                return []
            
            # Parse results
            results = self._parse_startpage_results(response.text, max_results)
            
            if results:
                logging.info(f"✅ StartPage search successful: {len(results)} results")
            else:
                logging.info("⚠️ No results found in StartPage response")
                
            return results
            
        except Exception as e:
            logging.error(f"❌ Error in StartPage search: {e}", exc_info=True)
            return []

    def _parse_startpage_results(self, html_content: str, max_results: int) -> List[Dict[str, str]]:
        """
        Parse StartPage HTML results.
        
        Args:
            html_content: HTML content from StartPage
            max_results: Maximum number of results to extract
            
        Returns:
            List[Dict[str, str]]: Parsed search results
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            # StartPage result selectors — try in priority order
            result_selectors = [
                '.w-gl__result',
                '.result',
                '.web-result',
                '[data-testid="result"]'
            ]
            
            result_elements = []
            for selector in result_selectors:
                result_elements = soup.select(selector)
                if result_elements:
                    logging.debug(f"Found {len(result_elements)} results using selector '{selector}'")
                    break
            
            if not result_elements:
                # Fallback — extract any external links
                logging.debug("Using fallback link extraction for StartPage")
                result_elements = soup.find_all('a', href=True)
                
            # Get extra elements so we have backups after filtering
            for element in result_elements[:max_results * 2]:
                try:
                    # Extract URL
                    url = None
                    if element.name == 'a':
                        url = element.get('href', '')
                    else:
                        link_elem = element.find('a', href=True)
                        if link_elem:
                            url = link_elem.get('href', '')
                    
                    if not url or not url.startswith('http'):
                        continue
                    
                    # Skip blacklisted domains
                    if self._is_domain_blacklisted(url):
                        continue
                    
                    # Extract title
                    title = ""
                    if element.name == 'a':
                        title = element.get_text(strip=True)
                    else:
                        title_elem = element.find(['h2', 'h3', 'h4', '.title', '.result-title'])
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                        else:
                            title = element.get_text(strip=True)[:100]
                    
                    if title and len(title) >= self.min_content_length:
                        results.append({
                            'url': url,
                            'title': title
                        })
                        
                        if len(results) >= max_results:
                            break
                            
                except Exception as parse_error:
                    logging.debug(f"Error parsing StartPage result element: {parse_error}")
                    continue
            
            return results
            
        except Exception as e:
            logging.error(f"Error parsing StartPage results: {e}")
            return []

    def _search_wikipedia(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search Wikipedia for articles matching the query using the OpenSearch API.
        Returns results in the same format as _search_startpage for consistency.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, str]]: List of search results with 'url' and 'title' keys
        """
        try:
            logging.info(f"🔍 Starting Wikipedia search for: '{query}'")
            
            # Wikipedia's OpenSearch API has generous rate limits — short delay is fine
            time.sleep(random.uniform(1.0, 2.0))
            
            # Long compound queries return 0 results on Wikipedia — trim to 60 chars
            search_query = query
            if len(query) > 60:
                trimmed = query[:60]
                last_space = trimmed.rfind(' ')
                search_query = trimmed[:last_space].strip() if last_space > 20 else trimmed.strip()
                logging.info(f"🔍 Wikipedia query trimmed: '{query}' → '{search_query}'")
            
            # Wikipedia OpenSearch API — returns: [query, [titles], [descriptions], [urls]]
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': search_query,
                'limit': max_results,
                'namespace': 0,       # Main namespace (articles only)
                'format': 'json',
                'redirects': 'resolve' # Automatically follow redirects
            }
            
            headers = self._get_headers_rotation()
            
            try:
                response = requests.get(search_url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if len(data) >= 4:
                        titles = data[1]  # List of article titles
                        urls = data[3]    # List of article URLs
                        
                        results = []
                        for title, url in zip(titles, urls):
                            results.append({
                                'title': title,
                                'url': url
                            })
                        
                        logging.info(f"✅ Wikipedia search successful: {len(results)} results")
                        return results
                    else:
                        logging.warning(f"⚠️ Wikipedia API returned unexpected format: {len(data)} elements")
                        return []
                        
                elif response.status_code == 429:
                    logging.warning(f"🚫 Rate limited by Wikipedia API")
                    return []
                    
                else:
                    logging.warning(f"⚠️ Wikipedia API returned HTTP {response.status_code}")
                    return []
                    
            except requests.exceptions.Timeout:
                logging.warning(f"⏰ Wikipedia API search timeout for query: '{query}'")
                return []
                
            except requests.exceptions.ConnectionError:
                logging.warning(f"🔌 Wikipedia API connection error for query: '{query}'")
                return []
                
            except requests.exceptions.RequestException as req_error:
                logging.error(f"Request error during Wikipedia search: {req_error}")
                return []
                
            except json.JSONDecodeError as json_error:
                logging.error(f"Failed to parse Wikipedia API JSON response: {json_error}")
                return []
                
            except Exception as request_error:
                logging.error(f"Unexpected error making Wikipedia API request: {request_error}")
                return []
                
        except Exception as e:
            logging.error(f"Error in Wikipedia search for '{query}': {e}", exc_info=True)
            return []
    
    def _get_wikipedia_content_direct(self, page_title: str) -> Optional[Dict[str, Any]]:
        """
        Get structured content directly from Wikipedia API.
        Bypasses web scraping for clean, structured plain-text data.
        
        Args:
            page_title: Wikipedia page title
            
        Returns:
            Optional[Dict[str, Any]]: Structured Wikipedia content, or None on failure.
            Keys: extract, content, categories, related_links, images, url, title
        """
        try:
            logging.info(f"📚 Fetching Wikipedia content for: '{page_title}'")
            
            # Wikipedia API is generous — shorter delay is acceptable
            time.sleep(random.uniform(1.0, 2.0))
            
            api_url = "https://en.wikipedia.org/w/api.php"
            
            # Get extract (summary/intro) plus metadata
            params = {
                'action': 'query',
                'titles': page_title,
                'prop': 'extracts|categories|links|images|info',
                'exintro': True,           # Introduction only for extract
                'explaintext': True,       # Plain text (no HTML)
                'exsectionformat': 'plain',
                'inprop': 'url',           # Get canonical URL
                'pllimit': 10,             # Limit related links
                'cllimit': 10,             # Limit categories
                'imlimit': 5,              # Limit images
                'format': 'json',
                'formatversion': 2
            }
            
            headers = {
                'User-Agent': 'WebKnowledgeSeeker/2.0 (Educational purposes; enhanced API access)',
                'Accept': 'application/json'
            }
            
            response = requests.get(api_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract page data
            if 'query' in data and 'pages' in data['query']:
                pages = data['query']['pages']
                
                if pages and len(pages) > 0:
                    page = pages[0]
                    
                    # Check if page exists
                    if 'missing' in page:
                        logging.warning(f"⚠️ Wikipedia page not found: {page_title}")
                        return None
                    
                    # Now get the full article text (not just intro)
                    full_params = {
                        'action': 'query',
                        'titles': page_title,
                        'prop': 'extracts',
                        'explaintext': True,
                        'format': 'json',
                        'formatversion': 2
                    }
                    
                    full_response = requests.get(api_url, params=full_params, headers=headers, timeout=15)
                    full_response.raise_for_status()
                    full_data = full_response.json()
                    
                    full_extract = ""
                    if 'query' in full_data and 'pages' in full_data['query']:
                        full_pages = full_data['query']['pages']
                        if full_pages and len(full_pages) > 0:
                            full_extract = full_pages[0].get('extract', '')
                    
                    # Build structured content dict
                    wiki_content = {
                        'title': page.get('title', page_title),
                        'extract': page.get('extract', ''),   # Summary/intro
                        'content': full_extract,               # Full article text
                        'url': page.get('canonicalurl', f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"),
                        'categories': [cat['title'].replace('Category:', '') 
                                    for cat in page.get('categories', [])],
                        'related_links': [link['title'] 
                                        for link in page.get('links', [])],
                        'images': [img['title'] 
                                for img in page.get('images', [])],
                        'page_id': page.get('pageid', 0),
                        'last_modified': page.get('touched', ''),
                    }
                    
                    content_length = len(wiki_content['content'])
                    extract_length = len(wiki_content['extract'])
                    logging.info(f"✅ Wikipedia API content retrieved: {content_length} chars full, {extract_length} chars extract")
                    logging.info(f"📊 Categories: {len(wiki_content['categories'])}, Links: {len(wiki_content['related_links'])}")
                    
                    return wiki_content
                else:
                    logging.warning(f"⚠️ No pages found for: {page_title}")
                    return None
            else:
                logging.warning(f"⚠️ Unexpected Wikipedia API response format")
                return None
                
        except requests.exceptions.Timeout:
            logging.warning(f"⏰ Wikipedia API timeout for: {page_title}")
            return None
        except requests.exceptions.RequestException as req_error:
            logging.error(f"❌ Wikipedia API request error for '{page_title}': {req_error}")
            return None
        except Exception as e:
            logging.error(f"❌ Error fetching Wikipedia content for '{page_title}': {e}", exc_info=True)
            return None

    def _create_wikipedia_knowledge_items(self, wiki_content: Dict[str, Any], topic: str) -> List[Dict[str, Any]]:
        """
        Convert Wikipedia API structured content into knowledge items.
        Creates multiple knowledge items from different aspects of the article:
        summary, full content, and metadata/relations.
        
        Args:
            wiki_content: Structured Wikipedia content from _get_wikipedia_content_direct
            topic: Search topic for context
            
        Returns:
            List[Dict[str, Any]]: Knowledge items ready for storage
        """
        try:
            knowledge_items = []
            source_url = wiki_content['url']
            title = wiki_content['title']
            
            # Item 1: Summary/Extract (high priority, concise)
            if wiki_content['extract']:
                extract_text = wiki_content['extract'].strip()
                if len(extract_text) >= self.min_content_length:
                    knowledge_items.append({
                        'content': extract_text,
                        'topic': topic,
                        'source': source_url,
                        'title': f"{title} (Summary)",
                        'search_query': topic,
                        'relevance_score': 0.95,  # High relevance for intro/summary
                        'extraction_method': 'wikipedia-api-extract',
                        'extracted_at': datetime.now().isoformat(),
                        'content_type': 'wikipedia_summary',
                        'source_quality': 'high',
                        'categories': wiki_content['categories']
                    })
                    logging.info(f"✅ Created Wikipedia summary item: {len(extract_text)} chars")
            
            # Item 2: Full content (detailed information)
            if wiki_content['content']:
                full_text = wiki_content['content'].strip()
                
                # Truncate if too long — keep more than standard pages (20KB vs 15KB)
                max_full_length = 20000
                if len(full_text) > max_full_length:
                    full_text = full_text[:max_full_length] + "... [Content truncated, see URL for complete article]"
                
                if len(full_text) >= self.min_content_length:
                    knowledge_items.append({
                        'content': full_text,
                        'topic': topic,
                        'source': source_url,
                        'title': f"{title} (Full Article)",
                        'search_query': topic,
                        'relevance_score': 0.85,  # Slightly lower than summary (less focused)
                        'extraction_method': 'wikipedia-api-full',
                        'extracted_at': datetime.now().isoformat(),
                        'content_type': 'wikipedia_article',
                        'source_quality': 'high',
                        'categories': wiki_content['categories'],
                        'related_topics': wiki_content['related_links'][:5]
                    })
                    logging.info(f"✅ Created Wikipedia full article item: {len(full_text)} chars")
            
            # Item 3: Structured metadata (categories and relations)
            if wiki_content['categories'] or wiki_content['related_links']:
                metadata_content = f"Wikipedia article '{title}' "
                
                if wiki_content['categories']:
                    cats = ', '.join(wiki_content['categories'][:5])
                    metadata_content += f"belongs to categories: {cats}. "
                
                if wiki_content['related_links']:
                    links = ', '.join(wiki_content['related_links'][:5])
                    metadata_content += f"Related topics include: {links}."
                
                if len(metadata_content) >= self.min_content_length:
                    knowledge_items.append({
                        'content': metadata_content,
                        'topic': topic,
                        'source': source_url,
                        'title': f"{title} (Metadata & Relations)",
                        'search_query': topic,
                        'relevance_score': 0.70,  # Contextual information
                        'extraction_method': 'wikipedia-api-metadata',
                        'extracted_at': datetime.now().isoformat(),
                        'content_type': 'wikipedia_metadata',
                        'source_quality': 'high',
                        'categories': wiki_content['categories'],
                        'related_topics': wiki_content['related_links']
                    })
                    logging.info(f"✅ Created Wikipedia metadata item")
            
            logging.info(f"📦 Created {len(knowledge_items)} knowledge items from Wikipedia API")
            return knowledge_items
            
        except Exception as e:
            logging.error(f"❌ Error creating Wikipedia knowledge items: {e}", exc_info=True)
            return []

    def _execute_search_by_engine(self, engine: str, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Execute search using the specified engine.
        Only 'startpage' and 'wikipedia' are active — other engine names will log
        a warning and return an empty list.
        
        Args:
            engine: Search engine name ('startpage' or 'wikipedia')
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List[Dict[str, str]]: Search results
        """
        try:
            if engine == 'startpage':
                return self._search_startpage(query, max_results)
            elif engine == 'wikipedia':
                return self._search_wikipedia(query, max_results)
            else:
                # Log clearly so future re-enablement attempts are obvious in logs
                logging.warning(f"Unknown or disabled search engine: '{engine}' — skipping")
                return []
                
        except Exception as e:
            logging.error(f"Error executing search with {engine}: {e}")
            return []

    # ========================================================================
    # PDF CONTENT EXTRACTION
    # ========================================================================

    def _is_pdf_url(self, url: str) -> bool:
        """
        Check if a URL points to a PDF based on extension or known PDF path patterns.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if URL appears to be a PDF
        """
        try:
            # Parse URL and check extension
            parsed = urlparse(url.lower())
            path = parsed.path
            
            # Direct extension check
            if path.endswith('.pdf'):
                return True
            
            # Common PDF serving paths
            pdf_indicators = ['/pdf/', '/pdfs/', '/download/pdf', '/files/pdf']
            if any(indicator in path for indicator in pdf_indicators):
                return True
            
            # Query string PDF indicators
            query = parsed.query.lower()
            if 'format=pdf' in query or 'type=pdf' in query:
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking PDF URL for {url}: {e}")
            return False

    def _fetch_pdf_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract text from a PDF URL using PyPDF2.
        Downloads to memory (no temp files) and extracts text page by page.
        
        Args:
            url: URL of the PDF to fetch
            
        Returns:
            Optional[str]: Extracted text content, or None on failure
        """
        try:
            if not PDF_SUPPORT:
                logging.warning(f"📄 PDF support unavailable (PyPDF2 not installed), skipping: {url}")
                return None
            
            logging.info(f"📄 Fetching PDF content from: {url}")
            
            headers = self._get_headers_rotation()
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Download PDF bytes into memory
            # pdf_bytes = io.BytesIO(response.content)  # DEAD CODE TEST 2026-05-17: duplicate of pdf_file allocated below — unused (ruff F841 + vulture)
            
            if len(response.content) == 0:
                logging.warning(f"📄 Empty PDF response from: {url}")
                return None
            
            # Wrap in a file-like object for PyPDF2
            pdf_file = io.BytesIO(response.content)
            
            # Extract text page by page
            reader = PyPDF2.PdfReader(pdf_file)
            
            if not reader.pages:
                logging.warning(f"📄 PDF has no readable pages: {url}")
                return None
            
            page_texts = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        page_texts.append(page_text.strip())
                except Exception as page_error:
                    # A single bad page should not abort the whole document
                    logging.warning(f"📄 Could not extract page {i} from {url}: {page_error}")
                    continue
            
            if not page_texts:
                logging.warning(f"📄 No extractable text found in PDF: {url}")
                return None
            
            # Join pages with clear separator
            full_text = "\n\n".join(page_texts)
            
            # Normalize whitespace — PDF extraction often produces excessive spacing
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
            full_text = re.sub(r' {2,}', ' ', full_text)
            full_text = full_text.strip()
            
            # Apply 15KB sentence-aware truncation
            max_length = 15000
            if len(full_text) > max_length:
                truncated = full_text[:max_length]
                last_period = max(
                    truncated.rfind('. '),
                    truncated.rfind('! '),
                    truncated.rfind('? ')
                )
                if last_period > max_length * 0.7:
                    full_text = truncated[:last_period + 1].strip()
                    logging.debug(f"📄 PDF sentence-truncated to {len(full_text)} chars")
                else:
                    last_space = truncated.rfind(' ')
                    full_text = truncated[:last_space].strip() if last_space > 0 else truncated.strip()
                    logging.debug(f"📄 PDF word-truncated to {len(full_text)} chars")
            
            logging.info(f"✅ PDF extraction successful: {len(full_text)} chars, {len(reader.pages)} pages from {url}")
            return full_text
            
        except requests.exceptions.HTTPError as e:
            # Log HTTP errors but do NOT blacklist — domain may serve HTML fine
            logging.warning(f"📄 PDF HTTP error for {url}: {e}")
            return None
        except requests.exceptions.Timeout:
            logging.warning(f"📄 PDF fetch timeout for {url}")
            return None
        except PyPDF2.errors.PdfReadError as e:
            # Corrupted or encrypted PDF — not a server problem
            logging.warning(f"📄 PDF read error for {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"📄 Unexpected error fetching PDF {url}: {e}")
            return None

    # ========================================================================
    # WEB CONTENT FETCHING AND PROCESSING
    # ========================================================================

    def _fetch_webpage_content_enhanced(self, url: str) -> Optional[str]:
        """
        Fetch webpage content with multiple strategies and strict validation.
        PDFs are detected early (by URL pattern or runtime Content-Type) and
        routed to _fetch_pdf_content() instead of the HTML pipeline.
        
        Args:
            url: URL to fetch
            
        Returns:
            Optional[str]: Sanitized page content or None
        """
        try:
            # PRE-FLIGHT 1: Config-blocked domains (structurally unscrapable — social media SPAs, etc.)
            # These are checked before everything else to avoid wasting any network time.
            if self._is_domain_config_blocked(url):
                logging.debug(f"⛔ Skipping config-blocked domain: {url}")
                return None
            
            # PRE-FLIGHT 2: Session blacklist (domains that failed or returned login walls recently)
            if self._is_domain_blacklisted(url):
                logging.info(f"⚠️ Skipping blacklisted domain: {url}")
                return None
            
            # PDF EARLY DETECTION — skip HTML pipeline entirely for PDF URLs
            # This prevents all three strategies from failing and incorrectly
            # blacklisting a domain that serves valid PDFs.
            if self._is_pdf_url(url):
                logging.info(f"📄 PDF URL detected, routing to PDF extractor: {url}")
                return self._fetch_pdf_content(url)
            
            # Try multiple HTML fetch strategies in priority order
            strategies = [
                ("Enhanced Session", self._fetch_with_enhanced_session),
                ("Retries with Backoff", self._fetch_with_retries_enhanced),
                ("Basic Fallback", self._fetch_basic_fallback)
            ]
            
            for strategy_name, strategy_func in strategies:
                try:
                    logging.info(f"🌐 Trying {strategy_name} for {url}")
                    
                    response_text = strategy_func(url)
                    
                    if response_text:
                        logging.info(f"✅ {strategy_name} succeeded for {url}")
                        
                        # Process and clean the HTML content
                        cleaned_content = self._process_webpage_content(response_text, url)
                        
                        if cleaned_content:
                            sanitized_content = self._sanitize_text_for_storage(cleaned_content, url)
                            if sanitized_content:
                                return sanitized_content
                            else:
                                logging.warning(f"⚠️ Content failed sanitization for {url}")
                                return None
                        else:
                            logging.warning(f"⚠️ Content processing returned None for {url}")
                
                except PDFContentDetected:
                    # Strategy detected PDF Content-Type at runtime (URL didn't end in .pdf
                    # but the server returned application/pdf). Route to PDF extractor
                    # immediately — no further HTML strategies needed, no blacklisting.
                    logging.info(f"📄 Runtime PDF detection in {strategy_name}, routing to PDF extractor: {url}")
                    return self._fetch_pdf_content(url)
                    
                except requests.exceptions.Timeout:
                    logging.warning(f"⏰ {strategy_name} timed out for {url}")
                    
                except requests.exceptions.ConnectionError:
                    logging.warning(f"🔌 {strategy_name} connection error for {url}")
                    
                except requests.exceptions.HTTPError as e:
                    status_code = getattr(e.response, 'status_code', None)
                    if status_code in [403, 429, 503]:
                        error_type = str(status_code)
                        logging.warning(f"🚫 {strategy_name} blocked for {url}: HTTP {status_code}")
                        self._blacklist_domain_graduated(url, error_type)
                        return None
                    else:
                        logging.warning(f"⚠️ {strategy_name} HTTP error for {url}: {e}")
                        
                except Exception as strategy_error:
                    logging.warning(f"❌ {strategy_name} failed for {url}: {strategy_error}")
                    continue
            
            # All strategies failed — blacklist domain
            logging.warning(f"❌ All fetch strategies failed for {url}")
            self._blacklist_domain_graduated(url, 'fetch_failed')
            return None
            
        except Exception as e:
            logging.error(f"❌ Error in enhanced webpage fetch for {url}: {e}", exc_info=True)
            return None
        
    def _fetch_with_enhanced_session(self, url: str) -> Optional[str]:
        """
        Fetch using enhanced session with realistic browser behavior.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Optional[str]: The page content as text, or None if fetch fails
        """
        try:
            # Update session headers to appear realistic
            self.session.headers.update(self._get_headers_rotation())
            
            # Add random delay to simulate human behavior
            time.sleep(random.uniform(0.5, 2.0))
            
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            # Validate Content-Type before processing
            is_valid_type, type_reason = self._validate_content_type(response, url)
            if not is_valid_type:
                # Runtime PDF detection — signal caller to route to PDF extractor
                if type_reason == "PDF_CONTENT":
                    raise PDFContentDetected(url)
                logging.info(f"⚠️ Skipping {url}: {type_reason}")
                return None
            
            # Handle encoding properly
            text = self._handle_response_encoding(response, url)
            if not text:
                logging.warning(f"⚠️ Failed to decode content from {url}")
                return None
            
            # Validate content appears to be HTML/text
            is_valid_content, content_reason = self._validate_content_is_html(text, url)
            if not is_valid_content:
                logging.info(f"⚠️ Skipping {url}: {content_reason}")
                return None
            
            return text
            
        except PDFContentDetected:
            # Re-raise so _fetch_webpage_content_enhanced can catch it
            raise
        except Exception as e:
            logging.debug(f"Enhanced session fetch failed for {url}: {e}")
            return None

    def _fetch_with_retries_enhanced(self, url: str, max_retries: int = 3) -> Optional[str]:
        """
        Enhanced retry mechanism with exponential backoff and header rotation.
        
        Args:
            url: URL to fetch
            max_retries: Maximum retry attempts
            
        Returns:
            Optional[str]: Page content or None
        """
        for attempt in range(max_retries):
            try:
                headers = self._get_headers_rotation()
                
                # Exponential backoff with jitter between retries
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                
                response = requests.get(url, headers=headers, timeout=15 + (attempt * 5))
                response.raise_for_status()
                
                # Validate Content-Type
                is_valid_type, type_reason = self._validate_content_type(response, url)
                if not is_valid_type:
                    # Runtime PDF detection — signal caller to route to PDF extractor
                    if type_reason == "PDF_CONTENT":
                        raise PDFContentDetected(url)
                    logging.info(f"Skipping {url}: {type_reason}")
                    return None
                
                # Handle encoding properly
                text = self._handle_response_encoding(response, url)
                if not text:
                    logging.warning(f"Failed to decode content from {url} on attempt {attempt + 1}")
                    continue  # Try again with next attempt
                
                # Validate content
                is_valid_content, content_reason = self._validate_content_is_html(text, url)
                if not is_valid_content:
                    logging.info(f"Skipping {url}: {content_reason}")
                    return None
                
                return text
                
            except PDFContentDetected:
                # PDF detection must propagate immediately — no retries needed
                raise
            except Exception as e:
                logging.debug(f"Retry attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    # Re-raise on final attempt so caller can handle it
                    raise
        
        return None

    def _fetch_basic_fallback(self, url: str) -> Optional[str]:
        """
        Basic fallback fetch method with minimal headers for maximum compatibility.
        
        Args:
            url: URL to fetch
            
        Returns:
            Optional[str]: Page content or None
        """
        try:
            basic_headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; WebKnowledgeSeeker/1.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            time.sleep(random.uniform(2, 4))
            response = requests.get(url, headers=basic_headers, timeout=25)
            response.raise_for_status()
            
            # Validate Content-Type
            is_valid_type, type_reason = self._validate_content_type(response, url)
            if not is_valid_type:
                # Runtime PDF detection — signal caller to route to PDF extractor
                if type_reason == "PDF_CONTENT":
                    raise PDFContentDetected(url)
                logging.info(f"Skipping {url}: {type_reason}")
                return None
            
            # Handle encoding
            text = self._handle_response_encoding(response, url)
            if not text:
                logging.warning(f"Failed to decode content from {url}")
                return None
            
            # Validate content
            is_valid_content, content_reason = self._validate_content_is_html(text, url)
            if not is_valid_content:
                logging.info(f"Skipping {url}: {content_reason}")
                return None
            
            return text
            
        except PDFContentDetected:
            # Re-raise so _fetch_webpage_content_enhanced can catch it
            raise
        except Exception as e:
            logging.debug(f"Basic fallback fetch failed for {url}: {e}")
            return None

    def _process_webpage_content(self, html_content: str, url: str) -> Optional[str]:
        """
        Process and clean webpage content with enhanced filtering.
        Removes navigation, ads, scripts, and other noise from HTML before
        returning clean plain text.
        
        Args:
            html_content: Raw HTML content
            url: Source URL
            
        Returns:
            Optional[str]: Cleaned text content or None
        """
        try:
            # Check for login walls before doing any expensive processing.
            # Blacklist the domain for 12 hours on detection — login walls do not
            # self-resolve and we should not waste fetch attempts on them this session.
            is_login_wall, login_reason = self._detect_login_wall(html_content, url)
            if is_login_wall:
                logging.info(f"⚠️ Skipping {url}: {login_reason}")
                self._blacklist_domain_graduated(url, 'login_wall')
                return None
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            unwanted_tags = [
                'script', 'style', 'nav', 'header', 'footer', 'aside',
                'iframe', 'noscript', 'svg', 'button', 'form'
            ]
            
            for tag in unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Remove common noise classes/ids
            noise_selectors = [
                '.advertisement', '.ad', '.sidebar', '.menu', '.cookie-notice',
                '#comments', '.social-share', '.related-posts', '#footer', '#header'
            ]
            
            for selector in noise_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Extract plain text
            text = soup.get_text(separator=' ', strip=True)
            
            # Remove multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Remove common web artifacts
            unwanted_phrases = [
                'Cookie Policy', 'Privacy Policy', 'Terms of Service',
                'Accept All Cookies', 'Manage Cookies', 'Cookie Settings',
                'Subscribe to our newsletter', 'Sign up for updates',
                'Follow us on', 'Like us on Facebook', 'Tweet',
                'Already have an account', 'Forgot password', 'Remember me',
                'Sign up for free', 'Join now', 'Get started',
                'Share on Twitter', 'Share on Facebook', 'Share on LinkedIn',
                'Click to share', 'Pin it', 'Email this',
                'Skip to content', 'Skip to main content', 'Back to top',
                'Previous article', 'Next article', 'Related articles',
                'Read more', 'See more', 'Load more', 'Show more'
            ]
            
            for phrase in unwanted_phrases:
                text = re.sub(rf'\b{re.escape(phrase)}\b', '', text, flags=re.IGNORECASE)
            
            # Final whitespace cleanup
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Validate content length
            if len(text) < self.min_content_length:
                logging.warning(f"Content too short ({len(text)} chars) for {url}")
                return None
            
            # Check for meaningful content — very short avg word length suggests nav junk
            words = text.split()
            if words:
                avg_word_length = sum(len(w) for w in words) / len(words)
                if avg_word_length < 3.5 and len(words) < 200:
                    logging.warning(f"Content appears to be mostly navigation ({avg_word_length:.1f} avg word len) for {url}")
                    return None
            
            # Sentence-aware truncation at 15KB
            max_length = 15000
            if len(text) > max_length:
                truncated = text[:max_length]
                last_period = max(
                    truncated.rfind('. '),
                    truncated.rfind('! '),
                    truncated.rfind('? ')
                )
                if last_period > max_length * 0.7:
                    text = truncated[:last_period + 1].strip()
                    logging.debug(f"📐 Content sentence-truncated to {len(text)} chars")
                else:
                    last_space = truncated.rfind(' ')
                    if last_space > max_length * 0.8:
                        text = truncated[:last_space].strip()
                    else:
                        text = truncated.strip()
                    logging.debug(f"📐 Content word-truncated to {len(text)} chars")
            
            logging.info(f"✅ Processed {len(text)} characters from {url}")
            return text
            
        except Exception as e:
            logging.error(f"Error processing webpage content from {url}: {e}")
            return None
    
    # ========================================================================
    # SEARCH AND KNOWLEDGE ACQUISITION METHODS
    # ========================================================================
    
    def _generate_search_queries(self, topic: str, description: str) -> List[str]:
        """
        Generate varied search queries for a topic to maximize coverage across engines.
        
        Args:
            topic: Main topic to search for
            description: Additional context description
            
        Returns:
            List[str]: Up to 5 generated search queries
        """
        try:
            queries = []
            
            # Base query with topic
            queries.append(topic)
            
            # Enhanced query generation based on description
            if description:
                # Extract meaningful keywords (longer than 3 chars, alphabetic)
                words = re.findall(r'\b[a-zA-Z]{4,}\b', description.lower())
                key_phrases = [word for word in words if word not in [
                    'what', 'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'said',
                    'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other'
                ]]
                
                if key_phrases:
                    # Combine topic with most relevant phrases
                    for phrase in key_phrases[:2]:
                        queries.append(f"{topic} {phrase}")
                    
                    queries.append(f"what is {topic}")
                    queries.append(f"{topic} guide tutorial")
                    queries.append(f"{topic} explained")
                    
                    # If it seems technical, add technical queries
                    if any(tech_term in description.lower() for tech_term in [
                        'api', 'code', 'programming', 'software', 'algorithm', 'method', 'function'
                    ]):
                        queries.append(f"{topic} best practices")
                        queries.append(f"{topic} examples documentation")
            
            # General educational queries
            queries.extend([
                f"{topic} overview",
                f"learn {topic}",
                f"{topic} fundamentals"
            ])
            
            # Deduplicate while preserving order
            seen = set()
            unique_queries = []
            for query in queries:
                query_clean = query.lower().strip()
                if query_clean not in seen and len(query_clean) > 2:
                    seen.add(query_clean)
                    unique_queries.append(query)
            
            logging.info(f"🔍 Generated {len(unique_queries)} search queries for topic '{topic}'")
            return unique_queries[:5]  # Limit to 5 queries max
            
        except Exception as e:
            logging.error(f"Error generating search queries: {e}")
            return [topic]  # Fallback to just the topic

    def search_for_knowledge(self, topic: str, description: str = "", max_results: int = 3) -> List[Dict[str, Any]]:
        """
        RETRIEVAL METHOD: Search the web and return knowledge items for direct LLM consumption.
        
        Called by deepseek.py _process_external_search_commands via [EXTERNAL_SEARCH:].
        Unlike search_and_learn(), this method does NOT store to memory DB — it returns
        clean text items immediately so the LLM can read and reason over them in the current turn.
        
        NOTE: To store results in memory, use search_and_learn() instead.
        
        Args:
            topic (str): The search query / topic to find information about
            description (str): Additional context to guide which content to prioritize
            max_results (int): Maximum number of content items to return (default 3)
            
        Returns:
            List[Dict[str, Any]]: List of knowledge items, each containing:
                - content (str): Extracted text content, clean and readable
                - source (str): URL of the source page
                - title (str): Title of the source page
                - engine (str): Which search engine found this result
                - word_count (int): Approximate word count of content
        """
        try:
            logging.info(f"🔎 search_for_knowledge: Starting retrieval for topic='{topic}', max_results={max_results}")
            
            # Validate input — empty topic wastes network requests
            if not topic or not topic.strip():
                logging.error("search_for_knowledge: Empty topic provided")
                return []
            
            topic = topic.strip()
            knowledge_items = []   # Final list to return to the LLM
            seen_urls = set()      # Deduplicate URLs across engines

            # ----------------------------------------------------------------
            # PHASE 1: Collect candidate URLs from all configured search engines
            # ----------------------------------------------------------------
            candidate_results = []
            
            for engine in self.search_engines:
                try:
                    logging.info(f"🔎 search_for_knowledge: Querying engine '{engine}' for '{topic}'")
                    
                    # Request slightly more than max_results to have backups for failed fetches
                    engine_results = self._execute_search_by_engine(engine, topic, max_results=max_results + 2)
                    
                    if engine_results:
                        logging.info(f"🔎 search_for_knowledge: Engine '{engine}' returned {len(engine_results)} URLs")
                        for result in engine_results:
                            url = result.get('url', '').strip()
                            # Skip empty, already-seen, session-blacklisted, or config-blocked URLs.
                            # Config-blocked check here prevents these domains from ever entering
                            # the candidate list — no fetch attempts, no logging noise.
                            if (url
                                    and url not in seen_urls
                                    and not self._is_domain_blacklisted(url)
                                    and not self._is_domain_config_blocked(url)):
                                seen_urls.add(url)
                                result['engine'] = engine   # Tag which engine found this
                                candidate_results.append(result)
                    else:
                        logging.warning(f"🔎 search_for_knowledge: Engine '{engine}' returned no URLs")
                        
                except Exception as engine_error:
                    logging.error(f"🔎 search_for_knowledge: Engine '{engine}' error: {engine_error}")
                    continue
                
                # Short pause between engines to avoid rate-limiting
                time.sleep(random.uniform(0.5, 1.5))
            
            if not candidate_results:
                logging.warning(f"🔎 search_for_knowledge: No candidate URLs found for '{topic}'")
                return []
            
            logging.info(f"🔎 search_for_knowledge: {len(candidate_results)} unique candidate URLs to fetch")
            
            # ----------------------------------------------------------------
            # PHASE 2: Fetch and extract text content from each candidate URL.
            # Stop early once we have max_results quality items.
            # ----------------------------------------------------------------
            for result in candidate_results:
                
                # Stop when we have enough quality items for the LLM
                if len(knowledge_items) >= max_results:
                    logging.info(f"🔎 search_for_knowledge: Reached max_results ({max_results}), stopping fetch loop")
                    break
                
                url = result.get('url', '')
                title = result.get('title', 'Unknown Title')
                engine = result.get('engine', 'unknown')
                
                try:
                    logging.info(f"🔎 search_for_knowledge: Fetching content from {url}")
                    content_text = None

                    # ── SESSION FETCH CACHE ──────────────────────────────────────────
                    # If this URL was already fetched during this WEB_SEARCH session,
                    # return the cached content immediately — no network call needed.
                    # This prevents the same article being re-fetched (and its 3-strategy
                    # chain re-run) every turn when multiple search queries surface the
                    # same top result.
                    if url in self.session_fetch_cache:
                        logging.info(
                            f"🔎 search_for_knowledge: Session cache HIT for {url} "
                            f"({len(self.session_fetch_cache[url])} chars) — skipping fetch"
                        )
                        content_text = self.session_fetch_cache[url]
                        # Restore the cached title if we stored it
                        cached_title = self.session_fetch_cache.get(f"__title__{url}")
                        if cached_title:
                            title = cached_title

                    else:
                        # ── WIKIPEDIA PATH ───────────────────────────────────────────
                        # Wikipedia gets special treatment — the structured API returns
                        # clean plain text without scraping noise, ads, and nav clutter.
                        if 'wikipedia.org/wiki/' in url:
                            logging.info(f"🔎 search_for_knowledge: Using Wikipedia API for {url}")
                            page_title = url.split('/wiki/')[-1].replace('_', ' ')
                            wiki_data = self._get_wikipedia_content_direct(page_title)

                            if wiki_data:
                                # Prefer the clean intro extract; fall back to full content if thin
                                extract = wiki_data.get('extract', '').strip()
                                full_content = wiki_data.get('content', '').strip()

                                if len(extract) >= self.min_content_length:
                                    content_text = extract
                                elif len(full_content) >= self.min_content_length:
                                    content_text = full_content[:8000]

                                # Use the canonical Wikipedia title
                                title = wiki_data.get('title', title)
                                logging.info(
                                    f"🔎 search_for_knowledge: Wikipedia API returned "
                                    f"{len(content_text or '')} chars"
                                )
                            else:
                                logging.warning(
                                    f"🔎 search_for_knowledge: Wikipedia API returned None for '{page_title}'"
                                )
                                continue   # Try next candidate

                        else:
                            # ── STANDARD WEB PAGES ───────────────────────────────────
                            # Use the existing multi-strategy fetch pipeline.
                            content_text = self._fetch_webpage_content_enhanced(url)
                            logging.info(
                                f"🔎 search_for_knowledge: Web fetch returned "
                                f"{len(content_text or '')} chars for {url}"
                            )

                        # Store successful fetch result in session cache so subsequent
                        # turns that surface the same URL skip the network entirely.
                        if content_text and len(content_text.strip()) >= self.min_content_length:
                            self.session_fetch_cache[url] = content_text
                            self.session_fetch_cache[f"__title__{url}"] = title
                            logging.debug(
                                f"🔎 search_for_knowledge: Cached {len(content_text)} chars for {url}"
                            )

                    # Validate we got enough usable content
                    if not content_text or len(content_text.strip()) < self.min_content_length:
                        logging.warning(
                            f"🔎 search_for_knowledge: Insufficient content from {url} "
                            f"({len(content_text or '')} chars, minimum={self.min_content_length}), skipping"
                        )
                        continue

                    # Final sanitization pass — only needed for Wikipedia content which
                    # bypasses _fetch_webpage_content_enhanced. Non-Wikipedia content was
                    # already sanitized inside _fetch_webpage_content_enhanced, so running
                    # it again would be redundant CPU work.
                    if 'wikipedia.org' in url:
                        sanitized = self._sanitize_text_for_storage(content_text, url)
                        if not sanitized:
                            logging.warning(
                                f"🔎 search_for_knowledge: Wikipedia content failed sanitization for {url}, skipping"
                            )
                            continue
                    else:
                        # Already sanitized by _fetch_webpage_content_enhanced
                        sanitized = content_text

                    # Build the knowledge item dict.
                    # 'content', 'source', 'title' are the keys expected by
                    # deepseek.py _format_external_search_results()
                    item = {
                        'content': sanitized,
                        'source': url,
                        'title': title,
                        'engine': engine,
                        'word_count': len(sanitized.split()),
                        'topic': topic,
                        'description': description,
                        'extraction_method': (
                            'wikipedia-api' if 'wikipedia.org' in url
                            else 'pdf-extract' if self._is_pdf_url(url)
                            else 'web-scrape'
                        ),
                    }

                    knowledge_items.append(item)
                    logging.info(
                        f"✅ search_for_knowledge: Added item {len(knowledge_items)}/{max_results} "
                        f"from '{title}' at {url} ({item['word_count']} words)"
                    )
                    
                except Exception as fetch_error:
                    # Log but continue — one bad URL should not abort the whole search
                    logging.error(f"🔎 search_for_knowledge: Error processing {url}: {fetch_error}")
                    continue
            
            # ----------------------------------------------------------------
            # PHASE 3: Log summary and return
            # ----------------------------------------------------------------
            logging.info(
                f"🔎 search_for_knowledge: Complete. Returning {len(knowledge_items)}/{max_results} items "
                f"from {len(candidate_results)} candidates for topic='{topic}'"
            )
            return knowledge_items
            
        except Exception as e:
            logging.error(f"❌ search_for_knowledge: Critical error for topic='{topic}': {e}", exc_info=True)
            return []   # Always return a list, never raise — caller expects list or empty list

    def search_and_learn(self, topic: str, description: str = "", max_sources: int = 5) -> Dict[str, Any]:
        """
        MAIN STORAGE METHOD: Search the web, extract knowledge, and store to memory DB.
        Uses chatbot's two-phase commit storage (store_memory_with_transaction).
        
        NOTE: Use search_for_knowledge() if you only need results returned without storage.
        
        Args:
            topic: The topic to learn about
            description: Additional context or specific aspects to focus on
            max_sources: Maximum number of sources to fetch and process
            
        Returns:
            Dict[str, Any]: Results summary with transaction details
        """
        try:
            logging.info(f"🎯 Starting knowledge search for topic: '{topic}'")
            
            # Validate chatbot availability for AI extraction AND storage
            ai_available = self.chatbot is not None and hasattr(self.chatbot, 'llm')
            storage_available = self.chatbot is not None and hasattr(self.chatbot, 'store_memory_with_transaction')
            
            if not ai_available:
                logging.warning("⚠️ AI chatbot not available, will use content-based extraction")
            if not storage_available:
                logging.error("❌ Chatbot storage method not available - cannot store knowledge!")
                return {
                    'transaction_id': str(uuid.uuid4()),
                    'topic': topic,
                    'error': 'Storage unavailable - chatbot.store_memory_with_transaction not found',
                    'items_stored': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Generate search queries
            search_queries = self._generate_search_queries(topic, description)
            
            # Track results
            transaction_id = str(uuid.uuid4())
            all_search_results = []
            processed_sources = 0
            items_stored = 0
            
            # Search using all available engines
            for query in search_queries:
                if processed_sources >= max_sources:
                    break
                    
                for engine in self.search_engines:
                    try:
                        logging.info(f"🔍 Searching {engine} for: '{query}'")
                        results = self._execute_search_by_engine(engine, query, max_results=3)
                        
                        if results:
                            all_search_results.extend(results)
                            logging.info(f"✅ {engine} returned {len(results)} results")
                        else:
                            logging.info(f"⚠️ {engine} returned no results")
                            
                    except Exception as search_error:
                        logging.error(f"Error searching {engine}: {search_error}")
                        continue
                
                # Polite pause between query rounds
                time.sleep(random.uniform(1, 3))
            
            # Deduplicate URLs
            unique_results = []
            seen_urls = set()
            for result in all_search_results:
                url = result.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)
            
            logging.info(f"📊 Found {len(unique_results)} unique sources to process")
            
            # Process each source
            for result in unique_results[:max_sources]:
                if processed_sources >= max_sources:
                    break
                
                try:
                    url = result['url']
                    title = result['title']
                    
                    logging.info(f"📄 Processing source {processed_sources + 1}/{max_sources}: {title}")
                    
                    # Wikipedia — use direct API for clean content
                    if 'wikipedia.org' in url:
                        logging.info(f"🔍 Detected Wikipedia source, using direct API access")
                        
                        # URL format: https://en.wikipedia.org/wiki/Page_Title
                        page_title = url.split('/wiki/')[-1].replace('_', ' ')
                        
                        wiki_content = self._get_wikipedia_content_direct(page_title)
                        
                        if wiki_content:
                            processed_sources += 1
                            knowledge_items = self._create_wikipedia_knowledge_items(wiki_content, topic)
                            logging.info(f"📚 Wikipedia API: Created {len(knowledge_items)} knowledge items")
                        else:
                            logging.warning(f"⚠️ Could not fetch Wikipedia content via API for: {page_title}")
                            continue
                    else:
                        # Non-Wikipedia — use existing web scraping pipeline
                        content = self._fetch_webpage_content_enhanced(url)
                        
                        if not content:
                            logging.warning(f"⚠️ Could not fetch content from {url}")
                            continue
                        
                        processed_sources += 1
                        
                        # Extract knowledge using AI or content-based method
                        if ai_available:
                            knowledge_items = self._extract_knowledge_with_ai_improved(
                                content, topic, description, url, title
                            )
                        else:
                            knowledge_items = self._create_content_based_knowledge_item(
                                content, topic, url, title
                            )
                    
                    # ================================================================
                    # STORAGE SECTION — uses chatbot's two-phase commit method
                    # ================================================================
                    if knowledge_items:
                        for item in knowledge_items if isinstance(knowledge_items, list) else [knowledge_items]:
                            try:
                                # Final sanitization check before storage
                                content_to_store = item.get('content', '')
                                sanitized = self._sanitize_text_for_storage(content_to_store, url)
                                
                                if sanitized:
                                    item['content'] = sanitized
                                    item['transaction_id'] = transaction_id
                                    
                                    # Build metadata dict (exclude content — passed separately)
                                    metadata = {k: v for k, v in item.items() if k != 'content'}
                                    
                                    # Storage context
                                    metadata['source'] = 'web_knowledge'   # Normalized source — matches WEB_SEARCH STORE path
                                    metadata['source_url'] = url           # Preserve actual URL for provenance/debugging
                                    metadata['source_type'] = 'web_search'
                                    metadata['search_topic'] = item.get('topic', topic)
                                    metadata['search_query'] = item.get('search_query', topic)
                                    metadata['transaction_id'] = transaction_id
                                    metadata['extraction_method'] = item.get('extraction_method', 'unknown')
                                    metadata['content_type'] = item.get('content_type', 'web_content')
                                    metadata['source_quality'] = item.get('source_quality', 'medium')
                                    
                                    # Tags for better retrieval
                                    tags = ['web_knowledge', topic.lower()]
                                    if item.get('categories'):
                                        tags.extend(item.get('categories', [])[:3])
                                    metadata['tags'] = tags
                                    
                                    # Confidence based on extraction method and source quality
                                    extraction_method = item.get('extraction_method', 'unknown')
                                    source_quality = item.get('source_quality', 'medium')
                                    
                                    if source_quality == 'high':
                                        confidence = 0.95
                                    elif 'ai_extraction' in extraction_method or 'wikipedia' in extraction_method:
                                        confidence = 0.90
                                    elif 'content_based' in extraction_method:
                                        confidence = 0.75
                                    else:
                                        confidence = 0.80
                                    
                                    # Store via chatbot's two-phase commit
                                    try:
                                        success, memory_id = self.chatbot.store_memory_with_transaction(
                                            content=sanitized,
                                            memory_type='web_knowledge',
                                            metadata=metadata,
                                            confidence=confidence,
                                            duplicate_threshold=0.95
                                        )
                                        
                                        if success:
                                            items_stored += 1
                                            logging.info(f"✅ Stored knowledge item from {url} with ID {memory_id}")
                                        else:
                                            logging.warning(f"⚠️ Failed to store knowledge item from {url}")
                                            
                                    except Exception as storage_error:
                                        logging.error(f"Error storing via chatbot.store_memory_with_transaction: {storage_error}")
                                else:
                                    logging.warning(f"⚠️ Content failed final sanitization for {url}")
                                    
                            except Exception as store_error:
                                logging.error(f"Error storing knowledge item: {store_error}")
                    
                except Exception as source_error:
                    logging.error(f"Error processing source {result.get('url', 'unknown')}: {source_error}")
                    continue
            
            # Build results summary
            results_summary = {
                'transaction_id': transaction_id,
                'topic': topic,
                'description': description,
                'sources_found': len(unique_results),
                'sources_processed': processed_sources,
                'items_stored': items_stored,
                'search_queries_used': search_queries,
                'timestamp': datetime.now().isoformat()
            }
            
            logging.info(
                f"✅ Knowledge search complete: {items_stored} items stored from "
                f"{processed_sources} sources (found {len(unique_results)} total sources)"
            )
            
            return results_summary
            
        except Exception as e:
            logging.error(f"❌ Critical error in search_and_learn: {e}", exc_info=True)
            return {
                'transaction_id': str(uuid.uuid4()),
                'topic': topic,
                'error': str(e),
                'items_stored': 0,
                'timestamp': datetime.now().isoformat()
            }

    # ========================================================================
    # AI KNOWLEDGE EXTRACTION METHODS
    # ========================================================================

    def _extract_knowledge_with_ai_improved(self, content: str, topic: str, description: str, 
                                        source_url: str, title: str) -> List[Dict[str, Any]]:
        """
        AI knowledge extraction with multiple fallback levels.
        Tries strict → moderate → flexible extraction prompts, then falls
        back to content-based extraction if AI is unavailable or all approaches fail.
        
        Args:
            content: Page content to extract from
            topic: Target topic
            description: Additional context
            source_url: Source URL
            title: Page title
            
        Returns:
            List[Dict[str, Any]]: Extracted knowledge items
        """
        try:
            if not self.chatbot or not hasattr(self.chatbot, 'llm'):
                logging.warning("AI chatbot not available, using content-based extraction")
                return self._create_content_based_knowledge_item(content, topic, source_url, title)
            
            # Use chunking for long content to avoid truncation
            if len(content) > 12000:
                logging.info(f"Long content ({len(content)} chars), using chunked extraction")
                return self._extract_knowledge_with_chunking(
                    content, topic, description, source_url, title
                )
            
            # Try multiple extraction approaches with decreasing strictness
            extraction_attempts = [
                ('strict', self._create_strict_extraction_prompt),
                ('moderate', self._create_moderate_extraction_prompt),
                ('flexible', self._create_flexible_extraction_prompt)
            ]
            
            for approach_name, prompt_creator in extraction_attempts:
                try:
                    logging.info(f"🤖 Trying {approach_name} AI extraction for {source_url}")
                    
                    prompt = prompt_creator(content, topic, description, source_url, title)
                    ai_response = self.chatbot.llm.invoke(prompt)
                    
                    if ai_response and "NO_RELEVANT_CONTENT_FOUND" not in ai_response:
                        # Strip AI thinking patterns (DeepSeek chain-of-thought, etc.)
                        cleaned_response = self._strip_ai_thinking(ai_response)
                        
                        knowledge_items = self._parse_ai_extracted_knowledge_enhanced(
                            cleaned_response, topic, source_url, title, description
                        )
                        
                        if knowledge_items:
                            logging.info(f"🤖 {approach_name} extraction successful: {len(knowledge_items)} items")
                            return knowledge_items
                        else:
                            logging.info(f"🤖 {approach_name} extraction returned no parseable items")
                    else:
                        logging.info(f"🤖 {approach_name} extraction found no relevant content")
                        
                except Exception as ai_error:
                    logging.warning(f"🤖 {approach_name} extraction failed: {ai_error}")
                    continue
            
            # Final fallback — content-based extraction
            logging.info("🤖 All AI extraction methods failed, using content-based fallback")
            return self._create_content_based_knowledge_item(content, topic, source_url, title)
            
        except Exception as e:
            logging.error(f"Error in improved AI knowledge extraction: {e}")
            return self._create_content_based_knowledge_item(content, topic, source_url, title)
        
    def _extract_knowledge_with_chunking(self, content: str, topic: str, description: str, 
                                        source_url: str, title: str) -> List[Dict[str, Any]]:
        """
        Extract knowledge by processing content in overlapping chunks.
        Ensures no information is lost due to context window truncation.
        
        Args:
            content: Full page content (expected to be > 12000 chars)
            topic: Target topic
            description: Additional context
            source_url: Source URL
            title: Page title
            
        Returns:
            List[Dict[str, Any]]: Deduplicated knowledge items from all chunks
        """
        try:
            chunk_size = 12000  # Characters per chunk
            overlap = 2000      # Overlap to avoid cutting mid-sentence
            
            # If content fits, process in one go
            if len(content) <= chunk_size:
                logging.info(f"Content short ({len(content)} chars), single chunk")
                return self._extract_knowledge_with_ai_improved(
                    content, topic, description, source_url, title
                )
            
            # Split into overlapping chunks at sentence boundaries
            chunks = []
            start = 0
            chunk_num = 0
            
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk = content[start:end]
                
                # Try to end on a sentence boundary
                if end < len(content):
                    last_period = chunk.rfind('.', -500)
                    if last_period > 0:
                        chunk = chunk[:last_period + 1]
                        end = start + last_period + 1
                
                chunks.append({
                    'content': chunk,
                    'start': start,
                    'end': end,
                    'number': chunk_num
                })
                
                chunk_num += 1
                start = end - overlap if end < len(content) else len(content)
            
            logging.info(f"📄 Processing {len(chunks)} chunks for {source_url}")
            
            # Extract from each chunk
            all_knowledge = []
            successful_chunks = 0
            
            for i, chunk_info in enumerate(chunks):
                try:
                    logging.info(f"  Chunk {i+1}/{len(chunks)}")
                    
                    chunk_knowledge = self._extract_knowledge_with_ai_improved(
                        chunk_info['content'], 
                        topic, 
                        description, 
                        source_url,
                        f"{title} (Section {i+1}/{len(chunks)})"
                    )
                    
                    if chunk_knowledge:
                        for item in chunk_knowledge:
                            item['chunk_number'] = i + 1
                            item['total_chunks'] = len(chunks)
                        
                        all_knowledge.extend(chunk_knowledge)
                        successful_chunks += 1
                        logging.info(f"    ✅ Extracted {len(chunk_knowledge)} items")
                        
                except Exception as chunk_error:
                    logging.error(f"    ❌ Chunk {i+1} error: {chunk_error}")
                    continue
            
            logging.info(f"📊 {successful_chunks}/{len(chunks)} chunks successful, {len(all_knowledge)} items total")
            
            # Deduplicate overlapping content from adjacent chunks
            if len(all_knowledge) > 1:
                deduplicated = self._deduplicate_knowledge_items(all_knowledge)
                logging.info(f"📊 Deduplicated: {len(all_knowledge)} → {len(deduplicated)}")
                return deduplicated
            
            return all_knowledge
            
        except Exception as e:
            logging.error(f"Chunking error: {e}", exc_info=True)
            # Fallback to first 15000 chars
            return self._extract_knowledge_with_ai_improved(
                content[:15000], topic, description, source_url, title
            )

    def _deduplicate_knowledge_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate or highly similar knowledge items using Jaccard similarity.
        Used after chunked extraction to eliminate overlapping content.
        
        Args:
            items: List of knowledge items to deduplicate
            
        Returns:
            List[Dict[str, Any]]: Deduplicated items
        """
        try:
            if not items:
                return []
            
            deduplicated = []
            seen_content = set()
            
            for item in items:
                content = item.get('content', '').strip()
                
                # Normalize for comparison
                simplified = content.lower()
                simplified = re.sub(r'\s+', ' ', simplified)
                simplified = re.sub(r'[^\w\s]', '', simplified)
                
                # Skip exact duplicates
                if simplified in seen_content:
                    continue
                
                # Skip near-duplicates (>85% similar)
                is_duplicate = False
                for existing in seen_content:
                    if self._calculate_content_similarity(simplified, existing) > 0.85:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    deduplicated.append(item)
                    seen_content.add(simplified)
            
            return deduplicated
            
        except Exception as e:
            logging.error(f"Deduplication error: {e}")
            return items

    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using Jaccard word overlap.
        
        Args:
            text1: First text (should already be normalized/lowercased)
            text2: Second text (should already be normalized/lowercased)
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        try:
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0

    # ========================================================================
    # EXTRACTION PROMPT BUILDERS
    # ========================================================================

    def _create_flexible_extraction_prompt(self, content: str, topic: str, description: str, 
                                        source_url: str, title: str) -> str:
        """
        Create a flexible extraction prompt that accepts broader relevance.
        Used as the last AI extraction attempt before content-based fallback.
        
        Args:
            content: Page content (will be truncated to 15000 chars)
            topic: Target topic
            description: Context description
            source_url: Source URL
            title: Page title
            
        Returns:
            str: Formatted prompt for AI
        """
        return f"""
        Extract useful information from this content that relates to: "{topic}"

        Content from: {title} ({source_url})
        Target: {topic}
        Context: {description}

        Content:
        ```
        {content[:15000]}
        ```

        Instructions:
        - Look for ANY information related to "{topic}" or similar concepts
        - Include background information, related topics, or contextual details
        - Each piece of information should be factual and complete
        - If you find relevant information, format as numbered points
        - If truly no relevant information exists, respond with: "NO_RELEVANT_CONTENT_FOUND"

        Extract information:
        """

    def _create_strict_extraction_prompt(self, content: str, topic: str, description: str, 
                                    source_url: str, title: str) -> str:
        """
        Create a strict extraction prompt that only accepts highly relevant content.
        Used as the first AI extraction attempt.
        
        Args:
            content: Page content (will be truncated to 15000 chars)
            topic: Target topic
            description: Context description
            source_url: Source URL
            title: Page title
            
        Returns:
            str: Formatted prompt for AI
        """
        return f"""
        I need to extract ONLY information that DIRECTLY addresses this specific knowledge gap.

        KNOWLEDGE GAP:
        Topic: {topic}
        Context: {description}

        SOURCE:
        Title: {title}
        URL: {source_url}
        
        Content:
        ```
        {content[:15000]}
        ```

        STRICT REQUIREMENTS:
        - Extract ONLY information that directly explains or describes "{topic}"
        - Ignore tangentially related information
        - Each extracted point must be a complete, factual statement
        - Format as numbered points (1., 2., 3., etc.)
        - If NO directly relevant information exists, respond ONLY with: "NO_RELEVANT_CONTENT_FOUND"

        Extract information:
        """

    def _create_moderate_extraction_prompt(self, content: str, topic: str, description: str,
                                         source_url: str, title: str) -> str:
        """
        Create a moderate extraction prompt that balances relevance and breadth.
        Used as the second AI extraction attempt if strict fails.
        
        Args:
            content: Page content (will be truncated to 15000 chars)
            topic: Target topic
            description: Context description
            source_url: Source URL
            title: Page title
            
        Returns:
            str: Formatted prompt for AI
        """
        return f"""
        Extract relevant information about "{topic}" from the content below.

        Source: {title} ({source_url})
        Focus: {topic}
        Context: {description}

        Content:
        ```
        {content[:15000]}
        ```

        Instructions:
        - Extract information that is clearly relevant to "{topic}"
        - Include key facts, explanations, or examples
        - Format as numbered points
        - If no relevant information found, respond with: "NO_RELEVANT_CONTENT_FOUND"

        Extract information:
        """

    def _parse_ai_extracted_knowledge_enhanced(self, ai_response: str, topic: str, 
                                              source_url: str, title: str, 
                                              description: str) -> List[Dict[str, Any]]:
        """
        Parse AI-extracted knowledge into structured items.
        Splits numbered points into individual items; treats the full response
        as a single item if no numbered points are found.
        
        Args:
            ai_response: Raw AI response (already cleaned by _strip_ai_thinking)
            topic: Original topic
            source_url: Source URL
            title: Page title
            description: Original description
            
        Returns:
            List[Dict[str, Any]]: Parsed knowledge items
        """
        try:
            knowledge_items = []
            
            # Split by numbered points (1., 2., 3., etc.)
            points = re.split(r'\n\s*\d+\.\s+', ai_response)
            
            # First element might be preamble — skip if too short
            if len(points) > 1 and len(points[0].strip()) < 50:
                points = points[1:]
            
            for point in points:
                point_clean = point.strip()
                
                # Strip any residual numbered list prefix left by the split
                # Handles "1. content", ". content" (orphaned period), "12. content"
                # The split regex consumes \n\d+.\s+ but can leave ". " when no leading \n exists
                point_clean = re.sub(r'^\d*\.\s+', '', point_clean)
                
                # Skip empty or very short points
                if len(point_clean) < self.min_content_length:
                    continue
                
                knowledge_items.append({
                    'content': point_clean,
                    'topic': topic,
                    'source': source_url,
                    'title': title,
                    'search_query': topic,
                    'relevance_score': self._calculate_relevance_score(point_clean, topic, description),
                    'extracted_at': datetime.now().isoformat()
                })
            
            # If no numbered points extracted, treat entire response as one item
            if not knowledge_items and len(ai_response.strip()) >= self.min_content_length:
                knowledge_items.append({
                    'content': ai_response.strip(),
                    'topic': topic,
                    'source': source_url,
                    'title': title,
                    'search_query': topic,
                    'relevance_score': self._calculate_relevance_score(ai_response, topic, description),
                    'extracted_at': datetime.now().isoformat()
                })
            
            return knowledge_items
            
        except Exception as e:
            logging.error(f"Error parsing AI extracted knowledge: {e}")
            return []

    def _create_content_based_knowledge_item(self, content: str, topic: str, 
                                            source_url: str, title: str) -> List[Dict[str, Any]]:
        """
        Create knowledge item directly from content when AI is unavailable.
        Truncates to 10000 chars and packages as a single item.
        
        Args:
            content: Page content
            topic: Target topic
            source_url: Source URL
            title: Page title
            
        Returns:
            List[Dict[str, Any]]: Content-based knowledge items
        """
        try:
            # Truncate content to a reasonable size
            max_content = 10000
            truncated_content = content[:max_content] if len(content) > max_content else content
            
            item = {
                'content': truncated_content,
                'topic': topic,
                'source': source_url,
                'title': title,
                'search_query': topic,
                'relevance_score': 0.6,  # Moderate relevance for content-based
                'extraction_method': 'content-based',
                'extracted_at': datetime.now().isoformat()
            }
            
            return [item]
            
        except Exception as e:
            logging.error(f"Error creating content-based knowledge item: {e}")
            return []

    def _calculate_relevance_score(self, content: str, topic: str, description: str) -> float:
        """
        Calculate relevance score for extracted content using keyword matching.
        
        Args:
            content: Extracted content
            topic: Target topic
            description: Original description
            
        Returns:
            float: Relevance score between 0.3 and 1.0
        """
        try:
            content_lower = content.lower()
            topic_lower = topic.lower()
            
            # Base score
            score = 0.5
            
            # Topic mention count — max +0.3
            topic_mentions = content_lower.count(topic_lower)
            score += min(topic_mentions * 0.05, 0.3)
            
            # Description keyword matches — max +0.2
            if description:
                desc_words = set(re.findall(r'\b\w{4,}\b', description.lower()))
                content_words = set(re.findall(r'\b\w{4,}\b', content_lower))
                matches = len(desc_words & content_words)
                score += min(matches * 0.02, 0.2)
            
            # Content length quality — prefer substantial content
            if len(content) > 200:
                score += 0.1
            
            # Technical depth indicators
            tech_indicators = ['how', 'why', 'example', 'method', 'process', 'step']
            tech_score = sum(0.02 for indicator in tech_indicators if indicator in content_lower)
            score += min(tech_score, 0.1)
            
            # Floor and cap
            score = max(score, 0.3)
            score = min(score, 1.0)
            
            return round(score, 2)
            
        except Exception as e:
            logging.error(f"Error calculating relevance score: {e}")
            return 0.5

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 4 web cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Docstring says "backwards compatibility wrapper" but no caller exists for any version of the API.
    # The active code paths build knowledge items as inline dicts (see search_for_knowledge and search_and_learn)
    # rather than calling this constructor.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__create_knowledge_item(self, content: str, topic: str, source_url: str, title: str) -> Dict[str, Any]:
        """
        Create a basic structured knowledge item (backwards compatibility wrapper).
        
        Args:
            content: Knowledge content
            topic: Topic
            source_url: Source URL
            title: Page title
            
        Returns:
            Dict[str, Any]: Structured knowledge item
        """
        try:
            return {
                'content': content.strip(),
                'topic': topic,
                'source': source_url,
                'title': title,
                'search_query': topic,
                'transaction_id': str(uuid.uuid4()),
                'items_stored': 1,
                'relevance_score': 0.7,
                'extracted_at': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error creating knowledge item: {e}")
            return {}

    # ========================================================================
    # UTILITY AND DEBUGGING METHODS
    # ========================================================================

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 4 web cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Diagnostic utility for surfacing blacklist state and search engine config to UI/admin.
    # Was likely intended for the admin dashboard but never wired up. No callers anywhere.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about search performance and blacklisted domains.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        try:
            stats = {
                'blacklisted_domains': len(self.failed_domains),
                'blacklisted_domain_list': list(self.failed_domains.keys()),
                'searx_instances': len(self.searx_instances),
                'searx_instance_list': self.searx_instances,
                'current_searx_instance': self.searx_instances[self.current_searx_index] if self.searx_instances else None,
                'search_engines': self.search_engines,
                'min_content_length': self.min_content_length,
                'request_interval': f"{self.min_request_interval}-{self.max_request_interval} seconds",
                'text_quality_thresholds': {
                    'max_replacement_char_ratio': self.max_replacement_char_ratio,
                    'max_non_printable_ratio': self.max_non_printable_ratio
                }
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting search statistics: {e}")
            return {}

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 4 web cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Manual reset of the domain blacklist. The blacklist self-manages via TTL expiry inside
    # _load_blacklist() and per-entry expiry checks in _is_domain_blacklisted(), so this
    # manual flush is never needed in the active flow. No callers anywhere.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_clear_blacklist(self):
        """
        Clear the domain blacklist entirely. Useful for testing or manual reset.
        """
        try:
            self.failed_domains.clear()
            self._save_blacklist()
            logging.info("🧹 Domain blacklist cleared")
            
        except Exception as e:
            logging.error(f"Error clearing blacklist: {e}")