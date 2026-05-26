# web_crawler.py
"""Web crawler for autonomous knowledge acquisition."""

import logging
import uuid
import re
import time
import datetime  
from threading import Semaphore
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from typing import List, Tuple
from document_reader import DocumentReader
from vector_db import VectorDB
from memory_db import MemoryDB
import os
from config import DOCS_PATH

# Configure logging for key events only
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lock file path — updated to reflect new module name
LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_crawler.lock")

# Path to the url_paths.txt file (in project root)
URL_PATHS_FILE = os.path.join(os.path.dirname(DOCS_PATH), "url_paths.txt")

# Decorator defined at module level
def with_qdrant_retry(func):
    """Decorator to add retry logic for Qdrant operations."""
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 2  # seconds
        
        # For file-based Qdrant, simply try with retries
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries:
                    logging.warning(f"Qdrant operation failed on attempt {attempt}, retrying: {e}")
                    time.sleep(retry_delay * attempt)  # Exponential backoff
                else:
                    logging.error(f"Qdrant operation failed after {max_retries} attempts: {e}")
                    return None  # Default fallback for all functions
    return wrapper

class WebCrawler:
    """Handles web crawling and AI-driven knowledge extraction using existing infrastructure."""
    
    def __init__(self, chatbot=None):
        """Initialize WebCrawler with optional chatbot reference."""
        self.doc_reader = DocumentReader()
        self.vector_db = VectorDB()
        self.memory_db = MemoryDB()
        self.chatbot = chatbot  # Store the chatbot reference

    def process_url_content(self, url: str) -> Tuple[str, bool]:
        """
        Fetch and process content from a URL.
        
        Args:
            url (str): URL to fetch content from
            
        Returns:
            Tuple[str, bool]: (content, success)
        """
        try:
            # Set timeout to prevent hanging
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                main_content = self._extract_main_content(soup)
                
                # Check if content is empty after extraction
                if not main_content:
                    logging.warning(f"Extracted empty content from URL {url}, AI processing will be skipped for this URL")
                    return "", False  # Return empty content and False to indicate failure
                    
                return main_content, True
            logging.error(f"Failed to fetch URL {url}: HTTP {response.status_code}")
            return "", False
        except requests.exceptions.Timeout:
            logging.error(f"Timeout while fetching URL {url}")
            return "", False
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error processing URL {url}: {e}")
            return "", False
        except Exception as e:
            logging.error(f"Error processing URL {url}: {e}")
            return "", False

    def process_web_content_as_web_knowledge(self, url: str, content: str, chatbot=None) -> Tuple[int, int]:
        """
        Process web content and store it specifically as web_knowledge type.
        
        Args:
            url (str): Source URL
            content (str): Web content to process
            chatbot (Chatbot): Chatbot instance for AI processing
            
        Returns:
            Tuple[int, int]: (Total chunks, stored items)
        """
        try:
            logging.info(f"🌐 EXTRACTING WEB KNOWLEDGE FROM: {url}")
            
            if not chatbot:
                logging.error("No chatbot available for web knowledge processing")
                return 0, 0
            
            # Enhanced AI extraction prompt for web content
            extraction_prompt = f"""Analyze this web content and extract the most valuable information for future reference.

    Source URL: {url}
    Content: {content[:4000]}...

    Extract key information in this format:
    1. **Main Topic:** [What is this content about?]
    2. **Key Points:** 
    - [Most important fact 1]
    - [Most important fact 2]
    - [Most important fact 3]
    3. **Practical Value:** [Why is this information useful?]
    4. **Summary:** [2-3 sentence comprehensive summary]

    Focus on factual, actionable information that would be valuable to remember and search for later."""

            try:
                extracted_info = chatbot.llm.invoke(extraction_prompt)
                
                if not extracted_info or not extracted_info.strip():
                    logging.warning(f"No information extracted from {url}")
                    return 0, 0
                
                # Create enhanced metadata for web knowledge
                domain = self._extract_domain_from_url(url)
                metadata = {
                    "type": "web_knowledge",
                    "source": url,
                    "original_url": url,
                    "domain": domain,
                    "extraction_method": "ai_driven",
                    "content_length": len(content),
                    "extracted_at": datetime.datetime.now().isoformat(),
                    "tags": f"web_knowledge,url,{domain}",
                    "confidence": 0.5  # Placeholder confidence score
                }
                
                # Store as web_knowledge using transaction coordination.
                # duplicate_threshold=0.95 matches web_knowledge_seeker.py — prevents
                # storing near-identical re-extractions of the same URL across runs.
                success, memory_id = chatbot.store_memory_with_transaction(
                    content=f"Web Knowledge from {url}:\n\n{extracted_info}",
                    memory_type="web_knowledge",
                    metadata=metadata,
                    confidence=0.5,
                    duplicate_threshold=0.95
                )
                
                if success:
                    logging.info(f"✅ Successfully stored web knowledge from {url} with ID {memory_id}")
                    return 1, 1
                else:
                    logging.error(f"❌ Failed to store web knowledge from {url}")
                    return 0, 0
                    
            except Exception as ai_error:
                logging.error(f"AI extraction error for {url}: {ai_error}")
                return 0, 0
                
        except Exception as e:
            logging.error(f"Error processing web content as web knowledge: {e}", exc_info=True)
            return 0, 0

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain name from URL for tagging."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            return domain.replace('.', '_')  # Replace dots for tag compatibility
        except:
            return "unknown_domain"

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from webpage and clean whitespace.
        
        Args:
            soup (BeautifulSoup): Parsed HTML soup object
            
        Returns:
            str: Extracted main content
        """
        try:
            # Special handling for specific sites
            page_html = str(soup)
            
            if 'wikipedia.org' in page_html:
                content_div = soup.find('div', {'id': 'mw-content-text'})
                if content_div:
                    text = content_div.get_text()
                else:
                    text = soup.get_text()
            elif 'arxiv.org' in page_html:
                # ArXiv-specific extraction
                # Try to get the paper listings
                listings = soup.select('dl')
                if listings:
                    text = ""
                    for listing in listings:
                        # Extract titles and abstracts
                        titles = listing.select('dt')
                        abstracts = listing.select('dd')
                        
                        # Process each paper
                        for i in range(min(len(titles), len(abstracts))):
                            title_text = titles[i].get_text(strip=True)
                            abstract_text = abstracts[i].get_text(strip=True)
                            paper_text = f"Title: {title_text}\nAbstract: {abstract_text}\n\n"
                            text += paper_text
                else:
                    # If it's a single paper page
                    title = soup.select_one('.title')
                    abstract = soup.select_one('.abstract')
                    authors = soup.select_one('.authors')
                    
                    text = ""
                    if title:
                        text += f"Title: {title.get_text(strip=True)}\n"
                    if authors:
                        text += f"Authors: {authors.get_text(strip=True)}\n"
                    if abstract:
                        text += f"Abstract: {abstract.get_text(strip=True)}\n"
                    
                    if not text:
                        # Fallback to body content
                        text = soup.get_text()
            else:
                # Try to find main content for other sites
                main_content = None
                for tag in ['main', 'article', 'div.content', 'div.main', '#content', '.content']:
                    # Use soup.select_one for CSS selectors or soup.find for standard tag selection
                    if '.' in tag or '#' in tag:  # It's a CSS selector
                        main_content = soup.select_one(tag)
                    else:  # It's a tag name
                        main_content = soup.find(tag)
                    
                    if main_content:
                        text = main_content.get_text()
                        break
                else:  # No specific content container found
                    text = soup.get_text()
            
            # Clean up whitespace
            cleaned_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
            
            # If we ended up with empty content, fall back to full page
            if not cleaned_text:
                logging.warning("Content extraction resulted in empty text, falling back to full page")
                cleaned_text = soup.get_text()
                
            return cleaned_text
        except Exception as e:
            logging.error(f"Error extracting main content: {e}", exc_info=True)
            # Return empty string instead of None when extraction fails
            return ""
        
    def read_url_paths(self) -> List[str]:
        """
        Read URLs from the url_paths.txt file.

        Returns:
            List[str]: List of URLs to crawl
        """
        urls = []
        try:
            # Check if the file exists
            if not os.path.exists(URL_PATHS_FILE):
                logging.error(f"URL paths file not found: {URL_PATHS_FILE}")
                # File should exist in project root — missing file indicates a configuration issue
                return []
        
            # Read the file
            with open(URL_PATHS_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        urls.append(line)
        
            logging.info(f"Read {len(urls)} URLs from url_paths.txt")
            return urls
        except Exception as e:
            logging.error(f"Error reading url_paths.txt: {e}", exc_info=True)
            return []

    @with_qdrant_retry    
    def process_web_content_with_ai(self, url: str, content: str, chatbot=None) -> Tuple[int, int]:
        """
        Process web content using AI selection and store as web_knowledge type.

        Args:
            url (str): Source URL
            content (str): Web content to process
            chatbot (Chatbot): Chatbot instance for AI processing

        Returns:
            Tuple[int, int]: (Total chunks, stored items)
        """
        logging.info(f"🌐 STARTING AI-DRIVEN WEB KNOWLEDGE PROCESSING for URL: {url}")

        try:
            if not chatbot:
                logging.error("No chatbot available for web knowledge processing")
                return 0, 0
            
            # Process web content and store as web_knowledge (single storage approach)
            chunks, stored_items = self.process_web_content_as_web_knowledge(url, content, chatbot)
            
            if stored_items > 0:
                logging.info(f"🌐 WEB KNOWLEDGE PROCESSING COMPLETE for {url}: {stored_items} items stored as web_knowledge")
            else:
                logging.warning(f"🌐 WEB KNOWLEDGE PROCESSING COMPLETED BUT NO ITEMS STORED for {url}")
            
            return chunks, stored_items
            
        except Exception as e:
            logging.error(f"❌ ERROR in AI web knowledge processing: {e}", exc_info=True)
            return 0, 0
   
    @with_qdrant_retry
    def process_learning_path(self, url_or_max_pages=10, max_pages=None, context=None, bypass_lock=False):
        """
        Process URLs from url_paths.txt with AI-driven selection only.
        No fallback to basic chunking — requires AI processing for all content.
        
        Args:
            url_or_max_pages: Can be either a URL (string) or max pages limit (int)
            max_pages: Alternative way to specify max pages (int)
            context: Optional context dict with knowledge gap information
            bypass_lock: If True, bypass the lock file check
            
        Returns:
            bool: True if processing completed successfully, False otherwise
        """
        # Check if lock file exists, but bypass the check if requested
        if not bypass_lock and os.path.exists(LOCK_FILE):
            logging.info("Web crawler is already running. Skipping execution.")
            return False

        # Create lock file to prevent concurrent runs
        try:
            with open(LOCK_FILE, 'w') as f:
                f.write(str(os.getpid()))
            logging.info("🔒 Lock file created. Proceeding with crawl.")
        except Exception as e:
            logging.error(f"Failed to create lock file: {e}", exc_info=True)
            return False

        try:
            # Validate AI processing availability upfront — no fallback exists
            if not self.chatbot:
                logging.error("❌ CRITICAL: AI processing required but chatbot not available. Cannot proceed without AI-driven content selection.")
                return False
                
            # Log the input parameters and context if available
            context_log = ""
            if context and context.get('is_knowledge_gap'):
                context_log = f" with knowledge gap focus: '{context.get('topic')}'"
                
            logging.info(f"🧠 AI-ONLY processing mode: url_or_max_pages={url_or_max_pages}, max_pages={max_pages}{context_log}")

            # Better URL detection - check if it's a valid URL string
            is_url = isinstance(url_or_max_pages, str) and (
                url_or_max_pages.startswith('http://') or 
                url_or_max_pages.startswith('https://') or
                url_or_max_pages.startswith('www.')
            )

            # Determine the actual max_pages value to use
            actual_max_pages = max_pages if max_pages is not None else (
                10 if is_url else url_or_max_pages if isinstance(url_or_max_pages, int) else 10
            )

            # Determine the URLs to process
            if is_url:
                # If url_or_max_pages is a URL, process just that URL
                start_urls = [url_or_max_pages]
                logging.info(f"🧠 Processing single URL with AI selection: {url_or_max_pages}")
            else:
                # Otherwise, read URLs from url_paths.txt
                start_urls = self.read_url_paths()
                logging.info(f"🧠 Processing URLs from url_paths.txt with AI selection, max_pages={actual_max_pages}")

            if not start_urls:
                logging.warning("❌ No URLs found to process.")
                return False

            processed_urls = set()
            pages_processed = 0
            successful_ai_processing = 0
            failed_ai_processing = 0

            # Process each starting URL
            for start_url in start_urls:
                try:
                    # Validate URL format
                    if not (start_url.startswith('http://') or start_url.startswith('https://')):
                        start_url = 'https://' + start_url.lstrip('www.')
                        logging.info(f"URL format adjusted to: {start_url}")
        
                    urls_to_process = [start_url]
                    logging.info(f"🧠 Starting AI-driven crawl with URL: {start_url}")
                
                    # Process URLs breadth-first up to the maximum number of pages
                    while urls_to_process and pages_processed < actual_max_pages:
                        current_url = urls_to_process.pop(0)
                    
                        # Skip if already processed
                        if current_url in processed_urls:
                            logging.info(f"Skipping already processed URL: {current_url}")
                            continue
                    
                        # Process the URL content
                        logging.info(f"🧠 Processing URL with AI selection: {current_url}")
                        content, success = self.process_url_content(current_url)
                    
                        if not success:
                            logging.warning(f"❌ Failed to fetch content from URL: {current_url}")
                            failed_ai_processing += 1
                            continue
                            
                        if not content.strip():
                            logging.warning(f"❌ Empty content retrieved from URL: {current_url}")
                            failed_ai_processing += 1
                            continue
                        
                        # Add to processed URLs
                        processed_urls.add(current_url)
                        pages_processed += 1
                    
                        # Add a clear divider in logs to make AI processing very visible
                        logging.info(f"{'='*60}")
                        logging.info(f"🧠 AI-DRIVEN CONTENT SELECTION FOR: {current_url}")
                        logging.info(f"{'='*60}")
                    
                        # Process content using AI-driven approach (only method)
                        chunks = 0
                        stored = 0
                        
                        try:
                            if context and context.get('is_knowledge_gap'):
                                # Knowledge gap focused processing
                                logging.info(f"🎯 KNOWLEDGE GAP FOCUSED AI PROCESSING")
                                logging.info(f"🎯 Target topic: '{context.get('topic')}'")
                                logging.info(f"🎯 Description: '{context.get('description', 'Not specified')}'")
                                
                                # Create a temporary file with the content for AI processing
                                temp_filename = f"knowledge_gap_{context.get('topic', 'unknown').replace(' ', '_')}_{current_url.replace('://', '_').replace('/', '_')[:30]}.txt"
                                temp_path = os.path.join(DOCS_PATH, temp_filename)
                                
                                try:
                                    with open(temp_path, 'w', encoding='utf-8') as f:
                                        f.write(content)
                                    
                                    # Prepare a specialized prompt for knowledge gap focus
                                    custom_prompt = f"""
                                    Analyze this web content and extract ONLY information related to: {context.get('topic')}
                                    
                                    Context about this knowledge gap:
                                    {context.get('description')}
                                    
                                    ONLY extract information specifically addressing this topic.
                                    Ignore all irrelevant information.
                                    
                                    Format each insight as a complete, self-contained piece of knowledge.
                                    """
                                    
                                    # Process document directly through document reader
                                    logging.info(f"🎯 Processing knowledge gap document: topic={context.get('topic')}, source={current_url}")

                                    # Process the document (generates summary)
                                    result = self.chatbot.doc_reader.process_uploaded_document(temp_filename)

                                    # If successful, store additional knowledge gap context as a separate memory
                                    if result and "successfully" in result.lower():
                                        try:
                                            # Create a knowledge gap context entry
                                            gap_context_content = (
                                                f"Knowledge Gap Research - {context.get('topic')}\n"
                                                f"Source: {current_url}\n"
                                                f"Description: {context.get('description', 'Not specified')}\n"
                                                f"Document: {temp_filename}\n"
                                                f"This content addresses the identified knowledge gap about: {context.get('topic')}"
                                            )
                                            
                                            # Store the context with metadata
                                            success, context_id = self.chatbot.store_memory_with_transaction(
                                                content=gap_context_content,
                                                memory_type="knowledge_gap_research",
                                                metadata={
                                                    "type": "knowledge_gap_research",
                                                    "topic": context.get('topic'),
                                                    "source_url": current_url,
                                                    "source_document": temp_filename,
                                                    "tags": f"knowledge_gap,{context.get('topic')},research"
                                                },
                                                confidence=0.5
                                            )
                                            
                                            if success:
                                                logging.info(f"🎯 Knowledge gap context stored: {context_id}")
                                            else:
                                                logging.warning(f"⚠️ Failed to store knowledge gap context")
                                                
                                        except Exception as context_error:
                                            logging.error(f"❌ Error storing knowledge gap context: {context_error}", exc_info=True)
                                    
                                    # Clean up temp file
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                        
                                    # Parse the result to determine success
                                    if isinstance(result, str):
                                        if "Successfully processed" in result or "Document Summary" in result:
                                            # Extract stored items count if available
                                            count_match = re.search(r'Extracted (\d+) pieces', result)
                                            stored = int(count_match.group(1)) if count_match else 1
                                            chunks = 1
                                            
                                            # Log success with context info
                                            logging.info(f"🎯 KNOWLEDGE GAP SUCCESS: {stored} targeted items about '{context.get('topic')}' stored from {current_url}")
                                            successful_ai_processing += 1
                                        else:
                                            logging.warning(f"🎯 KNOWLEDGE GAP NO RESULTS: AI found no relevant content for '{context.get('topic')}' in {current_url}")
                                            logging.warning(f"AI response preview: {result[:150]}...")
                                            failed_ai_processing += 1
                                    else:
                                        logging.error(f"🎯 KNOWLEDGE GAP ERROR: Unexpected result type {type(result)} from AI processing")
                                        failed_ai_processing += 1
                                            
                                except Exception as gap_error:
                                    logging.error(f"❌ Error in knowledge gap AI processing for {current_url}: {gap_error}", exc_info=True)
                                    failed_ai_processing += 1
                                    # Clean up temp file in case of error
                                    if 'temp_path' in locals() and os.path.exists(temp_path):
                                        try:
                                            os.remove(temp_path)
                                        except:
                                            pass
                                            
                            else:
                                # Standard AI-driven processing
                                logging.info(f"🧠 STANDARD AI-DRIVEN CONTENT SELECTION")
                                logging.info(f"🧠 AI will intelligently select high-value information from this content")
                                
                                try:
                                    chunks, stored = self.process_web_content_with_ai(current_url, content, self.chatbot)
                                    
                                    if stored > 0:
                                        logging.info(f"🧠 STANDARD AI SUCCESS: {stored} valuable items selected and stored from {current_url}")
                                        successful_ai_processing += 1
                                    else:
                                        logging.warning(f"🧠 STANDARD AI NO RESULTS: AI found no valuable content to store from {current_url}")
                                        failed_ai_processing += 1
                                        
                                except Exception as ai_error:
                                    logging.error(f"❌ Error in standard AI processing for {current_url}: {ai_error}", exc_info=True)
                                    failed_ai_processing += 1
                            
                        except Exception as processing_error:
                            logging.error(f"❌ Critical error during AI processing of {current_url}: {processing_error}", exc_info=True)
                            failed_ai_processing += 1
                        
                        # Log the completion with clear success/failure indicator
                        if stored > 0:
                            if context and context.get('is_knowledge_gap'):
                                method_indicator = "🎯 KNOWLEDGE GAP AI"
                            else:
                                method_indicator = "🧠 STANDARD AI"
                            logging.info(f"✅ {method_indicator} PROCESSING COMPLETE: {chunks} chunks processed, {stored} items stored from {current_url}")
                        else:
                            logging.warning(f"⚠️ AI PROCESSING YIELDED NO STORED CONTENT from {current_url}")
                        
                        logging.info(f"{'='*60}")
                    
                        # Extract more links to process (only if we haven't hit the page limit)
                        if pages_processed < actual_max_pages:
                            try:
                                # If focusing on knowledge gap, use specialized link extraction
                                if context and context.get('is_knowledge_gap'):
                                    new_links = self._extract_topic_focused_links(
                                        content, current_url, 
                                        context.get('topic'), 
                                        context.get('description', '')
                                    )
                                    logging.info(f"🎯 Extracted {len(new_links)} topic-focused links for '{context.get('topic')}'")
                                else:
                                    new_links = self._extract_relevant_links(content, current_url)
                                    logging.info(f"🧠 Extracted {len(new_links)} relevant links for continued crawling")
                                    
                                # Add new links that haven't been processed yet
                                added_links = 0
                                for link in new_links:
                                    if link not in processed_urls and link not in urls_to_process:
                                        urls_to_process.append(link)
                                        added_links += 1
                                        
                                if added_links > 0:
                                    logging.info(f"📋 Added {added_links} new URLs to processing queue")
                                    
                            except Exception as link_error:
                                logging.error(f"❌ Error extracting links from {current_url}: {link_error}", exc_info=True)
                
                except Exception as url_error:
                    logging.error(f"❌ Error processing starting URL {start_url}: {url_error}", exc_info=True)
                    failed_ai_processing += 1
                    continue

            # Final summary with detailed statistics
            total_attempts = successful_ai_processing + failed_ai_processing
            success_rate = (successful_ai_processing / total_attempts * 100) if total_attempts > 0 else 0
            
            logging.info(f"🏁 AI-DRIVEN CRAWL COMPLETE")
            logging.info(f"📊 STATISTICS:")
            logging.info(f"   • Total pages processed: {pages_processed}")
            logging.info(f"   • Successful AI extractions: {successful_ai_processing}")
            logging.info(f"   • Failed AI extractions: {failed_ai_processing}")
            logging.info(f"   • Success rate: {success_rate:.1f}%")
            
            if context and context.get('is_knowledge_gap'):
                logging.info(f"🎯 Knowledge gap topic: '{context.get('topic')}'")
                
            # Return True if we had any successful AI processing
            return successful_ai_processing > 0

        except Exception as e:
            logging.error(f"❌ CRITICAL ERROR in AI-driven crawl: {e}", exc_info=True)
            return False

        finally:
            # Remove lock file on exit (success or failure)
            try:
                if os.path.exists(LOCK_FILE):
                    os.remove(LOCK_FILE)
                    logging.info("🔓 Lock file removed.")
            except Exception as e:
                logging.error(f"❌ Failed to remove lock file: {e}", exc_info=True)

    def _extract_topic_focused_links(self, content: str, base_url: str, topic: str, description: str = '') -> List[str]:
        """
        Extract links from content that are specifically relevant to the knowledge gap topic.
        
        Args:
            content (str): HTML content to extract links from
            base_url (str): Base URL for resolving relative links
            topic (str): The knowledge gap topic to focus on
            description (str): Additional description of the knowledge gap
            
        Returns:
            List[str]: List of extracted topic-relevant URLs
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            candidate_links = []
            
            # Get the domain of the base URL for comparison
            from urllib.parse import urlparse
            base_domain = urlparse(base_url).netloc
            
            # Create keywords from the topic and description
            topic_words = set(topic.lower().split())
            description_words = set(description.lower().split())
            keywords = topic_words.union(set([word for word in description_words if len(word) > 3]))
            
            # Remove common words that aren't useful for matching
            common_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'what', 'how', 'when', 'where'}
            keywords = keywords - common_words
            
            logging.info(f"Searching for links relevant to topic keywords: {keywords}")
            
            for link in soup.find_all('a', href=True):
                try:
                    url = urljoin(base_url, link['href'])
                    
                    # Basic filtering
                    if not url.startswith(('http://', 'https://')):
                        continue
                    if url.split('#')[0] == base_url.split('#')[0]:  # Skip self-references
                        continue
                    if any(ext in url.lower() for ext in ['.pdf', '.jpg', '.png', '.mp4', '.zip']):
                        continue
                    
                    # Get the link text and surrounding context
                    link_text = link.get_text(strip=True).lower()
                    parent_text = link.parent.get_text(strip=True).lower() if link.parent else ""
                    
                    # Calculate topic relevance score
                    topic_score = 0.0
                    
                    # Check if topic words appear in URL, link text, or parent text
                    for keyword in keywords:
                        if keyword in url.lower():
                            topic_score += 2.0  # Strong indicator if topic is in URL
                        if keyword in link_text:
                            topic_score += 3.0  # Very strong indicator if topic is in link text
                        if keyword in parent_text:
                            topic_score += 1.0  # Moderate indicator if topic is in surrounding text
                    
                    # Boost score for links from the same domain
                    if urlparse(url).netloc == base_domain:
                        topic_score += 0.5
                    
                    # Only consider links with significant topic relevance
                    if topic_score >= 2.0:
                        candidate_links.append((url, topic_score))
                        
                except Exception as link_error:
                    continue
            
            # Sort by relevance score and take top 5
            candidate_links.sort(key=lambda x: x[1], reverse=True)
            top_links = [url for url, score in candidate_links[:5]]
            
            logging.info(f"Extracted {len(top_links)} links specifically relevant to '{topic}' from {base_url}")
            return top_links
            
        except Exception as e:
            logging.error(f"Error extracting topic-focused links for '{topic}' from {base_url}: {e}", exc_info=True)
            return []

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 4 web cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Helper that splits AI-extracted content into separate knowledge items by numbered
    # lists, bullet points, or blank-line paragraphs. Has zero callers internal or external.
    # The active extraction pipeline (process_web_content_with_ai → process_web_content_as_web_knowledge)
    # stores extracted content as a single unit instead of splitting.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED__split_into_knowledge_items(self, extracted_text):
        """Split extracted information into separate knowledge items."""
        # Try to split by numbered items
        if re.search(r'^\d+\.', extracted_text, re.MULTILINE):
            items = re.split(r'\n\d+\.', '\n' + extracted_text)
            if len(items) > 1:
                return [item.strip() for item in items[1:] if item.strip()]
        
        # Try to split by bullet points
        if re.search(r'^\s*[\*\-•]', extracted_text, re.MULTILINE):
            items = re.split(r'\n\s*[\*\-•]', '\n' + extracted_text)
            if len(items) > 1:
                return [item.strip() for item in items[1:] if item.strip()]
        
        # Try to split by blank lines (paragraphs)
        items = re.split(r'\n\s*\n', extracted_text)
        if len(items) > 1:
            return [item.strip() for item in items if item.strip()]
        
        # If all else fails, return the whole text as one item
        return [extracted_text.strip()]

    def _extract_relevant_links(self, content: str, base_url: str) -> List[str]:
        """
        Extract relevant links from content, with improved filtering for higher-quality sources.
    
        Args:
            content (str): HTML content to extract links from
            base_url (str): Base URL for resolving relative links
        
        Returns:
            List[str]: List of extracted relevant URLs
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            candidate_links = []
        
            # Get the domain of the base URL for comparison
            from urllib.parse import urlparse
            base_domain = urlparse(base_url).netloc
        
            # Define patterns for low-quality or irrelevant pages
            low_value_patterns = [
                'login', 'logout', 'signin', 'signup', 'register', 
                'cart', 'checkout', 'account', 'privacy', 'terms',
                'contact', 'shop', 'store', 'buy', 'pricing'
            ]
        
            # Define patterns that suggest higher-value content
            high_value_indicators = [
                'guide', 'tutorial', 'learn', 'article', 'blog', 
                'doc', 'resource', 'information', 'overview', 'about',
                'research', 'paper', 'study', 'report', 'analysis'
            ]
        
            for link in soup.find_all('a', href=True):
                try:
                    url = urljoin(base_url, link['href'])
                
                    # Basic filtering
                    if not url.startswith(('http://', 'https://')):
                        continue
                    if url.split('#')[0] == base_url.split('#')[0]:  # Skip self-references
                        continue
                    
                    # Skip low-value pages
                    if any(pattern in url.lower() for pattern in low_value_patterns):
                        continue
                
                    # Get the link text
                    link_text = link.get_text(strip=True)
                
                    # Calculate a simple relevance score
                    relevance_score = 1.0  # Base score
                
                    # Boost score for links from the same domain (likely more relevant)
                    link_domain = urlparse(url).netloc
                    if link_domain == base_domain:
                        relevance_score += 0.5
                
                    # Boost score for links with high-value indicators in URL or text
                    if any(indicator in url.lower() for indicator in high_value_indicators):
                        relevance_score += 0.7
                    if link_text and any(indicator in link_text.lower() for indicator in high_value_indicators):
                        relevance_score += 1.0
                    
                    # Append to candidates with score
                    candidate_links.append((url, relevance_score))
                
                except Exception as link_error:
                    logging.warning(f"Error processing link in {base_url}: {link_error}")
                    continue
        
            # Sort by relevance score and take top 5
            candidate_links.sort(key=lambda x: x[1], reverse=True)
            top_links = [url for url, score in candidate_links[:5]]
        
            logging.info(f"Extracted {len(top_links)} high-value links from {base_url}")
            return top_links
        
        except Exception as e:
            logging.error(f"Error extracting links from {base_url}: {e}", exc_info=True)
            return []


if __name__ == "__main__":
    try:
        # Check if another instance is running via lock file
        if os.path.exists(LOCK_FILE):
            file_age = time.time() - os.path.getmtime(LOCK_FILE)
            if file_age < 300:  # 5 minutes in seconds
                print(f"Another web crawler is already running (lock file age: {file_age:.1f}s)")
                exit(0)
            else:
                print(f"Found stale lock file (age: {file_age:.1f}s), removing")
                os.remove(LOCK_FILE)
        
        # Note: direct execution without chatbot will fail at AI processing stage.
        # Use main.py to run with a full chatbot instance.
        crawler = WebCrawler()
        crawler.process_learning_path(10)  # Process up to 10 pages
    except Exception as e:
        logging.critical(f"Critical error in web crawler main execution: {e}", exc_info=True)
        # Ensure lock file is removed in case of critical errors
        if os.path.exists(LOCK_FILE):
            try:
                os.remove(LOCK_FILE)
                logging.info("Lock file removed after critical error.")
            except:
                pass