# document_reader.py
"""Document reading and processing functionality."""

import os
import logging
import re
import json
import uuid
import datetime
from typing import List, Tuple, Dict, Any, Optional
import PyPDF2
import docx
from config import DOCS_PATH, SUPPORTED_EXTENSIONS, DEFAULT_CHUNK_SIZE

# =============================================================================
# FIX 4: Content size guard — maximum characters to send to the LLM for
# summarization.  With num_ctx 65536 (~4 chars/token ≈ 16K tokens for content)
# we leave headroom for the prompt wrapper and the generated response.
# =============================================================================
MAX_SUMMARY_CONTENT_CHARS = 50000


class DocumentReader:
    """Handles reading and processing different document types."""
    
    def __init__(self, docs_path: str = DOCS_PATH, chatbot=None):
        """
        Initialize the DocumentReader with a documents directory path and chatbot reference.
        
        Args:
            docs_path (str): Path to documents directory
            chatbot: Reference to the main chatbot instance for accessing LLM, memory functions, etc.
        """
        logging.info("Initializing DocumentReader")
        try:
            self.docs_path = os.path.abspath(docs_path)
            # Create LocalDocs directory if it doesn't exist
            os.makedirs(self.docs_path, exist_ok=True)
            self.supported_extensions = SUPPORTED_EXTENSIONS
            
            # Store reference to the chatbot instance
            self.chatbot = chatbot
            
            logging.info(f"Documents directory confirmed: {self.docs_path}")
        except Exception as e:
            logging.error(f"DocumentReader initialization error: {e}")
            raise

    def find_actual_file(self, partial_name: str) -> Optional[str]:
        """Find the actual filename from a partial or case-insensitive match."""
        try:
            # Guard against None or empty filename
            if not partial_name or not isinstance(partial_name, str):
                logging.warning(f"Invalid filename provided to find_actual_file: {partial_name}")
                return None
                
            available_files = os.listdir(self.docs_path)
            search_term = partial_name.lower().strip()
            
            # Try exact match first (case insensitive)
            for file in available_files:
                if file.lower() == search_term:
                    return file
                    
            # Try with common extensions if no extension provided
            if '.' not in search_term:
                for ext in self.supported_extensions:
                    test_name = search_term + ext
                    for file in available_files:
                        if file.lower() == test_name:
                            return file
            
            # Try matching just the name part (without extension)
            for file in available_files:
                name_only = os.path.splitext(file)[0].lower()
                if name_only == search_term:
                    return file
                    
            # Try more lenient matching
            for file in available_files:
                name_only = os.path.splitext(file)[0].lower()
                if search_term in name_only or name_only in search_term:
                    return file
            
            return None
            
        except Exception as e:
            logging.error(f"Error in file search: {e}")
            return None

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 3 media cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Helper for listing supported documents in DOCS_PATH. The active File Import UI in
    # utils.display_file_import_widget reads uploaded_file.name directly and never
    # consults this lister. No internal callers within document_reader.py either.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_list_documents(self) -> List[str]:
        """List all available documents in the LocalDocs directory."""
        try:
            return [file for file in os.listdir(self.docs_path) 
                   if file.lower().endswith(tuple(self.supported_extensions))]
        except Exception as e:
            logging.error(f"Error listing documents: {e}")
            return []

    def read_file(self, filename: str) -> Tuple[str, bool]:
        """Read and extract text from a file."""
        try:
            # Guard against None or empty filename
            if filename is None or not isinstance(filename, str) or not filename.strip():
                logging.error(f"Attempted to read a None or empty filename")
                return "", False
                
            actual_file = self.find_actual_file(filename)
            if not actual_file:
                logging.error(f"File not found: {filename}")
                return "", False
                
            file_path = os.path.join(self.docs_path, actual_file)
            file_ext = os.path.splitext(actual_file)[1].lower()
            
            if file_ext == '.pdf':
                content = self._read_pdf(file_path)
            elif file_ext == '.txt':
                content = self._read_txt(file_path)
            elif file_ext == '.docx':
                content = self._read_docx(file_path)
            else:
                logging.error(f"Unsupported file type: {file_ext}")
                return "", False
                
            if content:
                logging.info(f"Successfully read file: {actual_file}")
                return content, True
            return "", False
            
        except Exception as e:
            logging.error(f"Error reading file {filename}: {e}")
            return "", False

    def _read_pdf(self, file_path: str) -> str:
        """Read PDF file and extract text.
        If text extraction returns minimal content, assumes scanned/image-based
        PDF and falls back to vision-based page analysis via ImageProcessor.
        """
        try:
            # Guard against None or empty file_path
            if file_path is None or not isinstance(file_path, str) or not file_path.strip():
                logging.error("Attempted to read PDF with None or empty file_path")
                return ""
                
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content.append(text)
                text_content = "\n".join(content)
            
            # ✅ If extracted text is below threshold, treat as scanned/image-based PDF
            # and fall back to vision analysis
            if len(text_content.strip()) < 100:
                logging.info(
                    f"PDF text extraction returned minimal content ({len(text_content)} chars) "
                    f"— assuming scanned PDF, attempting vision-based fallback"
                )
                return self._read_pdf_as_images(file_path)
            
            logging.info(f"PDF text extraction successful: {len(text_content)} chars")
            return text_content
            
        except Exception as e:
            logging.error(f"Error reading PDF {file_path}: {e}")
            return ""
        
    def _read_pdf_as_images(self, file_path: str) -> str:
        """
        Fallback reader for scanned/image-based PDFs.
        Renders each page as an image using PyMuPDF and passes it through
        ImageProcessor vision analysis to extract both text and visual content.
        
        Captures the full page — transcribed text AND visual descriptions of
        illustrations, diagrams, comics, charts, or any non-text elements.
        
        Requires: pip install pymupdf
        
        Args:
            file_path (str): Path to the scanned PDF file
            
        Returns:
            str: Concatenated vision analysis text from all pages,
                or empty string on failure
        """
        # ✅ Maximum pages to process — prevents very long waits on large scanned PDFs
        MAX_PAGES = 20
        
        try:
            # ✅ Lazy imports — only needed for scanned PDF fallback path
            import fitz  # PyMuPDF
            import tempfile
            from image_processor import ImageProcessor
            
            logging.info(f"Starting vision-based PDF analysis for: {file_path}")
            
            # ✅ Initialize ImageProcessor — uses same vision model as image uploads
            image_processor = ImageProcessor()
            
            if not image_processor.api_available:
                logging.error("❌ ImageProcessor API unavailable — cannot analyze scanned PDF")
                return ""
            
            # ✅ Open PDF with PyMuPDF
            pdf_document = fitz.open(file_path)
            total_pages = len(pdf_document)
            pages_to_process = min(total_pages, MAX_PAGES)
            
            if total_pages > MAX_PAGES:
                logging.warning(
                    f"PDF has {total_pages} pages — processing first {MAX_PAGES} only "
                    f"to avoid excessive processing time"
                )
            
            logging.info(f"PDF opened: {total_pages} total pages, processing {pages_to_process}")
            
            page_analyses = []
            temp_files = []
            
            try:
                for page_num in range(pages_to_process):
                    try:
                        page = pdf_document[page_num]
                        
                        # ✅ Render page at 200 DPI — higher than 150 for better legibility
                        # on pages that mix text with illustrations, comics, or diagrams.
                        # 200/72 ≈ 2.78x screen resolution — good balance of quality vs file size
                        mat = fitz.Matrix(200 / 72, 200 / 72)
                        pix = page.get_pixmap(matrix=mat)
                        
                        # ✅ Save rendered page as temp PNG for ImageProcessor
                        temp_file = tempfile.NamedTemporaryFile(
                            suffix='.png',
                            prefix=f'pdf_page_{page_num + 1}_',
                            delete=False
                        )
                        temp_file.close()
                        pix.save(temp_file.name)
                        temp_files.append(temp_file.name)
                        
                        logging.info(f"Analyzing PDF page {page_num + 1}/{pages_to_process}")
                        
                        # ✅ Prompt captures BOTH text transcription and visual content.
                        # Previous prompt was text-only which caused comics, diagrams,
                        # and mixed-content pages to lose all visual context.
                        result = image_processor.analyze_image(
                            temp_file.name,
                            prompt=(
                                f"This is page {page_num + 1} of {pages_to_process} "
                                f"of a scanned document. "
                                f"First, transcribe ALL text visible on this page exactly "
                                f"as written, including headings, body text, captions, "
                                f"speech bubbles, labels, and any other readable text. "
                                f"Then describe the visual content in detail — any images, "
                                f"illustrations, diagrams, charts, comics panels, photographs, "
                                f"or other non-text elements visible, including what is depicted "
                                f"and how elements relate to each other. "
                                f"Preserve the logical reading order throughout. "
                                f"If no readable text is present, describe only what is "
                                f"shown visually."
                            )
                        )
                        
                        if result.get("success") and result.get("description"):
                            page_analyses.append(
                                f"--- Page {page_num + 1} of {pages_to_process} ---\n"
                                f"{result['description']}"
                            )
                            logging.info(f"✅ Page {page_num + 1} analysis complete")
                        else:
                            # ✅ Log failure but continue processing remaining pages
                            error = result.get('error', 'Unknown error')
                            logging.warning(
                                f"⚠️ Vision analysis failed for page {page_num + 1}: {error}"
                            )
                            page_analyses.append(
                                f"--- Page {page_num + 1} of {pages_to_process} --- "
                                f"[Analysis failed: {error}]"
                            )
                            
                    except Exception as page_error:
                        logging.error(
                            f"❌ Error processing page {page_num + 1}: {page_error}",
                            exc_info=True
                        )
                        page_analyses.append(
                            f"--- Page {page_num + 1} of {pages_to_process} --- "
                            f"[Error: {str(page_error)}]"
                        )
                        continue  # ✅ Keep going — don't let one bad page kill the whole document
            
            finally:
                # ✅ Always close PDF and clean up all temp image files regardless of outcome
                pdf_document.close()
                for temp_path in temp_files:
                    try:
                        os.remove(temp_path)
                        logging.debug(f"Cleaned up temp file: {temp_path}")
                    except Exception as cleanup_err:
                        logging.warning(
                            f"⚠️ Could not clean up temp file {temp_path}: {cleanup_err}"
                        )
            
            if page_analyses:
                combined = "\n\n".join(page_analyses)
                logging.info(
                    f"✅ Vision-based PDF analysis complete: "
                    f"{len(page_analyses)}/{pages_to_process} pages analyzed, "
                    f"{len(combined)} chars extracted"
                )
                return combined
            else:
                logging.error("❌ Vision-based PDF analysis produced no content")
                return ""
                
        except ImportError:
            # ✅ Graceful failure if PyMuPDF not installed
            logging.error(
                "❌ PyMuPDF not installed — cannot analyze scanned PDF. "
                "Install with: pip install pymupdf"
            )
            return ""
        except Exception as e:
            logging.error(
                f"❌ Unexpected error in vision-based PDF analysis: {e}",
                exc_info=True
            )
            return ""

    def _read_txt(self, file_path: str) -> str:
        """Read text file with encoding fallback support."""
        try:
            # Guard against None or empty file_path
            if file_path is None or not isinstance(file_path, str) or not file_path.strip():
                logging.error(f"Attempted to read TXT with None or empty file_path")
                return ""
                
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logging.error(f"Error reading text file with latin-1 encoding: {e}")
                return ""
        except Exception as e:
            logging.error(f"Error reading text file: {e}")
            return ""

    def _read_docx(self, file_path: str) -> str:
        """Read DOCX file and extract text."""
        try:
            # Guard against None or empty file_path
            if file_path is None or not isinstance(file_path, str) or not file_path.strip():
                logging.error(f"Attempted to read DOCX with None or empty file_path")
                return ""
                
            doc = docx.Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
            return content
        except Exception as e:
            logging.error(f"Error reading DOCX file {file_path}: {e}")
            return ""

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 3 media cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Sentence-aware text chunker. Was part of the old per-chunk extraction pipeline that
    # was replaced by the document-summary pipeline (see also: FIX 3 'DEAD CODE REMOVED'
    # block at bottom of this file). No callers internal or external.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """Break text into manageable chunks for processing."""
        try:
            # Guard against None or empty text
            if text is None:
                logging.error(f"Error chunking text: 'NoneType' object has no attribute 'split'")
                return []
                
            if not isinstance(text, str) or not text.strip():
                logging.warning(f"Attempted to chunk empty text")
                return []
                
            sentences = text.split('.')
            chunks = []
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                # FIX 5: Only append period if the sentence doesn't already end with one
                # and isn't empty (which would create orphan periods from consecutive dots)
                if not sentence:
                    continue
                if not sentence.endswith('.'):
                    sentence += '.'
                    
                sentence_size = len(sentence)
                
                if current_size + sentence_size > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_size
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            return chunks
        except Exception as e:
            logging.error(f"Error chunking text: {e}")
            return [text] if text else []

    def _truncate_content_for_summary(self, content: str) -> Tuple[str, bool]:
        """
        FIX 4: Truncate oversized document content for LLM summarization.
        
        Preserves the first 60% and last 20% of the allowed budget to keep
        both the introduction (purpose/context) and conclusion (results/summary)
        while cutting the middle where detail density is typically highest.
        
        Args:
            content (str): Full document text
            
        Returns:
            Tuple[str, bool]: (possibly truncated content, was_truncated flag)
        """
        if len(content) <= MAX_SUMMARY_CONTENT_CHARS:
            return content, False
            
        # Calculate split sizes: 60% head, 20% tail of the budget
        head_size = int(MAX_SUMMARY_CONTENT_CHARS * 0.60)
        tail_size = int(MAX_SUMMARY_CONTENT_CHARS * 0.20)
        
        head = content[:head_size]
        tail = content[-tail_size:]
        
        # Calculate how much was removed for logging
        removed_chars = len(content) - head_size - tail_size
        
        truncated = (
            f"{head}\n\n"
            f"[... {removed_chars:,} characters truncated for summarization ...]\n\n"
            f"{tail}"
        )
        
        logging.warning(
            f"Document content truncated from {len(content):,} to ~{MAX_SUMMARY_CONTENT_CHARS:,} chars "
            f"({removed_chars:,} chars removed from middle)"
        )
        
        return truncated, True

    def process_uploaded_document(self, filename: str) -> str:
        """
        Process an uploaded document by generating and storing only a document summary.
        This method skips storing individual chunks/lines and focuses only on creating a
        comprehensive document summary.

        Args:
            filename (str): Name of the uploaded file in LocalDocs.

        Returns:
            str: Status message for the user, including document summary information.
        """
        try:
            # Guard against None or empty filename
            if filename is None or not isinstance(filename, str) or not filename.strip():
                return "Please provide a valid filename."

            # Ensure we have a chatbot reference
            if not self.chatbot:
                return "Error: Chatbot reference not available for document processing."

            # Find the actual file
            actual_filename = self.find_actual_file(filename)
            if not actual_filename:
                return f"Document not found: {filename}"

            # FIX 6: Generate a transaction ID and wire it into metadata for traceability
            document_transaction_id = str(uuid.uuid4())
            logging.info(f"Starting document processing transaction {document_transaction_id} for {actual_filename}")

            # Check if the document has been processed before
            previously_processed = self.chatbot.memory_db.check_document_processed(actual_filename)
            if previously_processed:
                logging.info(f"Document {actual_filename} has been processed before, checking for existing summary")
                
                # Try to retrieve existing summary with metadata filters
                # Using flat keys — vector_db.search() normalizes both formats
                try:
                    vector_results = self.chatbot.vector_db.search(
                        query="",  # Empty query for metadata-only search (uses scroll path)
                        mode="selective",
                        metadata_filters={"type": "document_summary", "source": actual_filename}
                    )
                    
                    if vector_results and len(vector_results) > 0:
                        logging.info(f"Found existing summary for {actual_filename}")
                        return (
                            f"Document {actual_filename} has already been processed. "
                            f"You can retrieve its summary with:\n\n"
                            f"[SEARCH: {actual_filename} | type=document_summary]"
                        )
                except Exception as e:
                    logging.warning(f"Error checking for existing summary: {e}")

            # Read the document content
            content, success = self.read_file(actual_filename)
            if not success:
                return f"Failed to read {actual_filename}."

            if not content or not content.strip():
                return f"No content found in {actual_filename} to process."

            # Get file size for reporting
            file_size_kb = os.path.getsize(os.path.join(self.docs_path, actual_filename)) / 1024

            # Create tracking info for reporting
            extraction_info = {
                "document_name": actual_filename,
                "file_size_kb": file_size_kb,
                "previously_processed": previously_processed,
                "document_summary": "Not created",
                "summary_preview": ""
            }
                
            # Log the processing start with document details
            logging.info(f"Started document processing: {actual_filename}, Size: {file_size_kb:.1f} KB")

            # Log content preview at DEBUG level to reduce log noise on large documents
            logging.debug(f"Content preview: {content[:500]}...")

            # FIX 4: Truncate oversized content before sending to LLM
            content_for_summary, was_truncated = self._truncate_content_for_summary(content)
            if was_truncated:
                logging.info(f"Document content was truncated for summarization (original: {len(content):,} chars)")

            # Create a specific prompt for document summary generation
            summary_prompt = (
                "/no_think\n\n"    
                f"As an autonomous AI system, you're analyzing document '{actual_filename}'.\n\n"
                f"TASK: Create a comprehensive summary of this document.\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Create a summary (500 to 1000 words MAXIMUM)\n"
                f"2. Make the summary concise but informative, prioritizing the most important content\n"
                f"3. Begin with a clear statement of the document's purpose\n"
                f"4. Write in a clear, direct style optimized for retrieval\n"
                f"5. Do NOT include specific commands or implementation details in the summary\n\n"
                f"DOCUMENT CONTENT:\n{content_for_summary}"
            )

            logging.info(f"Summary prompt created with length: {len(summary_prompt)}")

            # Generate the summary using chatbot's LLM
            try:
                logging.info("Invoking LLM to generate document summary")
                doc_summary = self.chatbot.llm.invoke(summary_prompt)
                
                # Log the summary generation result
                if doc_summary is None:
                    logging.error("LLM returned None for document summary generation")
                    extraction_info["document_summary"] = "Error: LLM returned None"
                    return f"Error: Failed to generate summary for {actual_filename}"
                else:
                    logging.info(f"Generated summary length: {len(doc_summary)}")
                    logging.info(f"Summary preview: {doc_summary[:200]}...")
                
                # Verify the summary is valid and meets quality standards
                if doc_summary and isinstance(doc_summary, str) and len(doc_summary.strip()) > 50:
                    # Format the summary with a clear prefix that identifies it as a document summary
                    formatted_summary = f"Document Summary - {actual_filename}: {doc_summary.strip()}"
                    
                    logging.info("Summary meets quality standards, proceeding with storage")
                    
                    # FIX 2: Single confidence value used in both metadata and store call
                    summary_confidence = 0.5  # Neutral — we don't know how accurate the LLM summary is
                    
                    # FIX 6: Include transaction_id in metadata for traceability
                    # Using flat keys — vector_db.search() normalizes both formats
                    summary_metadata = {
                        "type": "document_summary",          # Standard type for search filtering
                        "source": actual_filename,            # Document source/name for search filtering
                        "tags": "summary,document",           # Helpful tags for broad searches
                        "confidence": summary_confidence,     # FIX 2: Consistent confidence value
                        "transaction_id": document_transaction_id  # FIX 6: Traceable to this processing run
                    }

                    # Store the document summary using chatbot's transaction coordination
                    # FIX 2: confidence param matches metadata confidence
                    success, summary_id = self.chatbot.store_memory_with_transaction(
                        content=formatted_summary,
                        memory_type="document_summary",
                        metadata=summary_metadata,
                        confidence=summary_confidence
                    )
                    
                    logging.info(f"Storing document summary with type=document_summary")
                    logging.info(f"Summary metadata: {summary_metadata}")

                    # =========================================================
                    # FIX 1: Single verification block (was duplicated before)
                    # Only runs when store succeeded with a valid summary_id
                    # =========================================================
                    if success and summary_id:
                        logging.info(f"Successfully stored document summary with ID {summary_id}")
                        extraction_info["document_summary"] = "Created and stored successfully"
                        extraction_info["summary_preview"] = doc_summary.strip()[:200] + "..."

                        # Store the search command for sidebar display
                        search_command = f"[SEARCH: {actual_filename} | type=document_summary]"
                        
                        # Try to update Streamlit session state for sidebar display
                        # NOTE: UI-layer coupling — could be refactored to a callback later
                        try:
                            import streamlit as st
                            if hasattr(st, 'session_state'):
                                st.session_state.recent_document_search = {
                                    'filename': actual_filename,
                                    'search_command': search_command,
                                    'processed_time': datetime.datetime.now().strftime("%H:%M:%S")
                                }
                                logging.info(f"Stored search command in session state: {search_command}")
                        except Exception as session_error:
                            # Non-fatal: Streamlit may not be available in all execution contexts
                            logging.warning(
                                f"Could not store search command in session state: {session_error}"
                            )

                        # =================================================
                        # VERIFICATION: Confirm the summary is retrievable
                        # Tests both vector DB and memory DB storage paths
                        # =================================================
                        logging.info("Verifying summary storage with direct search")
                        verification_results = []

                        try:
                            # Test 1: Metadata filter search (the primary retrieval path)
                            # Using flat keys — vector_db.search() normalizes both formats
                            vector_results = self.chatbot.vector_db.search(
                                query="",  # Empty query → scroll path for metadata-only search
                                mode="selective",
                                metadata_filters={"type": "document_summary", "source": actual_filename}
                            )
                            
                            if vector_results and len(vector_results) > 0:
                                verification_results.append("✓ Found in vector database by metadata filter")
                                logging.info(f"Vector DB verification success: Found summary by metadata")
                            else:
                                verification_results.append("✗ Not found in vector database by metadata filter")
                                logging.warning(f"Vector DB verification failed: No results with metadata filter")
                                    
                                # Test 2 (fallback): Semantic text search
                                text_results = self.chatbot.vector_db.search(
                                    query=f"document summary {actual_filename}",
                                    mode="comprehensive",
                                    k=5
                                )
                                    
                                if text_results and len(text_results) > 0:
                                    verification_results.append("✓ Found in vector database by text search")
                                    logging.info(f"Vector DB text search found {len(text_results)} results")
                                else:
                                    verification_results.append("✗ Not found in vector database by text search")
                                    logging.warning(f"Vector DB text search failed to find summary")
                            
                            # Test 3: Memory DB check
                            mem_results = self.chatbot.memory_db.get_memories_by_type("document_summary", limit=5)
                            if mem_results:
                                found_match = False
                                for mem in mem_results:
                                    if actual_filename in mem.get('content', '') or actual_filename in mem.get('source', ''):
                                        found_match = True
                                        break
                                        
                                if found_match:
                                    verification_results.append("✓ Found in memory database")
                                    logging.info(f"Memory DB verification success: Found summary")
                                else:
                                    verification_results.append("✗ Found summaries in memory DB but none match this document")
                                    logging.warning(f"Memory DB verification partial: Found summaries but none match")
                            else:
                                verification_results.append("✗ No document summaries found in memory database")
                                logging.warning(f"Memory DB verification failed: No summaries found")
                                
                            # Store verification results in tracking info
                            extraction_info["verification"] = verification_results
                            logging.info(f"Verification results: {', '.join(verification_results)}")
                            
                        except Exception as verify_err:
                            logging.error(f"Error during summary verification: {verify_err}", exc_info=True)
                            extraction_info["verification"] = [f"Error during verification: {str(verify_err)}"]
                            
                        # Build the user-friendly response showing the search command
                        formatted_report = (
                            f"# Document Summary: {actual_filename}\n\n"
                            f"**Status:** {extraction_info['document_summary']}\n\n"
                            f"**Summary:**\n{doc_summary.strip()}\n\n"
                            f"**To retrieve this summary, use this command:**\n"
                            f"```\n[SEARCH: {actual_filename} | type=document_summary]\n```\n\n"
                        )

                        if "verification" in extraction_info:
                            formatted_report += "**Verification:**\n"
                            for result in extraction_info["verification"]:
                                formatted_report += f"- {result}\n"
                                
                        return formatted_report
                    else:
                        # Store call returned failure or no summary_id
                        logging.error(f"Failed to store document summary for {actual_filename}")
                        extraction_info["document_summary"] = "Failed to store"
                        return f"Error: Failed to store document summary for {actual_filename}"
                else:
                    # Summary didn't meet quality standards (None, wrong type, or too short)
                    logging.warning(f"Generated summary was too short or invalid for {actual_filename}")
                    if doc_summary:
                        logging.warning(f"Invalid summary content: {doc_summary[:200]}")
                    extraction_info["document_summary"] = "Generated summary was invalid or too short"
                    return f"Error: Generated summary was too short or invalid for {actual_filename}"
            except Exception as summary_error:
                logging.error(f"Error generating document summary: {summary_error}", exc_info=True)
                extraction_info["document_summary"] = f"Error: {str(summary_error)}"
                return f"Error generating document summary: {str(summary_error)}"
                
        except Exception as e:
            logging.error(f"Error processing document {filename}: {e}", exc_info=True)
            return f"Error processing document: {str(e)}"

    # =========================================================================
    # FIX 3: DEAD CODE REMOVED
    # The following methods were deleted in this update:
    #
    #   _format_document_extraction_report() — Never called. Expected keys
    #       (total_chunks, processed_chunks, stored_items, extraction_rate,
    #       important, general, skipped, by_confidence) that
    #       process_uploaded_document never populated. Leftover from the old
    #       per-chunk extraction pipeline.
    #
    #   _store_extraction_report() — Never called. Companion to the above
    #       formatting method, also from the old pipeline.
    #
    #   test_document_summary_search() — Dead code. Called
    #       self.chatbot.deepseek_enhancer._handle_retrieve_command() (a private
    #       method). grep confirmed no other caller exists in the codebase.
    # =========================================================================