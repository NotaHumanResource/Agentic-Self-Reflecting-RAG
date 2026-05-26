"""Image processing module using Ollama's multimodal vision model (configured via IMAGE_MODEL in config.py) with enhanced timeout handling and progress tracking."""
import os
import logging
import uuid
import datetime
import base64
import requests
import json
import time
import streamlit as st
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
# Pull image analysis model name from central config rather than hardcoding
# Keeps model selection in one place alongside OLLAMA_MODEL and EMBEDDING_MODEL
from config import IMAGE_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)

class ImageProcessor:
    """Handles image uploads and processing using the multimodal vision model configured in config.py (IMAGE_MODEL) with enhanced timeout and progress handling."""
    
    def __init__(self, image_storage_path: str = "image_uploads", api_url: str = "http://localhost:11434/api/generate"):
        """Initialize the image processor with enhanced settings for the configured multimodal vision model (IMAGE_MODEL).
        
        Args:
            image_storage_path (str): Path to store uploaded images
            api_url (str): URL for Ollama API
        """
        self.image_storage_path = image_storage_path
        self.api_url = api_url
        # Vision model imported from config.py (see IMAGE_MODEL)
        # Multimodal — handles image analysis via Ollama's /api/generate endpoint
        self.model_name = IMAGE_MODEL

        # ✅ Register HEIC support via pillow-heif so Image.open() handles iPhone photos
        # This must be called once at startup — all subsequent Image.open() calls
        # will transparently support .heic files without any other code changes
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
            logging.info("✅ HEIC image support registered (pillow-heif)")
        except ImportError:
            logging.warning("⚠️ pillow-heif not installed — HEIC images will not be supported. Run: pip install pillow-heif")
        
        # ✅ Adaptive timeout settings optimized for 27B model with hardware constraints
        self.base_timeout = 180  # Base timeout for image analysis (3 minutes, doubled from 90)
        self.timeout_per_mb = 60  # Additional seconds per MB of image size (doubled from 30)
        self.max_timeout = 1200  # Maximum timeout (20 minutes for very large images, doubled from 600)
        
        # ✅ Retry settings for image processing
        self.max_retries = 2
        self.retry_delay = 2  # Shorter delay for images
        
        # ✅ Image size limits and processing settings
        self.max_image_size_mb = 50  # Maximum image file size
        self.max_resolution = (2048, 2048)  # Maximum resolution for processing
        
        self.ensure_storage_path()
        
        # Test connection to Ollama API
        self.api_available = self._test_api_connection()
        if self.api_available:
            logging.info(f"✅ Successfully connected to Ollama API at {api_url} using {self.model_name}")
        else:
            logging.error(f"❌ Failed to connect to Ollama API at {api_url}")
    
    def _test_api_connection(self) -> bool:
        """Test connection to Ollama API and verify the configured IMAGE_MODEL is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name') for model in models]
                
                # ✅ Check for the specific model we require
                if self.model_name in model_names:
                    logging.info(f"✅ Found and verified target model: {self.model_name}")
                    return True
                else:
                    # ❌ Model not found - provide helpful error message
                    logging.error(
                        f"❌ Required image processing model '{self.model_name}' not found in Ollama.\n"
                        f"   Available models: {model_names}\n"
                        f"   Please verify model installation with: ollama list\n"
                        f"   Or check running models with: ollama ps"
                    )
                    return False
            else:
                logging.error(f"❌ API connection failed with status code: {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error testing Ollama API connection: {e}")
            return False
    
    def ensure_storage_path(self):
        """Ensure the image storage directory exists with enhanced error handling."""
        try:
            os.makedirs(self.image_storage_path, exist_ok=True)
            
            # ✅ Test write permissions
            test_file = os.path.join(self.image_storage_path, "test_write.tmp")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logging.info(f"✅ Image storage path ready: {self.image_storage_path}")
            except Exception as write_error:
                logging.warning(f"⚠️ Image storage path may not be writable: {write_error}")
                
        except Exception as e:
            logging.error(f"❌ Error creating image storage directory: {e}")
            # ✅ Fallback to temp directory
            import tempfile
            self.image_storage_path = tempfile.mkdtemp(prefix="image_analysis_")
            logging.warning(f"⚠️ Using fallback temp directory: {self.image_storage_path}")
    
    def _calculate_adaptive_timeout(self, image_size_mb: float, image_resolution: Tuple[int, int]) -> int:
        """
        Calculate adaptive timeout based on image properties.
        
        Args:
            image_size_mb (float): Image file size in megabytes
            image_resolution (Tuple[int, int]): Image resolution (width, height)
            
        Returns:
            int: Calculated timeout in seconds
        """
        try:
            # ✅ Start with base timeout
            timeout = self.base_timeout
            
            # ✅ Add time based on file size
            timeout += int(image_size_mb * self.timeout_per_mb)
            
            # ✅ Add time based on resolution (higher resolution = more processing time)
            total_pixels = image_resolution[0] * image_resolution[1]
            if total_pixels > 2000000:  # > 2MP
                timeout = int(timeout * 1.5)
            elif total_pixels > 1000000:  # > 1MP
                timeout = int(timeout * 1.2)
            
            # ✅ Ensure timeout is within reasonable bounds
            timeout = max(30, min(timeout, self.max_timeout))  # Minimum 30s, maximum as configured
            
            logging.info(f"Adaptive timeout calculated: {timeout}s (size: {image_size_mb:.1f}MB, resolution: {image_resolution[0]}x{image_resolution[1]})")
            return timeout
            
        except Exception as e:
            logging.error(f"Error calculating adaptive timeout: {e}")
            return self.base_timeout
    
    def _optimize_image_for_processing(self, image_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize image for processing if needed (resize large images, etc.).
        
        Returns:
            Tuple[str, Dict]: (optimized_image_path, optimization_info)
        """
        try:
            with Image.open(image_path) as img:
                # ✅ Convert HEIC/non-RGB images to RGB before processing
                # HEIC from iPhones can arrive in YCbCr or other color modes
                # JPEG saving requires RGB — this is a safe no-op for normal images
                if img.mode not in ("RGB", "L"):
                    logging.info(f"Converting image mode {img.mode} → RGB for processing")
                    img = img.convert("RGB")

                original_size = img.size
                original_format = img.format
                file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                
                optimization_info = {
                    "original_size": original_size,
                    "original_format": original_format,
                    "original_file_size_mb": round(file_size_mb, 2),
                    "optimized": False
                }
                
                # ✅ Check if optimization is needed
                needs_resize = (original_size[0] > self.max_resolution[0] or 
                              original_size[1] > self.max_resolution[1])
                
                if needs_resize or file_size_mb > self.max_image_size_mb:
                    # ✅ Calculate new size maintaining aspect ratio
                    ratio = min(self.max_resolution[0] / original_size[0], 
                              self.max_resolution[1] / original_size[1])
                    
                    if ratio < 1.0:  # Only resize if we're making it smaller
                        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                        
                        # ✅ Create optimized version
                        optimized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        # ✅ Save optimized version
                        optimized_path = image_path.replace('.', '_optimized.')
                        quality = 85 if file_size_mb > 10 else 90
                        
                        if original_format in ['JPEG', 'JPG']:
                            optimized_img.save(optimized_path, 'JPEG', quality=quality, optimize=True)
                        else:
                            optimized_img.save(optimized_path, 'JPEG', quality=quality)
                        
                        # ✅ Update optimization info
                        optimized_file_size_mb = os.path.getsize(optimized_path) / (1024 * 1024)
                        optimization_info.update({
                            "optimized": True,
                            "new_size": new_size,
                            "new_file_size_mb": round(optimized_file_size_mb, 2),
                            "compression_ratio": round(file_size_mb / optimized_file_size_mb, 2)
                        })
                        
                        logging.info(f"Image optimized: {original_size} -> {new_size}, {file_size_mb:.1f}MB -> {optimized_file_size_mb:.1f}MB")
                        return optimized_path, optimization_info
                
                return image_path, optimization_info
                
        except Exception as e:
            logging.error(f"Error optimizing image: {e}")
            return image_path, {"error": str(e)}
    
    def _make_api_call_with_retry(self, message: dict, timeout: int, progress_bar=None, status_placeholder=None) -> Tuple[bool, dict, str]:
        """
        Make API call with retry logic and progress tracking.
        
        Returns:
            Tuple[bool, dict, str]: (success, response_data, error_message)
        """
        for attempt in range(self.max_retries):
            try:
                logging.info(f"Image analysis API call attempt {attempt + 1}/{self.max_retries} (timeout: {timeout}s)")
                
                # ✅ Update progress and status
                if status_placeholder:
                    status_placeholder.info(f"🧠 Analyzing image with {self.model_name} (attempt {attempt + 1}/{self.max_retries})...")
                
                if progress_bar:
                    # ✅ Show progress during API call
                    progress_value = 0.7 + (attempt / self.max_retries) * 0.2
                    progress_bar.progress(progress_value)
                
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(message),
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get('response', '').strip()
                    
                    if generated_text:
                        logging.info(f"✅ Successful image analysis on attempt {attempt + 1}")
                        return True, result, ""
                    else:
                        error_msg = f"Empty response from API (attempt {attempt + 1})"
                        logging.warning(error_msg)
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            continue
                        return False, {}, error_msg
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    logging.warning(f"Image analysis failed on attempt {attempt + 1}: {error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return False, {}, error_msg
                    
            except requests.exceptions.Timeout:
                error_msg = f"Timeout after {timeout}s on attempt {attempt + 1}"
                logging.warning(error_msg)
                if attempt < self.max_retries - 1:
                    # ✅ Increase timeout for next attempt (but more conservatively than video)
                    timeout = min(timeout + 30, self.max_timeout)
                    time.sleep(self.retry_delay)
                    continue
                return False, {}, f"Image analysis timeout after {self.max_retries} attempts"
                
            except Exception as e:
                error_msg = f"API call error on attempt {attempt + 1}: {str(e)}"
                logging.error(error_msg)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return False, {}, error_msg
        
        return False, {}, f"All {self.max_retries} image analysis attempts failed"
    
    def save_uploaded_image(self, image_file) -> Tuple[bool, str, str]:
        """Save an uploaded image file to disk with enhanced validation.
        
        Args:
            image_file: A file-like object from Streamlit or other source
            
        Returns:
            Tuple[bool, str, str]: (success, file_path or error message, image_id)
        """
        try:
            # ✅ Generate unique ID for the image
            image_id = str(uuid.uuid4())
            
            # ✅ Check file size before saving
            if hasattr(image_file, 'size'):
                file_size_mb = image_file.size / (1024 * 1024)
            else:
                # ✅ Get size from buffer if available
                file_content = image_file.getvalue() if hasattr(image_file, 'getvalue') else image_file.getbuffer()
                file_size_mb = len(file_content) / (1024 * 1024)
            
            if file_size_mb > self.max_image_size_mb:
                return False, f"Image file too large: {file_size_mb:.1f}MB (max: {self.max_image_size_mb}MB)", image_id
            
            # ✅ Extract file extension from original filename if possible
            if hasattr(image_file, 'name'):
                original_filename = image_file.name
                file_ext = os.path.splitext(original_filename)[1].lower()
                
                # ✅ Validate file extension
                # ✅ Supported image formats — includes HEIC (iPhone) and BMP
                # pillow-heif registered in __init__ handles HEIC transparently
                allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic']
                if file_ext not in allowed_extensions:
                    file_ext = ".jpg"  # Default to jpg for unknown extensions
            else:
                file_ext = ".jpg"
                
            # ✅ Create full file path
            file_name = f"{image_id}{file_ext}"
            file_path = os.path.join(self.image_storage_path, file_name)
            
            # ✅ Save the file
            with open(file_path, "wb") as f:
                if hasattr(image_file, 'getbuffer'):
                    f.write(image_file.getbuffer())
                else:
                    f.write(image_file.getvalue())
                
            logging.info(f"✅ Image saved successfully: {file_path} ({file_size_mb:.1f}MB)")
            return True, file_path, image_id
            
        except Exception as e:
            error_msg = f"Error saving image: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return False, error_msg, ""
    
    
    def analyze_image(self, image_path: str, prompt: str = "Describe what you see in this image in detail.") -> Dict[str, Any]:
        """Analyze an image using Ollama's multimodal API with enhanced progress tracking and timeout handling.
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Text prompt for analysis
            
        Returns:
            Dict: Analysis results with text description and metadata
        """
        # ✅ Initialize progress tracking
        progress_container = st.container()
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        try:
            if not self.api_available:
                progress_bar.empty()
                status_placeholder.error("❌ Ollama API not available")
                return {
                    "success": False,
                    "error": "Ollama API not available",
                    "description": "Image saved successfully, but Ollama API is not available. Please ensure Ollama is running with a multimodal model."
                }
            
            status_placeholder.info("📸 Preparing image for analysis...")
            progress_bar.progress(0.1)
            
            # ✅ Get image metadata and optimize if needed
            try:
                with Image.open(image_path) as img:
                    original_size = img.size
                    original_format = img.format
                    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            except Exception as e:
                progress_bar.empty()
                status_placeholder.error(f"❌ Error reading image: {str(e)}")
                return {"success": False, "error": f"Error reading image: {str(e)}", "description": ""}
            
            status_placeholder.info("🔧 Optimizing image if needed...")
            progress_bar.progress(0.2)
            
            # ✅ Optimize image for processing
            optimized_path, optimization_info = self._optimize_image_for_processing(image_path)
            
            status_placeholder.info("⏱️ Calculating processing time...")
            progress_bar.progress(0.3)
            
            # ✅ Calculate adaptive timeout
            final_size = optimization_info.get("new_size", original_size)
            final_file_size = optimization_info.get("new_file_size_mb", file_size_mb)
            
            adaptive_timeout = self._calculate_adaptive_timeout(final_file_size, final_size)
            
            status_placeholder.info("📤 Encoding image...")
            progress_bar.progress(0.5)
            
            # ✅ Load and encode the image
            try:
                with open(optimized_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
            except Exception as e:
                progress_bar.empty()
                status_placeholder.error(f"❌ Error encoding image: {str(e)}")
                return {"success": False, "error": f"Error encoding image: {str(e)}", "description": ""}
            
            status_placeholder.info("🧠 Analyzing image with AI...")
            progress_bar.progress(0.6)
            
            # ✅ Format the message for Ollama API
            message = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20
                }
            }
            
            # ✅ Log API call details
            logging.info(f"Analyzing image with model: {self.model_name}, timeout: {adaptive_timeout}s, prompt: '{prompt[:50]}...'")
            
            # ✅ Make API call with retry logic and progress tracking
            success, result, error_msg = self._make_api_call_with_retry(
                message, adaptive_timeout, progress_bar, status_placeholder
            )
            
            if success:
                generated_text = result.get('response', '')
                
                progress_bar.progress(1.0)
                status_placeholder.success("✅ Image analysis completed successfully!")
                
                # ✅ Clean up progress indicators after a short delay
                time.sleep(2)
                progress_bar.empty()
                status_placeholder.empty()
                
                logging.info(f"✅ Image analysis completed: '{generated_text[:100]}...'")
                
                # ✅ Prepare comprehensive metadata
                metadata = {
                    "filename": os.path.basename(image_path),
                    "original_size": original_size,
                    "final_size": final_size,
                    "original_format": original_format,
                    "original_file_size_mb": round(file_size_mb, 2),
                    "final_file_size_mb": round(final_file_size, 2),
                    "model": self.model_name,
                    "processing_timeout": adaptive_timeout,
                    "optimization_applied": optimization_info.get("optimized", False),
                    "analysis_timestamp": datetime.datetime.now().isoformat()
                }
                
                # ✅ Clean up optimized file if it was created
                if optimized_path != image_path and os.path.exists(optimized_path):
                    try:
                        os.remove(optimized_path)
                        logging.info(f"Cleaned up optimized image: {optimized_path}")
                    except:
                        pass  # Ignore cleanup errors
                
                return {
                    "success": True,
                    "description": generated_text,
                    "metadata": metadata,
                    "image_path": image_path,
                    "optimization_info": optimization_info
                }
            else:
                progress_bar.empty()
                status_placeholder.error(f"❌ Image analysis failed: {error_msg}")
                
                # ✅ Clean up optimized file on failure
                if optimized_path != image_path and os.path.exists(optimized_path):
                    try:
                        os.remove(optimized_path)
                    except:
                        pass
                
                return {
                    "success": False, 
                    "error": error_msg,
                    "description": "",
                    "timeout_used": adaptive_timeout
                }
                
        except Exception as e:
            progress_bar.empty()
            status_placeholder.error(f"❌ Unexpected error: {str(e)}")
            error_msg = f"Error analyzing image with Ollama: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg,
                "description": ""
            }
    
    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 3 media cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Helper to pre-flight an image's adaptive timeout/size info — useful for UI estimates
    # but never wired to any caller. analyze_image() handles its own timeout internally.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_estimate_processing_time(self, image_path: str) -> Dict[str, Any]:
        """
        Estimate processing time for an image based on its properties.
        
        Returns:
            Dict containing estimated processing time and image info
        """
        try:
            with Image.open(image_path) as img:
                size = img.size
                file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                
                # ✅ Calculate estimated timeout
                estimated_timeout = self._calculate_adaptive_timeout(file_size_mb, size)
                
                return {
                    "estimated_processing_time_seconds": estimated_timeout,
                    "estimated_processing_time_minutes": round(estimated_timeout / 60, 2),
                    "image_size": size,
                    "file_size_mb": round(file_size_mb, 2),
                    "total_pixels": size[0] * size[1],
                    "within_size_limits": file_size_mb <= self.max_image_size_mb,
                    "model": self.model_name
                }
                
        except Exception as e:
            logging.error(f"Error estimating processing time: {e}")
            return {"error": str(e)}
        
    def store_enhanced_image_analysis(
        self, 
        chatbot, 
        analysis_result: Dict[str, Any],
        user_context: str = ""
    ) -> Tuple[bool, str]:
        """
        Store image analysis with optional user-provided context.
        Combines AI analysis with personal details from the user.
        
        Args:
            chatbot: Reference to the main chatbot instance
            analysis_result: Result from analyze_image containing AI analysis
            user_context: Additional details provided by user (names, dates, locations, etc.)
            
        Returns:
            Tuple[bool, str]: (success, memory_id or error message)
        """
        if not analysis_result["success"]:
            return False, "Cannot store failed analysis"
        
        try:
            # ✅ Prepare enhanced content combining AI analysis and user context
            ai_analysis = analysis_result['description']
            
            if user_context and user_context.strip():
                # User provided additional context - create enriched content
                content = (
                    f"Image Analysis:\n\n"
                    f"AI Description: {ai_analysis}\n\n"
                    f"Personal Context: {user_context}"
                )
                has_user_context = True
            else:
                # No additional context - store AI analysis only
                content = f"Image Analysis: {ai_analysis}"
                has_user_context = False
            
            # ✅ Prepare enhanced metadata
            metadata = {
                "type": "image_analysis",
                "source": analysis_result["image_path"],
                "image_metadata": analysis_result["metadata"],
                "model": analysis_result["metadata"].get("model", self.model_name),
                "processing_timeout": analysis_result["metadata"].get("processing_timeout"),
                "optimization_applied": analysis_result.get("optimization_info", {}).get("optimized", False),
                "has_user_context": has_user_context,
                "user_context_length": len(user_context) if user_context else 0,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            # ✅ If user provided context, add it to metadata for future reference
            if has_user_context:
                metadata["user_provided_context"] = user_context
            
            # ✅ Store in memory using transaction coordinator
            # Higher confidence if user added personal context
            confidence = 1.0 if has_user_context else 0.5
            
            success, memory_id = chatbot.store_memory_with_transaction(
                content=content,
                memory_type="image_analysis",
                metadata=metadata,
                confidence=confidence
            )
            
            if success:
                context_note = "with user context" if has_user_context else "AI analysis only"
                logging.info(f"✅ Stored enhanced image analysis ({context_note}) with ID {memory_id}")
                return True, memory_id
            else:
                return False, "Failed to store image analysis in memory system"
                
        except Exception as e:
            error_msg = f"Error storing enhanced image analysis: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return False, error_msg
    
    