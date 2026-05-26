# video_processor.py
"""Video processing module using Ollama's multimodal vision model (configured via VIDEO_MODEL in config.py) with enhanced timeout handling and progress tracking."""

import os
import io
import logging
import tempfile
import hashlib
import base64
import requests
import json
import cv2  # For video frame extraction
import numpy as np
import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from PIL import Image
# Pull video analysis model name from central config rather than hardcoding
# Keeps model selection in one place alongside OLLAMA_MODEL, EMBEDDING_MODEL, IMAGE_MODEL
from config import VIDEO_MODEL

class VideoProcessor:
    """Handles video upload, processing, and analysis using the multimodal vision model configured in config.py (VIDEO_MODEL) with enhanced timeout and progress handling."""
    
    def __init__(self):
        """Initialize the video processor with enhanced settings for large model support."""
        self.supported_formats = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv']
        self.max_file_size_mb = 200  # Reasonable limit for long videos
        self.max_duration_seconds = 1800  # ✅ Increased to 30 minutes (1800 seconds)
        
        # Use specific directory path for video uploads
        self.temp_dir = r"C:\Users\kenba\source\repos\Ollama3\video_uploads"
        
        # ✅ Conservative timeout settings retained for large video file processing
        self.base_timeout = 300  # ✅ Increased base timeout to 5 minutes (was 120s)
        self.timeout_per_frame = 45  # ✅ Increased per-frame timeout to 45s (was 15s)
        self.max_timeout = 3600  # ✅ Increased maximum timeout to 1 hour (was 900s)
        
        # Retry settings - conservative for large video files
        self.max_retries = 1  # ✅ Reduced retries to avoid excessive wait times (was 3)
        self.retry_delay = 10  # ✅ Increased delay between retries (was 5s)
                # Vision model imported from config.py (see VIDEO_MODEL)
        # Multimodal — handles video frame analysis via Ollama's /api/generate endpoint
        # Stored as instance attribute so all methods reference one source of truth
        self.model_name = VIDEO_MODEL
        
        try:
            # Ensure the directory exists
            os.makedirs(self.temp_dir, exist_ok=True)
            logging.info(f"VideoProcessor initialized with directory: {self.temp_dir}")
            
            # Verify we can write to the directory
            test_file = os.path.join(self.temp_dir, "test_write.tmp")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logging.info("✅ Video upload directory is writable")
            except Exception as write_error:
                logging.warning(f"⚠️ Video upload directory may not be writable: {write_error}")
                
        except Exception as e:
            logging.error(f"❌ Error creating video upload directory: {e}")
            # Fallback to temp directory if the specific path fails
            import tempfile
            self.temp_dir = tempfile.mkdtemp(prefix="video_analysis_")
            logging.warning(f"⚠️ Using fallback temp directory: {self.temp_dir}")
    
    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 3 media cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Format-validation helper for video uploads. Streamlit's st.file_uploader already
    # constrains uploads via its `type=` parameter, so this validation never runs.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_is_supported_format(self, filename: str) -> bool:
        """Check if video format is supported."""
        return any(filename.lower().endswith(fmt) for fmt in self.supported_formats)
    
    def _check_video_duration(self, video_path: str) -> Tuple[bool, float, str]:
        """
        Check video duration to ensure it's within processing limits (now 30 minutes).
        
        Returns:
            Tuple[bool, float, str]: (is_valid, duration, message)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, 0.0, "Could not open video for duration check"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            if duration > self.max_duration_seconds:
                return False, duration, f"Video duration ({duration:.1f}s / {duration/60:.1f} min) exceeds maximum limit ({self.max_duration_seconds}s / 30 minutes)"
            
            return True, duration, f"Video duration: {duration:.1f}s / {duration/60:.1f} min (within 30-minute limit)"
            
        except Exception as e:
            logging.error(f"Error checking video duration: {e}")
            return False, 0.0, f"Error checking duration: {str(e)}"
    
    def save_temp_video(self, uploaded_file) -> Tuple[bool, str, str]:
        """
        Save uploaded video to temporary location for processing with duration validation.
        
        Returns:
            Tuple[bool, str, str]: (success, file_path_or_message, video_id)
        """
        try:
            # Generate unique video ID
            file_content = uploaded_file.getvalue()
            video_id = hashlib.md5(file_content).hexdigest()[:12]
            
            # Check file size
            file_size_mb = len(file_content) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return False, f"Video file too large: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)", video_id
            
            # Save to temp directory
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            temp_filename = f"video_{video_id}{file_extension}"
            temp_path = os.path.join(self.temp_dir, temp_filename)
            
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            # Check video duration
            is_valid, duration, message = self._check_video_duration(temp_path)
            if not is_valid:
                # Clean up the file if duration check fails
                self.cleanup_temp_file(temp_path)
                return False, message, video_id
            
            logging.info(f"Video saved temporarily: {temp_filename} ({file_size_mb:.1f}MB, {duration:.1f}s)")
            return True, temp_path, video_id
            
        except Exception as e:
            logging.error(f"Error saving video: {e}", exc_info=True)
            return False, f"Error saving video: {str(e)}", ""
    
    def _calculate_adaptive_timeout(self, duration: float, frame_count: int, file_size_mb: float) -> int:
        """
        Calculate adaptive timeout based on video properties - optimized for large model constraints.
        
        Args:
            duration (float): Video duration in seconds
            frame_count (int): Number of frames to be processed
            file_size_mb (float): File size in megabytes
            
        Returns:
            int: Calculated timeout in seconds
        """
        try:
            # ✅ Base timeout calculation — conservative for large video files
            timeout = self.base_timeout  # Start with 5 minutes
            
            # ✅ Add time based on frame count (45s per frame)
            timeout += frame_count * self.timeout_per_frame
            
            # ✅ Add extra time based on video duration for complex analysis
            if duration > 60:  # Videos longer than 1 minute need more processing time
                duration_multiplier = min(3.0, duration / 60.0)  # Up to 3x for very long videos
                timeout = int(timeout * (1 + duration_multiplier * 0.3))
            
            # ✅ Add time based on file size (higher quality = more processing time)
            if file_size_mb > 75:
                timeout = int(timeout * 1.5)  # 50% more time for large files
            elif file_size_mb > 50:
                timeout = int(timeout * 1.3)  # 30% more time for medium files
            elif file_size_mb > 25:
                timeout = int(timeout * 1.2)  # 20% more time for medium-small files
            
            # ✅ Conservative multiplier retained for large video file processing
            resource_multiplier = 1.8  # Retained as conservative buffer for large/complex videos
            timeout = int(timeout * resource_multiplier)
            
            # ✅ Ensure timeout is within reasonable bounds
            timeout = max(self.base_timeout, min(timeout, self.max_timeout))
            
            logging.info(f"Adaptive timeout calculated: {timeout}s (duration: {duration:.1f}s, frames: {frame_count}, size: {file_size_mb:.1f}MB, conservative-adjusted)")
            return timeout
            
        except Exception as e:
            logging.error(f"Error calculating adaptive timeout: {e}")
            return self.base_timeout
        
    def _calculate_optimal_frames(self, duration: float, file_size_mb: float) -> int:
        """
        Calculate optimal number of frames based on video duration and file size.
        
        Args:
            duration (float): Video duration in seconds
            file_size_mb (float): File size in megabytes
            
        Returns:
            int: Optimal number of frames to extract
        """
        try:
            # Enhanced adaptive frame calculation for longer videos
            if duration <= 5:
                base_frames = min(6, max(3, int(duration * 1.2)))
            elif duration <= 15:
                base_frames = 8
            elif duration <= 30:
                base_frames = 12
            elif duration <= 60:
                base_frames = 16
            elif duration <= 120:
                base_frames = 20
            elif duration <= 300:  # 5 minutes
                base_frames = 24
            elif duration <= 600:  # 10 minutes
                base_frames = 30  # Maximum frames for longest videos
            else:
                base_frames = 30  # Cap at 30 frames
            
            # Adjust based on file size
            if file_size_mb > 75:
                size_multiplier = 1.2
            elif file_size_mb > 50:
                size_multiplier = 1.1
            else:
                size_multiplier = 1.0
            
            # Calculate final frame count
            optimal_frames = int(base_frames * size_multiplier)
            
            # Ensure reasonable bounds
            min_frames = 3
            max_frames = 35  # Increased slightly for very long videos
            
            final_frames = max(min_frames, min(optimal_frames, max_frames))
            
            logging.info(f"Enhanced adaptive frame calculation: duration={duration:.1f}s, size={file_size_mb:.1f}MB -> {final_frames} frames")
            return final_frames
            
        except Exception as e:
            logging.error(f"Error calculating optimal frames: {e}")
            return 8
    
    def _make_api_call_with_retry(self, message: dict, timeout: int) -> Tuple[bool, dict, str]:
        """
        Make API call with retry logic and enhanced progress updates for long waits.
        
        Returns:
            Tuple[bool, dict, str]: (success, response_data, error_message)
        """
        api_url = "http://localhost:11434/api/generate"
        
        for attempt in range(self.max_retries):
            try:
                # ✅ Enhanced logging for long timeouts
                timeout_minutes = timeout / 60
                logging.info(f"API call attempt {attempt + 1}/{self.max_retries} (timeout: {timeout}s / {timeout_minutes:.1f} min)")
                
                # ✅ Show timeout info in Streamlit for user awareness
                if hasattr(st, 'session_state'):
                    try:
                        with st.container():
                            # Model name from self.model_name so display matches actual config
                            st.info(f"⏱️ Processing with {timeout_minutes:.1f} minute timeout (large model: {self.model_name})")
                            if timeout_minutes > 10:
                                st.warning("⚠️ This may take a while. Please be patient...")
                    except:
                        pass  # Ignore if Streamlit context not available
                
                response = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(message),
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get('response', '').strip()
                    
                    if generated_text:
                        logging.info(f"✅ Successful API response on attempt {attempt + 1} after {timeout}s timeout")
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
                    logging.warning(f"API call failed on attempt {attempt + 1}: {error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return False, {}, error_msg
                    
            except requests.exceptions.Timeout:
                error_msg = f"Timeout after {timeout}s / {timeout/60:.1f} min on attempt {attempt + 1}"
                logging.warning(error_msg)
                if attempt < self.max_retries - 1:
                    # ✅ Increase timeout on retry for large video files
                    timeout = min(timeout + 300, self.max_timeout)  # Add 5 more minutes per retry
                    logging.info(f"Retrying with increased timeout: {timeout}s / {timeout/60:.1f} min")
                    time.sleep(self.retry_delay)
                    continue
                return False, {}, f"API timeout after {self.max_retries} attempts (final timeout: {timeout/60:.1f} min)"
                
            except Exception as e:
                error_msg = f"API call error on attempt {attempt + 1}: {str(e)}"
                logging.error(error_msg)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return False, {}, error_msg
        
        return False, {}, f"All {self.max_retries} API attempts failed"

    def analyze_video_with_qwen(self, video_path: str, prompt: str, chatbot) -> Dict[str, Any]:
        """
        Analyze video by extracting frames and analyzing them with the configured multimodal vision model (VIDEO_MODEL) with enhanced progress tracking.
        
        Args:
            video_path (str): Path to the video file
            prompt (str): Analysis prompt
            chatbot: Chatbot instance with LLM access
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Initialize progress tracking
        if 'stqdm' not in locals():
            try:
                from stqdm import stqdm
            except ImportError:
                # Fallback to regular progress updates
                stqdm = None
        
        progress_container = st.container()
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_placeholder.info("🎬 Starting video analysis...")
            progress_bar.progress(0.1)
            
            # Get file size for adaptive calculations
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            status_placeholder.info("🎞️ Extracting video frames...")
            progress_bar.progress(0.2)
            
            # Extract frames from video with adaptive frame count
            frames_data = self._extract_video_frames(video_path, file_size_mb, progress_bar, status_placeholder)
            
            if not frames_data["success"]:
                progress_bar.empty()
                status_placeholder.empty()
                return {
                    "success": False,
                    "description": None,
                    "error": f"Frame extraction failed: {frames_data['error']}",
                    "video_path": video_path
                }
            
            frames = frames_data["frames"]
            video_info = frames_data["video_info"]
            frames_extracted = frames_data.get("frames_extracted", len(frames))
            
            progress_bar.progress(0.6)
            # Status text uses self.model_name so it always reflects actual model in use
            status_placeholder.info(f"🧠 Analyzing {frames_extracted} frames with {self.model_name}...")
            # Calculate adaptive timeout
            adaptive_timeout = self._calculate_adaptive_timeout(
                video_info['duration'], 
                frames_extracted, 
                file_size_mb
            )
            
            # Create comprehensive video analysis prompt
            video_prompt = f"""Analyze this video sequence represented by {frames_extracted} key frames (adaptively selected) extracted from the video. {prompt}

Video Information:
- Duration: {video_info['duration']:.1f} seconds
- Total Frames: {video_info['total_frames']}
- FPS: {video_info['fps']:.1f}
- Frames Analyzed: {frames_extracted} (adaptively selected based on video length)

Please analyze the sequence of frames and describe:
1. What is happening throughout the video
2. Key objects, people, or scenes visible
3. Any actions, movements, or changes occurring over time
4. Overall context and setting
5. Notable details or interesting observations
6. How the scene evolves from beginning to end

Provide a comprehensive analysis that captures the temporal flow and content of the entire video."""

            # Prepare API message — model name sourced from config (VIDEO_MODEL)
            message = {
                "model": self.model_name,  # Multimodal vision model from config.py
                "prompt": video_prompt,
                "images": frames,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 20
                }
            }
            
            progress_bar.progress(0.7)
            status_placeholder.info(f"⏱️ Processing with adaptive timeout ({adaptive_timeout}s)...")
            
            # Make API call with retry logic
            success, result, error_msg = self._make_api_call_with_retry(message, adaptive_timeout)
            
            if success:
                generated_text = result.get('response', '')
                progress_bar.progress(1.0)
                status_placeholder.success("✅ Video analysis completed successfully!")
                
                # Clean up progress indicators after a short delay
                time.sleep(2)
                progress_bar.empty()
                status_placeholder.empty()
                
                # Enhanced response with processing metadata
                enhanced_description = f"{generated_text.strip()}\n\n[Analysis based on {frames_extracted} adaptively selected frames from {video_info['duration']:.1f} second video using {adaptive_timeout}s timeout]"
                
                return {
                    "success": True,
                    "description": enhanced_description,
                    "video_path": video_path,
                    "prompt_used": prompt,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "model": self.model_name,  # Recorded from config so memory metadata stays accurate if model swapped
                    "frames_analyzed": frames_extracted,
                    "video_duration": video_info['duration'],
                    "adaptive_frames": True,
                    "adaptive_timeout": adaptive_timeout,
                    "processing_time": adaptive_timeout,
                    "error": None
                }
            else:
                progress_bar.empty()
                status_placeholder.error(f"❌ Analysis failed: {error_msg}")
                return {
                    "success": False,
                    "description": None,
                    "error": error_msg,
                    "video_path": video_path,
                    "timeout_used": adaptive_timeout
                }
                
        except Exception as e:
            progress_bar.empty()
            status_placeholder.error(f"❌ Unexpected error: {str(e)}")
            logging.error(f"Error in video analysis: {e}", exc_info=True)
            return {
                "success": False,
                "description": None,
                "error": f"Analysis error: {str(e)}",
                "video_path": video_path
            }

    def _extract_video_frames(self, video_path: str, file_size_mb: float = None, 
                            progress_bar=None, status_placeholder=None) -> Dict[str, Any]:
        """
        Extract key frames from video for analysis with progress tracking and memory optimization.
        
        Args:
            video_path (str): Path to video file
            file_size_mb (float): File size in MB (for adaptive calculation)
            progress_bar: Streamlit progress bar widget
            status_placeholder: Streamlit status placeholder
            
        Returns:
            Dict[str, Any]: Extracted frames and video info
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {
                    "success": False,
                    "error": f"Could not open video file: {video_path}",
                    "frames": [],
                    "video_info": {}
                }
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_info = {
                "total_frames": total_frames,
                "fps": fps,
                "duration": duration,
                "width": width,
                "height": height
            }
            
            # Calculate optimal number of frames adaptively
            if file_size_mb is not None:
                max_frames = self._calculate_optimal_frames(duration, file_size_mb)
            else:
                max_frames = 8
                logging.warning("File size not provided for adaptive frame calculation, using default 8 frames")
            
            # Calculate frame intervals
            if total_frames <= max_frames:
                frame_indices = list(range(0, total_frames, max(1, total_frames // max_frames)))
            else:
                interval = total_frames // max_frames
                frame_indices = list(range(0, total_frames, interval))[:max_frames]
            
            frames = []
            
            # Determine optimal resolution for processing (balance quality vs. performance)
            target_width = min(800, width)  # Cap at 800px width for efficiency
            scale_factor = target_width / width if width > target_width else 1.0
            target_height = int(height * scale_factor)
            
            if status_placeholder:
                status_placeholder.info(f"🎞️ Extracting {len(frame_indices)} frames (resolution: {target_width}x{target_height})...")
            
            for i, frame_idx in enumerate(frame_indices):
                try:
                    # Update progress
                    if progress_bar:
                        frame_progress = 0.2 + (i / len(frame_indices)) * 0.4  # Progress from 20% to 60%
                        progress_bar.progress(frame_progress)
                    
                    # Set frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    
                    # Read frame
                    ret, frame = cap.read()
                    
                    if ret:
                        # Resize frame if needed for efficiency
                        if scale_factor < 1.0:
                            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                        
                        # Convert BGR to RGB (OpenCV uses BGR by default)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Convert to PIL Image for consistent processing
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Adaptive JPEG quality based on file size and frame count
                        if file_size_mb and file_size_mb > 75:
                            quality = 85  # Slightly reduced for large files to manage memory
                        elif file_size_mb and file_size_mb > 50:
                            quality = 80
                        else:
                            quality = 75
                        
                        # Convert to base64
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
                        frame_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        frames.append(frame_data)
                        
                        # Clear memory
                        del frame, frame_rgb, pil_image, buffer
                        
                        logging.debug(f"Extracted frame {frame_idx} ({i+1}/{len(frame_indices)})")
                
                except Exception as frame_error:
                    logging.warning(f"Error extracting frame {frame_idx}: {frame_error}")
                    continue
            
            cap.release()
            
            if frames:
                logging.info(f"Successfully extracted {len(frames)} frames from video (duration: {duration:.1f}s, adaptive, resolution: {target_width}x{target_height})")
                return {
                    "success": True,
                    "frames": frames,
                    "video_info": video_info,
                    "frames_extracted": len(frames),
                    "adaptive_calculation": True,
                    "processing_resolution": f"{target_width}x{target_height}",
                    "scale_factor": scale_factor,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "error": "No frames could be extracted from video",
                    "frames": [],
                    "video_info": video_info
                }
                
        except Exception as e:
            logging.error(f"Error extracting frames from video: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Frame extraction error: {str(e)}",
                "frames": [],
                "video_info": {}
            }
    
    def cleanup_temp_file(self, video_path: str) -> bool:
        """Clean up temporary video file after processing with enhanced error handling."""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logging.info(f"✅ Cleaned up temporary video: {video_path}")
                return True
            else:
                logging.warning(f"⚠️ Temp video file not found for cleanup: {video_path}")
                return False
        except Exception as e:
            logging.error(f"❌ Error cleaning up video file: {e}")
            return False
    
    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 3 media cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Returns video metadata dict (duration, fps, resolution, etc.). The only call site is
    # commented out in main.py L2085, and its sole internal caller is _UNUSED_estimate_processing_time.
    # Transitively dead through a dead parent.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract comprehensive video metadata with enhanced information."""
        try:
            file_stats = os.stat(video_path)
            
            # Get video properties using OpenCV
            cap = cv2.VideoCapture(video_path)
            video_props = {}
            
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                video_props = {
                    "duration_seconds": round(duration, 2),
                    "fps": round(fps, 2),
                    "total_frames": total_frames,
                    "resolution": f"{width}x{height}",
                    "width": width,
                    "height": height
                }
                cap.release()
            
            return {
                "filename": os.path.basename(video_path),
                "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                "file_extension": os.path.splitext(video_path)[1],
                "created_time": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                **video_props
            }
        except Exception as e:
            logging.error(f"Error getting video metadata: {e}")
            return {"error": str(e)}

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 3 media cleanup pass).
    # Verified against main.py, chatbot.py, utils.py, deepseek.py, autonomous_cognition.py.
    # Pre-flight estimate helper — never wired to UI. analyze_video_with_qwen() handles its own
    # adaptive timeout internally. Calls get_video_metadata + _calculate_optimal_frames +
    # _calculate_adaptive_timeout but doesn't itself surface results to any caller.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_estimate_processing_time(self, video_path: str) -> Dict[str, Any]:
        """
        Estimate processing time for a video based on its properties - calibrated for the configured multimodal vision model (VIDEO_MODEL).
        
        Returns:
            Dict containing estimated processing time and breakdown
        """
        try:
            metadata = self.get_video_metadata(video_path)
            
            if "error" in metadata:
                return {"error": metadata["error"]}
            
            duration = metadata.get("duration_seconds", 0)
            file_size_mb = metadata.get("file_size_mb", 0)
            
            # Calculate estimated frames
            estimated_frames = self._calculate_optimal_frames(duration, file_size_mb)
            
            # Calculate estimated timeout
            estimated_timeout = self._calculate_adaptive_timeout(duration, estimated_frames, file_size_mb)
            
            # ✅ Estimates based on dense 27B vision model (~17GB, VRAM-resident at 64K context)
            return {
                "estimated_processing_time_seconds": estimated_timeout,
                "estimated_processing_time_minutes": round(estimated_timeout / 60, 1),
                "estimated_frames_to_analyze": estimated_frames,
                "video_duration_seconds": duration,
                "video_duration_minutes": round(duration / 60, 1),
                "file_size_mb": file_size_mb,
                "within_limits": duration <= self.max_duration_seconds,
                "model_info": self.model_name,  # Sourced from config (VIDEO_MODEL)
                "timeout_includes_resource_adjustment": True,  # ✅ Conservative multiplier retained pending video test
                "max_allowed_duration_minutes": 30
            }
                        
        except Exception as e:
            logging.error(f"Error estimating processing time: {e}")
            return {"error": str(e)}