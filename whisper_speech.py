# whisper_speech.py
import pyaudio
import numpy as np
import threading
# import queue  # DEAD CODE TEST 2026-05-17: unused per ruff F401 + vulture
import time
import logging
import ollama
# import soundfile as sf  # DEAD CODE TEST 2026-05-17: shadowed by L520 re-import inside speak_text() (ruff F401)
from faster_whisper import WhisperModel


class WhisperSpeechUtils:
    def __init__(self, whisper_model_size="base"):
        """
        Initialize WhisperSpeechUtils with default audio device validation
        
        Args:
            whisper_model_size: Size of Whisper model to use ("tiny", "base", "small", etc.)
        """
        # Whisper STT model configuration
        self.whisper_model = None
        self.whisper_model_size = whisper_model_size
        self.command_model_size = "small"   # Better model for commands
        self.kokoro_pipeline = None
        self.kokoro_voice = None

        # Pre-initialize Whisper model at startup to avoid delays during recording
        self._pre_initialize_whisper()
        
        # Ollama/Gemma3 integration
        self.ollama_client = None
        
        # Speech state management (button-based only)
        self.is_listening = False
        self.status_callback = None
        
        # Audio configuration for default device
        self.sample_rate = 16000
        self.chunk_size = 512
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize PyAudio
        try:
            self.audio = pyaudio.PyAudio()
            logging.info("PyAudio initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize PyAudio: {e}")
            self.audio = None
        
        # Validate default input device exists and is accessible
        if self.audio is not None:
            if self._validate_default_input_device():
                logging.info("Default audio input device validated successfully")
            else:
                logging.error("Default audio input device validation failed - speech recognition may not work")
        else:
            logging.error("PyAudio not available - speech recognition disabled")
        
        # Status tracking
        self.recognition_status = "inactive"
        
        logging.info("WhisperSpeechUtils initialization complete")

    
    def _validate_default_input_device(self):
        """
        Validate that a default input device exists and is accessible
        
        Returns:
            bool: True if default device is valid and accessible, False otherwise
        """
        if self.audio is None:
            logging.error("PyAudio not initialized - cannot validate device")
            return False
        
        try:
            # Get default input device info
            device_info = self.audio.get_default_input_device_info()
            
            # Check if device has input channels
            if device_info['maxInputChannels'] <= 0:
                logging.error(f"Default device '{device_info['name']}' has no input channels")
                return False
            
            # Log device information
            logging.info("=" * 60)
            logging.info("DEFAULT AUDIO INPUT DEVICE:")
            logging.info(f"  Name: {device_info['name']}")
            logging.info(f"  Index: {device_info['index']}")
            logging.info(f"  Channels: {device_info['maxInputChannels']}")
            logging.info(f"  Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
            logging.info("=" * 60)
            
            # Test if we can actually open the device
            test_stream = None
            try:
                test_stream = self.audio.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                
                # Try to read a small amount of data
                test_data = test_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(test_data, dtype=np.int16)
                test_volume = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
                
                logging.info(f"Device test successful - test volume: {test_volume:.2f}")
                return True
                
            except OSError as e:
                logging.error(f"Cannot open default input device: {e}")
                logging.error("  Possible causes:")
                logging.error("    - Microphone is being used by another application")
                logging.error("    - Microphone permissions not granted")
                logging.error("    - Device doesn't support 16kHz sample rate")
                return False
                
            finally:
                # Always close test stream
                if test_stream is not None:
                    try:
                        test_stream.stop_stream()
                        test_stream.close()
                    except Exception as e:
                        logging.debug(f"Error closing test stream: {e}")
            
        except OSError as e:
            logging.error(f"No default input device available: {e}")
            logging.error("  Please connect a microphone and set it as default in system settings")
            return False
            
        except Exception as e:
            logging.error(f"Error validating default input device: {e}")
            import traceback
            logging.debug(f"Validation traceback: {traceback.format_exc()}")
            return False

    def _pre_initialize_whisper(self):
        """Pre-initialize Whisper model at startup to avoid UI delays"""
        try:
            logging.info("Pre-initializing Whisper model at startup...")
            success = self._init_whisper()
            if success:
                logging.info("✅ Whisper model pre-loaded successfully")
            else:
                logging.warning("⚠️ Whisper model pre-load failed")
        except Exception as e:
            logging.error(f"Error pre-initializing Whisper: {e}")

        
    def _init_whisper(self):
        """ENHANCED: Initialize Whisper model with better error handling and CPU optimization."""
        if self.whisper_model is None:
            try:
                logging.info(f"🔧 Loading Whisper {self.whisper_model_size} model for CPU...")
                
                # Enhanced CPU-optimized settings
                self.whisper_model = WhisperModel(
                    self.whisper_model_size, 
                    device="cpu",
                    compute_type="int8",      # Use int8 for better CPU performance
                    cpu_threads=4,            # Limit CPU threads to prevent blocking
                    num_workers=1             # Single worker for stability
                )
                
                # Test the model with a small audio sample
                test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                segments, info = self.whisper_model.transcribe(test_audio, language="en")
                
                logging.info(f"✅ Whisper {self.whisper_model_size} model loaded and tested successfully")
                return True
                
            except Exception as e:
                logging.error(f"❌ Failed to load Whisper {self.whisper_model_size} model: {e}")
                
                # Try fallback to smaller model
                if self.whisper_model_size != "tiny":
                    logging.info("🔄 Attempting fallback to 'tiny' model...")
                    try:
                        self.whisper_model = WhisperModel(
                            "base", 
                            device="cpu",
                            compute_type="int8",
                            cpu_threads=2,
                            num_workers=1
                        )
                        self.whisper_model_size = "tiny"
                        logging.info("✅ Fallback to tiny model successful")
                        return True
                    except Exception as fallback_e:
                        logging.error(f"❌ Fallback model also failed: {fallback_e}")
                
                self.whisper_model = None
                return False
        return True
    
    def _init_ollama(self):
        """Initialize Ollama client for Gemma3"""
        if self.ollama_client is None:
            try:
                self.ollama_client = ollama.Client()
                # Test connection — return value intentionally discarded (ruff F841 cleanup 2026-05-17)
                self.ollama_client.list()
                logging.info("Ollama client initialized successfully")
                return True
            except Exception as e:
                logging.error(f"Failed to initialize Ollama client: {e}")
                return False
        return True
    
    def _init_kokoro(self):
        """
        Initialize Kokoro TTS using kokoro-onnx package.
        ENHANCED with comprehensive error handling and validation.
        
        Model files must be downloaded to kokoro_models folder.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # =========================================================================
        # Check if already initialized
        # =========================================================================
        if self.kokoro_pipeline is not None:
            logging.debug("TTS: Kokoro already initialized")
            return True
        
        try:
            # =====================================================================
            # STEP 1: Import required modules with error handling
            # =====================================================================
            try:
                from kokoro_onnx import Kokoro
                import os
            except ImportError as import_error:
                logging.error(f"❌ Kokoro not installed: {import_error}")
                logging.error("   Install with: pip install kokoro-onnx soundfile")
                return False
            
            logging.info("🔧 Initializing Kokoro TTS (ONNX)...")
            
            # =====================================================================
            # STEP 2: Validate model file paths
            # =====================================================================
            cache_dir = r"C:\Users\kenba\source\repos\Ollama3\kokoro_models"
            model_path = os.path.join(cache_dir, "kokoro-v1.0.onnx")
            voices_path = os.path.join(cache_dir, "voices-v1.0.bin")
            
            # Check if cache directory exists
            if not os.path.exists(cache_dir):
                logging.error(f"❌ Kokoro models directory not found: {cache_dir}")
                logging.error("   Please create the directory and download model files")
                return False
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logging.error(f"❌ Kokoro model file not found: {model_path}")
                logging.error("   Download from: https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
                return False
            
            # Validate model file size (should be substantial, not corrupted)
            try:
                model_size = os.path.getsize(model_path)
                if model_size < 1000000:  # Less than 1MB is suspicious
                    logging.error(f"❌ Kokoro model file appears corrupted (size: {model_size} bytes)")
                    logging.error("   Expected size: ~200MB+")
                    return False
                logging.info(f"📂 Model file size: {model_size / 1024 / 1024:.1f} MB")
            except Exception as size_error:
                logging.error(f"❌ Cannot check model file size: {size_error}")
                return False
            
            # Check if voices file exists
            if not os.path.exists(voices_path):
                logging.error(f"❌ Kokoro voices file not found: {voices_path}")
                logging.error("   Download from: https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
                return False
            
            # Validate voices file size
            try:
                voices_size = os.path.getsize(voices_path)
                if voices_size < 100000:  # Less than 100KB is suspicious
                    logging.error(f"❌ Kokoro voices file appears corrupted (size: {voices_size} bytes)")
                    return False
                logging.info(f"📂 Voices file size: {voices_size / 1024:.1f} KB")
            except Exception as size_error:
                logging.error(f"❌ Cannot check voices file size: {size_error}")
                return False
            
            # =====================================================================
            # STEP 3: Initialize Kokoro pipeline with error handling
            # =====================================================================
            try:
                logging.info(f"📂 Loading Kokoro from: {cache_dir}")
                
                # CRITICAL: Initialize the Kokoro pipeline
                # This can fail for various reasons (corrupted files, missing dependencies, etc.)
                self.kokoro_pipeline = Kokoro(model_path, voices_path)
                
                # Validate that pipeline was actually created
                if self.kokoro_pipeline is None:
                    logging.error("❌ Kokoro pipeline initialization returned None")
                    return False
                
                logging.info("✅ Kokoro pipeline object created")
                
            except FileNotFoundError as file_error:
                logging.error(f"❌ Kokoro file not found during initialization: {file_error}")
                self.kokoro_pipeline = None
                return False
                
            except PermissionError as perm_error:
                logging.error(f"❌ Permission denied accessing Kokoro files: {perm_error}")
                self.kokoro_pipeline = None
                return False
                
            except Exception as pipeline_error:
                logging.error(f"❌ Kokoro pipeline creation failed: {pipeline_error}")
                import traceback
                logging.error(f"Pipeline initialization traceback:\n{traceback.format_exc()}")
                self.kokoro_pipeline = None
                return False
            
            # =====================================================================
            # STEP 4: Set and validate voice selection
            # =====================================================================
            # Voice selection - using af_heart as default
            self.kokoro_voice = 'af_heart'
            
            # Available voices for reference:
            # American Female: af_heart, af_sarah, af_bella, af_nicole, af_sky
            # American Male: am_adam, am_michael
            # British Female: bf_emma, bf_isabella
            # British Male: bm_george, bm_lewis
            
            # List of valid voices for validation
            valid_voices = [
                'af_heart', 'af_sarah', 'af_bella', 'af_nicole', 'af_sky',
                'am_adam', 'am_michael',
                'bf_emma', 'bf_isabella',
                'bm_george', 'bm_lewis'
            ]
            
            if self.kokoro_voice not in valid_voices:
                logging.warning(f"⚠️ Voice '{self.kokoro_voice}' may not be valid")
                logging.warning(f"   Valid voices: {', '.join(valid_voices)}")
            
            logging.info(f"✅ Kokoro voice set to: {self.kokoro_voice}")
            
            # =====================================================================
            # STEP 5: Test the pipeline with a simple phrase
            # =====================================================================
            try:
                logging.info("🧪 Testing Kokoro pipeline with sample text...")
                
                test_text = "Hello, this is a test."
                test_samples, test_rate = self.kokoro_pipeline.create(
                    test_text,
                    voice=self.kokoro_voice,
                    speed=1.0
                )
                
                # Validate test output
                if test_samples is None or len(test_samples) == 0:
                    logging.error("❌ Kokoro test generation failed - no audio produced")
                    self.kokoro_pipeline = None
                    return False
                
                if test_rate is None or test_rate <= 0:
                    logging.error(f"❌ Kokoro test generation failed - invalid sample rate: {test_rate}")
                    self.kokoro_pipeline = None
                    return False
                
                test_duration = len(test_samples) / test_rate
                logging.info(f"✅ Kokoro test successful - generated {test_duration:.2f}s of audio at {test_rate}Hz")
                
            except AttributeError as attr_error:
                logging.error(f"❌ Kokoro pipeline missing 'create' method: {attr_error}")
                self.kokoro_pipeline = None
                return False
                
            except TypeError as type_error:
                logging.error(f"❌ Kokoro API parameter error during test: {type_error}")
                self.kokoro_pipeline = None
                return False
                
            except Exception as test_error:
                logging.error(f"❌ Kokoro pipeline test failed: {test_error}")
                import traceback
                logging.error(f"Test traceback:\n{traceback.format_exc()}")
                self.kokoro_pipeline = None
                return False
            
            # =====================================================================
            # Success!
            # =====================================================================
            logging.info(f"✅ Kokoro TTS fully initialized and tested successfully")
            logging.info(f"   Voice: {self.kokoro_voice}")
            logging.info(f"   Model: {model_path}")
            
            return True
            
        except Exception as outer_error:
            # =====================================================================
            # OUTER EXCEPTION HANDLER - Catches any unexpected errors
            # =====================================================================
            logging.error(f"❌ Unexpected error in Kokoro initialization: {outer_error}")
            import traceback
            logging.error(f"Kokoro init traceback:\n{traceback.format_exc()}")
            self.kokoro_pipeline = None
            self.kokoro_voice = None
            return False

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 2 speech cleanup pass).
    # Sets self.status_callback which is checked inside _update_status() — but since this
    # setter is never called, status_callback stays None forever and the conditional branch
    # inside _update_status never fires. Entire callback hook is dead.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_set_status_callback(self, callback):
        """Set callback for status updates"""
        self.status_callback = callback
        logging.info("Status callback set")
    
    def _update_status(self, status, data=None):
        """Update recognition status and notify callback"""
        if data is None:
            data = {}
        
        self.recognition_status = status
        data['timestamp'] = time.time()
        
        # Log status changes
        if status != 'recognizing' or logging.getLogger().level <= logging.DEBUG:
            logging.info(f"Speech status: {status} - {data}")
        
        # Call status callback if set
        if self.status_callback:
            try:
                self.status_callback(status, data)
            except Exception as e:
                logging.error(f"Error in status callback: {e}")

        
    
    def text_to_speech(self, text):
        """
        Convert text to speech using Kokoro TTS (offline, no internet required).
        ENHANCED with comprehensive error handling to prevent Streamlit crashes.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            bool: True if speech started successfully, False otherwise
        """
        # =========================================================================
        # STEP 1: Input validation
        # =========================================================================
        if not text or not text.strip():
            logging.warning("TTS: Empty text provided, skipping")
            return False
        
        # Track timing for debugging
        start_time = time.time()
        
        try:
            # =====================================================================
            # STEP 2: Clean the text before speech synthesis
            # =====================================================================
            try:
                cleaned_text = self.clean_response_for_tts(text)
            except Exception as clean_error:
                logging.error(f"TTS: Error cleaning text: {clean_error}")
                # Try to use original text as fallback
                cleaned_text = text
            
            if not cleaned_text or not cleaned_text.strip():
                logging.warning("TTS: Text empty after cleaning, skipping")
                logging.info(f"TTS: Original text preview: '{text[:200]}...'")
                return False
            
            # Log what we're speaking
            original_len = len(text)
            cleaned_len = len(cleaned_text)
            logging.info(f"TTS: Speaking response of {original_len} characters")
            logging.info(f"TTS: Text cleaned. Original length: {original_len}, Cleaned length: {cleaned_len}")
            logging.info(f"TTS: Cleaned text preview: '{cleaned_text[:150]}...'")
            
            # =====================================================================
            # STEP 3: Define thread-safe speech function with comprehensive error handling
            # =====================================================================
            def speak_text():
                """
                Inner function to run TTS in separate thread.
                ENHANCED with comprehensive error handling to prevent crashes.
                """
                nonlocal start_time
                
                # Thread-level exception handler
                try:
                    # =============================================================
                    # Initialize required modules with error handling
                    # =============================================================
                    try:
                        import pygame
                        import tempfile
                        import os
                        import soundfile as sf
                        import numpy as np
                    except ImportError as import_error:
                        logging.error(f"TTS: Missing required module: {import_error}")
                        logging.error(f"TTS: Install with: pip install pygame soundfile numpy")
                        return
                    
                    # =============================================================
                    # Validate Kokoro initialization
                    # =============================================================
                    try:
                        if not self._init_kokoro():
                            logging.error("TTS: Failed to initialize Kokoro")
                            return
                        
                        # Double-check that pipeline was actually created
                        if self.kokoro_pipeline is None:
                            logging.error("TTS: Kokoro pipeline is None after initialization")
                            return
                        
                        # Verify voice is set
                        if not self.kokoro_voice:
                            logging.error("TTS: Kokoro voice not set")
                            return
                            
                    except Exception as init_error:
                        logging.error(f"TTS: Kokoro initialization error: {init_error}")
                        import traceback
                        logging.error(f"TTS: Initialization traceback: {traceback.format_exc()}")
                        return
                    
                    # =============================================================
                    # CRITICAL: Generate audio with Kokoro (wrapped in try-catch)
                    # =============================================================
                    samples = None
                    sample_rate = None
                    
                    try:
                        logging.info(f"TTS: Generating speech with Kokoro voice '{self.kokoro_voice}'")
                        logging.info(f"TTS: Text preview: '{cleaned_text[:80]}...'")
                        
                        generation_start = time.time()
                        
                        # CRITICAL OPERATION: This is where crashes often occur
                        # Generate audio using kokoro-onnx API
                        samples, sample_rate = self.kokoro_pipeline.create(
                            cleaned_text, 
                            voice=self.kokoro_voice,
                            speed=1.0
                        )
                        
                        generation_time = time.time() - generation_start
                        logging.info(f"TTS: Audio generation completed in {generation_time:.2f}s")
                        
                    except AttributeError as attr_error:
                        logging.error(f"TTS: Kokoro pipeline method error: {attr_error}")
                        logging.error(f"TTS: Pipeline object: {type(self.kokoro_pipeline)}")
                        return
                        
                    except TypeError as type_error:
                        logging.error(f"TTS: Kokoro API parameter error: {type_error}")
                        logging.error(f"TTS: Check voice name '{self.kokoro_voice}' is valid")
                        return
                        
                    except Exception as gen_error:
                        logging.error(f"TTS: Kokoro audio generation failed: {gen_error}")
                        import traceback
                        logging.error(f"TTS: Generation traceback: {traceback.format_exc()}")
                        return
                    
                    # =============================================================
                    # Validate generated audio
                    # =============================================================
                    try:
                        if samples is None:
                            logging.error("TTS: Kokoro returned None for samples")
                            return
                        
                        if len(samples) == 0:
                            logging.error("TTS: Kokoro generated empty audio (0 samples)")
                            return
                        
                        if sample_rate is None or sample_rate <= 0:
                            logging.error(f"TTS: Invalid sample rate: {sample_rate}")
                            return
                        
                        audio_duration = len(samples) / sample_rate
                        logging.info(f"TTS: Audio validated - duration: {audio_duration:.1f}s, sample_rate: {sample_rate}")
                        
                    except Exception as validation_error:
                        logging.error(f"TTS: Audio validation error: {validation_error}")
                        return
                    
                    # =============================================================
                    # Save to temporary WAV file with error handling
                    # =============================================================
                    tmp_path = None
                    
                    try:
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        
                        # Write audio to WAV file
                        sf.write(tmp_path, samples, sample_rate)
                        
                        # Verify file was created
                        if not os.path.exists(tmp_path):
                            logging.error(f"TTS: Temp file not created: {tmp_path}")
                            return
                        
                        file_size = os.path.getsize(tmp_path)
                        if file_size == 0:
                            logging.error(f"TTS: Temp file is empty: {tmp_path}")
                            return
                        
                        logging.info(f"TTS: Audio saved to temp file ({file_size} bytes)")
                        
                    except OSError as os_error:
                        logging.error(f"TTS: File system error writing WAV: {os_error}")
                        return
                        
                    except Exception as write_error:
                        logging.error(f"TTS: Error writing WAV file: {write_error}")
                        import traceback
                        logging.error(f"TTS: Write traceback: {traceback.format_exc()}")
                        return
                    
                    # =============================================================
                    # Play audio with pygame - wrapped in comprehensive try-finally
                    # =============================================================
                    try:
                        # =========================================================
                        # Initialize pygame mixer with error handling
                        # =========================================================
                        try:
                            # Clean shutdown of any existing mixer
                            try:
                                pygame.mixer.quit()
                            except:
                                pass  # Ignore errors if mixer wasn't initialized
                            
                            # Initialize with settings matching Kokoro output
                            pygame.mixer.pre_init(frequency=sample_rate, size=-16, channels=1, buffer=2048)
                            pygame.mixer.init()
                            
                            logging.info("TTS: Pygame mixer initialized successfully")
                            
                        except pygame.error as pygame_init_error:
                            logging.error(f"TTS: Pygame mixer initialization failed: {pygame_init_error}")
                            return
                            
                        except Exception as mixer_error:
                            logging.error(f"TTS: Unexpected mixer initialization error: {mixer_error}")
                            return
                        
                        # =========================================================
                        # Load and play the audio file
                        # =========================================================
                        try:
                            # Load the audio file
                            pygame.mixer.music.load(tmp_path)
                            pygame.mixer.music.set_volume(1.0)
                            
                            logging.info("TTS: Audio file loaded successfully")
                            
                            # Start playback
                            pygame.mixer.music.play()
                            
                            logging.info("TTS: Playback started")
                            
                        except pygame.error as load_error:
                            logging.error(f"TTS: Pygame load/play error: {load_error}")
                            return
                            
                        except Exception as play_error:
                            logging.error(f"TTS: Unexpected playback error: {play_error}")
                            return
                        
                        # =========================================================
                        # Wait for playback to complete with timeout
                        # =========================================================
                        try:
                            playback_start = time.time()
                            
                            # Set reasonable timeout (3x audio duration + buffer)
                            timeout = max(30, audio_duration * 3 + 10)
                            
                            # Wait for playback to finish
                            while pygame.mixer.music.get_busy():
                                elapsed = time.time() - playback_start
                                
                                # Check for timeout
                                if elapsed > timeout:
                                    logging.warning(f"TTS: Playback timeout after {elapsed:.1f}s")
                                    pygame.mixer.music.stop()
                                    break
                                
                                # Small sleep to prevent CPU spinning
                                time.sleep(0.1)
                            
                            playback_time = time.time() - playback_start
                            logging.info(f"TTS: Playback completed in {playback_time:.1f}s")
                            
                        except Exception as wait_error:
                            logging.error(f"TTS: Error during playback wait: {wait_error}")
                            # Try to stop playback on error
                            try:
                                pygame.mixer.music.stop()
                            except:
                                pass
                        
                    finally:
                        # =========================================================
                        # CRITICAL: Cleanup pygame and temp file (ALWAYS runs)
                        # =========================================================
                        logging.debug("TTS: Starting cleanup...")
                        
                        # Cleanup pygame
                        try:
                            pygame.mixer.music.stop()
                            pygame.mixer.music.unload()
                            pygame.mixer.quit()
                            logging.debug("TTS: Pygame cleaned up")
                        except Exception as pygame_cleanup_error:
                            logging.debug(f"TTS: Pygame cleanup note: {pygame_cleanup_error}")
                        
                        # Cleanup temporary file
                        if tmp_path:
                            try:
                                time.sleep(0.5)  # Brief delay to ensure file is released
                                if os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                                    logging.debug("TTS: Temp file deleted")
                            except Exception as file_cleanup_error:
                                logging.debug(f"TTS: Temp file cleanup note: {file_cleanup_error}")
                    
                    # =============================================================
                    # Log successful completion
                    # =============================================================
                    total_time = time.time() - start_time
                    logging.info(f"✅ TTS: Complete. Total time: {total_time:.2f}s")
                    
                except Exception as thread_error:
                    # =============================================================
                    # GLOBAL THREAD EXCEPTION HANDLER - Catches any uncaught errors
                    # =============================================================
                    elapsed = time.time() - start_time
                    logging.error(f"❌ TTS THREAD EXCEPTION after {elapsed:.1f}s: {thread_error}")
                    import traceback
                    logging.error(f"TTS THREAD TRACEBACK:\n{traceback.format_exc()}")
                    
                    # Attempt emergency cleanup
                    try:
                        pygame.mixer.quit()
                    except:
                        pass
            
            # =====================================================================
            # STEP 4: Start TTS in separate daemon thread
            # =====================================================================
            try:
                tts_thread = threading.Thread(target=speak_text, daemon=True)
                tts_thread.start()
                logging.info("TTS: Thread started successfully")
                return True
                
            except Exception as thread_start_error:
                logging.error(f"TTS: Failed to start thread: {thread_start_error}")
                return False
            
        except Exception as setup_error:
            # =====================================================================
            # OUTER EXCEPTION HANDLER - Catches setup errors before thread starts
            # =====================================================================
            elapsed = time.time() - start_time
            logging.error(f"❌ TTS Setup Error after {elapsed:.1f}s: {setup_error}")
            import traceback
            logging.error(f"TTS Setup Traceback:\n{traceback.format_exc()}")
            return False

            
    # removes memory commands and bullet points from models spoke responses
    def clean_response_for_tts(self, text):
        """
        Thoroughly clean response text for text-to-speech, including bullet point conversion
        
        Args:
            text (str): Original text to clean
            
        Returns:
            str: Cleaned text ready for speech synthesis
        """
        if not text or not isinstance(text, str):
            return text
        
        try:
            import re
            
            cleaned = text
            
            # Remove memory commands and their results (from existing logic)
            memory_patterns = [
                # NEW: Remove <think> blocks (QWEN reasoning mode) - MUST BE FIRST
                r'<think>.*?</think>',
                
                r'\[SEARCH:.*?\].*?END OF SEARCH.*?\n\n',
                r'\[STORE:.*?\].*?✅',
                r'\[REFLECT\].*?\n\n',
                r'\[FORGET:.*?\].*?✅',
                r'\[CORRECT:.*?\].*?\n\n',
                r'\[SUMMARIZE_CONVERSATION\].*?\n\n',
                r'\[REMINDER:.*?\].*?\n\n',
                r'\[COMMAND:.*?\].*?\n\n',
                r'\[DISCUSS_WITH_CLAUDE:.*?\].*?\n\n',
                # Clean up memory command indicators
                r'\*\*===== .*? =====\*\*.*?\*\*===== END OF .*? =====\*\*',
                r'✅.*?\n',
                r'❌.*?\n',
                # Remove model reasoning sections
                r'<model_reasoning>.*?</model_reasoning>',
                r'</?model_reasoning[^>]*>',  # Any orphaned opening/closing tags
            ]
            
            for pattern in memory_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove any remaining command brackets
            cleaned = re.sub(r'\[[A-Z_]+:.*?\]', '', cleaned)
            
            # ENHANCED: More comprehensive removal of checkmark symbols
            # Remove checkmarks in various contexts - standalone, with spaces, at line ends, etc.
            checkmark_patterns = [
                r'✅\s*',           # Green checkmark with optional space
                r'❌\s*',           # Red X with optional space  
                r'\s*✅\s*',        # Checkmark with spaces around it
                r'\s*❌\s*',        # Red X with spaces around it
                r'✅$',             # Checkmark at end of line
                r'❌$',             # Red X at end of line
                r'^✅\s*',          # Checkmark at start of line
                r'^❌\s*',          # Red X at start of line
                r'✅\s*\n',         # Checkmark followed by newline
                r'❌\s*\n',         # Red X followed by newline
            ]
            
            for pattern in checkmark_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
            
            # NEW: Convert bullet points to numbered lists
            lines = cleaned.split('\n')
            processed_lines = []
            bullet_counter = 1
            in_bullet_list = False
            
            for line in lines:
                stripped_line = line.strip()
                
                # Skip empty lines but reset counter
                if not stripped_line:
                    processed_lines.append(line)
                    if in_bullet_list:
                        bullet_counter = 1
                        in_bullet_list = False
                    continue
                
                # Check for bullet points (asterisk, dash, or bullet character)
                bullet_patterns = [
                    r'^\s*\*\s+(.+)',     # * bullet
                    r'^\s*-\s+(.+)',      # - bullet  
                    r'^\s*•\s+(.+)',      # • bullet
                    r'^\s*\+\s+(.+)',     # + bullet
                ]
                
                bullet_found = False
                for pattern in bullet_patterns:
                    match = re.match(pattern, line)
                    if match:
                        # Extract the content after the bullet
                        content = match.group(1)
                        # Replace with numbered format
                        processed_line = f"{bullet_counter}. {content}"
                        processed_lines.append(processed_line)
                        bullet_counter += 1
                        in_bullet_list = True
                        bullet_found = True
                        logging.debug(f"TTS: Converted bullet '{stripped_line}' to '{processed_line}'")
                        break
                
                # If no bullet pattern found, add line as-is
                if not bullet_found:
                    processed_lines.append(line)
                    # Reset counter if we were in a list
                    if in_bullet_list:
                        bullet_counter = 1
                        in_bullet_list = False
            
            # Join lines back together
            cleaned = '\n'.join(processed_lines)
            
            # Clean up markdown formatting for speech (existing logic)
            cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove bold
            cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Remove italic
            cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)        # Remove code backticks
            cleaned = re.sub(r'#+ ', '', cleaned)               # Remove markdown headers
            
            # Remove === divider lines (3 or more equals signs used as section separators)
            # Safe: single = (x = 5) and double == (Python equality) are untouched.
            # Only triggers on ===, =====, === Title ===, etc. — never natural prose.
            cleaned = re.sub(r'={3,}', '', cleaned)             # Remove === dividers
            
            # Additional cleaning for better speech
            replacements = {
                # Handle special characters
                '&': ' and ',       # Ampersand
                '@': ' at ',        # At symbol
                '%': ' percent ',   # Percent
                
                # Handle common abbreviations
                'e.g.': 'for example',
                'i.e.': 'that is',
                'etc.': 'and so on',
                'vs.': 'versus',
                'Dr.': 'Doctor',
                'Mr.': 'Mister',
                'Mrs.': 'Missus',
                'Ms.': 'Miss',
                
                # ENHANCED: Additional symbol cleanup
                '→': ' to ',         # Arrow symbol
                '←': ' from ',       # Left arrow  
                '↓': ' down ',       # Down arrow
                '↑': ' up ',         # Up arrow
                # Remove all emojis (comprehensive)
                '📅': '',            # Calendar
                '🎄': '',            # Christmas tree
                '💡': '',            # Light bulb
                '🎯': '',            # Target
                '🔧': '',            # Wrench
                '⚙️': '',            # Gear
                '🔍': '',            # Magnifying glass
                '📁': '',            # Folder
                '📄': '',            # Document
                '💾': '',            # Save disk
                '🗑️': '',            # Trash
                '⭐': '',            # Star (remove, not "star")
                '⚠️': '',            # Warning (remove, not "warning")
                '🎉': '',            # Party
                '👋': '',            # Wave
                '😊': '',            # Smile
                '🤔': '',            # Thinking
                '✨': '',            # Sparkles
                '🚀': '',            # Rocket
                '💬': '',            # Speech bubble
                '📝': '',            # Memo
                '🔔': '',            # Bell
                '📢': '',            # Loudspeaker
                '👍': '',            # Thumbs up
                '👎': '',            # Thumbs down
                '🎁': '',            # Gift
            }
            
            for old, new in replacements.items():
                cleaned = cleaned.replace(old, new)
            
            # Clean up extra whitespace (existing logic)
            cleaned = re.sub(r'\n{3,}', ' ', cleaned)          # Multiple newlines to space
            cleaned = re.sub(r'  +', ' ', cleaned)             # Multiple spaces to single space
            cleaned = re.sub(r'^\s+|\s+$', '', cleaned)        # Leading/trailing whitespace
            
            # Final cleanup - remove any remaining isolated symbols
            cleaned = re.sub(r'\s+[✅❌]\s+', ' ', cleaned)     # Isolated checkmarks with spaces
            cleaned = re.sub(r'[✅❌]', '', cleaned)            # Any remaining checkmarks
            
            # Clean up any double spaces that might have been created
            cleaned = re.sub(r'  +', ' ', cleaned)
            cleaned = cleaned.strip()
            
            # Log the cleaning results
            if text != cleaned:
                logging.info(f"TTS: Text cleaned. Original length: {len(text)}, Cleaned length: {len(cleaned)}")
                logging.debug(f"TTS: Cleaned text sample: '{cleaned[:100]}...'")
            
            return cleaned
            
        except Exception as e:
            logging.error(f"Error cleaning text for TTS: {e}")
            # Return original text if cleaning fails
            return text
            
    def speech_to_text(self, max_duration=30, adaptive_timeout=True):
        """
        Convert speech to text using Whisper with adaptive noise floor calibration.
        
        Changes from previous version:
        - Increased silence_duration from 1.5s to 2.5s to prevent premature cutoff
        - Increased min_audio_length from 0.3s to 1.0s to require meaningful audio
        - Enabled vad_filter=True so Whisper strips non-speech internally
        - Added temperature fallback: retries at 0.2 if greedy decode returns empty
        """
        if not self._init_whisper():
            logging.error("Failed to initialize Whisper model")
            return None
        
        try:
            logging.info(f"Starting speech recognition (max: {max_duration}s)")
            
            # Record audio with pre-roll buffer and adaptive noise floor
            audio_data = self._record_with_simple_silence_detection(
                max_duration=max_duration,
                silence_threshold=300,   # Starting value — overridden by adaptive calibration
                silence_duration=2.5,    # INCREASED from 1.5s — allows natural mid-thought pauses
                min_audio_length=1.0     # INCREASED from 0.3s — requires at least 1 second of speech
            )
            
            if audio_data is None:
                logging.info("No speech detected")
                return None
            
            # --- PASS 1: Greedy decode (temperature=0.0, fastest and most confident) ---
            try:
                segments, info = self.whisper_model.transcribe(
                    audio_data,
                    language="en",
                    beam_size=3,
                    best_of=2,
                    temperature=0.0,           # Greedy decode first
                    condition_on_previous_text=False,
                    word_timestamps=False,
                    vad_filter=True,           # NEW: Let Whisper strip silence/noise internally
                    vad_parameters=dict(
                        min_silence_duration_ms=500   # Whisper VAD: 500ms silence = segment boundary
                    )
                )
                
                # Extract text from segments
                result_text = ""
                for segment in segments:
                    segment_text = segment.text.strip()
                    if len(segment_text) >= 2:
                        result_text += segment_text + " "
                
                result = result_text.strip()
                
            except Exception as transcribe_error:
                logging.error(f"Whisper transcription error on pass 1: {transcribe_error}")
                result = ""
            
            # --- PASS 2: Temperature fallback if greedy decode returned nothing ---
            # This handles cases where background noise confused the greedy decoder
            if not result or len(result) < 2:
                logging.info("Pass 1 returned empty — retrying with temperature=0.2 fallback")
                try:
                    segments, info = self.whisper_model.transcribe(
                        audio_data,
                        language="en",
                        beam_size=3,
                        best_of=2,
                        temperature=0.2,           # Softer decode — recovers uncertain speech
                        condition_on_previous_text=False,
                        word_timestamps=False,
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=500
                        )
                    )
                    
                    result_text = ""
                    for segment in segments:
                        segment_text = segment.text.strip()
                        if len(segment_text) >= 2:
                            result_text += segment_text + " "
                    
                    result = result_text.strip()
                    
                    if result:
                        logging.info(f"Pass 2 (temperature fallback) recovered: '{result}'")
                    else:
                        logging.info("Pass 2 also returned empty — no valid speech found")
                        
                except Exception as fallback_error:
                    logging.error(f"Whisper transcription error on pass 2: {fallback_error}")
                    result = ""
            
            # --- Return final result ---
            if result:
                logging.info(f"Speech recognized: '{result}'")
                return result
            else:
                logging.info("No valid speech found after both passes")
                return None
                
        except Exception as e:
            logging.error(f"Speech-to-text error: {e}")
            return None
        
    def _record_with_simple_silence_detection(self, max_duration=30, silence_threshold=300,
                                            silence_duration=2.5, min_audio_length=1.0):
        """
        Improved audio recording with three key enhancements:
        
        1. PRE-ROLL RING BUFFER: Continuously captures the last 0.5s of audio before
        speech detection fires. Prepended to the recording so the first syllable
        is never clipped.
        
        2. ADAPTIVE NOISE FLOOR: Samples ambient audio for 0.5s at startup and sets
        the detection threshold to ambient_rms * 2.5. Prevents false triggers from
        background farm/environment noise and false stops from quiet moments.
        
        3. UPDATED DEFAULTS: silence_duration=2.5s and min_audio_length=1.0s give
        more room for natural mid-thought pauses before recording stops.
        """
        try:
            # Open audio stream
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # -----------------------------------------------------------------------
            # STEP 1: Adaptive noise floor calibration
            # Sample 0.5 seconds of ambient audio to establish background RMS level.
            # Threshold = ambient_rms * 2.5, clamped between 200 and 800.
            # -----------------------------------------------------------------------
            CALIBRATION_SECONDS = 0.5
            calibration_frames = int(CALIBRATION_SECONDS * self.sample_rate / self.chunk_size)
            calibration_rms_values = []
            
            logging.info("Calibrating ambient noise floor...")
            
            for _ in range(calibration_frames):
                try:
                    cal_data = stream.read(self.chunk_size, exception_on_overflow=False)
                    cal_chunk = np.frombuffer(cal_data, dtype=np.int16)
                    cal_rms = np.sqrt(np.mean(cal_chunk.astype(np.float32) ** 2))
                    calibration_rms_values.append(cal_rms)
                except Exception as cal_err:
                    logging.warning(f"Calibration read error (skipping frame): {cal_err}")
                    continue
            
            if calibration_rms_values:
                # Use median to resist single loud transients during calibration
                ambient_rms = float(np.median(calibration_rms_values))
                # Dynamic threshold: 2.5x ambient, clamped to safe range
                adaptive_threshold = max(200, min(800, ambient_rms * 2.5))
                logging.info(
                    f"Noise floor calibrated — ambient RMS: {ambient_rms:.1f}, "
                    f"detection threshold: {adaptive_threshold:.1f}"
                )
            else:
                # Fallback to passed-in threshold if calibration had no data
                adaptive_threshold = silence_threshold
                logging.warning(f"Calibration failed — using default threshold: {adaptive_threshold}")
            
            # -----------------------------------------------------------------------
            # STEP 2: Set up pre-roll ring buffer
            # Holds the last ~0.5s of audio in a circular deque. When speech is
            # detected, we prepend this buffer so the first syllable is not lost.
            # -----------------------------------------------------------------------
            from collections import deque
            
            PREROLL_SECONDS = 0.5
            frames_per_second = self.sample_rate // self.chunk_size
            preroll_capacity = int(PREROLL_SECONDS * frames_per_second)
            preroll_buffer = deque(maxlen=preroll_capacity)  # Auto-discards oldest frames
            
            # Recording state variables
            audio_buffer = []
            min_frames = int(min_audio_length * frames_per_second)
            max_frames = int(max_duration * frames_per_second)
            silence_frames_threshold = int(silence_duration * frames_per_second)
            
            silence_count = 0
            recording_started = False
            
            logging.info(
                f"Recording started — threshold: {adaptive_threshold:.0f}, "
                f"silence cutoff: {silence_duration}s, min length: {min_audio_length}s"
            )
            self._update_status('listening_for_speech', {'mode': 'adaptive'})
            
            # -----------------------------------------------------------------------
            # STEP 3: Main recording loop
            # -----------------------------------------------------------------------
            while len(audio_buffer) < max_frames:
                try:
                    # Read audio chunk
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Compute RMS volume for this chunk
                    volume = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
                    
                    if volume > adaptive_threshold:
                        # --- Speech detected ---
                        if not recording_started:
                            # Prepend pre-roll buffer so we don't clip the first syllable
                            preroll_frames = list(preroll_buffer)
                            audio_buffer.extend(preroll_frames)
                            recording_started = True
                            logging.info(
                                f"Speech detected (RMS: {volume:.0f}) — "
                                f"prepended {len(preroll_frames)} pre-roll frames"
                            )
                            self._update_status('recording_speech', {'mode': 'adaptive'})
                        
                        audio_buffer.append(data)
                        silence_count = 0  # Reset silence counter on speech
                        
                    else:
                        # --- Silence detected ---
                        if recording_started:
                            # After speech has started, keep buffering silence
                            # (we include silence frames in the buffer so speech sounds natural)
                            audio_buffer.append(data)
                            silence_count += 1
                            
                            # Check if we've had enough silence to consider speech done
                            if len(audio_buffer) >= min_frames and silence_count >= silence_frames_threshold:
                                actual_silence_s = silence_count / frames_per_second
                                logging.info(
                                    f"End of speech — {actual_silence_s:.1f}s silence detected, "
                                    f"total frames: {len(audio_buffer)}"
                                )
                                break
                        else:
                            # Still waiting for speech — keep pre-roll buffer rolling
                            preroll_buffer.append(data)
                            
                except Exception as chunk_err:
                    logging.error(f"Error reading audio chunk: {chunk_err}")
                    break
            
            # -----------------------------------------------------------------------
            # STEP 4: Clean up and return
            # -----------------------------------------------------------------------
            stream.stop_stream()
            stream.close()
            
            if audio_buffer and len(audio_buffer) >= min_frames:
                # Convert byte frames to normalized float32 array for Whisper
                audio_data = (
                    np.frombuffer(b''.join(audio_buffer), dtype=np.int16)
                    .astype(np.float32) / 32768.0
                )
                duration = len(audio_data) / self.sample_rate
                logging.info(f"Recording complete — {duration:.2f} seconds captured")
                self._update_status('processing_speech', {'duration': duration, 'mode': 'adaptive'})
                return audio_data
            else:
                logging.info("No sufficient speech detected")
                self._update_status('no_speech_detected', {'mode': 'adaptive'})
                return None
                
        except Exception as e:
            logging.error(f"Error in adaptive audio recording: {e}")
            self._update_status('recording_error', {'error': str(e), 'mode': 'adaptive'})
            return None
       
    
    def test_whisper_integration(self):
        """Test Whisper integration and all speech components"""
        try:
            result = {
                "whisper_available": False,
                "ollama_available": False,
                "audio_available": False,
                "tts_available": False,  # Add TTS testing
                "error": None
            }
            
            # Test Whisper
            if self._init_whisper():
                result["whisper_available"] = True
                result["whisper_model"] = self.whisper_model_size
            
            # Test Ollama
            if self._init_ollama():
                result["ollama_available"] = True
                models = self.ollama_client.list()
                result["available_models"] = [model.model for model in models.models]
            
            # Test audio input
            try:
                test_stream = self.audio.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                test_stream.close()
                result["audio_available"] = True
            except Exception as audio_err:
                result["audio_error"] = str(audio_err)
            
            # Test TTS (Edge-TTS)
            try:
                import edge_tts
                import pygame
                result["tts_available"] = True
                result["tts_engine"] = "Edge-TTS"
                logging.info("TTS test: Edge-TTS available")
            except ImportError:  # DEAD CODE TEST 2026-05-17: dropped unused 'as e' capture (ruff F841)
                # Try fallback to SAPI
                try:
                    import win32com.client
                    result["tts_available"] = True
                    result["tts_engine"] = "SAPI (fallback)"
                    logging.info("TTS test: SAPI available as fallback")
                except ImportError:
                    result["tts_available"] = False
                    result["tts_error"] = "Neither Edge-TTS nor SAPI available"
                    logging.error("TTS test: No TTS engines available")
            except Exception as e:
                result["tts_available"] = False
                result["tts_error"] = str(e)
                logging.error(f"TTS test error: {e}")
            
            return result
            
        except Exception as e:
            result = {"error": str(e)}
            return result
        

# Create singleton instance
whisper_speech_utils = WhisperSpeechUtils()