# wake_word_listener.py
"""
Wake word detection module for QWEN.

Runs a background thread that continuously listens for a configurable
wake phrase (default: "I have a question") using the already-loaded
Whisper model from whisper_speech_utils. When detected, sets a
threading.Event that main.py polls on each Streamlit refresh cycle.

Architecture:
  - Borrows the WhisperSpeechUtils singleton (no new model loading)
  - Own PyAudio instance (independent from STT pipeline)
  - Rolling 3-second buffer with 1-second slide for overlapping detection
  - RMS energy gate skips silent windows to reduce CPU load
  - On wake detection: releases mic (Stop-and-Restart handoff)
    so existing speech_to_text() can open a fresh stream cleanly
  - No changes required to whisper_speech.py, speech_utils.py, or chatbot.py

Integration points:
  - Instantiated in main.py at startup, passed whisper_speech_utils singleton
  - Polled via is_triggered() on each Streamlit refresh cycle
  - reset() called by main.py after handing off to speech_to_text()
"""

import threading
import logging
import time
import numpy as np
import pyaudio


class WakeWordListener:
    """
    Continuous background listener that detects a wake phrase via Whisper.

    Usage:
        listener = WakeWordListener(whisper_utils_instance)
        listener.start()
        ...
        if listener.is_triggered():
            listener.reset()
            # hand off to existing speech_to_text()
        ...
        listener.stop()
    """

    # ------------------------------------------------------------------
    # Audio capture constants — must match whisper_speech.py config
    # ------------------------------------------------------------------
    SAMPLE_RATE  = 16000         # Hz — Whisper expects 16kHz
    CHUNK_SIZE   = 512           # Frames per read — same as whisper_speech.py
    CHANNELS     = 1             # Mono
    AUDIO_FORMAT = pyaudio.paInt16

    # How many seconds of audio to transcribe per detection window.
    # 3 seconds is long enough to catch "I have a question" naturally
    # even if spoken slowly, and short enough for fast Whisper inference.
    WINDOW_SECONDS = 3.0

    # How often (in seconds) to transcribe the rolling buffer.
    # 1 second means each phrase gets at least 2 overlapping windows
    # that could contain it completely (3s window - 1s slide = 2s overlap).
    # Lower = more responsive but more CPU; higher = less CPU but slower.
    SLIDE_SECONDS = 1.0

    # RMS energy gate — chunks below this level are treated as silence
    # and skipped entirely (saves CPU). Lowered from 200 to 100 because
    # successful detections were barely clearing (RMS=207 vs threshold=200).
    # Normal speech at arm's length from a USB mic is typically RMS 150-400.
    SILENCE_THRESHOLD = 100

    def __init__(self, whisper_utils, wake_phrase: str = "I have a question"):
        """
        Initialize the wake word listener.

        Args:
            whisper_utils: The WhisperSpeechUtils singleton instance
                           (already has Whisper model + PyAudio loaded).
                           Passed in from main.py to avoid circular imports.
            wake_phrase:   The phrase to listen for. Case-insensitive.
                           Default: "I have a question"
        """
        # ---- Core references ----
        self._whisper_utils = whisper_utils   # Borrow existing singleton
        self.wake_phrase = wake_phrase.lower().strip()

        # ---- Thread control ----
        self._stop_event    = threading.Event()  # Set externally to stop the loop
        self._trigger_event = threading.Event()  # Set internally when wake detected
        self._thread        = None               # Background thread handle

        # ---- State tracking (for UI status badge in Voice Settings) ----
        # Valid values: "stopped" | "starting" | "listening" | "triggered"
        self.status       = "stopped"
        self._status_lock = threading.Lock()     # Protects status from race conditions

        # ---- Audio stream handle ----
        # Owned exclusively by the listener thread. stop() signals the
        # thread via _stop_event and the thread closes its own stream
        # in _listen_loop cleanup (never from the main thread).
        self._stream = None
        self._stream_lock = threading.Lock()  # Serializes all stream open/close ops

        # ---- Stop guard ----
        # Prevents duplicate stop() calls from Streamlit rapid reruns
        # from hammering the thread with concurrent join() attempts
        self._stopping = False

        # ---- Own PyAudio instance ----
        # CRITICAL: Do NOT borrow whisper_utils.audio — sharing a single
        # PyAudio instance causes stream conflicts that kill both pipelines.
        # We create our own instance so wake word detection and the existing
        # STT pipeline are completely independent.
        try:
            self._audio = pyaudio.PyAudio()
            logging.info("WakeWordListener: own PyAudio instance created")
        except Exception as e:
            self._audio = None
            logging.error(f"WakeWordListener: failed to create PyAudio instance: {e}")

        logging.info(
            f"WakeWordListener initialized — wake phrase: '{self.wake_phrase}'"
        )

    # ------------------------------------------------------------------
    # Public interface — called from main.py
    # ------------------------------------------------------------------

    def start(self):
        """
        Start the background listening thread.
        Safe to call multiple times — ignores duplicate calls if already running.
        Recreates the PyAudio instance if stop() previously destroyed it.
        """
        # Guard against duplicate starts
        if self._thread and self._thread.is_alive():
            logging.warning("WakeWordListener: already running — ignoring start()")
            return

        # Clear any stale flags from a previous session
        self._stop_event.clear()
        self._trigger_event.clear()
        self._stopping = False  # Reset stop guard for fresh start/stop cycle
        self._set_status("starting")

        # ---------------------------------------------------------------
        # FIX 5: Recreate PyAudio instance if stop() terminated it.
        # stop() calls self._audio.terminate() and sets self._audio = None.
        # Without this, _open_stream() fails on every attempt because
        # self._audio is None, and the listener thread loops forever
        # printing "own PyAudio instance not available".
        # ---------------------------------------------------------------
        if self._audio is None:
            try:
                self._audio = pyaudio.PyAudio()
                logging.info("WakeWordListener: PyAudio instance recreated for restart")
            except Exception as e:
                logging.error(f"WakeWordListener: failed to recreate PyAudio instance: {e}")
                self._set_status("stopped")
                return  # Don't start thread if we can't open audio

        # Daemon=True ensures the thread dies automatically if the main
        # Streamlit process exits, preventing zombie threads on app restart
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="WakeWordListenerThread",
            daemon=True
        )
        self._thread.start()
        logging.info("WakeWordListener: background thread started")

    def stop(self):
        """
        Signal the background thread to stop and wait for it to exit cleanly.
        
        CRITICAL: Do NOT call _close_stream() here. The listener thread calls
        _close_stream() in its own cleanup path (_listen_loop exit). If we
        also call it from this thread, both threads race into PyAudio's C
        code simultaneously — stream.close() frees PortAudio buffers that
        the other thread is still using, causing heap corruption (Windows
        error 0xc0000374) and an instant segfault that kills Streamlit.
        
        With the rolling buffer approach, each stream.read() is only 512
        frames at 16kHz = 32ms. The thread will see _stop_event within
        one chunk cycle and handle its own stream cleanup safely.
        """
        # ---------------------------------------------------------------
        # FIX 14: Guard against duplicate stop() calls.
        # Streamlit reruns can fire stop() multiple times in rapid
        # succession before the thread has exited. Without this guard,
        # each call blocks on join(3s), stacking up and potentially
        # interfering with thread cleanup. Return early if already stopping.
        # ---------------------------------------------------------------
        if self._stopping:
            logging.info("WakeWordListener: stop() already in progress — skipping duplicate call")
            return
        self._stopping = True

        logging.info("WakeWordListener: stop() called")
        self._stop_event.set()
        self._set_status("stopped")

        # ---------------------------------------------------------------
        # FIX 13: Do NOT call self._close_stream() here.
        #
        # Previously this was called "defensively" to unblock stream.read()
        # in case the thread was stuck on a long blocking read. But with
        # the rolling buffer (CHUNK_SIZE=512 at 16kHz = 32ms per read),
        # the thread checks _stop_event every 32ms and exits promptly.
        #
        # The crash log (crash_fault.log) confirmed the race condition:
        #   Thread A (main):     stop() → _close_stream() → stream.close()
        #   Thread B (listener): _listen_loop exit → _close_stream() → stream.close()
        # Both hit PyAudio's C code concurrently → heap corruption → segfault.
        #
        # The listener thread owns the stream — let it close its own stream.
        # ---------------------------------------------------------------

        # Wait up to 3 seconds for graceful thread exit
        _thread_exited = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logging.warning(
                    "WakeWordListener: thread did not exit within 3s — "
                    "it will die when the process exits (daemon=True)"
                )
                # -------------------------------------------------------
                # FIX 12: Do NOT set self._thread = None here. The thread
                # is still alive, so is_running() must return True to
                # prevent start() from spawning a duplicate thread.
                # Also do NOT terminate PyAudio — the zombie thread may
                # still call self._audio.open() or stream.read(), and
                # accessing a terminated PyAudio instance causes a
                # segfault that kills the Streamlit process.
                # The daemon flag ensures it dies with the process.
                # -------------------------------------------------------
            else:
                logging.info("WakeWordListener: thread stopped cleanly")
                _thread_exited = True
                self._thread = None
        else:
            # Thread was already dead or never started
            _thread_exited = True
            self._thread = None

        # Clean up our own PyAudio instance — ONLY if thread is confirmed dead.
        # If thread is still alive (zombie), leave _audio intact so the thread
        # can exit gracefully. start() will recreate it if _audio is None.
        if _thread_exited and self._audio is not None:
            try:
                self._audio.terminate()
                logging.info("WakeWordListener: PyAudio instance terminated")
            except Exception as e:
                logging.warning(f"WakeWordListener: error terminating PyAudio: {e}")
            finally:
                self._audio = None
        elif not _thread_exited:
            logging.info("WakeWordListener: skipping PyAudio terminate (thread still alive)")

        # Clear the stopping guard so start()/stop() can cycle again
        self._stopping = False

    def is_running(self) -> bool:
        """Return True if the listener thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    def is_triggered(self) -> bool:
        """
        Return True if the wake phrase was detected since the last reset().
        Call this on every Streamlit refresh cycle in main.py.
        """
        return self._trigger_event.is_set()

    # QUARANTINED 2026-05-19: No callers found in cross-repo scan (batch 2 speech cleanup pass).
    # Orphan by architecture drift. The module docstring and method docstring claim main.py
    # calls reset() after handing off to speech_to_text(), but main.py's actual trigger
    # handler at L2418-2520 does stop() then start() instead — recreating the listener
    # rather than resetting it. Zero callers anywhere.
    # Renamed to detect any silent dispatch; safe to delete after one session cycle without errors.
    def _UNUSED_reset(self):
        """
        Clear the trigger flag after main.py has handled the wake event.
        Resumes listening automatically — no need to call start() again.
        Call this AFTER handing off to speech_to_text() so the mic is free.
        """
        self._trigger_event.clear()

        # Only update status if thread is still alive
        if self.is_running():
            self._set_status("listening")

        logging.info("WakeWordListener: trigger cleared — resuming listening")

    def get_status(self) -> str:
        """
        Return current listener status string for the UI status badge.

        Returns:
            str: One of 'stopped' | 'starting' | 'listening' | 'triggered'
        """
        with self._status_lock:
            return self.status

    def set_wake_phrase(self, phrase: str):
        """
        Update the wake phrase at runtime (e.g. from Voice Settings UI input).
        Takes effect on the next detection window — no restart required.

        Args:
            phrase: New wake phrase string. Will be lowercased and stripped.
        """
        self.wake_phrase = phrase.lower().strip()
        logging.info(f"WakeWordListener: wake phrase updated to '{self.wake_phrase}'")

    # ------------------------------------------------------------------
    # Internal helpers — not called from outside this module
    # ------------------------------------------------------------------

    def _set_status(self, new_status: str):
        """Thread-safe status update."""
        with self._status_lock:
            self.status = new_status

    def _close_stream(self):
        """
        Safely close the PyAudio input stream if one is currently open.
        
        Hardened against PortAudio heap corruption on Windows:
        - Checks is_active() before calling stop_stream() to avoid
          stopping an already-stopped stream (corrupts PortAudio C buffers)
        - Separate try/except for stop_stream() vs close() so a failure
          in stop doesn't skip close
        - Lock serializes access to prevent any cross-thread races
        """
        with self._stream_lock:
            if self._stream is None:
                return

            # Step 1: Stop the stream only if it's actively capturing
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                    logging.debug("WakeWordListener: stream stopped")
            except Exception as e:
                # Non-fatal — stream may already be in error state
                logging.warning(f"WakeWordListener: error stopping stream: {e}")

            # Step 2: Close the stream (releases PortAudio resources)
            # This runs even if stop_stream() failed above
            try:
                self._stream.close()
                logging.debug("WakeWordListener: stream closed")
            except Exception as e:
                logging.warning(f"WakeWordListener: error closing stream: {e}")

            # Always clear the reference regardless of close success
            self._stream = None

    def _open_stream(self):
        """
        Open a fresh PyAudio input stream using the same configuration
        as whisper_speech.py (SAMPLE_RATE, CHUNK_SIZE, CHANNELS).

        Uses our own dedicated PyAudio instance (self._audio) to avoid
        resource conflicts with the STT pipeline's PyAudio instance.

        Returns:
            pyaudio.Stream on success, None on failure.
        """
        try:
            # Use our own PyAudio instance — not whisper_utils.audio
            if self._audio is None:
                logging.error(
                    "WakeWordListener: own PyAudio instance not available"
                )
                return None

            stream = self._audio.open(
                format=self.AUDIO_FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )

            logging.debug("WakeWordListener: audio stream opened successfully")
            return stream

        except OSError as e:
            # Most common cause: mic is held by another stream
            # (e.g. active speech_to_text() session)
            logging.error(
                f"WakeWordListener: cannot open audio stream (mic busy?): {e}"
            )
            return None

        except Exception as e:
            logging.error(
                f"WakeWordListener: unexpected error opening stream: {e}",
                exc_info=True
            )
            return None

    def _transcribe_window(self, audio_data: np.ndarray) -> str:
        """
        Run Whisper on a single audio window (WINDOW_SECONDS long).

        Uses the already-loaded model from whisper_utils — no reload cost.
        Uses beam_size=1 and best_of=1 for fastest inference (accuracy is
        sufficient for a short, natural-language wake phrase).

        Args:
            audio_data: float32 numpy array normalized to [-1.0, 1.0]

        Returns:
            Lowercase transcript string, or "" on any failure.
        """
        try:
            model = self._whisper_utils.whisper_model

            if model is None:
                logging.warning(
                    "WakeWordListener: Whisper model not loaded in whisper_utils"
                )
                return ""

            segments, _ = model.transcribe(
                audio_data,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,


            )

            # Combine all segment text into a single lowercase string
            transcript = " ".join(seg.text for seg in segments).lower().strip()

            if transcript:
                # Promoted from debug → info so transcripts are always visible
                logging.info(f"WakeWordListener: Whisper transcript: '{transcript}'")
            else:
                # Whisper produced no output — VAD filter may have rejected the window
                logging.info("WakeWordListener: Whisper returned empty transcript "
                             "(VAD filtered or no speech detected in window)")

            return transcript

        except Exception as e:
            logging.error(
                f"WakeWordListener: transcription error: {e}",
                exc_info=True
            )
            return ""

    def _phrase_detected(self, transcript: str) -> bool:
        """
        Check if the wake phrase (or a known variant) appears in the transcript.

        Uses substring matching to handle cases where Whisper transcribes
        surrounding words alongside the wake phrase.

        Whisper occasionally drops the article or contracts "have" in fast
        speech — aliases below catch the most common variations observed
        in practice.

        Args:
            transcript: Lowercase transcript string from _transcribe_window()

        Returns:
            True if wake phrase or a variant is detected, False otherwise.
        """
        if not transcript:
            return False

        # ---- Primary match ----
        if self.wake_phrase in transcript:
            logging.info(
                f"WakeWordListener: primary match — '{self.wake_phrase}' "
                f"in '{transcript}'"
            )
            return True

        # ---- Phonetic / fast-speech aliases ----
        # Whisper occasionally drops the article or contracts "have"
        # when speech is fast or slightly mumbled.
        # Add confirmed mishearings here as you observe them in real use.
        # NOTE: All aliases MUST be lowercase — _transcribe_window()
        # lowercases the transcript, and Python 'in' is case-sensitive.
        aliases = [
            "i've got a question",   # Natural contraction variant
            "i have question",       # Dropped article — common in fast speech
            "i have a questions",    # Pluralization artifact from Whisper
            "i have a quest",        # Truncation if speech ends abruptly
        ]

        for alias in aliases:
            if alias in transcript:
                logging.info(
                    f"WakeWordListener: alias match '{alias}' "
                    f"in '{transcript}' — treating as wake phrase"
                )
                return True

        # No match — log what Whisper heard so we can see near-misses
        logging.info(f"WakeWordListener: no match — heard: '{transcript}' "
                     f"(looking for: '{self.wake_phrase}')")
        return False

    # ------------------------------------------------------------------
    # Main background thread loop
    # ------------------------------------------------------------------

    def _listen_loop(self):
        """
        Core background thread — rolling buffer with overlapping windows.

        Instead of capturing discrete 2-second blocks (which miss phrases
        that straddle a boundary), this version continuously feeds audio
        into a ring buffer and transcribes every SLIDE_SECONDS using the
        last WINDOW_SECONDS of audio.

        With WINDOW_SECONDS=3 and SLIDE_SECONDS=1, each phrase gets at
        least 2 overlapping windows where it could appear in full:

            [Window 1: 0-3s]
                  [Window 2: 1-4s]    ← 2s overlap
                        [Window 3: 2-5s]

        "I have a question" takes ~1.5-2s to say, so even if spoken
        across a window boundary, at least one window captures it entirely.

        Outer loop:
          Opens an audio stream, runs inner capture loop, handles errors.

        Inner loop:
          1. Read one audio chunk into the ring buffer (continuous)
          2. Every SLIDE_SECONDS, check the buffer:
             a. RMS gate — skip if silent (saves CPU)
             b. Transcribe with Whisper (reuses loaded model)
             c. Check transcript for wake phrase or alias
          3. If detected: close stream, set trigger, wait for reset
        """
        from collections import deque

        logging.info("WakeWordListener: _listen_loop started")
        self._set_status("listening")

        # Pre-compute chunk counts from timing constants
        # chunks_per_window: how many chunks fill WINDOW_SECONDS of audio
        # chunks_per_slide:  how many chunks fill SLIDE_SECONDS (check cadence)
        chunks_per_window = int(self.SAMPLE_RATE * self.WINDOW_SECONDS) // self.CHUNK_SIZE
        chunks_per_slide = int(self.SAMPLE_RATE * self.SLIDE_SECONDS) // self.CHUNK_SIZE

        while not self._stop_event.is_set():
            try:
                # ----------------------------------------------------------
                # Open a fresh stream at the start of each outer iteration.
                # ----------------------------------------------------------
                self._stream = self._open_stream()

                if self._stream is None:
                    logging.warning(
                        "WakeWordListener: mic unavailable — retrying in 2s"
                    )
                    time.sleep(2.0)
                    continue

                logging.info("WakeWordListener: stream open — entering capture loop")

                # Ring buffer: auto-discards oldest chunks when full.
                # Holds exactly WINDOW_SECONDS of audio at all times.
                ring_buffer = deque(maxlen=chunks_per_window)
                chunks_since_check = 0

                # ----------------------------------------------------------
                # Inner capture loop — continuous chunk reading
                # ----------------------------------------------------------
                while (not self._stop_event.is_set() and
                       not self._trigger_event.is_set()):

                    # Read one chunk into the ring buffer
                    try:
                        data = self._stream.read(
                            self.CHUNK_SIZE, exception_on_overflow=False
                        )
                        ring_buffer.append(data)
                        chunks_since_check += 1
                    except OSError as e:
                        logging.warning(
                            f"WakeWordListener: stream read error: {e}"
                        )
                        self._stream = None
                        break  # Break to outer loop to reopen stream

                    # -------------------------------------------------------
                    # Every SLIDE_SECONDS worth of chunks, check the buffer.
                    # Require at least half a window of data so we don't
                    # transcribe near-empty buffers at startup.
                    # -------------------------------------------------------
                    if (chunks_since_check >= chunks_per_slide and
                            len(ring_buffer) >= chunks_per_window // 2):

                        chunks_since_check = 0

                        # Combine ring buffer into a single int16 array
                        audio_int16 = np.frombuffer(
                            b"".join(ring_buffer), dtype=np.int16
                        )

                        # ---- RMS energy gate ----
                        rms = np.sqrt(
                            np.mean(audio_int16.astype(np.float32) ** 2)
                        )

                        if rms < self.SILENCE_THRESHOLD:
                            # Silent window — skip Whisper to save CPU
                            logging.debug(
                                f"WakeWordListener: buffer RMS={rms:.0f} "
                                f"below threshold ({self.SILENCE_THRESHOLD})"
                                f" — skipping"
                            )
                            continue

                        logging.info(
                            f"WakeWordListener: buffer RMS={rms:.0f}"
                            f" — sending to Whisper"
                        )

                        # Normalize to float32 [-1.0, 1.0] for Whisper
                        audio_float32 = (
                            audio_int16.astype(np.float32) / 32768.0
                        )

                        # Transcribe the rolling window
                        transcript = self._transcribe_window(audio_float32)

                        # Check for wake phrase
                        if self._phrase_detected(transcript):
                            logging.info(
                                f"WakeWordListener: ✅ Wake phrase detected!"
                                f" Transcript: '{transcript}'"
                            )

                            # Close stream BEFORE setting trigger —
                            # ensures mic is free for speech_to_text()
                            self._close_stream()
                            self._set_status("triggered")
                            self._trigger_event.set()

                            # Wait here until stop() or reset() clears us
                            while (not self._stop_event.is_set() and
                                   self._trigger_event.is_set()):
                                time.sleep(0.1)

                            # Break inner loop → outer loop reopens stream
                            break

                # Close stream when inner loop exits
                self._close_stream()

            except Exception as e:
                logging.error(
                    f"WakeWordListener: error in listen loop: {e}",
                    exc_info=True
                )
                self._close_stream()
                time.sleep(2.0)  # Backoff before retry

        # ------------------------------------------------------------------
        # Thread exit — clean up
        # ------------------------------------------------------------------
        self._close_stream()
        self._set_status("stopped")
        logging.info("WakeWordListener: _listen_loop exited cleanly")