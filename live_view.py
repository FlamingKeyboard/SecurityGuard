"""
Live Multi-Camera View - Real-time video streaming from all Vivint cameras.

Displays all cameras in a grid layout with low-latency RTSP streaming.
Uses OpenCV for rendering and threading for parallel camera capture.
Supports multi-camera audio playback via ffplay and two-way audio via WebRTC.

Usage:
    python live_view.py                    # All cameras in grid (SD)
    python live_view.py Doorbell           # Single camera with audio
    python live_view.py --hd               # HD streams (better quality, more latency)
    python live_view.py --no-audio         # Disable audio playback
    python live_view.py --audio-device "Microphone (Name)"  # Specify microphone

Controls:
    q     - Quit
    s     - Save snapshot of all cameras
    f     - Toggle fullscreen
    a     - Toggle all audio on/off
    n     - Toggle native resolution (in fullscreen)
    1-3   - Focus on specific camera
    !@#   - Toggle audio for camera 1/2/3 (Shift+1/2/3)
    0     - Return to grid view
    t     - Push-to-talk (speak through doorbell via WebRTC)
    SPACE - Push-to-talk (alternative key)

Audio:
    - Press Shift+1/2/3 to toggle audio for individual cameras
    - Press 'a' to toggle all audio on/off
    - Multiple cameras can play audio simultaneously
    - Press 't' or SPACE to speak through the doorbell (push-to-talk)
    - Full-duplex: you can hear the doorbell while speaking

Two-Way Audio:
    - Uses WebRTC via Vivint's TURN relay servers
    - Works from anywhere (not just LAN)
    - Requires aiortc and doorbell with two-way audio support

Requires: opencv-python (pip install opencv-python)
          aiortc (pip install aiortc) - for WebRTC two-way audio
Run with: python live_view.py
"""

import asyncio
import subprocess
import sys
import threading
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
_LOGGER = logging.getLogger(__name__)

# Try to import WebRTC two-way audio (works from anywhere via TURN relay)
try:
    from vivint_webrtc import VivintWebRTCClient, VivintWebRTCConfig
    HAS_TWO_WAY_WEBRTC = True
except ImportError:
    HAS_TWO_WAY_WEBRTC = False
    _LOGGER.debug("WebRTC two-way audio not available (aiortc not installed)")


class WebRTCTwoWayAudio:
    """
    WebRTC-based two-way audio for Vivint cameras.

    Works from anywhere (not just LAN) via TURN relay.
    Runs the async WebRTC client in a background thread.
    """

    def __init__(self, oauth_token: str, camera_uuid: str, audio_device: str = None):
        """
        Args:
            oauth_token: Vivint OAuth id_token
            camera_uuid: Camera device UUID (from camera.data['uuid'])
            audio_device: Windows DirectShow audio device name (optional)
        """
        self.oauth_token = oauth_token
        self.camera_uuid = camera_uuid
        self.audio_device = audio_device
        self.client: Optional[VivintWebRTCClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._error: Optional[str] = None

    def start(self) -> bool:
        """Start WebRTC two-way audio session (blocking until connected)."""
        if self._running:
            return self._connected

        self._running = True
        self._error = None

        # Create event to wait for connection
        connected_event = threading.Event()

        def run_async():
            """Run the async WebRTC client in a new event loop."""
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            try:
                self._loop.run_until_complete(self._connect_and_start(connected_event))
            except Exception as e:
                _LOGGER.error(f"WebRTC thread error: {e}")
                self._error = str(e)
            finally:
                self._loop.close()
                self._running = False

        self._thread = threading.Thread(target=run_async, daemon=True)
        self._thread.start()

        # Wait for connection (with timeout)
        if connected_event.wait(timeout=30.0):
            return self._connected
        else:
            self._error = "Connection timeout"
            return False

    async def _connect_and_start(self, connected_event: threading.Event):
        """Connect to camera and start two-way audio."""
        try:
            config = VivintWebRTCConfig(
                oauth_token=self.oauth_token,
                camera_uuid=self.camera_uuid,
                audio_device=self.audio_device,
            )
            self.client = VivintWebRTCClient(config)

            _LOGGER.info("WebRTC: Connecting...")
            if await self.client.connect():
                _LOGGER.info("WebRTC: Connected, starting two-way talk...")
                if await self.client.start_two_way_talk():
                    self._connected = True
                    _LOGGER.info("WebRTC: Two-way audio ACTIVE")
                    connected_event.set()

                    # Keep running until stopped
                    while self._running and self.client.connected:
                        await asyncio.sleep(0.5)
                else:
                    self._error = "Failed to start two-way talk"
                    _LOGGER.error("WebRTC: Failed to start two-way talk")
            else:
                self._error = "Failed to connect"
                _LOGGER.error("WebRTC: Failed to connect")

            connected_event.set()  # Unblock even on failure

        except Exception as e:
            self._error = str(e)
            _LOGGER.error(f"WebRTC connection error: {e}")
            connected_event.set()
        finally:
            if self.client:
                await self.client.disconnect()
            self._connected = False

    def stop(self):
        """Stop WebRTC two-way audio session."""
        _LOGGER.info("WebRTC: Stopping...")
        self._running = False

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._connected = False
        self.client = None
        _LOGGER.info("WebRTC: Stopped")

    @property
    def is_active(self) -> bool:
        """Check if two-way audio is active."""
        return self._connected and self._running


# Try to import OpenCV
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("ERROR: opencv-python not installed")
    print("Install with: pip install opencv-python")
    print("Then run with: C:\\Python312\\python.exe live_view.py")
    sys.exit(1)


@dataclass
class CameraStream:
    """Represents a camera's stream state."""
    name: str
    url: str
    frame: Optional[np.ndarray] = None
    last_update: float = 0
    fps: float = 0
    connected: bool = False
    error: Optional[str] = None
    width: int = 0
    height: int = 0
    latency_ms: float = 0


class LowLatencyCapture(threading.Thread):
    """
    Threaded camera capture with low-latency RTSP settings.

    Each camera runs in its own thread to maximize parallelism.
    """

    def __init__(self, camera: CameraStream, frame_queue: queue.Queue, hd: bool = False):
        super().__init__(daemon=True)
        self.camera = camera
        self.frame_queue = frame_queue
        self.hd = hd
        self.running = True
        self.stopping = False  # Graceful shutdown flag
        self.cap = None

    def run(self):
        """Main capture loop."""
        url = self.camera.url

        _LOGGER.info(f"[{self.camera.name}] Connecting...")

        # Set FFmpeg options via environment for lower latency
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            self.camera.error = "Failed to open stream"
            self.camera.connected = False
            _LOGGER.error(f"[{self.camera.name}] Failed to connect")
            return

        # Set capture properties for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer

        # Get resolution
        self.camera.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.camera.connected = True
        _LOGGER.info(f"[{self.camera.name}] Connected ({self.camera.width}x{self.camera.height})")

        frame_count = 0
        fps_start = time.time()
        fps_frames = 0
        reconnect_attempts = 0
        max_reconnect_attempts = 5

        while self.running:
            try:
                frame_start = time.time()
                ret, frame = self.cap.read()

                if not ret:
                    raise Exception("Failed to read frame")

                # Calculate frame latency (time to receive frame)
                self.camera.latency_ms = (time.time() - frame_start) * 1000

                frame_count += 1
                fps_frames += 1
                reconnect_attempts = 0  # Reset on success

                # Calculate FPS every second
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    self.camera.fps = fps_frames / elapsed
                    fps_frames = 0
                    fps_start = time.time()

                # Update resolution if it changed
                h, w = frame.shape[:2]
                if w != self.camera.width or h != self.camera.height:
                    self.camera.width = w
                    self.camera.height = h

                # Update camera state
                self.camera.frame = frame
                self.camera.last_update = time.time()
                self.camera.error = None

                # Put frame in queue (non-blocking, drop old frames)
                try:
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.frame_queue.put_nowait((self.camera.name, frame))
                except queue.Full:
                    pass

            except Exception as e:
                # Don't log errors if we're shutting down
                if self.stopping:
                    break

                # Handle stream errors and reconnect
                self.camera.error = "Reconnecting..."
                self.camera.connected = False
                reconnect_attempts += 1

                if reconnect_attempts > max_reconnect_attempts:
                    self.camera.error = "Connection failed"
                    _LOGGER.error(f"[{self.camera.name}] Max reconnect attempts reached")
                    time.sleep(5)
                    reconnect_attempts = 0  # Reset and keep trying

                _LOGGER.warning(f"[{self.camera.name}] Stream error, reconnecting... ({reconnect_attempts})")
                time.sleep(1)

                if self.stopping:
                    break

                try:
                    if self.cap:
                        self.cap.release()
                    self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if self.cap.isOpened():
                        self.camera.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        self.camera.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        self.camera.connected = True
                        self.camera.error = None
                        _LOGGER.info(f"[{self.camera.name}] Reconnected")
                except Exception as reconnect_err:
                    if not self.stopping:
                        _LOGGER.error(f"[{self.camera.name}] Reconnect failed: {reconnect_err}")

    def stop(self):
        """Stop the capture thread gracefully."""
        self.stopping = True  # Signal graceful shutdown first
        self.running = False
        time.sleep(0.1)  # Give thread time to see the flag
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass  # Ignore errors during shutdown


class AudioPlayer:
    """Plays audio from RTSP stream using ffplay subprocess."""

    def __init__(self, url: str, camera_name: str):
        self.url = url
        self.camera_name = camera_name
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        """Start audio playback."""
        try:
            # ffplay for audio only (no video window)
            cmd = [
                "ffplay",
                "-rtsp_transport", "tcp",
                "-nodisp",  # No video display
                "-vn",  # No video processing
                "-loglevel", "quiet",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-i", self.url,
            ]
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _LOGGER.info(f"[{self.camera_name}] Audio started")
        except FileNotFoundError:
            _LOGGER.warning("ffplay not found - audio disabled")
        except Exception as e:
            _LOGGER.warning(f"[{self.camera_name}] Audio failed: {e}")

    def stop(self):
        """Stop audio playback."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None


class MultiCameraViewer:
    """
    Multi-camera viewer with grid layout and multi-audio support.
    """

    def __init__(self, cameras: dict[str, str], hd: bool = False, audio: bool = True,
                 webrtc_credentials: dict = None, audio_device: str = None):
        """
        Args:
            cameras: Dict mapping camera names to RTSP URLs
            hd: Use HD streams (higher quality but more latency)
            audio: Enable audio playback
            webrtc_credentials: {'oauth_token': str, 'camera_uuid': str} for WebRTC two-way audio
            audio_device: Windows DirectShow audio device name for WebRTC microphone
        """
        self.camera_urls = cameras  # Keep URLs for audio
        self.cameras = {
            name: CameraStream(name=name, url=url)
            for name, url in cameras.items()
        }
        self.hd = hd
        self.audio_enabled = audio
        self.captures: list[LowLatencyCapture] = []
        self.frame_queues: dict[str, queue.Queue] = {}
        self.running = False
        self.focused_camera: Optional[str] = None  # None = grid view
        self.fullscreen = False
        self.native_resolution = False  # Show native res in fullscreen
        self.window_name = "Vivint Live View"
        # Multi-audio support: dict of camera_name -> AudioPlayer
        self.audio_players: dict[str, AudioPlayer] = {}
        self.camera_list: list[str] = list(cameras.keys())  # Ordered list for indexing

        # Two-way audio (push-to-talk) - WebRTC only (SIP doesn't support mic input)
        self.webrtc_credentials = webrtc_credentials
        self.audio_device = audio_device
        self.webrtc_two_way = None  # WebRTC-based two-way audio
        self.ptt_active = False  # Push-to-talk active
        self.ptt_connected = False  # Session established
        self.ptt_last_toggle = 0  # Debounce timestamp
        self.ptt_failed = False  # Don't retry if failed

    def start(self):
        """Start all camera capture threads."""
        _LOGGER.info(f"Starting {len(self.cameras)} camera streams...")

        for name, camera in self.cameras.items():
            q = queue.Queue(maxsize=2)
            self.frame_queues[name] = q

            capture = LowLatencyCapture(camera, q, hd=self.hd)
            capture.start()
            self.captures.append(capture)

        self.running = True

    def stop(self):
        """Stop all capture threads and audio."""
        _LOGGER.info("Shutting down...")
        self.running = False

        # Stop push-to-talk if active
        if self.ptt_active:
            self._stop_ptt()

        # Stop all audio first
        self._stop_all_audio()

        # Stop all capture threads
        for capture in self.captures:
            capture.stop()
        for capture in self.captures:
            capture.join(timeout=1)

    def _start_audio(self, camera_name: str):
        """Start audio for a specific camera (adds to active audio streams)."""
        if not self.audio_enabled:
            return

        if camera_name in self.audio_players:
            return  # Already playing

        url = self.camera_urls.get(camera_name)
        if url:
            player = AudioPlayer(url, camera_name)
            player.start()
            self.audio_players[camera_name] = player

    def _stop_audio(self, camera_name: str):
        """Stop audio for a specific camera."""
        if camera_name in self.audio_players:
            self.audio_players[camera_name].stop()
            del self.audio_players[camera_name]
            _LOGGER.info(f"[{camera_name}] Audio stopped")

    def _toggle_audio(self, camera_name: str):
        """Toggle audio for a specific camera."""
        if camera_name in self.audio_players:
            self._stop_audio(camera_name)
        else:
            self._start_audio(camera_name)

    def _stop_all_audio(self):
        """Stop all audio playback."""
        for name in list(self.audio_players.keys()):
            self._stop_audio(name)

    def _get_audio_status(self) -> str:
        """Get audio status string showing which cameras have audio."""
        if not self.audio_players:
            return "🔇"
        active = [name for name in self.camera_list if name in self.audio_players]
        if len(active) == len(self.camera_list):
            return "🔊 All"
        indices = [str(self.camera_list.index(name) + 1) for name in active]
        return f"🔊 {','.join(indices)}"

    def _init_two_way_audio(self):
        """Initialize WebRTC two-way audio."""
        if not HAS_TWO_WAY_WEBRTC:
            _LOGGER.warning("WebRTC two-way audio not available (aiortc not installed)")
            return False

        if not self.webrtc_credentials:
            _LOGGER.warning("No WebRTC credentials for two-way audio")
            return False

        oauth_token = self.webrtc_credentials.get('oauth_token')
        camera_uuid = self.webrtc_credentials.get('camera_uuid')

        if not oauth_token or not camera_uuid:
            _LOGGER.warning("Invalid WebRTC credentials")
            return False

        try:
            self.webrtc_two_way = WebRTCTwoWayAudio(
                oauth_token=oauth_token,
                camera_uuid=camera_uuid,
                audio_device=self.audio_device,
            )
            _LOGGER.info("WebRTC two-way audio initialized")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to initialize WebRTC two-way audio: {e}")
            return False

    def _start_ptt(self):
        """Start push-to-talk via WebRTC."""
        if self.ptt_active:
            return

        if self.ptt_failed:
            _LOGGER.warning("🎤 PTT: Previously failed, restart app to retry")
            return

        # Debounce - ignore rapid key presses (500ms)
        now = time.time()
        if now - self.ptt_last_toggle < 0.5:
            return
        self.ptt_last_toggle = now

        if not HAS_TWO_WAY_WEBRTC or not self.webrtc_credentials:
            _LOGGER.error("🎤 PTT: WebRTC not available")
            return

        _LOGGER.info("🎤 PTT: Connecting via WebRTC (may take a few seconds)...")

        try:
            if not self.webrtc_two_way:
                self.webrtc_two_way = WebRTCTwoWayAudio(
                    oauth_token=self.webrtc_credentials.get('oauth_token'),
                    camera_uuid=self.webrtc_credentials.get('camera_uuid'),
                    audio_device=self.audio_device,
                )

            if self.webrtc_two_way.start():
                self.ptt_active = True
                self.ptt_connected = True
                _LOGGER.info("🎤 PTT: ACTIVE - Speak now!")
            else:
                _LOGGER.error("🎤 PTT: Failed to connect")
                if self.webrtc_two_way._error:
                    _LOGGER.error(f"🎤 PTT: {self.webrtc_two_way._error}")
                self.ptt_failed = True
                self.webrtc_two_way = None
        except Exception as e:
            _LOGGER.error(f"🎤 PTT: Error - {e}")
            self.ptt_failed = True
            self.webrtc_two_way = None

    def _stop_ptt(self):
        """Stop push-to-talk."""
        if not self.ptt_active:
            return

        # Debounce
        now = time.time()
        if now - self.ptt_last_toggle < 0.5:
            return
        self.ptt_last_toggle = now

        _LOGGER.info("🎤 PTT: Stopping...")

        if self.webrtc_two_way:
            try:
                self.webrtc_two_way.stop()
            except Exception as e:
                _LOGGER.warning(f"PTT stop error: {e}")
            self.webrtc_two_way = None

        self.ptt_active = False
        self.ptt_connected = False
        _LOGGER.info("🎤 PTT: Stopped")

    def create_grid(self, width: int, height: int) -> np.ndarray:
        """Create a grid layout of all camera feeds."""
        num_cameras = len(self.cameras)

        if num_cameras == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate grid dimensions
        if num_cameras == 1:
            cols, rows = 1, 1
        elif num_cameras == 2:
            cols, rows = 2, 1
        elif num_cameras <= 4:
            cols, rows = 2, 2
        elif num_cameras <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 3, 3

        cell_width = width // cols
        cell_height = height // rows

        # Create output image
        grid = np.zeros((height, width, 3), dtype=np.uint8)

        camera_list = list(self.cameras.values())

        for idx, camera in enumerate(camera_list):
            row = idx // cols
            col = idx % cols

            x = col * cell_width
            y = row * cell_height

            # Get the frame or create placeholder
            if camera.frame is not None:
                frame = camera.frame.copy()
                # Resize to cell size
                frame = cv2.resize(frame, (cell_width, cell_height))
            else:
                # Create placeholder
                frame = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                cv2.putText(
                    frame, "Connecting...",
                    (cell_width // 4, cell_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2
                )

            # Add overlay info
            self._add_overlay(frame, camera, idx + 1)

            # Place in grid
            grid[y:y+cell_height, x:x+cell_width] = frame

            # Draw border
            cv2.rectangle(grid, (x, y), (x+cell_width-1, y+cell_height-1), (50, 50, 50), 1)

        return grid

    def create_single_view(self, camera_name: str, width: int, height: int) -> np.ndarray:
        """Create a single camera view."""
        camera = self.cameras.get(camera_name)

        if not camera:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera not found: {camera_name}",
                       (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        if camera.frame is not None:
            frame = camera.frame.copy()

            # In native resolution mode, don't resize (or resize to fit while maintaining aspect)
            if self.native_resolution and self.fullscreen:
                # Keep native resolution, just pad/center if needed
                src_h, src_w = frame.shape[:2]
                if src_w != width or src_h != height:
                    # Calculate scaling to fit while maintaining aspect ratio
                    scale = min(width / src_w, height / src_h)
                    new_w = int(src_w * scale)
                    new_h = int(src_h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    # Center on black background
                    output = np.zeros((height, width, 3), dtype=np.uint8)
                    x_offset = (width - new_w) // 2
                    y_offset = (height - new_h) // 2
                    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame
                    frame = output
            else:
                frame = cv2.resize(frame, (width, height))
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, "Connecting...", (width // 3, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        self._add_overlay(frame, camera, 0, large=True)
        return frame

    def _add_overlay(self, frame: np.ndarray, camera: CameraStream, idx: int, large: bool = False):
        """Add status overlay to frame."""
        h, w = frame.shape[:2]
        font_scale = 0.7 if large else 0.5
        thickness = 2 if large else 1
        small_font = font_scale * 0.8

        # Status color
        if camera.connected:
            color = (0, 255, 0)  # Green
        elif camera.error:
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 255)  # Yellow

        # Camera name with index (top left)
        name_text = f"[{idx}] {camera.name}" if idx > 0 else camera.name
        cv2.putText(frame, name_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Resolution (top left, below name)
        if camera.width > 0 and camera.height > 0:
            res_text = f"{camera.width}x{camera.height}"
            cv2.putText(frame, res_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, small_font, (200, 200, 200), thickness)

        # Status indicator (circle in top right)
        cv2.circle(frame, (w - 20, 20), 8, color, -1)

        # Bottom status bar
        if camera.connected:
            # FPS (bottom left)
            fps_text = f"{camera.fps:.1f} FPS"
            cv2.putText(frame, fps_text, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, small_font, (0, 255, 0), thickness)

            # Latency (bottom right) - using frame decode latency
            latency = camera.latency_ms
            latency_color = (0, 255, 0) if latency < 50 else (0, 255, 255) if latency < 100 else (0, 165, 255) if latency < 200 else (0, 0, 255)
            latency_text = f"{latency:.0f}ms"
            cv2.putText(frame, latency_text, (w - 70, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, small_font, latency_color, thickness)
        elif camera.error:
            # Error message (bottom left)
            cv2.putText(frame, camera.error, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, small_font, (0, 0, 255), thickness)
        else:
            cv2.putText(frame, "Connecting...", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, small_font, (0, 255, 255), thickness)

    def save_snapshot(self):
        """Save snapshot of current view."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, camera in self.cameras.items():
            if camera.frame is not None:
                filename = f"snapshot_{name}_{timestamp}.jpg"
                cv2.imwrite(filename, camera.frame)
                _LOGGER.info(f"Saved: {filename}")

    def run(self, single_camera: Optional[str] = None):
        """Main display loop."""
        self.start()

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        if single_camera:
            self.focused_camera = single_camera
            # Start audio for the focused camera
            if self.audio_enabled:
                self._start_audio(single_camera)

        print("\nControls:")
        print("  q - Quit")
        print("  s - Save snapshots")
        print("  f - Toggle fullscreen")
        print("  a - Toggle all audio on/off")
        print("  n - Toggle native resolution (in fullscreen)")
        print("  1-3 - Focus on camera")
        print("  !@# - Toggle audio for camera 1/2/3 (Shift+1/2/3)")
        print("  0 - Grid view")
        if HAS_TWO_WAY_WEBRTC and self.webrtc_credentials:
            print(f"  t/SPACE - Push-to-talk (WebRTC)")
        print()

        try:
            while self.running:
                # Get window size
                try:
                    rect = cv2.getWindowImageRect(self.window_name)
                    width, height = rect[2], rect[3]
                    if width <= 0 or height <= 0:
                        width, height = 1280, 720
                except:
                    width, height = 1280, 720

                # Create display frame
                if self.focused_camera:
                    frame = self.create_single_view(self.focused_camera, width, height)
                else:
                    frame = self.create_grid(width, height)

                # Add global info bar
                audio_status = self._get_audio_status()
                res_mode = "Native" if self.native_resolution else "Fit"
                ptt_status = "🎤 TALKING" if self.ptt_active else ""
                info_text = f"Vivint Live | {audio_status} | {res_mode}"
                if ptt_status:
                    info_text += f" | {ptt_status}"
                info_text += " | Press 'q' to quit"
                # PTT indicator color (green when active)
                info_color = (0, 255, 0) if self.ptt_active else (150, 150, 150)
                cv2.putText(frame, info_text, (width - 450, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)

                cv2.imshow(self.window_name, frame)

                # Handle key presses (1ms wait for low latency)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_snapshot()
                elif key == ord('f'):
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                elif key == ord('a'):
                    # Toggle all audio
                    if self.audio_players:
                        self._stop_all_audio()
                        _LOGGER.info("All audio stopped")
                    else:
                        # Start audio for focused camera or all cameras in grid
                        if self.focused_camera:
                            self._start_audio(self.focused_camera)
                        else:
                            for name in self.camera_list:
                                self._start_audio(name)
                            _LOGGER.info("All audio started")
                elif key == ord('n'):
                    # Toggle native resolution
                    self.native_resolution = not self.native_resolution
                    _LOGGER.info(f"Native resolution: {'ON' if self.native_resolution else 'OFF'}")
                # Shift+1/2/3 = !/@ /# to toggle individual camera audio
                elif key == ord('!') and len(self.camera_list) >= 1:
                    self._toggle_audio(self.camera_list[0])
                elif key == ord('@') and len(self.camera_list) >= 2:
                    self._toggle_audio(self.camera_list[1])
                elif key == ord('#') and len(self.camera_list) >= 3:
                    self._toggle_audio(self.camera_list[2])
                elif key == ord('0'):
                    self.focused_camera = None
                    # Don't stop audio when returning to grid - keep it playing
                elif ord('1') <= key <= ord('9'):
                    idx = key - ord('1')
                    if idx < len(self.camera_list):
                        self.focused_camera = self.camera_list[idx]
                        _LOGGER.info(f"Focused on: {self.focused_camera}")
                # Push-to-talk: 't' or SPACE (toggle mode)
                elif key == ord('t') or key == 32:  # 32 = SPACE
                    if HAS_TWO_WAY_WEBRTC and self.webrtc_credentials:
                        if self.ptt_active:
                            self._stop_ptt()
                        else:
                            self._start_ptt()
                    else:
                        _LOGGER.warning("Two-way audio not available (need WebRTC)")

                # Check if window was closed
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            cv2.destroyAllWindows()


async def get_camera_urls(prefer_hd: bool = False) -> tuple[dict[str, str], dict]:
    """
    Connect to Vivint and get RTSP URLs for all cameras.

    Returns:
        (urls_dict, webrtc_credentials) where:
        - webrtc_credentials is {'oauth_token': str, 'camera_uuid': str} for WebRTC two-way audio
    """
    # Add project to path
    sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])

    from vivint_client import VivintClient
    import config

    # Temporarily override HD preference
    original_hd = config.RTSP_PREFER_HD
    config.RTSP_PREFER_HD = prefer_hd

    print("Connecting to Vivint...")
    client = VivintClient()

    if not await client.connect():
        print("Failed to connect to Vivint")
        return {}, None

    urls = {}
    doorbell_uuid = None  # For WebRTC two-way audio

    for cam in client.cameras:
        url = client.get_rtsp_url(cam.id)
        if url:
            urls[cam.name] = url
            print(f"  Found: {cam.name} ({'HD' if prefer_hd else 'SD'})")
            # Track doorbell UUID for WebRTC two-way audio
            if 'doorbell' in cam.name.lower():
                doorbell_uuid = cam.data.get('uuid') if hasattr(cam, 'data') else None
                if doorbell_uuid:
                    print(f"    Doorbell UUID: {doorbell_uuid}")

    # Get WebRTC credentials for two-way audio
    webrtc_credentials = None
    if doorbell_uuid:
        try:
            # Get OAuth id_token for WebRTC
            api = client.account._api
            tokens = api.tokens
            oauth_token = tokens.get("id_token") or tokens.get("access_token")
            if oauth_token:
                webrtc_credentials = {
                    'oauth_token': oauth_token,
                    'camera_uuid': doorbell_uuid,
                }
                print(f"  Two-way audio: WebRTC ready")
        except Exception as e:
            _LOGGER.warning(f"Failed to get WebRTC credentials: {e}")

    await client.disconnect()

    # Restore setting
    config.RTSP_PREFER_HD = original_hd

    return urls, webrtc_credentials


async def main():
    # Parse args
    use_hd = "--hd" in sys.argv
    use_audio = "--no-audio" not in sys.argv
    camera_filter = None
    audio_device = None

    # Parse --audio-device "Device Name"
    for i, arg in enumerate(sys.argv):
        if arg == "--audio-device" and i + 1 < len(sys.argv):
            audio_device = sys.argv[i + 1]
        elif not arg.startswith("--") and i > 0 and sys.argv[i-1] != "--audio-device":
            camera_filter = arg

    # Get camera URLs and WebRTC credentials
    urls, webrtc_credentials = await get_camera_urls(prefer_hd=use_hd)

    if not urls:
        print("No cameras found")
        return

    print(f"\nStarting live view with {len(urls)} cameras...")
    print(f"Quality: {'HD' if use_hd else 'SD (lower latency)'}")
    print(f"Audio: {'Enabled' if use_audio else 'Disabled'}")

    # Two-way audio status (WebRTC only)
    if HAS_TWO_WAY_WEBRTC and webrtc_credentials:
        print(f"Two-way audio: WebRTC ready")
        if audio_device:
            print(f"  Audio device: {audio_device}")
    else:
        if not HAS_TWO_WAY_WEBRTC:
            print(f"Two-way audio: Not available (aiortc not installed)")
        else:
            print(f"Two-way audio: Not available (no doorbell found)")

    # Create and run viewer
    viewer = MultiCameraViewer(
        urls,
        hd=use_hd,
        audio=use_audio,
        webrtc_credentials=webrtc_credentials,
        audio_device=audio_device,
    )

    # If single camera specified, validate it
    if camera_filter:
        matched = None
        for name in urls.keys():
            if name.lower() == camera_filter.lower():
                matched = name
                break
        if not matched:
            print(f"Camera '{camera_filter}' not found")
            print(f"Available: {', '.join(urls.keys())}")
            return
        viewer.run(single_camera=matched)
    else:
        viewer.run()


if __name__ == "__main__":
    asyncio.run(main())
