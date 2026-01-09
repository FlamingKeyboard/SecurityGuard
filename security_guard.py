"""
Vivint Security Guard Service

Main entry point that coordinates:
- Vivint connection and event subscription
- Frame capture on motion/doorbell events
- Gemini vision analysis
- Notifications based on risk assessment
- GCP integration for logging and image archival

IMPORTANT SAFETY NOTE (2026-01):
    This system does NOT and MUST NOT automatically trigger the Vivint alarm
    panel, even when critical threats are detected. The Vivint alarm system
    may dispatch 911 automatically depending on monitoring plan configuration.

    We are in a testing/validation phase for the AI detection pipeline.
    False-positive 911 calls are completely unacceptable. All physical alarm
    triggering must go through explicit human confirmation via the Vivint app.

    The vivintpy library includes a trigger_alarm() method, but it is
    intentionally not wired into this automation. Do not add automatic alarm
    triggering without extensive testing AND a human-in-the-loop confirmation
    mechanism.
"""

import asyncio
import logging
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

import config

# Cloud Logging for heartbeat monitoring
_cloud_logger = None
try:
    from google.cloud import logging as cloud_logging
    if config.GCP_PROJECT_ID:
        _logging_client = cloud_logging.Client(project=config.GCP_PROJECT_ID)
        _cloud_logger = _logging_client.logger("security-guard-heartbeat")
except Exception:
    pass  # Cloud Logging optional
from gcp_logging import (
    get_or_create_event_id,
    get_or_create_conversation_id,
    run_sync as run_gcp_sync,
    test_bigquery_connection,
)
from gcp_storage import upload_media, archive_old_media, test_gcs_connection
from health_check import start_health_server, stop_health_server


def check_hub_connectivity() -> bool:
    """
    Check if the Vivint hub is reachable on the network.

    Returns True if reachable, False otherwise.
    """
    hub_ip = config.VIVINT_HUB_IP
    hub_port = config.VIVINT_HUB_RTSP_PORT

    if not hub_ip:
        print("ERROR: VIVINT_HUB_IP is not configured")
        print("  Set it in config.py or via environment variable")
        print("  Find your hub IP in your router's DHCP leases")
        return False

    print(f"Checking hub connectivity: {hub_ip}:{hub_port}")

    # Try to connect to the RTSP port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)

    try:
        result = sock.connect_ex((hub_ip, hub_port))
        if result == 0:
            print(f"  [OK] Hub is reachable at {hub_ip}:{hub_port}")
            return True
        else:
            print(f"  [FAIL] Cannot connect to {hub_ip}:{hub_port} (error code: {result})")
            _print_connectivity_help(hub_ip)
            return False
    except socket.timeout:
        print(f"  [FAIL] Connection timed out to {hub_ip}:{hub_port}")
        _print_connectivity_help(hub_ip)
        return False
    except socket.gaierror as e:
        print(f"  [FAIL] DNS/address error for {hub_ip}: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Connection error: {e}")
        _print_connectivity_help(hub_ip)
        return False
    finally:
        sock.close()


def _print_connectivity_help(hub_ip: str) -> None:
    """Print helpful troubleshooting tips for connectivity issues."""
    print()
    print("Troubleshooting:")
    print(f"  1. Verify the hub IP is correct: {hub_ip}")
    print("  2. Make sure the hub is powered on and connected")
    print("  3. If running in cloud/container:")
    print("     - Check Tailscale is connected: tailscale status")
    print("     - Verify subnet routing is enabled on your router")
    print("     - Try pinging the hub: ping " + hub_ip)
    print("  4. If running locally:")
    print("     - Ensure you're on the same network as the hub")
    print("     - Check for firewall blocking port 8554")


from vivint_client import VivintClient, load_credentials, VivintAuthInterventionRequired
from frame_capture import (
    capture_frames,
    capture_with_fallback,
    capture_multiple_cameras,
    cleanup_old_frames,
    CaptureResult,
    MultiCameraCapture,
)
from gemini_analyzer import (
    analyze_multiple_frames,
    analyze_video,
    analyze_multiple_videos,
    SecurityAnalysis,
)

# Doorbell AI agent imports
try:
    from doorbell_agent import DoorbellAgent
    from vivint_webrtc import (
        VivintWebRTCClient,
        VivintWebRTCConfig,
        prefetch_webrtc_credentials,
        PrefetchedCredentials,
    )
    DOORBELL_AI_AVAILABLE = True
except ImportError as e:
    _LOGGER = logging.getLogger(__name__)
    _LOGGER.warning(f"Doorbell AI agent not available: {e}")
    DOORBELL_AI_AVAILABLE = False
    DoorbellAgent = None
    VivintWebRTCClient = None
    VivintWebRTCConfig = None
    prefetch_webrtc_credentials = None
    PrefetchedCredentials = None

# Pushover credentials (loaded at startup)
_pushover_token: str | None = None
_pushover_user: str | None = None


def load_stored_credentials():
    """Load API keys and notification credentials from stored credentials."""
    global _pushover_token, _pushover_user

    creds = load_credentials() or {}

    # Gemini API key
    if creds.get("gemini_api_key") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = creds["gemini_api_key"]

    # Eleven Labs API key (for TTS doorbell responses)
    if creds.get("eleven_labs_api_key") and not os.environ.get("ELEVEN_LABS_API_KEY"):
        os.environ["ELEVEN_LABS_API_KEY"] = creds["eleven_labs_api_key"]

    # GCP credentials
    if creds.get("gcp_project_id") and not os.environ.get("GCP_PROJECT_ID"):
        os.environ["GCP_PROJECT_ID"] = creds["gcp_project_id"]
    if creds.get("gcp_service_account_file") and not os.environ.get("GCP_SERVICE_ACCOUNT_FILE"):
        os.environ["GCP_SERVICE_ACCOUNT_FILE"] = creds["gcp_service_account_file"]

    # Reload config to pick up new env vars
    import importlib
    importlib.reload(config)

    # Pushover credentials (must use global declaration above)
    _pushover_token = creds.get("pushover_token") or None
    _pushover_user = creds.get("pushover_user") or None


def check_gcp_connectivity() -> dict[str, tuple[bool, str]]:
    """
    Test GCP connectivity for both GCS and BigQuery.

    Returns a dict with connection status for each service:
        {
            'gcs': (success: bool, message: str),
            'bigquery': (success: bool, message: str)
        }
    """
    results = {}

    # Check if GCP is configured at all
    if not config.GCP_PROJECT_ID and not config.GCP_SERVICE_ACCOUNT_FILE:
        return {
            'gcs': (False, "GCP not configured"),
            'bigquery': (False, "GCP not configured")
        }

    # Test GCS connection
    print("Checking GCS connectivity...")
    results['gcs'] = test_gcs_connection()
    if results['gcs'][0]:
        print(f"  [OK] {results['gcs'][1]}")
    else:
        print(f"  [WARN] {results['gcs'][1]}")

    # Test BigQuery connection
    print("Checking BigQuery connectivity...")
    results['bigquery'] = test_bigquery_connection()
    if results['bigquery'][0]:
        print(f"  [OK] {results['bigquery'][1]}")
    else:
        print(f"  [WARN] {results['bigquery'][1]}")

    return results


_LOGGER = logging.getLogger(__name__)

# Track recent events to implement cooldown
_last_event_time: dict[int, float] = {}  # camera_id -> timestamp


def should_process_event(camera_id: int) -> bool:
    """Check if we should process an event (respecting cooldown)."""
    now = time.time()
    last = _last_event_time.get(camera_id, 0)

    if now - last < config.MOTION_COOLDOWN_SECONDS:
        return False

    _last_event_time[camera_id] = now
    return True


def format_notification(camera_name: str, analysis: SecurityAnalysis) -> str:
    """Format a notification message from analysis results."""
    lines = [
        f"[{analysis.risk_tier.upper()}] {camera_name}",
        f"Summary: {analysis.summary}",
    ]

    if analysis.person_detected:
        lines.append(f"People detected: {analysis.person_count}")

    if analysis.activity_observed:
        lines.append(f"Activity: {', '.join(analysis.activity_observed)}")

    if analysis.potential_concerns:
        lines.append(f"Concerns: {', '.join(analysis.potential_concerns)}")

    if analysis.context_clues:
        lines.append(f"Context: {', '.join(analysis.context_clues)}")

    if analysis.weapon_visible.get("detected"):
        lines.append(f"WEAPON ALERT: {analysis.weapon_visible.get('description', 'Detected')}")

    lines.append(f"Action: {analysis.recommended_action}")

    return "\n".join(lines)


async def send_pushover(
    title: str,
    message: str,
    image_path: Path | None = None,
    priority: int = 0,
) -> bool:
    """
    Send a push notification via Pushover.

    Args:
        title: Notification title
        message: Notification body
        image_path: Optional path to image to attach
        priority: -2 (silent) to 2 (emergency). 1 = high priority, bypasses quiet hours

    Returns:
        True if sent successfully
    """
    if not _pushover_token or not _pushover_user:
        _LOGGER.debug("Pushover not configured, skipping push notification")
        return False

    url = "https://api.pushover.net/1/messages.json"

    try:
        data = aiohttp.FormData()
        data.add_field("token", _pushover_token)
        data.add_field("user", _pushover_user)
        data.add_field("title", title)
        data.add_field("message", message)
        data.add_field("priority", str(priority))

        # Emergency priority (2) requires retry and expire parameters
        if priority == 2:
            data.add_field("retry", "30")   # Retry every 30 seconds
            data.add_field("expire", "300") # Stop after 5 minutes
            data.add_field("sound", "siren")  # Use siren sound for emergencies

        # Attach image if provided
        if image_path and image_path.exists():
            image_bytes = image_path.read_bytes()
            data.add_field(
                "attachment",
                image_bytes,
                filename=image_path.name,
                content_type="image/jpeg",
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                if resp.status == 200:
                    _LOGGER.info("Pushover notification sent successfully")
                    return True
                else:
                    body = await resp.text()
                    _LOGGER.error("Pushover failed: %s - %s", resp.status, body)
                    return False

    except Exception as e:
        _LOGGER.error("Failed to send Pushover notification: %s", e)
        return False


async def send_notification(
    message: str,
    title: str = "Security Alert",
    priority: int = 0,
    image_path: Path | None = None,
) -> None:
    """
    Send a notification via Pushover and console.

    Args:
        message: Notification body
        title: Notification title
        priority: Pushover priority (-2 to 2). 1=high, 2=emergency
        image_path: Optional image to attach
    """
    # Determine urgency for console display
    if priority >= 2:
        prefix = "[EMERGENCY] "
    elif priority >= 1:
        prefix = "[URGENT] "
    else:
        prefix = ""

    # Always log to console
    print(f"\n{'='*60}")
    print(f"{prefix}SECURITY NOTIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"{title}")
    print("-" * 40)
    print(message)
    if image_path:
        print(f"Image: {image_path}")
    print("=" * 60 + "\n")

    # Send via Pushover
    await send_pushover(title, message, image_path=image_path, priority=priority)


class SecurityGuard:
    """Main security guard service."""

    def __init__(self):
        self.client = VivintClient()
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()

        # Doorbell AI agent state
        self._doorbell_agent: DoorbellAgent | None = None
        self._doorbell_webrtc: VivintWebRTCClient | None = None
        self._doorbell_conversation_active = False
        self._doorbell_conversation_lock = asyncio.Lock()
        self._webrtc_credentials: PrefetchedCredentials | None = None

    def _get_cameras_to_capture(self, trigger_camera_name: str) -> dict[str, str]:
        """
        Get dict of camera names to RTSP URLs for multi-camera capture.

        Returns URLs for the trigger camera plus any adjacent cameras
        defined in config.CAMERA_ADJACENCY.

        Args:
            trigger_camera_name: Name of the camera that triggered the event

        Returns:
            Dict mapping camera names to RTSP URLs
        """
        camera_urls = {}

        # Find the trigger camera
        trigger_camera = None
        for cam in self.client.cameras:
            if cam.name == trigger_camera_name:
                trigger_camera = cam
                break

        if not trigger_camera:
            _LOGGER.warning("Trigger camera %s not found", trigger_camera_name)
            return camera_urls

        # Add trigger camera
        trigger_url = self.client.get_rtsp_url(trigger_camera.id)
        if trigger_url:
            camera_urls[trigger_camera_name] = trigger_url

        # Add adjacent cameras if multi-camera is enabled
        if config.MULTI_CAMERA_ENABLED:
            adjacent_names = config.CAMERA_ADJACENCY.get(trigger_camera_name, [])

            for adj_name in adjacent_names:
                # Respect max camera limit
                if len(camera_urls) >= config.MAX_CONCURRENT_CAMERAS:
                    _LOGGER.debug("Max concurrent cameras reached (%d)", config.MAX_CONCURRENT_CAMERAS)
                    break

                # Find the adjacent camera
                for cam in self.client.cameras:
                    if cam.name == adj_name:
                        url = self.client.get_rtsp_url(cam.id)
                        if url:
                            camera_urls[adj_name] = url
                            _LOGGER.debug("Added adjacent camera: %s", adj_name)
                        break

        return camera_urls

    async def start(self) -> bool:
        """Start the security guard service."""
        _LOGGER.info("Starting Security Guard service...")

        # Connect to Vivint
        if not await self.client.connect():
            _LOGGER.error("Failed to connect to Vivint")
            return False

        # Register event handlers
        self.client.on_motion(self._on_motion)
        self.client.on_doorbell(self._on_doorbell)

        self._running = True

        # Start background tasks
        asyncio.create_task(self._event_processor())
        asyncio.create_task(self._keepalive_loop())
        asyncio.create_task(self._cleanup_loop())
        asyncio.create_task(self._sync_loop())
        asyncio.create_task(self._heartbeat_loop())

        # Pre-fetch WebRTC credentials for doorbell AI (reduces latency)
        if DOORBELL_AI_AVAILABLE and config.DOORBELL_AI_ENABLED:
            await self._prefetch_webrtc_credentials()

        _LOGGER.info("Security Guard service started")
        _LOGGER.info("Monitoring %d cameras", len(self.client.cameras))
        for cam in self.client.cameras:
            _LOGGER.info("  - %s (ID: %d)", cam.name, cam.id)

        return True

    async def _prefetch_webrtc_credentials(self):
        """Pre-fetch Firebase credentials for faster doorbell AI connection."""
        try:
            # Find the doorbell camera
            doorbell_camera = None
            for cam in self.client.cameras:
                if 'doorbell' in cam.name.lower():
                    doorbell_camera = cam
                    break

            if not doorbell_camera:
                _LOGGER.warning("No doorbell camera found for WebRTC prefetch")
                return

            # Get OAuth token from Vivint
            api = self.client.account._api
            tokens = api.tokens
            oauth_token = tokens.get("id_token") or tokens.get("access_token")

            if not oauth_token:
                _LOGGER.warning("No OAuth token available for WebRTC prefetch")
                return

            # Get camera UUID
            camera_uuid = doorbell_camera.data.get('uuid') if hasattr(doorbell_camera, 'data') else None
            if not camera_uuid:
                camera_uuid = doorbell_camera.serial_number or str(doorbell_camera.id)

            # Prefetch credentials
            _LOGGER.info("Pre-fetching WebRTC credentials for doorbell AI...")
            self._webrtc_credentials = await prefetch_webrtc_credentials(
                oauth_token=oauth_token,
                camera_uuid=camera_uuid,
            )
            _LOGGER.info("WebRTC credentials pre-fetched successfully")

        except Exception as e:
            _LOGGER.warning(f"Failed to prefetch WebRTC credentials: {e}")

    async def stop(self) -> None:
        """Stop the security guard service."""
        _LOGGER.info("Stopping Security Guard service...")
        self._running = False

        # Stop any active doorbell AI conversation
        await self._stop_doorbell_conversation()

        await self.client.disconnect()
        _LOGGER.info("Security Guard service stopped")

    async def _start_doorbell_conversation(self, camera) -> bool:
        """
        Start an AI conversation with a doorbell visitor.

        Sets up:
        - WebRTC two-way audio with the doorbell
        - RTSP video capture for visual context
        - Gemini Live API session for conversation

        Args:
            camera: The doorbell camera device

        Returns:
            True if conversation started successfully
        """
        if not DOORBELL_AI_AVAILABLE:
            _LOGGER.warning("Doorbell AI not available - missing dependencies")
            return False

        # Use lock to prevent concurrent conversation starts
        async with self._doorbell_conversation_lock:
            if self._doorbell_conversation_active:
                _LOGGER.info("Doorbell conversation already active, skipping")
                return False

            _LOGGER.info("Starting doorbell AI conversation...")

            try:
                # Get OAuth token
                api = self.client.account._api
                tokens = api.tokens
                oauth_token = tokens.get("id_token") or tokens.get("access_token")

                if not oauth_token:
                    _LOGGER.error("No OAuth token available for doorbell AI")
                    return False

                # Get camera UUID
                camera_uuid = camera.data.get('uuid') if hasattr(camera, 'data') else None
                if not camera_uuid:
                    camera_uuid = camera.serial_number or str(camera.id)

                # Get RTSP URL for video
                rtsp_url = self.client.get_rtsp_url(camera.id)

                # Create WebRTC config with pre-fetched credentials if available
                config_kwargs = {
                    'oauth_token': oauth_token,
                    'camera_uuid': camera_uuid,
                }

                if self._webrtc_credentials:
                    config_kwargs.update({
                        'prefetched_firebase_token': self._webrtc_credentials.firebase_token,
                        'prefetched_firebase_id_token': self._webrtc_credentials.firebase_id_token,
                        'prefetched_firebase_uid': self._webrtc_credentials.firebase_uid,
                        'firebase_db_url': self._webrtc_credentials.firebase_db_url,
                        'system_id': self._webrtc_credentials.system_id,
                    })

                webrtc_config = VivintWebRTCConfig(**config_kwargs)

                # Create WebRTC client and enable AI mode
                self._doorbell_webrtc = VivintWebRTCClient(webrtc_config)
                self._doorbell_webrtc.enable_ai_conversation_mode()

                # Connect WebRTC
                _LOGGER.info("Connecting WebRTC for doorbell AI...")
                if not await self._doorbell_webrtc.connect():
                    _LOGGER.error("Failed to connect WebRTC for doorbell AI")
                    self._doorbell_webrtc = None
                    return False

                # Create and start doorbell agent
                self._doorbell_agent = DoorbellAgent()

                _LOGGER.info("Starting Gemini Live session...")
                if not await self._doorbell_agent.start_conversation():
                    _LOGGER.error("Failed to start Gemini session for doorbell AI")
                    await self._doorbell_webrtc.disconnect()
                    self._doorbell_webrtc = None
                    self._doorbell_agent = None
                    return False

                # Connect agent to WebRTC (audio piping)
                await self._doorbell_agent.connect_to_webrtc(self._doorbell_webrtc)

                # Start video capture if RTSP URL available
                if rtsp_url:
                    _LOGGER.info("Starting video capture for doorbell AI...")
                    await self._doorbell_agent.start_video_capture(
                        rtsp_url=rtsp_url,
                        camera_name=camera.name,
                        interval_seconds=config.DOORBELL_AI_VIDEO_INTERVAL,
                    )

                # Start two-way talk
                _LOGGER.info("Starting two-way talk...")
                if not await self._doorbell_webrtc.start_two_way_talk():
                    _LOGGER.error("Failed to start two-way talk")
                    await self._doorbell_agent.end_conversation()
                    await self._doorbell_webrtc.disconnect()
                    self._doorbell_webrtc = None
                    self._doorbell_agent = None
                    return False

                self._doorbell_conversation_active = True
                _LOGGER.info("Doorbell AI conversation active!")

                # Send initial context about the doorbell press
                await self._doorbell_agent.inject_context(
                    "Someone just pressed the doorbell button. Greet them warmly and ask how you can help."
                )

                # Schedule conversation timeout
                asyncio.create_task(self._doorbell_conversation_timeout())

                return True

            except Exception as e:
                _LOGGER.error(f"Error starting doorbell AI conversation: {e}")
                import traceback
                traceback.print_exc()

                # Clean up on error
                if self._doorbell_agent:
                    await self._doorbell_agent.end_conversation()
                    self._doorbell_agent = None
                if self._doorbell_webrtc:
                    await self._doorbell_webrtc.disconnect()
                    self._doorbell_webrtc = None

                return False

    async def _doorbell_conversation_timeout(self):
        """End the doorbell conversation after the configured timeout."""
        await asyncio.sleep(config.DOORBELL_AI_CONVERSATION_DURATION)
        if self._doorbell_conversation_active:
            _LOGGER.info("Doorbell conversation timeout reached")
            await self._stop_doorbell_conversation()

    async def _stop_doorbell_conversation(self):
        """Stop the active doorbell AI conversation."""
        if not self._doorbell_conversation_active:
            return

        _LOGGER.info("Stopping doorbell AI conversation...")

        async with self._doorbell_conversation_lock:
            self._doorbell_conversation_active = False

            if self._doorbell_agent:
                self._doorbell_agent.disconnect_from_webrtc()
                await self._doorbell_agent.end_conversation()
                self._doorbell_agent = None

            if self._doorbell_webrtc:
                await self._doorbell_webrtc.stop_two_way_talk()
                await self._doorbell_webrtc.disconnect()
                self._doorbell_webrtc = None

        _LOGGER.info("Doorbell AI conversation stopped")

    def _on_motion(self, camera, message: dict) -> None:
        """Handle motion event."""
        if camera and should_process_event(camera.id):
            self._event_queue.put_nowait(("motion", camera, message))

    def _on_doorbell(self, camera, message: dict) -> None:
        """Handle doorbell event (always process, no cooldown)."""
        if camera:
            _last_event_time[camera.id] = time.time()  # Update for motion cooldown
            self._event_queue.put_nowait(("doorbell", camera, message))

    async def _event_processor(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event_type, camera, message = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                await self._process_event(event_type, camera, message)
            except Exception as e:
                _LOGGER.error("Error processing event: %s", e)

    async def _process_event(self, event_type: str, camera, message: dict) -> None:
        """Process a single event."""
        _LOGGER.info("Processing %s event from %s", event_type, camera.name)

        # Generate event and conversation IDs for logging/grouping
        event_id = get_or_create_event_id()
        conversation_id = get_or_create_conversation_id()
        event_timestamp = datetime.now(timezone.utc)

        _LOGGER.debug("Event ID: %s, Conversation ID: %s", event_id, conversation_id)

        # Start doorbell AI conversation if enabled and this is a doorbell event
        is_doorbell_event = event_type == "doorbell" or 'doorbell' in camera.name.lower()
        if is_doorbell_event and config.DOORBELL_AI_ENABLED:
            # Start AI conversation in background (don't block normal processing)
            asyncio.create_task(self._start_doorbell_conversation(camera))
            _LOGGER.info("Doorbell AI conversation triggered")

        # Determine capture mode
        use_multi_camera = (
            config.VIDEO_CAPTURE_ENABLED and
            config.MULTI_CAMERA_ENABLED
        )

        if use_multi_camera:
            # Multi-camera video capture
            camera_urls = self._get_cameras_to_capture(camera.name)

            if not camera_urls:
                _LOGGER.warning("No camera URLs available for %s", camera.name)
                return

            _LOGGER.info("Multi-camera capture: %s", list(camera_urls.keys()))

            # Capture from all cameras simultaneously
            multi_capture = await capture_multiple_cameras(
                camera_urls,
                primary_camera=camera.name,
            )

            if not multi_capture.success:
                _LOGGER.warning("Multi-camera capture failed: %s", multi_capture.error)
                # Fall back to single camera
                rtsp_url = self.client.get_rtsp_url(camera.id)
                if rtsp_url:
                    capture_result = await capture_with_fallback(rtsp_url, camera.name)
                    if capture_result.success:
                        # Use single video analysis
                        analysis, image_path = await self._analyze_single_capture(
                            capture_result, camera.name, event_type,
                            event_id, conversation_id, event_timestamp
                        )
                    else:
                        return
                else:
                    return
            else:
                # Upload all videos to GCS
                video_uris = {}
                for cam_name, video_path in multi_capture.videos.items():
                    uri = upload_media(
                        local_path=video_path,
                        camera_name=cam_name,
                        event_id=event_id,
                        timestamp=event_timestamp,
                        frame_index=0,
                    )
                    if uri:
                        video_uris[cam_name] = uri

                # Analyze all videos together
                _LOGGER.info("Analyzing %d videos with Gemini...", len(multi_capture.videos))
                analysis = await analyze_multiple_videos(
                    video_paths=multi_capture.videos,
                    primary_camera=camera.name,
                    event_type=event_type,
                    event_id=event_id,
                    conversation_id=conversation_id,
                    video_uris=video_uris,
                )

                # Extract frame from primary camera video for notification
                primary_video = multi_capture.videos.get(camera.name)
                if primary_video:
                    image_path = await self._extract_frame_from_video(primary_video)
                else:
                    image_path = None

        else:
            # Single camera capture (original flow)
            rtsp_url = self.client.get_rtsp_url(camera.id)
            if not rtsp_url:
                _LOGGER.warning("No RTSP URL available for %s", camera.name)
                return

            _LOGGER.info("Capturing from %s...", camera.name)
            capture_result = await capture_with_fallback(rtsp_url, camera.name)

            if not capture_result.success:
                _LOGGER.warning("Capture failed for %s: %s", camera.name, capture_result.error)
                return

            analysis, image_path = await self._analyze_single_capture(
                capture_result, camera.name, event_type,
                event_id, conversation_id, event_timestamp
            )

        # Rest of the notification logic continues below...
        if not analysis:
            _LOGGER.warning("Analysis failed for %s, sending raw notification", camera.name)
            await send_notification(
                message=(
                    f"Motion detected but analysis failed.\n"
                    f"Event type: {event_type}\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                ),
                title=f"⚠️ {camera.name}",
                image_path=image_path,
            )
            return

        # Log the analysis
        _LOGGER.info("Analysis for %s: risk=%s, action=%s",
                     camera.name, analysis.risk_tier, analysis.recommended_action)
        _LOGGER.info("Summary: %s", analysis.summary)

        # Get risk configuration
        risk_config = config.RISK_LEVELS.get(analysis.risk_tier, {})

        # Notify if person detected OR risk level warrants it
        # ALWAYS notify for doorbell events (button press or motion) - anyone at the door is important
        is_doorbell = camera.name.lower() == "doorbell" or event_type == "doorbell"
        should_notify = is_doorbell or analysis.person_detected or risk_config.get("notify", False)

        # Get priority from config (default to 0 = normal)
        priority = risk_config.get("priority", 0)

        if should_notify:
            # Build title with emoji based on risk
            risk_emoji = {
                "low": "✅",
                "medium": "⚠️",
                "high": "🚨",
                "critical": "🆘",
            }.get(analysis.risk_tier, "")
            title = f"{risk_emoji} {camera.name}"

            # Add CRITICAL prefix for emergency situations
            if analysis.risk_tier == "critical":
                title = f"🆘 CRITICAL: {camera.name}"

            notification = format_notification(camera.name, analysis)
            await send_notification(
                message=notification,
                title=title,
                priority=priority,
                image_path=image_path,
            )

    async def _analyze_single_capture(
        self,
        capture_result: CaptureResult,
        camera_name: str,
        event_type: str,
        event_id: str,
        conversation_id: str,
        event_timestamp,
    ) -> tuple[SecurityAnalysis | None, Path | None]:
        """Analyze a single camera capture (video or frames)."""

        if capture_result.is_video and capture_result.video_path:
            _LOGGER.info("Captured video, analyzing with Gemini...")

            # Upload video to GCS
            video_uri = upload_media(
                local_path=capture_result.video_path,
                camera_name=camera_name,
                event_id=event_id,
                timestamp=event_timestamp,
                frame_index=0,
            )

            # Analyze video with Gemini
            analysis = await analyze_video(
                capture_result.video_path,
                camera_name=camera_name,
                event_type=event_type,
                event_id=event_id,
                conversation_id=conversation_id,
                video_uri=video_uri,
            )

            # Extract frame for notification
            image_path = await self._extract_frame_from_video(capture_result.video_path)

            return analysis, image_path

        else:
            # Frame-based capture
            frames = capture_result.frame_paths
            _LOGGER.info("Captured %d frames, analyzing...", len(frames))

            # Upload frames to GCS
            image_uris = []
            for i, frame_path in enumerate(frames):
                uri = upload_media(
                    local_path=frame_path,
                    camera_name=camera_name,
                    event_id=event_id,
                    timestamp=event_timestamp,
                    frame_index=i,
                )
                image_uris.append(uri)

            # Analyze with Gemini
            analysis = await analyze_multiple_frames(
                frames,
                camera_name=camera_name,
                event_type=event_type,
                event_id=event_id,
                conversation_id=conversation_id,
                image_uris=image_uris,
            )

            # Use first frame as the notification image
            image_path = frames[0] if frames else None

            return analysis, image_path

    async def _extract_frame_from_video(self, video_path: Path) -> Path | None:
        """Extract a single frame from a video for notification image."""
        try:
            frame_path = video_path.with_suffix(".jpg")
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-ss", "1",  # 1 second into the video
                "-frames:v", "1",
                "-q:v", "2",
                "-y",
                str(frame_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            await asyncio.wait_for(process.communicate(), timeout=10)

            if frame_path.exists():
                return frame_path
        except Exception as e:
            _LOGGER.warning("Failed to extract frame from video: %s", e)

        return None

    async def _keepalive_loop(self) -> None:
        """Periodically refresh the session to keep PubNub alive."""
        while self._running:
            await asyncio.sleep(config.KEEPALIVE_INTERVAL_SECONDS)
            if self._running:
                try:
                    await self.client.refresh()
                    _LOGGER.debug("Session keepalive completed")
                except Exception as e:
                    _LOGGER.error("Keepalive failed: %s", e)

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old frame files."""
        while self._running:
            await asyncio.sleep(3600)  # Every hour
            if self._running:
                cleanup_old_frames(max_age_hours=1)

    async def _sync_loop(self) -> None:
        """Periodically sync logs to BigQuery and archive old images to GCS."""
        while self._running:
            await asyncio.sleep(config.SYNC_INTERVAL_SECONDS)
            if self._running:
                try:
                    # Sync logs to BigQuery
                    synced, failed, cleaned = await run_gcp_sync()
                    if synced > 0 or failed > 0 or cleaned > 0:
                        _LOGGER.info(
                            "GCP sync: %d logs synced, %d failed, %d cleaned up",
                            synced, failed, cleaned
                        )

                    # Archive old media (images and videos) to GCS
                    uploaded, deleted = await asyncio.to_thread(archive_old_media)
                    if uploaded > 0 or deleted > 0:
                        _LOGGER.info(
                            "Media archival: %d uploaded to GCS, %d deleted locally",
                            uploaded, deleted
                        )
                except Exception as e:
                    _LOGGER.error("GCP sync failed: %s", e)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to Cloud Logging with service health."""
        # Heartbeat interval: 5 minutes
        heartbeat_interval = 300

        while self._running:
            await asyncio.sleep(heartbeat_interval)
            if self._running and _cloud_logger:
                try:
                    # Gather health information
                    from health_check import get_error_counts

                    # Get health status
                    health_data = {
                        "service": "security-guard",
                        "status": "running",
                        "cameras_monitored": len(self.client.cameras) if self.client else 0,
                        "camera_names": [c.name for c in self.client.cameras] if self.client else [],
                        "errors": get_error_counts(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    # Check external services (quick connectivity test)
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                "http://localhost:8080/health",
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as resp:
                                if resp.status == 200:
                                    health_response = await resp.json()
                                    health_data["health_status"] = health_response.get("status", "unknown")
                                    health_data["checks"] = {
                                        k: v.get("status") if isinstance(v, dict) else "ok"
                                        for k, v in health_response.get("checks", {}).items()
                                        if isinstance(v, dict)
                                    }
                    except Exception:
                        health_data["health_status"] = "unknown"

                    # Send to Cloud Logging
                    _cloud_logger.log_struct(
                        health_data,
                        severity="INFO",
                        labels={
                            "service": "security-guard",
                            "type": "heartbeat",
                        }
                    )
                    _LOGGER.debug("Heartbeat sent to Cloud Logging")
                except Exception as e:
                    _LOGGER.warning("Failed to send heartbeat: %s", e)


async def main():
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from external libraries
    logging.getLogger("pubnub").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    # Load stored credentials (API keys + Pushover)
    load_stored_credentials()

    # Check GCP connectivity (if configured)
    gcp_status = None
    if config.GCP_PROJECT_ID or config.GCP_SERVICE_ACCOUNT_FILE:
        print()
        gcp_status = check_gcp_connectivity()

    # Check hub connectivity before starting
    print()
    if not check_hub_connectivity():
        print("\nHub connectivity check failed. Exiting.")
        return

    # Start health check server
    await start_health_server()

    guard = SecurityGuard()

    try:
        if not await guard.start():
            await stop_health_server()
            return
    except VivintAuthInterventionRequired as e:
        # Auth requires user intervention - send Pushover alert and exit
        _LOGGER.error("Vivint auth intervention required: %s", e)

        # Rate-limit auth failure alerts (max once per 30 minutes)
        auth_alert_file = config.DATA_DIR / ".last_auth_alert"
        should_alert = True
        alert_cooldown_seconds = 1800  # 30 minutes

        try:
            if auth_alert_file.exists():
                last_alert_time = float(auth_alert_file.read_text().strip())
                if time.time() - last_alert_time < alert_cooldown_seconds:
                    _LOGGER.info("Auth alert rate-limited (last sent %.0f seconds ago)",
                                time.time() - last_alert_time)
                    should_alert = False
        except Exception:
            pass  # File doesn't exist or invalid content

        # DISABLED: MFA Pushover alerts were causing spam
        # TODO: Re-enable with better rate limiting or manual trigger only
        # if should_alert:
        #     sent = await send_pushover(
        #         title="🔐 Vivint Authentication Required",
        #         message=(
        #             f"{str(e)}\n\n"
        #             "The security guard service cannot connect to Vivint and "
        #             "requires manual re-authentication. The service will restart "
        #             "and retry automatically."
        #         ),
        #         priority=1,
        #     )
        #     if sent:
        #         try:
        #             auth_alert_file.write_text(str(time.time()))
        #         except Exception:
        #             pass
        should_alert = False  # Disabled

        print("\n" + "=" * 60)
        print("AUTHENTICATION REQUIRED")
        print("=" * 60)
        print(str(e))
        if should_alert:
            print("\nA Pushover notification has been sent.")
        else:
            print("\n(Pushover alert rate-limited - already sent recently)")
        print("The service will exit and systemd will restart it.")
        print("=" * 60 + "\n")

        await stop_health_server()
        # Exit with code 1 so systemd knows to restart
        import sys
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Security Guard Service Running")
    print("=" * 60)
    print("Monitoring cameras for motion and doorbell events.")
    if _pushover_token and _pushover_user:
        print("Pushover notifications: ENABLED")
    else:
        print("Pushover notifications: DISABLED (console only)")
        print("  Run setup_credentials.py to configure push notifications")

    # GCP status
    if config.GCP_PROJECT_ID or config.GCP_SERVICE_ACCOUNT_FILE:
        # Check if GCP is actually reachable
        gcs_ok = gcp_status and gcp_status.get('gcs', (False, ''))[0]
        bq_ok = gcp_status and gcp_status.get('bigquery', (False, ''))[0]

        if gcs_ok and bq_ok:
            print(f"GCP integration: ENABLED - Connected (sync every {config.SYNC_INTERVAL_SECONDS}s)")
        elif gcs_ok or bq_ok:
            print(f"GCP integration: ENABLED - Partially Connected (sync every {config.SYNC_INTERVAL_SECONDS}s)")
        else:
            print(f"GCP integration: ENABLED - Connection Failed (will retry)")

        # Show GCS status
        if config.GCS_BUCKET_NAME:
            gcs_status_icon = "[OK]" if gcs_ok else "[FAIL]"
            print(f"  GCS {gcs_status_icon}: {config.GCS_BUCKET_NAME}")
            if not gcs_ok and gcp_status:
                print(f"    Error: {gcp_status.get('gcs', (False, 'Unknown'))[1]}")

        # Show BigQuery status
        bq_status_icon = "[OK]" if bq_ok else "[FAIL]"
        print(f"  BigQuery {bq_status_icon}: {config.BQ_DATASET}.{config.BQ_TABLE}")
        if not bq_ok and gcp_status:
            print(f"    Error: {gcp_status.get('bigquery', (False, 'Unknown'))[1]}")
    else:
        print("GCP integration: DISABLED (local storage only)")

    # Doorbell AI status
    if config.DOORBELL_AI_ENABLED:
        if DOORBELL_AI_AVAILABLE:
            print(f"Doorbell AI: ENABLED (conversation: {config.DOORBELL_AI_CONVERSATION_DURATION}s)")
        else:
            print("Doorbell AI: ENABLED but unavailable (missing dependencies)")
    else:
        print("Doorbell AI: DISABLED")
        print("  Set DOORBELL_AI_ENABLED=true to enable AI doorbell conversations")

    # Cloud Logging heartbeat status
    if _cloud_logger:
        print("Cloud Logging heartbeat: ENABLED (every 5 min)")
    else:
        print("Cloud Logging heartbeat: DISABLED")

    print("Health check: http://localhost:8080/health")
    print("Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await guard.stop()
        await stop_health_server()


if __name__ == "__main__":
    asyncio.run(main())
