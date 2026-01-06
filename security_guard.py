"""
Vivint Security Guard Service

Main entry point that coordinates:
- Vivint connection and event subscription
- Frame capture on motion/doorbell events
- Gemini vision analysis
- Notifications based on risk assessment
"""

import asyncio
import logging
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import aiohttp

import config


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


from vivint_client import VivintClient, load_credentials
from frame_capture import capture_frames, cleanup_old_frames
from gemini_analyzer import analyze_multiple_frames, SecurityAnalysis

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
        import importlib
        importlib.reload(config)

    # Pushover credentials
    _pushover_token = creds.get("pushover_token")
    _pushover_user = creds.get("pushover_user")

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

        _LOGGER.info("Security Guard service started")
        _LOGGER.info("Monitoring %d cameras", len(self.client.cameras))
        for cam in self.client.cameras:
            _LOGGER.info("  - %s (ID: %d)", cam.name, cam.id)

        return True

    async def stop(self) -> None:
        """Stop the security guard service."""
        _LOGGER.info("Stopping Security Guard service...")
        self._running = False
        await self.client.disconnect()
        _LOGGER.info("Security Guard service stopped")

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

        # Get RTSP URL for this camera
        rtsp_url = self.client.get_rtsp_url(camera.id)
        if not rtsp_url:
            _LOGGER.warning("No RTSP URL available for %s", camera.name)
            return

        # Capture frames
        _LOGGER.info("Capturing frames from %s...", camera.name)
        frames = await capture_frames(rtsp_url, camera.name)

        if not frames:
            _LOGGER.warning("No frames captured from %s", camera.name)
            return

        _LOGGER.info("Captured %d frames, analyzing...", len(frames))

        # Analyze with Gemini
        analysis = await analyze_multiple_frames(frames)

        # Use first frame as the notification image
        image_path = frames[0] if frames else None

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
        should_notify = analysis.person_detected or risk_config.get("notify", False)

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

    # Check hub connectivity before starting
    print()
    if not check_hub_connectivity():
        print("\nHub connectivity check failed. Exiting.")
        return

    guard = SecurityGuard()

    if not await guard.start():
        return

    print("\n" + "=" * 60)
    print("Security Guard Service Running")
    print("=" * 60)
    print("Monitoring cameras for motion and doorbell events.")
    if _pushover_token and _pushover_user:
        print("Pushover notifications: ENABLED")
    else:
        print("Pushover notifications: DISABLED (console only)")
        print("  Run setup_credentials.py to configure push notifications")
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


if __name__ == "__main__":
    asyncio.run(main())
