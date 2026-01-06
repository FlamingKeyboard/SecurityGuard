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
import time
from datetime import datetime

import config
from vivint_client import VivintClient, load_credentials
from frame_capture import capture_frames, cleanup_old_frames
from gemini_analyzer import analyze_multiple_frames, SecurityAnalysis


def load_stored_api_keys():
    """Load API keys from stored credentials into environment."""
    creds = load_credentials() or {}
    if creds.get("gemini_api_key") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = creds["gemini_api_key"]
        # Reload config to pick up the change
        import importlib
        importlib.reload(config)

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


def send_notification(message: str, urgent: bool = False) -> None:
    """
    Send a notification.

    This is a placeholder - implement your preferred notification method:
    - Windows toast notification
    - Discord webhook
    - Pushover/Pushbullet
    - SMS via Twilio
    - etc.
    """
    prefix = "[URGENT] " if urgent else ""
    print(f"\n{'='*60}")
    print(f"{prefix}SECURITY NOTIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(message)
    print("=" * 60 + "\n")

    # TODO: Implement actual notification
    # Examples:
    #
    # Windows toast:
    # from win10toast import ToastNotifier
    # toaster = ToastNotifier()
    # toaster.show_toast("Security Alert", message[:256], duration=10)
    #
    # Discord webhook:
    # import requests
    # requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
    #
    # Pushover:
    # import requests
    # requests.post("https://api.pushover.net/1/messages.json", data={
    #     "token": PUSHOVER_TOKEN,
    #     "user": PUSHOVER_USER,
    #     "message": message,
    #     "priority": 2 if urgent else 0,
    # })


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

        if not analysis:
            _LOGGER.warning("Analysis failed for %s, sending raw notification", camera.name)
            send_notification(
                f"Motion detected on {camera.name} but analysis failed.\n"
                f"Event type: {event_type}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return

        # Log the analysis
        _LOGGER.info("Analysis for %s: risk=%s, action=%s",
                     camera.name, analysis.risk_tier, analysis.recommended_action)
        _LOGGER.info("Summary: %s", analysis.summary)

        # Determine notification based on risk level
        risk_config = config.RISK_LEVELS.get(analysis.risk_tier, {})

        if risk_config.get("notify"):
            notification = format_notification(camera.name, analysis)
            urgent = risk_config.get("urgent", False)
            send_notification(notification, urgent=urgent)

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

    # Load stored API keys
    load_stored_api_keys()

    guard = SecurityGuard()

    if not await guard.start():
        return

    print("\n" + "=" * 60)
    print("Security Guard Service Running")
    print("=" * 60)
    print("Monitoring cameras for motion and doorbell events.")
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
