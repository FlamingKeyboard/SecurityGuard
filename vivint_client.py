"""Vivint client wrapper with token persistence and credential management."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

# Add vivintpy to path
sys.path.insert(0, str(Path(__file__).parent / "vivintpy"))

from vivintpy.account import Account
from vivintpy.devices.camera import Camera, MOTION_DETECTED, DOORBELL_DING, RtspUrlType
from vivintpy.exceptions import VivintSkyApiMfaRequiredError

import config

_LOGGER = logging.getLogger(__name__)


class VivintAuthInterventionRequired(Exception):
    """
    Raised when Vivint authentication requires user intervention.

    This happens when:
    - MFA is required but the system is running non-interactively
    - Login fails and manual credential entry is needed
    """
    pass


def is_interactive() -> bool:
    """Check if we're running in an interactive terminal."""
    try:
        return sys.stdin.isatty()
    except Exception:
        return False

# Check if we're on Windows (for DPAPI support)
_IS_WINDOWS = sys.platform == "win32"


def _encrypt_data(data: bytes) -> bytes:
    """Encrypt data using Windows DPAPI (Windows only)."""
    if not _IS_WINDOWS:
        return data
    try:
        import win32crypt
        return win32crypt.CryptProtectData(data, None, None, None, None, 0)
    except ImportError:
        _LOGGER.warning("pywin32 not available, storing tokens unencrypted")
        return data


def _decrypt_data(data: bytes) -> bytes:
    """Decrypt data using Windows DPAPI (Windows only)."""
    if not _IS_WINDOWS:
        return data
    try:
        import win32crypt
        _, decrypted = win32crypt.CryptUnprotectData(data, None, None, None, 0)
        return decrypted
    except ImportError:
        _LOGGER.warning("pywin32 not available, reading unencrypted tokens")
        return data


# Credentials file for all secure data
CREDENTIALS_FILE = config.DATA_DIR / "credentials.enc"

# Environment variable mappings for container deployment
_ENV_CREDENTIAL_MAP = {
    "username": "VIVINT_USERNAME",
    "password": "VIVINT_PASSWORD",
    "gemini_api_key": "GEMINI_API_KEY",
    "pushover_token": "PUSHOVER_TOKEN",
    "pushover_user": "PUSHOVER_USER",
}


def save_credentials(creds: dict) -> None:
    """Save all credentials to encrypted file (Windows) or plain file (Linux)."""
    # Load existing and merge
    existing = _load_credentials_from_file() or {}
    existing.update(creds)
    data = json.dumps(existing).encode()
    encrypted = _encrypt_data(data)
    CREDENTIALS_FILE.write_bytes(encrypted)
    _LOGGER.info("Credentials saved")


def _load_credentials_from_file() -> dict | None:
    """Load credentials from file only."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        encrypted = CREDENTIALS_FILE.read_bytes()
        decrypted = _decrypt_data(encrypted)
        return json.loads(decrypted.decode())
    except Exception as e:
        _LOGGER.warning("Failed to load credentials from file: %s", e)
        return None


def load_credentials() -> dict | None:
    """
    Load credentials from environment variables (preferred) or encrypted file.

    Environment variables take precedence for container deployments.
    """
    creds = {}

    # First, try to load from file
    file_creds = _load_credentials_from_file() or {}
    creds.update(file_creds)

    # Override with environment variables (for container deployment)
    for cred_key, env_var in _ENV_CREDENTIAL_MAP.items():
        env_value = os.environ.get(env_var)
        if env_value:
            creds[cred_key] = env_value

    return creds if creds else None


def get_stored_credential(key: str) -> str | None:
    """
    Get a specific stored credential.

    Checks environment variable first, then falls back to stored credentials.
    """
    # Check environment variable first
    env_var = _ENV_CREDENTIAL_MAP.get(key)
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value

    # Fall back to stored credentials
    creds = _load_credentials_from_file()
    return creds.get(key) if creds else None


# Legacy token functions for backward compatibility
def save_tokens(tokens: dict) -> None:
    """Save tokens to encrypted file (legacy, now uses credentials file)."""
    save_credentials(tokens)


def load_tokens() -> dict | None:
    """Load tokens from encrypted file."""
    # Try new credentials file first
    creds = load_credentials()
    if creds and creds.get("refresh_token"):
        return creds
    # Fall back to old token file
    if not config.TOKEN_FILE.exists():
        return None
    try:
        encrypted = config.TOKEN_FILE.read_bytes()
        decrypted = _decrypt_data(encrypted)
        return json.loads(decrypted.decode())
    except Exception as e:
        _LOGGER.warning("Failed to load tokens: %s", e)
        return None


class VivintClient:
    """Wrapper around vivintpy Account with token persistence and utilities."""

    def __init__(self):
        self.account: Account | None = None
        self._motion_callbacks: list[Callable] = []
        self._doorbell_callbacks: list[Callable] = []
        self._cameras: list[Camera] = []
        self._rtsp_urls: dict[int, str] = {}  # camera_id -> rtsp_url

    async def connect(self) -> bool:
        """Connect to Vivint, handling MFA and token persistence."""
        # Load stored credentials
        stored_creds = load_credentials() or {}

        # Get username from env or stored
        username = config.VIVINT_USERNAME or stored_creds.get("username", "")
        password = config.VIVINT_PASSWORD or stored_creds.get("password", "")

        # Try loading saved tokens first
        tokens = load_tokens()

        if tokens and tokens.get("refresh_token") and username:
            _LOGGER.info("Attempting connection with saved refresh token...")
            try:
                # vivintpy's connect() asserts password exists before trying refresh token
                # so we pass a placeholder - it won't be used if refresh token works
                self.account = Account(
                    username=username,
                    password=password or "placeholder_for_refresh_token",
                    refresh_token=tokens["refresh_token"],
                )
                await self.account.connect(
                    load_devices=True,
                    subscribe_for_realtime_updates=True
                )
                _LOGGER.info("Connected using saved refresh token")
                self._save_current_tokens()
                await self._setup_after_connect()
                return True
            except Exception as e:
                _LOGGER.warning("Refresh token failed: %s (%s), will try fresh login", e, type(e).__name__)
                # Clean up failed session
                if self.account:
                    try:
                        await self.account.disconnect()
                    except Exception:
                        pass
                    self.account = None

        # Fresh login required
        if not username or not password:
            _LOGGER.error("No credentials available. Run setup_credentials.py first.")
            if not is_interactive():
                raise VivintAuthInterventionRequired(
                    "Vivint credentials not configured. "
                    "Please run setup_credentials.py to configure your Vivint login."
                )
            return False

        _LOGGER.info("Performing fresh login...")
        self.account = Account(
            username=username,
            password=password,
        )

        try:
            await self.account.connect(
                load_devices=True,
                subscribe_for_realtime_updates=True
            )
        except VivintSkyApiMfaRequiredError:
            if is_interactive():
                # Running interactively - prompt for MFA code
                code = input("Enter MFA Code: ")
                await self.account.verify_mfa(code)
                _LOGGER.info("MFA verified")
            else:
                # Non-interactive mode - cannot prompt for MFA
                _LOGGER.error("MFA required but running non-interactively")
                raise VivintAuthInterventionRequired(
                    "Vivint MFA required. Please run setup_credentials.py or "
                    "configure mode to enter the MFA code and re-authenticate."
                )
        except Exception as e:
            # Handle other authentication failures
            error_msg = str(e)
            _LOGGER.error("Vivint login failed: %s", error_msg)
            if not is_interactive():
                # Non-interactive mode - raise auth intervention error
                raise VivintAuthInterventionRequired(
                    f"Vivint login failed: {error_msg}. "
                    "Please check credentials and re-authenticate."
                )
            raise  # Re-raise in interactive mode

        # Save tokens for future use
        self._save_current_tokens()
        await self._setup_after_connect()
        return True

    def _save_current_tokens(self) -> None:
        """Save current session tokens."""
        if self.account:
            refresh_token = self.account.refresh_token
            if refresh_token:
                tokens = {"refresh_token": refresh_token}
                save_tokens(tokens)

    async def _setup_after_connect(self) -> None:
        """Setup cameras and event handlers after connection."""
        self._cameras = []
        for system in self.account.systems:
            _LOGGER.info("System: %s (ID: %s)", system.name, system.id)
            for alarm_panel in system.alarm_panels:
                _LOGGER.info("  Alarm Panel: %s", alarm_panel.name)
                # Get panel credentials for RTSP access
                await alarm_panel.get_panel_credentials()
                _LOGGER.info("  Panel credentials retrieved")

                for device in alarm_panel.devices:
                    if isinstance(device, Camera):
                        self._cameras.append(device)
                        _LOGGER.info("    Camera: %s (ID: %d, Online: %s)",
                                     device.name, device.id, device.is_online)

                        # Register event handlers
                        device.on(MOTION_DETECTED, self._handle_motion)
                        device.on(DOORBELL_DING, self._handle_doorbell)

                        # Get RTSP URL
                        await self._cache_rtsp_url(device)

    async def _cache_rtsp_url(self, camera: Camera) -> None:
        """Cache RTSP URL for a camera."""
        try:
            # Try direct local access first (fastest)
            url = camera.get_rtsp_access_url(RtspUrlType.LOCAL, hd=config.RTSP_PREFER_HD)
            if url:
                self._rtsp_urls[camera.id] = url
                _LOGGER.info("    Cached LOCAL RTSP URL for %s", camera.name)
                return

            # Fall back to panel access (through hub)
            url = await camera.get_rtsp_url(internal=True, hd=config.RTSP_PREFER_HD)
            if url:
                self._rtsp_urls[camera.id] = url
                _LOGGER.info("    Cached PANEL RTSP URL for %s", camera.name)
                return

            # Fall back to external access
            url = await camera.get_rtsp_url(internal=False, hd=config.RTSP_PREFER_HD)
            if url:
                self._rtsp_urls[camera.id] = url
                _LOGGER.info("    Cached EXTERNAL RTSP URL for %s", camera.name)
                return

            _LOGGER.warning("    Could not get RTSP URL for %s", camera.name)
        except Exception as e:
            _LOGGER.error("    Error getting RTSP URL for %s: %s", camera.name, e)

    def _handle_motion(self, event: dict) -> None:
        """Handle motion detected event."""
        camera = event.get("device")
        message = event.get("message", {})
        _LOGGER.info("Motion detected on %s: %s", camera.name if camera else "unknown", message)
        for callback in self._motion_callbacks:
            try:
                callback(camera, message)
            except Exception as e:
                _LOGGER.error("Error in motion callback: %s", e)

    def _handle_doorbell(self, event: dict) -> None:
        """Handle doorbell ding event."""
        camera = event.get("device")
        message = event.get("message", {})
        _LOGGER.info("Doorbell ding on %s: %s", camera.name if camera else "unknown", message)
        for callback in self._doorbell_callbacks:
            try:
                callback(camera, message)
            except Exception as e:
                _LOGGER.error("Error in doorbell callback: %s", e)

    def on_motion(self, callback: Callable) -> None:
        """Register a motion detection callback."""
        self._motion_callbacks.append(callback)

    def on_doorbell(self, callback: Callable) -> None:
        """Register a doorbell callback."""
        self._doorbell_callbacks.append(callback)

    @property
    def cameras(self) -> list[Camera]:
        """Return list of cameras."""
        return self._cameras

    def get_rtsp_url(self, camera_id: int) -> str | None:
        """Get cached RTSP URL for a camera."""
        return self._rtsp_urls.get(camera_id)

    def get_camera_by_id(self, camera_id: int) -> Camera | None:
        """Get camera by ID."""
        for cam in self._cameras:
            if cam.id == camera_id:
                return cam
        return None

    async def refresh(self) -> None:
        """Refresh the session to keep PubNub alive."""
        if self.account:
            await self.account.refresh()
            self._save_current_tokens()
            _LOGGER.debug("Session refreshed")

    async def disconnect(self) -> None:
        """Disconnect from Vivint."""
        if self.account:
            await self.account.disconnect()
            _LOGGER.info("Disconnected from Vivint")


async def test_connection():
    """Test the Vivint connection and list cameras."""
    logging.basicConfig(level=logging.INFO)
    client = VivintClient()
    if await client.connect():
        print("\n=== Connection Successful ===")
        print(f"Found {len(client.cameras)} cameras:")
        for cam in client.cameras:
            print(f"  - {cam.name} (ID: {cam.id})")
            url = client.get_rtsp_url(cam.id)
            if url:
                # Mask password in output
                import re
                masked = re.sub(r':([^:@]+)@', ':****@', url)
                print(f"    RTSP: {masked}")
        await client.disconnect()
    else:
        print("Connection failed")


if __name__ == "__main__":
    asyncio.run(test_connection())
