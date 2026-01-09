"""
Vivint WebRTC Two-Way Audio Implementation

Uses Firebase Realtime Database for signaling and WebRTC for audio.
Discovered via reverse-engineering the Vivint Android app.

Protocol:
1. Exchange Vivint OAuth token for Firebase custom token
2. Sign into Firebase with custom token
3. Create WebRTC PeerConnection with STUN/TURN servers
4. Exchange SDP offer/answer via Firebase Realtime Database WebSocket
5. Exchange ICE candidates via Firebase
6. Send TwoWayTalkStart protobuf message over DataChannel
7. Enable local audio track

Requirements:
    pip install aiortc aiohttp httpx websockets
"""

import asyncio
import json
import logging
import struct
import time
import uuid
import base64
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from urllib.parse import urlparse

import httpx

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    websockets = None
    WebSocketClientProtocol = None

# aiortc imports
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc import RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.mediastreams import MediaStreamTrack

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
_LOGGER = logging.getLogger(__name__)

# STUN/TURN configuration discovered from APK
STUN_SERVER = "stun:v2.nuts.vivint.ai:80"
TURN_SERVER = "turn:v2.nuts.vivint.ai:80"
TURN_USERNAME = "coturn@vivint.com"
TURN_PASSWORD = "VivCamProCoturnAuthAndCred{3779}"

# Firebase configuration
FIREBASE_TOKEN_URL = "https://exchange.run.vivint.ai?custom-token=true"
FIREBASE_SIGNAL_API_KEY = "AIzaSyDYE1emKFQGxuJhxkYuol6wH8zL1_B6afc"  # From APK
FIREBASE_AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken"
# Firebase REST API base - this needs the actual Firebase project URL
# The shard URL is typically per-system and retrieved from the API
FIREBASE_DB_URL = "https://vivint-prod.firebaseio.com"  # Default, may need adjustment

# TTS settings - Eleven Labs Turbo v2.5
# Voice: Jarnathan - Confident and Versatile
TTS_DEFAULT_VOICE = "c6SfcYrb2t09NHXiT80T"  # Jarnathan
TTS_VOICES = {
    "jarnathan": "c6SfcYrb2t09NHXiT80T",  # Jarnathan - Confident and Versatile (default)
}


class TTSEngine:
    """
    Text-to-speech engine using Eleven Labs Turbo v2.5.

    High-quality, low-latency speech synthesis.
    Requires ELEVEN_LABS_API_KEY environment variable.
    """

    def __init__(self, voice_id: str = None, api_key: str = None):
        """
        Args:
            voice_id: Eleven Labs voice ID (default: Jarnathan)
            api_key: Eleven Labs API key (default: from ELEVEN_LABS_API_KEY env var)
        """
        import os
        self.voice_id = voice_id or os.getenv("ELEVEN_LABS_VOICE_ID", TTS_DEFAULT_VOICE)
        self.api_key = api_key or os.getenv("ELEVEN_LABS_API_KEY", "")
        self._temp_dir = None

        if not self.api_key:
            _LOGGER.warning("TTS: ELEVEN_LABS_API_KEY not set - TTS will not work")

    async def generate_speech(self, text: str, output_path: str = None) -> str:
        """
        Generate speech audio from text using Eleven Labs Turbo v2.5.

        Args:
            text: Text to speak
            output_path: Path to save audio (optional, uses temp file if not provided)

        Returns:
            Path to generated audio file (MP3)
        """
        import tempfile
        import os

        if not self.api_key:
            raise ValueError("ELEVEN_LABS_API_KEY not set")

        if output_path is None:
            if self._temp_dir is None:
                self._temp_dir = tempfile.mkdtemp(prefix="tts_")
            output_path = os.path.join(self._temp_dir, f"tts_{int(time.time() * 1000)}.mp3")

        _LOGGER.info(f"TTS: Generating speech for: '{text[:50]}...' " if len(text) > 50 else f"TTS: Generating speech for: '{text}'")

        # Use Eleven Labs client
        from elevenlabs import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=self.api_key)

        try:
            # Generate audio using Eleven Turbo v2.5 (fastest model)
            # Returns async generator, not coroutine
            audio_generator = client.text_to_speech.convert(
                voice_id=self.voice_id,
                model_id="eleven_turbo_v2_5",  # Turbo v2.5 - low latency
                text=text,
                output_format="mp3_44100_128",  # High quality MP3
            )

            # Collect all audio chunks
            audio_data = b""
            async for chunk in audio_generator:
                audio_data += chunk

            # Write to file
            with open(output_path, "wb") as f:
                f.write(audio_data)

            _LOGGER.info(f"TTS: Audio saved to {output_path} ({len(audio_data)} bytes)")
            return output_path

        finally:
            pass  # AsyncElevenLabs client manages connections automatically

    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self._temp_dir:
            try:
                shutil.rmtree(self._temp_dir)
                _LOGGER.info("TTS: Cleaned up temp files")
            except Exception as e:
                _LOGGER.warning(f"TTS: Failed to clean up temp files: {e}")
            self._temp_dir = None

    @staticmethod
    def list_voices():
        """List available TTS voice presets."""
        return list(TTS_VOICES.keys())


@dataclass
class VivintWebRTCConfig:
    """Configuration for Vivint WebRTC connection."""
    oauth_token: str  # Vivint OAuth bearer token
    camera_uuid: str  # Camera device UUID
    system_id: str = ""  # Optional system ID for multi-panel
    app_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    firebase_db_url: str = FIREBASE_DB_URL
    audio_device: str = None  # Windows: "Microphone (Name)" for dshow capture
    # Pre-fetched credentials (set by prefetch_webrtc_credentials)
    prefetched_firebase_token: str = None
    prefetched_firebase_id_token: str = None
    prefetched_firebase_uid: str = None
    # TTS mode: set tts_audio_file to use TTS instead of microphone
    tts_audio_file: str = None  # Path to pre-generated TTS audio file
    tts_voice_id: str = TTS_DEFAULT_VOICE  # Eleven Labs voice ID to use


@dataclass
class PrefetchedCredentials:
    """Pre-fetched Firebase credentials for faster PTT connection."""
    firebase_token: str  # Firebase custom token
    firebase_id_token: str  # Firebase ID token
    firebase_uid: str  # Firebase user ID
    firebase_db_url: str  # Signaling server URL
    system_id: str  # Vivint system ID


async def prefetch_webrtc_credentials(oauth_token: str, camera_uuid: str) -> PrefetchedCredentials:
    """
    Pre-fetch Firebase credentials for faster WebRTC connection.

    Call this when the app starts to reduce PTT latency.
    Returns credentials that can be passed to VivintWebRTCConfig.

    Args:
        oauth_token: Vivint OAuth id_token
        camera_uuid: Camera device UUID

    Returns:
        PrefetchedCredentials with tokens ready for use
    """
    import httpx

    _LOGGER.info("Pre-fetching WebRTC credentials...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Exchange Vivint token for Firebase custom token
        headers = {
            "Authorization": f"Bearer {oauth_token}",
            "User-Agent": "okhttp/4.12.0",
            "Accept": "application/json",
        }
        response = await client.get(FIREBASE_TOKEN_URL, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get Firebase token: {response.status_code}")

        data = response.json()
        firebase_token = data.get("custom-token")
        if not firebase_token:
            raise Exception("No custom-token in response")

        _LOGGER.info("Pre-fetch: Got Firebase custom token")

        # Parse JWT claims for system ID and signaling server
        system_id = ""
        firebase_db_url = FIREBASE_DB_URL
        firebase_uid = ""

        try:
            parts = firebase_token.split('.')
            if len(parts) >= 2:
                payload = parts[1]
                payload += '=' * (4 - len(payload) % 4)
                decoded = base64.urlsafe_b64decode(payload)
                claims = json.loads(decoded)

                firebase_uid = claims.get("uid", "")
                nested_claims = claims.get("claims", {})
                sig_servers = nested_claims.get("sig_server", {})

                if sig_servers:
                    system_id, firebase_db_url = next(iter(sig_servers.items()))
                    _LOGGER.info(f"Pre-fetch: Using signaling server for system {system_id}")
        except Exception as e:
            _LOGGER.warning(f"Pre-fetch: Failed to parse token claims: {e}")

        # Step 2: Exchange custom token for Firebase ID token
        url = f"{FIREBASE_AUTH_URL}?key={FIREBASE_SIGNAL_API_KEY}"
        payload = {"token": firebase_token, "returnSecureToken": True}
        response = await client.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            firebase_id_token = data.get("idToken", firebase_token)
            firebase_uid = data.get("localId", firebase_uid)
            _LOGGER.info(f"Pre-fetch: Got Firebase ID token! UID: {firebase_uid}")
        else:
            firebase_id_token = firebase_token
            _LOGGER.info("Pre-fetch: Using custom token as ID token")

    _LOGGER.info("Pre-fetch: Credentials ready!")

    return PrefetchedCredentials(
        firebase_token=firebase_token,
        firebase_id_token=firebase_id_token,
        firebase_uid=firebase_uid,
        firebase_db_url=firebase_db_url,
        system_id=system_id,
    )


class DataChannelProtocol:
    """
    Protobuf-based DataChannel protocol for Vivint cameras.

    Message types (field numbers in DataChannelMessage):
    - 1: Ping
    - 2: Pong
    - 9: TwoWayTalkStart
    - 10: TwoWayTalkStartResponse
    - 11: TwoWayTalkEnd
    - 12: TwoWayTalkEndResponse
    """

    @staticmethod
    def _encode_varint(value: int) -> bytes:
        """Encode an integer as a protobuf varint."""
        result = []
        while value > 127:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value)
        return bytes(result)

    @staticmethod
    def _make_tag(field_number: int, wire_type: int = 2) -> bytes:
        """Create a protobuf tag (field number + wire type)."""
        return DataChannelProtocol._encode_varint((field_number << 3) | wire_type)

    @staticmethod
    def build_ping() -> bytes:
        """Build Ping message (field 1, empty submessage)."""
        # DataChannelMessage with ping field (1) set to empty Ping message
        return DataChannelProtocol._make_tag(1) + b'\x00'

    @staticmethod
    def build_pong() -> bytes:
        """Build Pong message (field 2, empty submessage)."""
        return DataChannelProtocol._make_tag(2) + b'\x00'

    @staticmethod
    def build_two_way_talk_start() -> bytes:
        """Build TwoWayTalkStart message (field 9, empty submessage)."""
        return DataChannelProtocol._make_tag(9) + b'\x00'

    @staticmethod
    def build_two_way_talk_end() -> bytes:
        """Build TwoWayTalkEnd message (field 11, empty submessage)."""
        return DataChannelProtocol._make_tag(11) + b'\x00'

    @staticmethod
    def parse_message_type(data: bytes) -> tuple[str, int]:
        """Parse DataChannel message type from protobuf data.

        Returns (message_type, field_number)
        """
        if len(data) < 1:
            return "unknown", 0

        # Parse varint for tag
        tag = data[0]
        if tag & 0x80:  # Multi-byte varint
            tag = data[0] & 0x7F
            if len(data) > 1:
                tag |= (data[1] & 0x7F) << 7

        field_number = tag >> 3
        wire_type = tag & 0x07

        type_map = {
            1: "ping",
            2: "pong",
            9: "two_way_talk_start",
            10: "two_way_talk_start_response",
            11: "two_way_talk_end",
            12: "two_way_talk_end_response",
            15: "play_pause_seek",
            20: "get_metadata_request",
            21: "get_metadata_response",
        }

        return type_map.get(field_number, f"field_{field_number}"), field_number


class FirebaseSignaler:
    """
    Firebase REST API-based WebRTC signaler for Vivint cameras.

    Uses Firebase Realtime Database REST API to exchange SDP and ICE candidates.
    """

    def __init__(self, config: VivintWebRTCConfig):
        self.config = config
        self.firebase_token: Optional[str] = None
        self.firebase_id_token: Optional[str] = None
        self.firebase_uid: Optional[str] = None  # Firebase user ID for listener path
        self._http_client: Optional[httpx.AsyncClient] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._device_uuids: list = []  # Device UUIDs from token claims

        # Callbacks
        self.on_remote_sdp: Optional[Callable[[str, str, str], None]] = None  # type, sdp, transaction_uuid
        self.on_remote_ice: Optional[Callable[[str, int], None]] = None  # candidate, sdpMLineIndex

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        """Close HTTP client and listener."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def get_firebase_token(self) -> str:
        """Exchange Vivint OAuth token for Firebase custom token."""
        _LOGGER.info("Exchanging Vivint token for Firebase custom token...")

        client = await self._get_http_client()

        # Try with different headers - the app may use specific User-Agent
        headers = {
            "Authorization": f"Bearer {self.config.oauth_token}",
            "User-Agent": "okhttp/4.12.0",  # Match Android app
            "Accept": "application/json",
        }

        response = await client.get(FIREBASE_TOKEN_URL, headers=headers)

        _LOGGER.info(f"Token exchange response: {response.status_code}")
        _LOGGER.debug(f"Response body: {response.text[:500] if response.text else 'empty'}")

        if response.status_code != 200:
            # Log more details for debugging
            _LOGGER.error(f"Token exchange failed: {response.status_code}")
            _LOGGER.error(f"Response: {response.text[:500] if response.text else 'empty'}")
            raise Exception(f"Failed to get Firebase token: {response.status_code} {response.text}")

        data = response.json()
        self.firebase_token = data.get("custom-token")

        if not self.firebase_token:
            raise Exception("No custom-token in response")

        _LOGGER.info("Got Firebase custom token")

        # Parse the JWT to extract claims (sig_server and devices)
        self._parse_firebase_token_claims()

        return self.firebase_token

    def _parse_firebase_token_claims(self):
        """Parse Firebase custom token JWT to extract signaling server and device UUIDs."""
        import base64
        import json

        if not self.firebase_token:
            return

        try:
            # JWT format: header.payload.signature
            parts = self.firebase_token.split('.')
            if len(parts) < 2:
                return

            # Decode payload (add padding if needed)
            payload = parts[1]
            payload += '=' * (4 - len(payload) % 4)
            decoded = base64.urlsafe_b64decode(payload)
            claims = json.loads(decoded)

            _LOGGER.debug(f"JWT claims: {json.dumps(claims, indent=2)[:1000]}")

            # Extract sig_server URL for our system
            nested_claims = claims.get("claims", {})
            sig_servers = nested_claims.get("sig_server", {})

            if self.config.system_id and self.config.system_id in sig_servers:
                self.config.firebase_db_url = sig_servers[self.config.system_id]
                _LOGGER.info(f"Using signaling server: {self.config.firebase_db_url}")
            elif sig_servers:
                # Use first available sig_server
                system_id, server_url = next(iter(sig_servers.items()))
                self.config.firebase_db_url = server_url
                self.config.system_id = system_id
                _LOGGER.info(f"Using signaling server for system {system_id}: {server_url}")

            # Extract device UUIDs
            devices = nested_claims.get("devices", {})
            if self.config.system_id and self.config.system_id in devices:
                device_uuids = devices[self.config.system_id]
                _LOGGER.info(f"Available device UUIDs: {device_uuids}")

                # If camera_uuid looks like MAC address, try to find matching UUID
                if ":" in self.config.camera_uuid and device_uuids:
                    _LOGGER.info(f"Camera UUID looks like MAC, using first device UUID instead")
                    # Store the first device UUID - in practice, need to match by device ID
                    self._device_uuids = device_uuids

        except Exception as e:
            _LOGGER.warning(f"Failed to parse Firebase token claims: {e}")

    async def sign_in_firebase(self) -> str:
        """Sign into Firebase using the custom token to get an ID token."""
        if not self.firebase_token:
            await self.get_firebase_token()

        _LOGGER.info("Signing into Firebase with custom token...")

        # Use Firebase Auth REST API to sign in with custom token
        # https://firebase.google.com/docs/reference/rest/auth#section-verify-custom-token
        client = await self._get_http_client()

        url = f"{FIREBASE_AUTH_URL}?key={FIREBASE_SIGNAL_API_KEY}"
        payload = {
            "token": self.firebase_token,
            "returnSecureToken": True
        }

        response = await client.post(url, json=payload)

        if response.status_code != 200:
            _LOGGER.error(f"Firebase sign-in failed: {response.status_code} {response.text[:500]}")
            # Fall back to using custom token directly
            _LOGGER.info("Falling back to custom token for auth")
            self.firebase_id_token = self.firebase_token
            return self.firebase_id_token

        data = response.json()
        self.firebase_id_token = data.get("idToken")
        self.firebase_uid = data.get("localId")  # This is the Firebase UID

        if self.firebase_id_token:
            _LOGGER.info(f"Got Firebase ID token! UID: {self.firebase_uid}")
        else:
            _LOGGER.warning("No idToken in response, using custom token")
            self.firebase_id_token = self.firebase_token

        return self.firebase_id_token

    def _get_db_path(self, device_id: str) -> str:
        """Get the Firebase database path for a device."""
        if self.config.system_id:
            return f"groups/{self.config.system_id}/devices/{device_id}/msg"
        return f"devices/{device_id}/msg"

    def get_device_uuid(self) -> str:
        """Get the proper device UUID to use for signaling."""
        # If we have device UUIDs from the token and camera_uuid looks like MAC
        if self._device_uuids and ":" in self.config.camera_uuid:
            # Try to find matching UUID based on MAC address parts
            mac_clean = self.config.camera_uuid.replace(":", "").lower()
            for uuid in self._device_uuids:
                uuid_clean = uuid.replace("-", "").lower()
                # Check if MAC parts appear in UUID (Vivint seems to embed MAC in UUID)
                if mac_clean[:6] in uuid_clean or uuid_clean[-12:] == mac_clean:
                    _LOGGER.info(f"Matched device UUID {uuid} to MAC {self.config.camera_uuid}")
                    return uuid
            # No match found, use first device UUID
            _LOGGER.info(f"Using first device UUID: {self._device_uuids[0]}")
            return self._device_uuids[0]
        return self.config.camera_uuid

    async def send_sdp(self, sdp_type: str, sdp: str, transaction_uuid: str):
        """Send SDP offer/answer to camera via Firebase REST API."""
        device_uuid = self.get_device_uuid()
        # Use Firebase UID as "from" so camera knows where to send response
        from_id = self.firebase_uid or self.config.app_uuid

        msg = {
            "type": "sdp",
            "sdpType": sdp_type,
            "sdp": sdp,
            "transactionUuid": transaction_uuid,
            "from": from_id,
            "videoSource": {"type": "nosource"},
            "_timestamp": int(time.time() * 1000)
        }

        path = self._get_db_path(device_uuid)
        url = f"{self.config.firebase_db_url}/{path}.json"

        # Use Firebase ID token (not custom token) for REST API authentication
        auth_token = self.firebase_id_token or self.firebase_token
        if auth_token:
            url += f"?auth={auth_token}"

        _LOGGER.info(f"Sending SDP {sdp_type} to Firebase: {path}")

        client = await self._get_http_client()
        response = await client.put(url, json=msg)

        if response.status_code not in [200, 204]:
            _LOGGER.error(f"Failed to send SDP: {response.status_code} {response.text}")
            raise Exception(f"Firebase write failed: {response.status_code}")

        _LOGGER.info(f"SDP {sdp_type} sent successfully")

    async def send_ice_candidate(self, candidate: str, sdp_m_line_index: int):
        """Send ICE candidate to camera via Firebase REST API."""
        device_uuid = self.get_device_uuid()
        # Use Firebase UID as "from" so camera knows where to send response
        from_id = self.firebase_uid or self.config.app_uuid

        msg = {
            "type": "ice",
            "candidate": candidate,
            "sdpMLineIndex": sdp_m_line_index,
            "from": from_id,
            "_timestamp": int(time.time() * 1000)
        }

        path = self._get_db_path(device_uuid)
        url = f"{self.config.firebase_db_url}/{path}.json"

        # Use Firebase ID token (not custom token) for REST API authentication
        auth_token = self.firebase_id_token or self.firebase_token
        if auth_token:
            url += f"?auth={auth_token}"

        _LOGGER.debug(f"Sending ICE candidate to Firebase")

        client = await self._get_http_client()
        response = await client.put(url, json=msg)

        if response.status_code not in [200, 204]:
            _LOGGER.warning(f"Failed to send ICE: {response.status_code}")

    async def listen_for_messages(self):
        """Listen for messages from camera on the camera's Firebase path."""
        # Firebase rules only allow access to device UUIDs in claims
        # So we listen on the same path we send to - the camera's device path
        device_uuid = self.get_device_uuid()
        path = self._get_db_path(device_uuid)
        url = f"{self.config.firebase_db_url}/{path}.json"

        # Use Firebase ID token (not custom token) for REST API authentication
        auth_token = self.firebase_id_token or self.firebase_token
        if auth_token:
            url += f"?auth={auth_token}"

        _LOGGER.info(f"Listening for messages on camera path: {path}")

        client = await self._get_http_client()

        last_data = None
        while True:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data and data != last_data:
                        last_data = data
                        await self._handle_message(data)
            except Exception as e:
                _LOGGER.debug(f"Firebase poll error: {e}")

            await asyncio.sleep(0.5)  # Poll every 500ms

    async def _handle_message(self, data: dict):
        """Handle incoming message from camera."""
        msg_type = data.get("type")
        _LOGGER.info(f"Received message type: {msg_type}")

        if msg_type == "sdp":
            sdp_type = data.get("sdpType")
            sdp = data.get("sdp")
            transaction_uuid = data.get("transactionUuid")
            _LOGGER.info(f"Received SDP {sdp_type}")
            if self.on_remote_sdp:
                self.on_remote_sdp(sdp_type, sdp, transaction_uuid)

        elif msg_type == "ice":
            candidate = data.get("candidate")
            sdp_m_line_index = data.get("sdpMLineIndex", 0)
            _LOGGER.debug(f"Received ICE candidate")
            if self.on_remote_ice:
                self.on_remote_ice(candidate, sdp_m_line_index)

    def start_listening(self):
        """Start listening for messages in background."""
        self._listener_task = asyncio.create_task(self.listen_for_messages())


class FirebaseWebSocketSignaler:
    """
    Firebase WebSocket-based WebRTC signaler for Vivint cameras.

    Uses Firebase Realtime Database WebSocket protocol instead of REST API.
    This bypasses REST API security rules that block our access.

    Firebase WebSocket Protocol:
    - Connect to wss://<project>.firebaseio.com/.ws?v=5
    - Authenticate with custom token
    - Listen/write to paths via WebSocket messages
    """

    def __init__(self, config: VivintWebRTCConfig):
        self.config = config
        self._http_client: Optional[httpx.AsyncClient] = None
        self._ws: Optional[WebSocketClientProtocol] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._device_uuids: list = []
        self._request_id = 0
        self._pending_requests: dict = {}
        self._connected = asyncio.Event()
        self._authenticated = asyncio.Event()

        # Use pre-fetched credentials if available
        self.firebase_token: Optional[str] = config.prefetched_firebase_token
        self.firebase_id_token: Optional[str] = config.prefetched_firebase_id_token
        self.firebase_uid: Optional[str] = config.prefetched_firebase_uid

        if self.firebase_token:
            _LOGGER.info("Using pre-fetched Firebase credentials")

        # Callbacks
        self.on_remote_sdp: Optional[Callable[[str, str, str], None]] = None
        self.on_remote_ice: Optional[Callable[[str, int], None]] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        """Close connections."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def get_firebase_token(self) -> str:
        """Exchange Vivint OAuth token for Firebase custom token."""
        # Skip if we already have a pre-fetched token
        if self.firebase_token:
            _LOGGER.info("Using pre-fetched Firebase custom token")
            self._parse_firebase_token_claims()
            return self.firebase_token

        _LOGGER.info("Exchanging Vivint token for Firebase custom token...")

        client = await self._get_http_client()
        headers = {
            "Authorization": f"Bearer {self.config.oauth_token}",
            "User-Agent": "okhttp/4.12.0",
            "Accept": "application/json",
        }

        response = await client.get(FIREBASE_TOKEN_URL, headers=headers)
        _LOGGER.info(f"Token exchange response: {response.status_code}")

        if response.status_code != 200:
            raise Exception(f"Failed to get Firebase token: {response.status_code} {response.text}")

        data = response.json()
        self.firebase_token = data.get("custom-token")

        if not self.firebase_token:
            raise Exception("No custom-token in response")

        _LOGGER.info("Got Firebase custom token")
        self._parse_firebase_token_claims()
        return self.firebase_token

    def _parse_firebase_token_claims(self):
        """Parse Firebase custom token JWT to extract signaling server and device UUIDs."""
        if not self.firebase_token:
            return

        try:
            parts = self.firebase_token.split('.')
            if len(parts) < 2:
                return

            payload = parts[1]
            payload += '=' * (4 - len(payload) % 4)
            decoded = base64.urlsafe_b64decode(payload)
            claims = json.loads(decoded)

            _LOGGER.debug(f"JWT claims: {json.dumps(claims, indent=2)[:1000]}")

            nested_claims = claims.get("claims", {})
            sig_servers = nested_claims.get("sig_server", {})

            if self.config.system_id and self.config.system_id in sig_servers:
                self.config.firebase_db_url = sig_servers[self.config.system_id]
                _LOGGER.info(f"Using signaling server: {self.config.firebase_db_url}")
            elif sig_servers:
                system_id, server_url = next(iter(sig_servers.items()))
                self.config.firebase_db_url = server_url
                self.config.system_id = system_id
                _LOGGER.info(f"Using signaling server for system {system_id}: {server_url}")

            devices = nested_claims.get("devices", {})
            if self.config.system_id and self.config.system_id in devices:
                device_uuids = devices[self.config.system_id]
                _LOGGER.info(f"Available device UUIDs: {device_uuids}")

                if ":" in self.config.camera_uuid and device_uuids:
                    _LOGGER.info(f"Camera UUID looks like MAC, using first device UUID")
                    self._device_uuids = device_uuids

            # Extract UID from token
            self.firebase_uid = claims.get("uid")
            if self.firebase_uid:
                _LOGGER.info(f"Firebase UID from token: {self.firebase_uid}")

        except Exception as e:
            _LOGGER.warning(f"Failed to parse Firebase token claims: {e}")

    async def sign_in_firebase(self) -> str:
        """Get Firebase ID token (for fallback REST API use)."""
        # Skip if we already have a pre-fetched ID token
        if self.firebase_id_token:
            _LOGGER.info(f"Using pre-fetched Firebase ID token! UID: {self.firebase_uid}")
            return self.firebase_id_token

        if not self.firebase_token:
            await self.get_firebase_token()

        _LOGGER.info("Signing into Firebase with custom token...")

        client = await self._get_http_client()
        url = f"{FIREBASE_AUTH_URL}?key={FIREBASE_SIGNAL_API_KEY}"
        payload = {"token": self.firebase_token, "returnSecureToken": True}

        response = await client.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            self.firebase_id_token = data.get("idToken")
            self.firebase_uid = data.get("localId") or self.firebase_uid
            _LOGGER.info(f"Got Firebase ID token! UID: {self.firebase_uid}")
        else:
            _LOGGER.info("ID token exchange failed, using custom token for WebSocket auth")
            self.firebase_id_token = self.firebase_token

        return self.firebase_id_token or self.firebase_token

    def _get_db_path(self, device_id: str) -> str:
        """Get the Firebase database path for a device."""
        if self.config.system_id:
            return f"groups/{self.config.system_id}/devices/{device_id}/msg"
        return f"devices/{device_id}/msg"

    def get_device_uuid(self) -> str:
        """Get the proper device UUID to use for signaling."""
        if self._device_uuids and ":" in self.config.camera_uuid:
            mac_clean = self.config.camera_uuid.replace(":", "").lower()
            for device_uuid in self._device_uuids:
                uuid_clean = device_uuid.replace("-", "").lower()
                if mac_clean[:6] in uuid_clean or uuid_clean[-12:] == mac_clean:
                    _LOGGER.info(f"Matched device UUID {device_uuid} to MAC {self.config.camera_uuid}")
                    return device_uuid
            _LOGGER.info(f"Using first device UUID: {self._device_uuids[0]}")
            return self._device_uuids[0]
        return self.config.camera_uuid

    def _next_request_id(self) -> int:
        """Get next request ID for Firebase protocol."""
        self._request_id += 1
        return self._request_id

    async def connect_websocket(self):
        """Connect to Firebase via WebSocket."""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets library required: pip install websockets")

        # Parse the Firebase URL to get WebSocket URL
        parsed = urlparse(self.config.firebase_db_url)
        host = parsed.netloc or parsed.path.replace("https://", "").replace("/", "")

        # Firebase WebSocket URL format
        ws_url = f"wss://{host}/.ws?v=5&ns={host.split('.')[0]}"

        _LOGGER.info(f"Connecting to Firebase WebSocket: {ws_url}")

        try:
            self._ws = await websockets.connect(
                ws_url,
                additional_headers={
                    "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 12)",
                },
                ping_interval=30,  # Send WebSocket ping every 30 seconds
                ping_timeout=10,
            )
            _LOGGER.info("WebSocket connected!")
            self._connected.set()

            # Start listening for messages
            self._listener_task = asyncio.create_task(self._ws_listener())

            # Start keepalive task
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

            # Wait for initial server hello
            await asyncio.sleep(0.5)

            # Authenticate
            await self._authenticate()

        except Exception as e:
            _LOGGER.error(f"WebSocket connection failed: {e}")
            raise

    async def _keepalive_loop(self):
        """Send periodic keepalive to Firebase."""
        try:
            while self._ws:
                await asyncio.sleep(15)  # Send keepalive every 15 seconds
                if self._ws:
                    try:
                        # Check if WebSocket is still open
                        if hasattr(self._ws, 'open') and not self._ws.open:
                            _LOGGER.debug("WebSocket closed, stopping keepalive")
                            break
                        # Send a no-op message to keep connection alive
                        keepalive_msg = {
                            "t": "d",
                            "d": {
                                "r": 0,  # Request ID 0 for keepalive
                                "a": "n",  # No-op action
                                "b": {}
                            }
                        }
                        await self._ws.send(json.dumps(keepalive_msg))
                        _LOGGER.debug("Sent Firebase keepalive")
                    except Exception as e:
                        _LOGGER.warning(f"Keepalive send failed: {e}")
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _LOGGER.debug(f"Keepalive loop ended: {e}")

    async def _authenticate(self):
        """Authenticate with Firebase via WebSocket."""
        _LOGGER.info("Authenticating with Firebase...")

        # Firebase WebSocket requires ID token (not custom token)
        # The custom token is only for signInWithCustomToken REST API
        # After that, we use the ID token from the response
        auth_token = self.firebase_id_token or self.firebase_token

        _LOGGER.info(f"Using {'ID token' if self.firebase_id_token else 'custom token'} for WebSocket auth")

        # Firebase WebSocket auth message format
        # The protocol sends: {"t":"d","d":{"r":<id>,"a":"auth","b":{"cred":"<token>"}}}
        auth_msg = {
            "t": "d",
            "d": {
                "r": self._next_request_id(),
                "a": "auth",
                "b": {
                    "cred": auth_token
                }
            }
        }

        await self._ws.send(json.dumps(auth_msg))
        _LOGGER.info("Auth message sent, waiting for response...")

        # Wait for auth response (handled in _ws_listener)
        try:
            await asyncio.wait_for(self._authenticated.wait(), timeout=10.0)
            _LOGGER.info("Firebase authentication successful!")
        except asyncio.TimeoutError:
            _LOGGER.warning("Auth response timeout, continuing anyway...")

    async def _ws_listener(self):
        """Listen for WebSocket messages from Firebase."""
        try:
            async for message in self._ws:
                _LOGGER.debug(f"WS RAW: {message[:500] if len(message) > 500 else message}")
                await self._handle_ws_message(message)
        except websockets.ConnectionClosed as e:
            _LOGGER.warning(f"WebSocket closed: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _LOGGER.error(f"WebSocket listener error: {e}")

    async def _handle_ws_message(self, message: str):
        """Handle incoming WebSocket message from Firebase."""
        try:
            data = json.loads(message)
            msg_type = data.get("t")

            if msg_type == "c":
                # Control message (connection info)
                _LOGGER.debug(f"Firebase control message: {data}")

            elif msg_type == "d":
                # Data message
                inner = data.get("d", {})
                action = inner.get("a")
                body = inner.get("b", {})
                request_id = inner.get("r")

                _LOGGER.info(f"Firebase data message: action={action}, r={request_id}, body_type={type(body)}")

                if action == "h":
                    # Server handshake/hello
                    _LOGGER.info(f"Firebase server hello: {body}")

                elif action == "ok":
                    # Auth success
                    _LOGGER.info("Firebase auth OK")
                    self._authenticated.set()

                elif action == "error":
                    _LOGGER.error(f"Firebase error: {body}")

                elif action == "d":
                    # Data update - this is what we need for signaling!
                    path = body.get("p", "")
                    data_content = body.get("d")
                    if data_content:
                        await self._handle_data_update(path, data_content)

                elif action == "ac":
                    # Auth check result
                    if body.get("s") == "ok":
                        _LOGGER.info("Firebase auth check passed")
                        self._authenticated.set()
                    else:
                        _LOGGER.error(f"Firebase auth check failed: {body}")

                elif request_id is not None:
                    # Response to our request
                    _LOGGER.info(f"Response for request {request_id}: status={body.get('s') if isinstance(body, dict) else 'N/A'}, body={str(body)[:200]}")
                    if request_id in self._pending_requests:
                        future = self._pending_requests.pop(request_id)
                        if not future.done():
                            future.set_result(body)
                    # Check if auth response
                    if isinstance(body, dict) and body.get("s") == "ok":
                        self._authenticated.set()

            else:
                _LOGGER.debug(f"Unknown Firebase message type: {msg_type}")

        except json.JSONDecodeError:
            _LOGGER.debug(f"Non-JSON message: {message[:100]}")
        except Exception as e:
            _LOGGER.error(f"Error handling message: {e}")

    async def _handle_data_update(self, path: str, data: dict):
        """Handle data update from Firebase (SDP/ICE messages)."""
        _LOGGER.info(f"Data update on path {path}")
        _LOGGER.info(f"  Data: {json.dumps(data, indent=2)[:500] if isinstance(data, dict) else str(data)[:500]}")

        if isinstance(data, dict):
            msg_type = data.get("type")

            if msg_type == "sdp":
                sdp_type = data.get("sdpType")
                sdp = data.get("sdp")
                transaction_uuid = data.get("transactionUuid", "")
                _LOGGER.info(f"Received SDP {sdp_type} via WebSocket")
                if self.on_remote_sdp:
                    self.on_remote_sdp(sdp_type, sdp, transaction_uuid)

            elif msg_type == "ice":
                candidate = data.get("candidate")
                sdp_m_line_index = data.get("sdpMLineIndex", 0)
                _LOGGER.debug(f"Received ICE candidate via WebSocket")
                if self.on_remote_ice:
                    self.on_remote_ice(candidate, sdp_m_line_index)

    async def subscribe_to_path(self, path: str):
        """Subscribe to a Firebase path for real-time updates."""
        _LOGGER.info(f"Subscribing to path: {path}")

        # Firebase listen message
        listen_msg = {
            "t": "d",
            "d": {
                "r": self._next_request_id(),
                "a": "q",  # query/listen
                "b": {
                    "p": f"/{path}",
                    "h": ""  # hash for sync
                }
            }
        }

        await self._ws.send(json.dumps(listen_msg))
        _LOGGER.info(f"Subscribed to {path}")

    async def send_sdp(self, sdp_type: str, sdp: str, transaction_uuid: str):
        """Send SDP offer/answer to camera via Firebase WebSocket."""
        device_uuid = self.get_device_uuid()
        from_id = self.firebase_uid or self.config.app_uuid
        path = self._get_db_path(device_uuid)

        msg_data = {
            "type": "sdp",
            "sdpType": sdp_type,
            "sdp": sdp,
            "transactionUuid": transaction_uuid,
            "from": from_id,
            "videoSource": {"type": "nosource"},
            "_timestamp": int(time.time() * 1000)
        }

        # Firebase set message
        set_msg = {
            "t": "d",
            "d": {
                "r": self._next_request_id(),
                "a": "p",  # put/set
                "b": {
                    "p": f"/{path}",
                    "d": msg_data
                }
            }
        }

        _LOGGER.info(f"Sending SDP {sdp_type} via WebSocket to {path}")
        await self._ws.send(json.dumps(set_msg))
        _LOGGER.info(f"SDP {sdp_type} sent")

    async def send_ice_candidate(self, candidate: str, sdp_m_line_index: int):
        """Send ICE candidate to camera via Firebase WebSocket."""
        device_uuid = self.get_device_uuid()
        from_id = self.firebase_uid or self.config.app_uuid
        path = self._get_db_path(device_uuid)

        msg_data = {
            "type": "ice",
            "candidate": candidate,
            "sdpMLineIndex": sdp_m_line_index,
            "from": from_id,
            "_timestamp": int(time.time() * 1000)
        }

        set_msg = {
            "t": "d",
            "d": {
                "r": self._next_request_id(),
                "a": "p",
                "b": {
                    "p": f"/{path}",
                    "d": msg_data
                }
            }
        }

        _LOGGER.debug(f"Sending ICE candidate via WebSocket")
        await self._ws.send(json.dumps(set_msg))

    async def listen_for_messages(self):
        """Start listening for messages from camera."""
        device_uuid = self.get_device_uuid()

        # Subscribe to our listener path (where camera sends responses)
        # The camera sends responses to the "from" address in our SDP
        from_id = self.firebase_uid or self.config.app_uuid

        # Subscribe to camera's path (to see our messages and camera responses)
        camera_path = self._get_db_path(device_uuid)
        await self.subscribe_to_path(camera_path)

        # Also subscribe to our own path if different
        if from_id != device_uuid:
            our_path = self._get_db_path(from_id)
            await self.subscribe_to_path(our_path)

        _LOGGER.info("Listening for messages via WebSocket")

    def start_listening(self):
        """Start listening for messages (async wrapper for compatibility)."""
        asyncio.create_task(self.listen_for_messages())


class IncomingAudioProcessor:
    """
    Processes incoming audio from doorbell camera and forwards to callbacks.

    Used to pipe audio from WebRTC to Gemini Live for real-time conversation.
    """

    def __init__(self):
        self._callbacks: list[Callable[[bytes], None]] = []
        self._running = False
        self._process_task: Optional[asyncio.Task] = None

    def add_callback(self, callback: Callable[[bytes], None]):
        """Add a callback to receive audio chunks (16kHz PCM)."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[bytes], None]):
        """Remove an audio callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def process_track(self, track: MediaStreamTrack):
        """
        Process an incoming audio track and forward to callbacks.

        Args:
            track: aiortc MediaStreamTrack (audio)
        """
        _LOGGER.info(f"Starting incoming audio processor for track: {track.kind}, id={track.id}")
        _LOGGER.info(f"Track readyState: {track.readyState if hasattr(track, 'readyState') else 'unknown'}")
        self._running = True

        try:
            while self._running:
                try:
                    # Get audio frame from track
                    frame = await asyncio.wait_for(track.recv(), timeout=1.0)

                    # Convert to raw PCM bytes
                    # aiortc frames are av.AudioFrame, need to extract samples
                    if hasattr(frame, 'to_ndarray'):
                        # Get samples as numpy array
                        samples = frame.to_ndarray()
                        # Convert to bytes (assuming 16-bit PCM)
                        audio_bytes = samples.tobytes()

                        # Log occasionally to track incoming audio
                        if not hasattr(self, '_recv_count'):
                            self._recv_count = 0
                        self._recv_count += 1
                        if self._recv_count % 50 == 1:
                            _LOGGER.info(f"Received audio frame #{self._recv_count} from doorbell ({len(audio_bytes)} bytes)")

                        # Forward to all callbacks
                        for callback in self._callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(audio_bytes)
                                else:
                                    callback(audio_bytes)
                            except Exception as e:
                                _LOGGER.error(f"Audio callback error: {e}")

                except asyncio.TimeoutError:
                    # Log to show we're waiting for audio from doorbell
                    if not hasattr(self, '_timeout_count'):
                        self._timeout_count = 0
                    self._timeout_count += 1
                    if self._timeout_count % 5 == 1:  # Log every 5 seconds
                        _LOGGER.warning(f"Incoming audio: {self._timeout_count} timeouts, waiting for doorbell mic (track.readyState={getattr(track, 'readyState', 'unknown')})")
                    continue
                except Exception as e:
                    if "Connection" in str(e) or "closed" in str(e).lower():
                        break
                    _LOGGER.error(f"Error processing audio frame: {e}")

        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            _LOGGER.info("Incoming audio processor stopped")

    def stop(self):
        """Stop processing audio."""
        self._running = False


class TTSAudioTrack(MediaStreamTrack):
    """
    Custom audio track that plays queued TTS audio.

    Allows dynamic injection of TTS audio into WebRTC stream.
    Must match MediaPlayer output format: 48kHz stereo s16 for proper encoding.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 48000, channels: int = 2):
        super().__init__()
        from fractions import Fraction
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._current_buffer = b""
        self._pts = 0
        self._samples_per_frame = 960  # 20ms at 48kHz (matches Opus encoder expectation)
        self._time_base = Fraction(1, sample_rate)
        self._frame_duration = self._samples_per_frame / self._sample_rate  # 0.02s (20ms)
        self._start_time = None  # Set on first recv() call

    async def queue_audio(self, audio_data: bytes):
        """
        Queue audio data to be played.

        Args:
            audio_data: PCM audio bytes (48kHz, 16-bit, stereo)
                       Each sample is 4 bytes (2 bytes per channel)
        """
        import numpy as np

        # Log detailed audio characteristics
        samples = np.frombuffer(audio_data, dtype=np.int16)
        max_amp = np.max(np.abs(samples)) if len(samples) > 0 else 0
        _LOGGER.info(f"TTSAudioTrack.queue_audio: {len(audio_data)} bytes, {len(samples)} samples, max_amplitude={max_amp}")

        await self._audio_queue.put(audio_data)
        _LOGGER.info(f"TTSAudioTrack queue size now: {self._audio_queue.qsize()}")

    async def recv(self):
        """Generate next audio frame (called by aiortc)."""
        import av
        import numpy as np
        import time

        # Initialize timing on first call
        if self._start_time is None:
            self._start_time = time.time()

        # Pacing: calculate when this frame should be generated
        # This is critical for proper WebRTC audio streaming
        if not hasattr(self, '_audio_frame_count'):
            self._audio_frame_count = 0
            self._real_audio_count = 0
            self._silence_count = 0

        expected_time = self._start_time + (self._audio_frame_count * self._frame_duration)
        current_time = time.time()
        wait_time = expected_time - current_time

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        self._audio_frame_count += 1

        # Calculate bytes needed per frame (16-bit samples)
        bytes_per_frame = self._samples_per_frame * 2 * self._channels

        # Drain all available audio from queue into buffer
        while True:
            try:
                chunk = self._audio_queue.get_nowait()
                self._current_buffer += chunk
            except asyncio.QueueEmpty:
                break

        # Track if we're sending actual audio or silence
        has_audio_data = len(self._current_buffer) >= bytes_per_frame

        # If buffer is too small, pad with silence
        if len(self._current_buffer) < bytes_per_frame:
            silence_needed = bytes_per_frame - len(self._current_buffer)
            self._current_buffer += bytes(silence_needed)

        # Extract frame data
        frame_data = self._current_buffer[:bytes_per_frame]
        self._current_buffer = self._current_buffer[bytes_per_frame:]

        # Log recv() call rate periodically (now should be ~50/second with pacing)
        if self._audio_frame_count % 100 == 1:
            _LOGGER.info(f"TTSAudioTrack.recv() frame {self._audio_frame_count} (real={self._real_audio_count}, silence={self._silence_count})")

        if has_audio_data:
            self._real_audio_count += 1
            if self._real_audio_count % 20 == 1:
                _LOGGER.info(f"TTSAudioTrack: Sending audio frame #{self._real_audio_count} (buffer: {len(self._current_buffer)} bytes remaining)")
        else:
            self._silence_count += 1

        # Convert to numpy array (16-bit signed)
        samples = np.frombuffer(frame_data, dtype=np.int16)

        # Log amplitude to debug Opus encoding issue
        if has_audio_data:
            max_amp = np.max(np.abs(samples))
            if self._real_audio_count % 20 == 1:
                _LOGGER.info(f"TTSAudioTrack: Frame amplitude max={max_amp}, samples shape={samples.shape}")

        # For packed s16 format, PyAV expects shape (1, samples * channels)
        # The data is interleaved: [L0, R0, L1, R1, ...] for stereo
        samples = samples.reshape(1, -1)

        # Create av.AudioFrame
        frame = av.AudioFrame.from_ndarray(
            samples,
            format='s16',
            layout='stereo' if self._channels == 2 else 'mono'
        )
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base

        self._pts += self._samples_per_frame

        return frame

    def clear_queue(self):
        """Clear any pending audio and reset timing."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._current_buffer = b""

    def reset_timing(self):
        """Reset timing for fresh audio playback."""
        import time
        self._start_time = time.time()
        self._audio_frame_count = 0
        self._real_audio_count = 0
        self._silence_count = 0
        self._pts = 0
        _LOGGER.info("TTSAudioTrack timing reset")


class VivintWebRTCClient:
    """
    WebRTC client for two-way audio with Vivint cameras.

    Usage:
        config = VivintWebRTCConfig(
            oauth_token="...",
            camera_uuid="...",
        )
        client = VivintWebRTCClient(config)
        await client.connect()
        await client.start_two_way_talk()
        # ... audio is now flowing ...
        await client.stop_two_way_talk()
        await client.disconnect()
    """

    def __init__(self, config: VivintWebRTCConfig, use_websocket: bool = True):
        self.config = config
        self.use_websocket = use_websocket and HAS_WEBSOCKETS

        # Use WebSocket signaler by default (REST API is blocked by Firebase rules)
        if self.use_websocket:
            self.signaler = FirebaseWebSocketSignaler(config)
        else:
            self.signaler = FirebaseSignaler(config)

        self.protocol = DataChannelProtocol()

        self.pc: Optional[RTCPeerConnection] = None
        self.data_channel = None
        self.audio_sender = None
        self.audio_track = None
        self.media_player = None
        self.connected = False
        self.two_way_active = False

        self._transaction_uuid = str(uuid.uuid4())
        self._answer_received = asyncio.Event()
        self._pong_received = asyncio.Event()
        self._datachannel_open = asyncio.Event()

        # Audio device configuration (Windows DirectShow)
        self.audio_device = config.audio_device if hasattr(config, 'audio_device') else None

        # AI conversation support
        self.incoming_audio_processor = IncomingAudioProcessor()
        self.tts_track: Optional[TTSAudioTrack] = None
        self._use_tts_track = False  # Set to True to use TTSAudioTrack instead of file/mic

    def _create_rtc_configuration(self) -> RTCConfiguration:
        """Create RTCConfiguration with STUN/TURN servers."""
        ice_servers = [
            RTCIceServer(urls=[STUN_SERVER]),
            RTCIceServer(
                urls=[TURN_SERVER],
                username=TURN_USERNAME,
                credential=TURN_PASSWORD,
            ),
        ]
        return RTCConfiguration(iceServers=ice_servers)

    async def connect(self) -> bool:
        """Establish WebRTC connection to camera."""
        _LOGGER.info("="*60)
        _LOGGER.info("Starting Vivint WebRTC connection...")
        _LOGGER.info(f"Using {'WebSocket' if self.use_websocket else 'REST API'} signaling")
        _LOGGER.info("="*60)

        try:
            # Step 1: Get Firebase token
            await self.signaler.get_firebase_token()

            # Step 2: Sign into Firebase / connect WebSocket
            if self.use_websocket:
                await self.signaler.sign_in_firebase()  # Get UID
                await self.signaler.connect_websocket()  # Connect WS
            else:
                await self.signaler.sign_in_firebase()

            # Step 3: Set up message handlers
            self.signaler.on_remote_sdp = self._on_remote_sdp
            self.signaler.on_remote_ice = self._on_remote_ice

            # Step 4: Start listening for messages
            if self.use_websocket:
                await self.signaler.listen_for_messages()
            else:
                self.signaler.start_listening()

            # Step 5: Create PeerConnection
            _LOGGER.info("Creating RTCPeerConnection...")
            config = self._create_rtc_configuration()
            self.pc = RTCPeerConnection(configuration=config)

            # Set up event handlers
            self.pc.on("icecandidate", self._on_ice_candidate)
            self.pc.on("connectionstatechange", self._on_connection_state_change)
            self.pc.on("datachannel", self._on_data_channel)
            self.pc.on("track", self._on_track)

            # Step 6: Create DataChannel
            _LOGGER.info("Creating DataChannel...")
            self.data_channel = self.pc.createDataChannel("channel")
            self.data_channel.on("open", self._on_data_channel_open)
            self.data_channel.on("message", self._on_data_channel_message)

            # Step 6b: Set up audio track if device specified, TTS file, or AI mode enabled
            if self.audio_device or self._use_tts_track or self.config.tts_audio_file:
                await self._setup_audio_track()

            # Step 7: Create and send offer
            _LOGGER.info("Creating SDP offer...")
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            # Debug: Log our offer's audio direction
            _LOGGER.info("=== SDP OFFER CREATED ===")
            for line in offer.sdp.split('\n'):
                if 'm=audio' in line or 'a=sendrecv' in line or 'a=recvonly' in line or 'a=sendonly' in line:
                    _LOGGER.info(f"SDP offer audio: {line.strip()}")

            # Wait a bit for TURN allocation to complete
            await asyncio.sleep(5)

            # Get the updated local description with ICE candidates
            local_desc = self.pc.localDescription
            _LOGGER.info("Sending SDP offer...")
            await self.signaler.send_sdp("offer", local_desc.sdp, self._transaction_uuid)

            # Extract and send ICE candidates from our local description
            await self._send_local_ice_candidates(local_desc.sdp)

            # Step 8: Wait for answer
            _LOGGER.info("Waiting for SDP answer from camera...")
            try:
                await asyncio.wait_for(self._answer_received.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                _LOGGER.error("Timeout waiting for SDP answer")
                return False

            self.connected = True
            _LOGGER.info("WebRTC connection established!")
            return True

        except Exception as e:
            _LOGGER.error(f"WebRTC connection failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _on_remote_sdp(self, sdp_type: str, sdp: str, transaction_uuid: str):
        """Handle remote SDP from camera."""
        asyncio.create_task(self._handle_remote_sdp(sdp_type, sdp, transaction_uuid))

    async def _handle_remote_sdp(self, sdp_type: str, sdp: str, transaction_uuid: str):
        """Handle remote SDP from camera (async)."""
        _LOGGER.info(f"Processing remote SDP {sdp_type}")

        # Check if we have a local description set (we've sent our offer)
        if not self.pc or not self.pc.localDescription:
            _LOGGER.warning(f"Ignoring {sdp_type} - no local description set yet (stale data?)")
            return

        if sdp_type == "answer":
            # Verify this answer is for our current session
            if transaction_uuid and transaction_uuid != self._transaction_uuid:
                _LOGGER.warning(f"Ignoring answer with different transaction UUID")
                return

            # Debug: Log SDP answer audio section
            _LOGGER.info("=== SDP ANSWER RECEIVED ===")
            for line in sdp.split('\n'):
                if 'm=audio' in line or 'a=sendrecv' in line or 'a=recvonly' in line or 'a=sendonly' in line or 'a=inactive' in line:
                    _LOGGER.info(f"SDP audio line: {line.strip()}")

            answer = RTCSessionDescription(sdp=sdp, type="answer")
            await self.pc.setRemoteDescription(answer)
            self._answer_received.set()
            _LOGGER.info("Remote description set")

            # Debug: Log transceiver states after setting remote description
            if self.pc:
                for i, transceiver in enumerate(self.pc.getTransceivers()):
                    _LOGGER.info(f"Transceiver {i}: kind={transceiver.kind}, direction={transceiver.direction}, currentDirection={transceiver.currentDirection}")
                    if transceiver.receiver:
                        _LOGGER.info(f"  Receiver: track={transceiver.receiver.track}")
                    else:
                        _LOGGER.warning(f"  Receiver: None")

        elif sdp_type == "offer":
            # Camera sent us an offer - we need to create an answer
            offer = RTCSessionDescription(sdp=sdp, type="offer")
            await self.pc.setRemoteDescription(offer)

            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            await self.signaler.send_sdp("answer", answer.sdp, transaction_uuid)
            self._answer_received.set()

    def _on_remote_ice(self, candidate: str, sdp_m_line_index: int):
        """Handle remote ICE candidate from camera."""
        asyncio.create_task(self._handle_remote_ice(candidate, sdp_m_line_index))

    async def _handle_remote_ice(self, candidate: str, sdp_m_line_index: int):
        """Handle remote ICE candidate (async)."""
        try:
            # Check if we have a remote description set (SDP exchange complete)
            if not self.pc or not self.pc.remoteDescription:
                _LOGGER.debug(f"Ignoring ICE candidate - no remote description set yet")
                return

            # Parse the ICE candidate SDP string
            # Format: candidate:foundation component protocol priority ip port typ type ...
            _LOGGER.debug(f"Parsing ICE candidate: {candidate[:100]}...")

            # aiortc wants the parsed fields - parse from SDP format
            import re
            match = re.match(
                r'candidate:(\S+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)\s+typ\s+(\S+)',
                candidate
            )
            if match:
                foundation, component, protocol, priority, ip, port, typ = match.groups()
                ice = RTCIceCandidate(
                    component=int(component),
                    foundation=foundation,
                    ip=ip,
                    port=int(port),
                    priority=int(priority),
                    protocol=protocol.lower(),
                    type=typ,
                    sdpMLineIndex=sdp_m_line_index,
                )
                await self.pc.addIceCandidate(ice)
                _LOGGER.debug(f"Added remote ICE candidate: {ip}:{port} ({typ})")
            else:
                _LOGGER.warning(f"Could not parse ICE candidate: {candidate[:100]}")
        except Exception as e:
            _LOGGER.warning(f"Failed to add ICE candidate: {e}")

    def _on_ice_candidate(self, candidate):
        """Handle local ICE candidate (for browsers - not used in aiortc)."""
        if candidate:
            _LOGGER.info(f"Local ICE candidate gathered: {candidate.candidate[:50]}...")
            asyncio.create_task(self._send_single_ice_candidate(candidate.candidate, 0))

    async def _send_single_ice_candidate(self, candidate_str: str, sdp_m_line_index: int):
        """Send a single local ICE candidate to camera."""
        try:
            await self.signaler.send_ice_candidate(candidate_str, sdp_m_line_index)
            _LOGGER.debug(f"Sent ICE candidate")
        except Exception as e:
            _LOGGER.warning(f"Failed to send ICE candidate: {e}")

    async def _send_local_ice_candidates(self, sdp: str):
        """Extract ICE candidates from SDP and send them to the camera."""
        import re
        candidate_pattern = r'a=candidate:(.+)'
        candidates = re.findall(candidate_pattern, sdp)

        _LOGGER.info(f"Extracted {len(candidates)} local ICE candidates from SDP")

        for candidate in candidates:
            full_candidate = f"candidate:{candidate}"
            _LOGGER.info(f"Sending local ICE candidate: {full_candidate[:60]}...")
            await self.signaler.send_ice_candidate(full_candidate, 0)
            # Small delay between candidates
            await asyncio.sleep(0.05)

    async def _setup_audio_track(self):
        """Set up audio track from microphone, TTS file, or AI TTS for two-way audio."""
        import platform

        try:
            # AI conversation mode - use TTSAudioTrack for dynamic TTS playback
            if self._use_tts_track:
                # Use 48kHz stereo to match MediaPlayer/Opus encoder requirements
                _LOGGER.info("AI conversation mode: Using TTSAudioTrack for dynamic TTS (48kHz stereo)")
                self.tts_track = TTSAudioTrack(sample_rate=48000, channels=2)
                self.audio_track = self.tts_track

                # Use addTrack() like the working microphone mode does
                # This matches how MediaPlayer audio tracks are added
                self.audio_sender = self.pc.addTrack(self.audio_track)
                _LOGGER.info("TTSAudioTrack added to PeerConnection (using addTrack)")
                return

            # Check if TTS mode (play audio file instead of microphone)
            if self.config.tts_audio_file:
                _LOGGER.info(f"TTS mode: Playing audio file {self.config.tts_audio_file}")
                self.media_player = MediaPlayer(self.config.tts_audio_file)

                if self.media_player.audio:
                    self.audio_track = self.media_player.audio
                    self.audio_sender = self.pc.addTrack(self.audio_track)
                    _LOGGER.info("TTS audio track added to PeerConnection")
                else:
                    _LOGGER.warning("TTS MediaPlayer has no audio track")
                return

            # Normal microphone mode
            if platform.system() == "Windows":
                # Windows DirectShow audio capture
                device_url = f"audio={self.audio_device}"
                _LOGGER.info(f"Setting up audio capture: {device_url}")

                self.media_player = MediaPlayer(
                    device_url,
                    format="dshow",
                    options={
                        "audio_buffer_size": "50",  # Low latency
                    }
                )

                if self.media_player.audio:
                    self.audio_track = self.media_player.audio
                    self.audio_sender = self.pc.addTrack(self.audio_track)
                    _LOGGER.info("Audio track added to PeerConnection")
                else:
                    _LOGGER.warning("MediaPlayer has no audio track")

            elif platform.system() == "Linux":
                # ALSA or PulseAudio
                _LOGGER.info("Setting up Linux audio capture")
                self.media_player = MediaPlayer(
                    "default",
                    format="pulse",  # or "alsa"
                )
                if self.media_player.audio:
                    self.audio_track = self.media_player.audio
                    self.audio_sender = self.pc.addTrack(self.audio_track)
                    _LOGGER.info("Audio track added to PeerConnection")

            elif platform.system() == "Darwin":
                # macOS AVFoundation
                _LOGGER.info("Setting up macOS audio capture")
                self.media_player = MediaPlayer(
                    "none:0",  # Default audio input
                    format="avfoundation",
                )
                if self.media_player.audio:
                    self.audio_track = self.media_player.audio
                    self.audio_sender = self.pc.addTrack(self.audio_track)
                    _LOGGER.info("Audio track added to PeerConnection")

            else:
                _LOGGER.warning(f"Unsupported platform for audio: {platform.system()}")

        except Exception as e:
            _LOGGER.error(f"Failed to set up audio track: {e}")
            import traceback
            traceback.print_exc()

    def _on_connection_state_change(self):
        """Handle connection state changes."""
        state = self.pc.connectionState
        _LOGGER.info(f"Connection state: {state}")

        if state == "connected":
            self.connected = True
        elif state in ["failed", "closed"]:
            self.connected = False

    def _on_data_channel(self, channel):
        """Handle incoming data channel."""
        _LOGGER.info(f"Received data channel: {channel.label}")
        channel.on("message", self._on_data_channel_message)

    def _on_data_channel_open(self):
        """Handle data channel open."""
        _LOGGER.info("DataChannel opened!")
        self._datachannel_open.set()
        # Send initial ping
        asyncio.create_task(self._send_ping())

    async def _send_ping(self):
        """Send ping and wait for pong."""
        if self.data_channel and self.data_channel.readyState == "open":
            _LOGGER.info("Sending Ping...")
            msg = self.protocol.build_ping()
            self.data_channel.send(msg)

    def _on_data_channel_message(self, message):
        """Handle data channel message."""
        if isinstance(message, bytes):
            msg_type, field_num = self.protocol.parse_message_type(message)
            _LOGGER.info(f"DataChannel message: {msg_type} (field {field_num})")

            if msg_type == "pong":
                _LOGGER.info("Received Pong!")
                self._pong_received.set()

            elif msg_type == "two_way_talk_start_response":
                _LOGGER.info("Two-way talk started (confirmed by camera)")

            elif msg_type == "two_way_talk_end_response":
                _LOGGER.info("Two-way talk ended (confirmed by camera)")

    def _on_track(self, track):
        """Handle incoming media track (audio/video from doorbell)."""
        _LOGGER.info(f"Received track: {track.kind}")

        if track.kind == "audio":
            # Note: We don't use WebRTC for incoming audio - RTSP stream has audio
            # WebRTC is only used for OUTGOING audio (Gemini voice to doorbell speaker)
            _LOGGER.info(f"Received WebRTC audio track (ignored - using RTSP for incoming audio)")

    async def start_two_way_talk(self, wait_for_channel: bool = True) -> bool:
        """Start two-way audio session."""
        if not self.connected:
            _LOGGER.error("Not connected")
            return False

        # Wait for DataChannel to open if needed
        if wait_for_channel and (not self.data_channel or self.data_channel.readyState != "open"):
            _LOGGER.info("Waiting for DataChannel to open...")
            try:
                await asyncio.wait_for(self._datachannel_open.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                _LOGGER.error("Timeout waiting for DataChannel to open")
                return False

        if not self.data_channel or self.data_channel.readyState != "open":
            _LOGGER.error("DataChannel not open")
            return False

        _LOGGER.info("Starting two-way talk...")

        # Send TwoWayTalkStart message
        msg = self.protocol.build_two_way_talk_start()
        self.data_channel.send(msg)

        self.two_way_active = True
        _LOGGER.info("TwoWayTalkStart sent!")

        # Note: Incoming audio processing is started immediately in _on_track
        # so we don't need to start it again here

        # Debug: Check audio sender status
        if self.audio_sender:
            _LOGGER.info(f"Audio sender track: {self.audio_sender.track}")
            if self.audio_sender.track:
                _LOGGER.info(f"Audio track kind: {self.audio_sender.track.kind}")
                _LOGGER.info(f"Audio track id: {self.audio_sender.track.id}")
                # Start monitoring task
                asyncio.create_task(self._monitor_audio_stats())
        else:
            _LOGGER.warning("No audio sender - microphone audio not being sent!")

        return True

    async def _monitor_audio_stats(self):
        """Monitor audio transmission and reception stats periodically."""
        _LOGGER.info("Starting audio stats monitoring...")
        last_tx_packets = 0
        last_rx_packets = 0
        for i in range(30):  # Monitor for 30 seconds
            await asyncio.sleep(1)
            if not self.two_way_active or not self.pc:
                _LOGGER.info(f"Stats monitor stopping: two_way_active={self.two_way_active}, pc={self.pc is not None}")
                break
            try:
                stats = await self.pc.getStats()
                tx_found = False
                rx_found = False

                for key, value in stats.items():
                    # Check outbound RTP (our audio to doorbell)
                    if 'outbound-rtp' in key:
                        if i == 0:
                            _LOGGER.info(f"Outbound RTP stats ({key}): {value}")

                        if hasattr(value, 'packetsSent'):
                            packets = value.packetsSent
                            bytes_sent = value.bytesSent
                        elif isinstance(value, dict):
                            packets = value.get('packetsSent', 0)
                            bytes_sent = value.get('bytesSent', 0)
                        else:
                            packets = 0
                            bytes_sent = 0

                        packets_delta = packets - last_tx_packets
                        _LOGGER.info(f"Audio TX: {packets} pkts (+{packets_delta}/s), {bytes_sent} bytes")
                        last_tx_packets = packets
                        tx_found = True

                    # Check inbound RTP (doorbell audio to us)
                    if 'inbound-rtp' in key:
                        if i == 0:
                            _LOGGER.info(f"Inbound RTP stats ({key}): {value}")

                        if hasattr(value, 'packetsReceived'):
                            packets = value.packetsReceived
                            bytes_recv = value.bytesReceived
                        elif isinstance(value, dict):
                            packets = value.get('packetsReceived', 0)
                            bytes_recv = value.get('bytesReceived', 0)
                        else:
                            packets = 0
                            bytes_recv = 0

                        packets_delta = packets - last_rx_packets
                        if packets > 0 or i < 5:  # Always log first 5 seconds or if packets received
                            _LOGGER.info(f"Audio RX: {packets} pkts (+{packets_delta}/s), {bytes_recv} bytes")
                        last_rx_packets = packets
                        rx_found = True

                if i == 0:
                    if not tx_found:
                        _LOGGER.warning(f"No outbound-rtp stats found")
                    if not rx_found:
                        _LOGGER.warning(f"No inbound-rtp stats found - doorbell may not be sending audio")
                    _LOGGER.info(f"Available stats keys: {list(stats.keys())}")

            except Exception as e:
                _LOGGER.error(f"Stats error: {e}")
                import traceback
                traceback.print_exc()

    async def stop_two_way_talk(self) -> bool:
        """Stop two-way audio session."""
        if not self.two_way_active:
            return True

        _LOGGER.info("Stopping two-way talk...")

        if self.data_channel and self.data_channel.readyState == "open":
            msg = self.protocol.build_two_way_talk_end()
            self.data_channel.send(msg)

        self.two_way_active = False
        _LOGGER.info("TwoWayTalkEnd sent!")
        return True

    # =========================================================================
    # AI Agent Integration Methods
    # =========================================================================

    def enable_ai_conversation_mode(self):
        """
        Enable AI conversation mode for dynamic TTS playback.

        Call this BEFORE connect() to use TTSAudioTrack instead of microphone.
        """
        self._use_tts_track = True
        _LOGGER.info("AI conversation mode enabled")

    def set_audio_callback(self, callback: Callable[[bytes], None]):
        """
        Set callback to receive incoming audio from doorbell.

        The callback receives PCM audio bytes (16kHz, 16-bit).
        Use this to pipe audio to Gemini Live or other speech recognition.

        Args:
            callback: Async or sync function receiving audio bytes
        """
        self.incoming_audio_processor.add_callback(callback)

    def remove_audio_callback(self, callback: Callable[[bytes], None]):
        """Remove an audio callback."""
        self.incoming_audio_processor.remove_callback(callback)

    async def speak(self, text: str) -> bool:
        """
        Speak text through the doorbell using TTS.

        Generates audio using ElevenLabs and queues it for playback.
        Only works in AI conversation mode (enable_ai_conversation_mode).

        Args:
            text: Text to speak

        Returns:
            True if audio was queued successfully
        """
        if not self.tts_track:
            _LOGGER.warning("speak() requires AI conversation mode - call enable_ai_conversation_mode() before connect()")
            return False

        try:
            from elevenlabs import AsyncElevenLabs
            import os

            api_key = os.getenv("ELEVEN_LABS_API_KEY")
            voice_id = os.getenv("ELEVEN_LABS_VOICE_ID", TTS_DEFAULT_VOICE)

            if not api_key:
                _LOGGER.error("ELEVEN_LABS_API_KEY not set")
                return False

            client = AsyncElevenLabs(api_key=api_key)

            _LOGGER.info(f"Generating TTS for: {text[:50]}...")

            # Generate audio (PCM 16kHz for WebRTC)
            # Returns async generator, not coroutine
            audio_generator = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id="eleven_turbo_v2_5",
                text=text,
                output_format="pcm_16000",  # 16kHz PCM mono
            )

            # Collect audio data
            audio_data = b""
            async for chunk in audio_generator:
                audio_data += chunk

            _LOGGER.info(f"TTS generated {len(audio_data)} bytes, queuing for playback")

            # Queue audio for playback
            await self.tts_track.queue_audio(audio_data)
            return True

        except Exception as e:
            _LOGGER.error(f"TTS error: {e}")
            return False

    async def queue_audio(self, audio_data: bytes):
        """
        Queue raw audio data for playback through doorbell.

        Args:
            audio_data: PCM audio bytes (48kHz, 16-bit, stereo)
        """
        if not self.tts_track:
            _LOGGER.warning("queue_audio() called but tts_track is None - AI conversation mode not enabled?")
            return

        _LOGGER.info(f"VivintWebRTCClient.queue_audio: forwarding {len(audio_data)} bytes to TTSAudioTrack")
        await self.tts_track.queue_audio(audio_data)

    def clear_audio_queue(self):
        """Clear any pending audio in the playback queue."""
        if self.tts_track:
            self.tts_track.clear_queue()

    async def disconnect(self):
        """Disconnect WebRTC session."""
        _LOGGER.info("Disconnecting WebRTC...")

        if self.two_way_active:
            await self.stop_two_way_talk()

        # Stop incoming audio processor
        self.incoming_audio_processor.stop()

        # Clean up TTS track
        if self.tts_track:
            self.tts_track.stop()
            self.tts_track = None

        # Clean up media player
        if self.media_player:
            try:
                self.media_player.stop()
            except Exception as e:
                _LOGGER.debug(f"Error stopping media player: {e}")
            self.media_player = None

        self.audio_track = None
        self.audio_sender = None

        if self.data_channel:
            self.data_channel.close()

        if self.pc:
            await self.pc.close()

        await self.signaler.close()

        self.connected = False
        _LOGGER.info("Disconnected")


async def get_vivint_token_and_camera():
    """Get Vivint OAuth token and camera UUID from VivintClient."""
    import sys
    sys.path.insert(0, ".")

    from vivint_client import VivintClient

    _LOGGER.info("Connecting to Vivint API...")

    client = VivintClient()
    if not await client.connect():
        _LOGGER.error("Failed to connect to Vivint")
        return None, None

    # Get access token
    # vivintpy stores tokens in account.api.tokens
    access_token = None
    camera_uuid = None

    try:
        # Access the API's token
        api = client.account._api  # Access private API object
        tokens = api.tokens

        # The app uses id_token for Firebase authentication (getIdToken in OAuthClientToken)
        access_token = tokens.get("id_token")

        if not access_token:
            # Fallback to access_token
            access_token = tokens.get("access_token")

        _LOGGER.info(f"Got id_token: {access_token[:50]}..." if access_token else "No token found")
        _LOGGER.debug(f"Token keys available: {list(tokens.keys())}")

        # Find a camera - use the VivintClient's cameras property
        cameras = client.cameras
        if cameras:
            # Prefer doorbell for two-way audio
            for cam in cameras:
                if 'doorbell' in cam.name.lower():
                    # Get the actual device UUID from camera data (not MAC address)
                    camera_uuid = cam.data.get('uuid') if hasattr(cam, 'data') else None
                    if not camera_uuid:
                        camera_uuid = cam.serial_number or str(cam.id)
                    _LOGGER.info(f"Found doorbell camera: {cam.name}")
                    _LOGGER.info(f"  UUID: {camera_uuid}")
                    _LOGGER.info(f"  MAC: {cam.mac_address}")
                    _LOGGER.info(f"  IP: {cam.ip_address}")
                    break

            # If no doorbell, use first camera
            if not camera_uuid and cameras:
                cam = cameras[0]
                camera_uuid = cam.data.get('uuid') if hasattr(cam, 'data') else None
                if not camera_uuid:
                    camera_uuid = cam.serial_number or str(cam.id)
                _LOGGER.info(f"Found camera: {cam.name} (UUID: {camera_uuid})")

    except Exception as e:
        _LOGGER.error(f"Error getting tokens/camera: {e}")
        import traceback
        traceback.print_exc()

    await client.disconnect()
    return access_token, camera_uuid


async def test_connection(audio_device: str = None, ai_conversation: bool = False, duration: int = 60):
    """Test the WebRTC connection to a Vivint camera.

    Args:
        audio_device: Windows DirectShow audio device name (e.g., "Microphone (Realtek Audio)")
                      Use --list-audio to see available devices.
        ai_conversation: If True, enable AI-powered conversation with Gemini
        duration: Test duration in seconds (for AI mode)
    """
    _LOGGER.info("="*60)
    _LOGGER.info("Vivint WebRTC Two-Way Audio - Connection Test")
    _LOGGER.info("="*60)

    # Get token and camera from Vivint
    access_token, camera_uuid = await get_vivint_token_and_camera()

    if not access_token:
        _LOGGER.error("Could not get Vivint access token")
        _LOGGER.info("")
        _LOGGER.info("Make sure you have configured Vivint credentials:")
        _LOGGER.info("  python setup_credentials.py")
        return

    if not camera_uuid:
        _LOGGER.error("No camera found in Vivint account")
        return

    # Create WebRTC config
    config = VivintWebRTCConfig(
        oauth_token=access_token,
        camera_uuid=camera_uuid,
        audio_device=audio_device,
    )

    _LOGGER.info("")
    _LOGGER.info("Configuration:")
    _LOGGER.info(f"  Camera UUID: {camera_uuid}")
    _LOGGER.info(f"  STUN: {STUN_SERVER}")
    _LOGGER.info(f"  TURN: {TURN_SERVER}")
    _LOGGER.info("")

    # Create client and connect
    client = VivintWebRTCClient(config)

    # Enable AI conversation mode if requested
    if ai_conversation:
        _LOGGER.info("AI conversation mode enabled")
        client.enable_ai_conversation_mode()

    try:
        if await client.connect():
            _LOGGER.info("")
            _LOGGER.info("="*60)
            _LOGGER.info("CONNECTION SUCCESSFUL!")
            _LOGGER.info("="*60)

            # Try starting two-way talk
            if await client.start_two_way_talk():
                _LOGGER.info("")
                _LOGGER.info("Two-way talk active!")

                if ai_conversation:
                    # Start AI-powered conversation
                    from doorbell_agent import DoorbellAgent

                    # Load stored credentials (API keys)
                    try:
                        from vivint_client import load_credentials
                        import os
                        creds = load_credentials() or {}
                        if creds.get('gemini_api_key') and not os.getenv('GEMINI_API_KEY'):
                            os.environ['GEMINI_API_KEY'] = creds['gemini_api_key']
                    except Exception as e:
                        _LOGGER.warning(f"Could not load stored credentials: {e}")

                    agent = DoorbellAgent()
                    # Must start conversation (Gemini session) first
                    if not await agent.start_conversation():
                        _LOGGER.error("Failed to start Gemini session")
                    elif await agent.connect_to_webrtc(client):
                        _LOGGER.info("AI conversation active!")
                        _LOGGER.info(f"Test duration: {duration}s (Press Ctrl+C to stop)")

                        try:
                            # Run for specified duration
                            await asyncio.sleep(duration)
                        except KeyboardInterrupt:
                            pass
                        finally:
                            await agent.end_conversation()
                    else:
                        _LOGGER.error("Failed to connect DoorbellAgent")
                else:
                    _LOGGER.info("Press Ctrl+C to stop...")
                    try:
                        # Keep connection alive
                        while client.connected:
                            await asyncio.sleep(1)
                    except KeyboardInterrupt:
                        pass

            await client.stop_two_way_talk()
        else:
            _LOGGER.error("Connection failed")

    finally:
        await client.disconnect()


async def test_sine_wave(duration: int = 10, frequency: int = 880):
    """
    Test TTSAudioTrack by generating a sine wave directly.

    This bypasses Gemini to isolate whether TTSAudioTrack works.
    If you hear the tone, TTSAudioTrack is working and the issue is
    with Gemini audio format. If you don't hear the tone, TTSAudioTrack
    itself has issues.

    Args:
        duration: How long to play the tone in seconds
        frequency: Frequency of sine wave in Hz (default 880 = A5)
    """
    import numpy as np

    _LOGGER.info("="*60)
    _LOGGER.info("TTSAudioTrack Sine Wave Test")
    _LOGGER.info("="*60)
    _LOGGER.info(f"This will generate a {frequency}Hz tone for {duration} seconds")
    _LOGGER.info("through TTSAudioTrack (bypassing Gemini)")
    _LOGGER.info("")

    # Get token and camera from Vivint
    access_token, camera_uuid = await get_vivint_token_and_camera()

    if not access_token:
        _LOGGER.error("Could not get Vivint access token")
        return

    # Create WebRTC config
    config = VivintWebRTCConfig(
        oauth_token=access_token,
        camera_uuid=camera_uuid,
    )

    # Create client with TTSAudioTrack
    client = VivintWebRTCClient(config)
    client.enable_ai_conversation_mode()  # Use TTSAudioTrack

    try:
        if await client.connect():
            _LOGGER.info("Connected!")

            if await client.start_two_way_talk():
                _LOGGER.info("Two-way talk active!")

                # Wait a moment for aiortc to finish its initial buffering
                _LOGGER.info("Waiting 2 seconds for WebRTC to stabilize...")
                await asyncio.sleep(2)

                # Reset TTSAudioTrack timing for fresh playback
                if client.tts_track:
                    client.tts_track.clear_queue()
                    client.tts_track.reset_timing()
                    _LOGGER.info("TTSAudioTrack ready for audio")

                _LOGGER.info("Generating sine wave...")

                # Generate sine wave audio (48kHz, stereo, 16-bit)
                sample_rate = 48000
                num_samples = sample_rate * duration
                t = np.linspace(0, duration, num_samples, dtype=np.float32)

                # Generate sine wave
                amplitude = 0.8  # 80% of max to avoid clipping
                samples = (amplitude * 32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)

                # Convert mono to stereo
                stereo = np.column_stack((samples, samples)).flatten()
                audio_bytes = stereo.astype(np.int16).tobytes()

                _LOGGER.info(f"Generated {len(audio_bytes)} bytes of audio")
                _LOGGER.info(f"Max amplitude: {np.max(np.abs(stereo))}")

                # Queue the audio in chunks (like real audio would arrive)
                chunk_size = 48000 * 4  # 1 second chunks (48kHz * 2 bytes * 2 channels)
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await client.queue_audio(chunk)
                    _LOGGER.info(f"Queued chunk {i // chunk_size + 1} ({len(chunk)} bytes)")

                # Wait for audio to play
                _LOGGER.info(f"Waiting {duration + 2} seconds for audio to play...")
                await asyncio.sleep(duration + 2)

                _LOGGER.info("Test complete! Did you hear the tone?")

            await client.stop_two_way_talk()
        else:
            _LOGGER.error("Connection failed")

    finally:
        await client.disconnect()


async def show_protocol_info():
    """Show discovered protocol information."""
    _LOGGER.info("="*60)
    _LOGGER.info("Vivint WebRTC Protocol - Discovered Configuration")
    _LOGGER.info("="*60)
    _LOGGER.info("")
    _LOGGER.info("1. Token Exchange:")
    _LOGGER.info(f"   URL: {FIREBASE_TOKEN_URL}")
    _LOGGER.info("   Method: GET with Bearer token")
    _LOGGER.info("")
    _LOGGER.info("2. ICE Servers:")
    _LOGGER.info(f"   STUN: {STUN_SERVER}")
    _LOGGER.info(f"   TURN: {TURN_SERVER}")
    _LOGGER.info(f"   Username: {TURN_USERNAME}")
    _LOGGER.info(f"   Password: {TURN_PASSWORD}")
    _LOGGER.info("")
    _LOGGER.info("3. Firebase Signaling:")
    _LOGGER.info("   Path: devices/<camera_uuid>/msg")
    _LOGGER.info("   Messages: SdpMessage, IceMessage")
    _LOGGER.info("")
    _LOGGER.info("4. DataChannel Protocol:")
    _LOGGER.info("   Field 1: Ping")
    _LOGGER.info("   Field 2: Pong")
    _LOGGER.info("   Field 9: TwoWayTalkStart")
    _LOGGER.info("   Field 11: TwoWayTalkEnd")
    _LOGGER.info("")
    _LOGGER.info("See docs/VIVINT_WEBRTC_PROTOCOL.md for full documentation")
    _LOGGER.info("="*60)


def list_audio_devices():
    """List available audio input devices on Windows."""
    import platform
    import subprocess

    _LOGGER.info("="*60)
    _LOGGER.info("Available Audio Input Devices")
    _LOGGER.info("="*60)

    if platform.system() == "Windows":
        try:
            # Use ffmpeg to list DirectShow audio devices
            result = subprocess.run(
                ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # ffmpeg outputs to stderr
            output = result.stderr
            _LOGGER.info("")
            _LOGGER.info("DirectShow Audio Devices:")
            _LOGGER.info("-" * 40)

            in_audio_section = False
            for line in output.split('\n'):
                if 'DirectShow audio devices' in line:
                    in_audio_section = True
                    continue
                if in_audio_section:
                    if 'Alternative name' in line:
                        continue
                    if '"' in line:
                        # Extract device name
                        start = line.find('"') + 1
                        end = line.rfind('"')
                        if start > 0 and end > start:
                            device_name = line[start:end]
                            _LOGGER.info(f"  {device_name}")
                    elif 'DirectShow video devices' in line or line.strip() == '':
                        break

            _LOGGER.info("")
            _LOGGER.info("Usage:")
            _LOGGER.info('  python vivint_webrtc.py --audio "Microphone (Device Name)"')

        except FileNotFoundError:
            _LOGGER.error("ffmpeg not found. Install ffmpeg to list audio devices.")
            _LOGGER.info("Or use Windows Sound Settings to find your microphone name.")
        except Exception as e:
            _LOGGER.error(f"Error listing devices: {e}")

    elif platform.system() == "Linux":
        _LOGGER.info("Linux audio devices (PulseAudio):")
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.split('\n'):
                if line.strip():
                    _LOGGER.info(f"  {line}")
        except Exception as e:
            _LOGGER.error(f"Error listing devices: {e}")

    elif platform.system() == "Darwin":
        _LOGGER.info("macOS: Use 'none:0' for default audio input")

    _LOGGER.info("="*60)


async def speak_through_doorbell(text: str, oauth_token: str = None, camera_uuid: str = None,
                                  voice_id: str = None,
                                  prefetched_credentials: 'PrefetchedCredentials' = None) -> bool:
    """
    Speak text through the doorbell using Eleven Labs Turbo v2.5 TTS.

    This generates speech from text and sends it through WebRTC to the doorbell.
    Requires ELEVEN_LABS_API_KEY environment variable to be set.

    Args:
        text: Text to speak
        oauth_token: Vivint OAuth token (fetched automatically if not provided)
        camera_uuid: Camera UUID (fetched automatically if not provided)
        voice_id: Eleven Labs voice ID (default: Jarnathan)
        prefetched_credentials: Pre-fetched Firebase credentials for faster connection

    Returns:
        True if successful, False otherwise

    Example:
        await speak_through_doorbell("Hello! I'll be right there.")
    """
    # Get credentials if not provided
    if not oauth_token or not camera_uuid:
        oauth_token, camera_uuid = await get_vivint_token_and_camera()
        if not oauth_token or not camera_uuid:
            _LOGGER.error("Could not get Vivint credentials for TTS")
            return False

    # Generate TTS audio using Cartesia
    tts = TTSEngine(voice_id=voice_id)
    try:
        audio_file = await tts.generate_speech(text)
        _LOGGER.info(f"TTS audio generated: {audio_file}")

        # Build config
        config_kwargs = {
            'oauth_token': oauth_token,
            'camera_uuid': camera_uuid,
            'tts_audio_file': audio_file,
        }

        if prefetched_credentials:
            config_kwargs.update({
                'prefetched_firebase_token': prefetched_credentials.firebase_token,
                'prefetched_firebase_id_token': prefetched_credentials.firebase_id_token,
                'prefetched_firebase_uid': prefetched_credentials.firebase_uid,
                'firebase_db_url': prefetched_credentials.firebase_db_url,
                'system_id': prefetched_credentials.system_id,
            })

        config = VivintWebRTCConfig(**config_kwargs)

        # Connect and speak
        client = VivintWebRTCClient(config)

        try:
            if await client.connect():
                _LOGGER.info("TTS: WebRTC connected")

                if await client.start_two_way_talk():
                    _LOGGER.info(f"TTS: Speaking: '{text}'")

                    # Wait for audio to play (estimate based on text length)
                    # Approximate speaking rate: ~150 words per minute
                    word_count = len(text.split())
                    duration = max(2.0, word_count / 2.5)  # At least 2 seconds
                    _LOGGER.info(f"TTS: Waiting {duration:.1f}s for audio to play...")
                    await asyncio.sleep(duration)

                    await client.stop_two_way_talk()
                    _LOGGER.info("TTS: Done speaking")
                    return True
                else:
                    _LOGGER.error("TTS: Failed to start two-way talk")
            else:
                _LOGGER.error("TTS: Failed to connect")

        finally:
            await client.disconnect()

    finally:
        tts.cleanup()

    return False


async def test_tts(text: str = "Hello! This is a test of the Vivint doorbell text to speech system."):
    """Test TTS through the doorbell."""
    _LOGGER.info("="*60)
    _LOGGER.info("Vivint WebRTC TTS Test")
    _LOGGER.info("="*60)
    _LOGGER.info(f"Text: {text}")
    _LOGGER.info("")

    success = await speak_through_doorbell(text)

    if success:
        _LOGGER.info("TTS test successful!")
    else:
        _LOGGER.error("TTS test failed!")


if __name__ == "__main__":
    import sys

    if "--info" in sys.argv:
        asyncio.run(show_protocol_info())
    elif "--list-audio" in sys.argv:
        list_audio_devices()
    elif "--tts" in sys.argv:
        # TTS mode: speak text through doorbell
        # Usage: python vivint_webrtc.py --tts "Hello, I'll be right there!"
        text = None
        for i, arg in enumerate(sys.argv):
            if arg == "--tts" and i + 1 < len(sys.argv):
                text = sys.argv[i + 1]
                break
        if text:
            asyncio.run(test_tts(text))
        else:
            asyncio.run(test_tts())  # Use default text
    elif "--sine-test" in sys.argv:
        # Sine wave test: bypass Gemini and test TTSAudioTrack directly
        # Usage: python vivint_webrtc.py --sine-test
        duration = 10
        frequency = 880
        for i, arg in enumerate(sys.argv):
            if arg == "--duration" and i + 1 < len(sys.argv):
                try:
                    duration = int(sys.argv[i + 1])
                except ValueError:
                    pass
            elif arg == "--frequency" and i + 1 < len(sys.argv):
                try:
                    frequency = int(sys.argv[i + 1])
                except ValueError:
                    pass
        asyncio.run(test_sine_wave(duration=duration, frequency=frequency))
    else:
        # Parse arguments
        audio_device = None
        ai_conversation = "--ai-conversation" in sys.argv
        duration = 60  # Default duration

        for i, arg in enumerate(sys.argv):
            if arg == "--audio" and i + 1 < len(sys.argv):
                audio_device = sys.argv[i + 1]
            elif arg == "--duration" and i + 1 < len(sys.argv):
                try:
                    duration = int(sys.argv[i + 1])
                except ValueError:
                    pass

        if audio_device:
            _LOGGER.info(f"Using audio device: {audio_device}")

        asyncio.run(test_connection(
            audio_device=audio_device,
            ai_conversation=ai_conversation,
            duration=duration
        ))
