"""
Two-Way Audio PoC for Vivint Doorbell

Uses SIP protocol to send audio TO the doorbell while receiving audio via RTSP.
Based on research from:
- https://github.com/tommyjlong/app_rtsp_sip
- https://github.com/tommyjlong/doorvivint-card

Vivint/Vivotek doorbells use:
- SIP on port 5060 for sending audio TO the doorbell
- RTSP for receiving audio FROM the doorbell (already implemented in live_view.py)
- Digest authentication with realm "streaming_server"

Usage:
    python two_way_audio.py              # Interactive mode
    python two_way_audio.py --test       # Test SIP connection only

Requirements:
    pip install pyaudio
"""

import asyncio
import hashlib
import random
import socket
import struct
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
_LOGGER = logging.getLogger(__name__)

# Try to import PyAudio
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    _LOGGER.warning("PyAudio not installed - microphone capture disabled")


@dataclass
class SIPConfig:
    """SIP configuration for Vivint doorbell."""
    doorbell_ip: str
    sip_port: int = 5060
    username: str = "admin"  # Default Vivint/Vivotek username
    password: str = ""       # Panel credentials
    realm: str = "streaming_server"  # Vivotek default realm
    local_ip: str = ""
    local_rtp_port: int = 10000


class SIPClient:
    """
    Minimal SIP client for two-way audio with Vivint doorbell.

    Implements basic SIP INVITE flow with Digest authentication.
    """

    def __init__(self, config: SIPConfig):
        self.config = config
        self.socket: Optional[socket.socket] = None
        self.call_id = f"{random.randint(100000, 999999)}@{config.local_ip}"
        self.cseq = 1
        self.tag = f"{random.randint(100000, 999999)}"
        self.branch = f"z9hG4bK{random.randint(100000, 999999)}"
        self.remote_rtp_port: Optional[int] = None
        self.session_active = False

    def _get_local_ip(self) -> str:
        """Get local IP address that can reach the doorbell on the same subnet."""
        if self.config.local_ip:
            _LOGGER.info(f"Using configured local IP: {self.config.local_ip}")
            return self.config.local_ip

        doorbell_ip = self.config.doorbell_ip
        doorbell_prefix = ".".join(doorbell_ip.split(".")[:3])  # e.g., "192.168.8"

        # Collect all local IPs
        all_ips = []
        try:
            import socket as sock
            hostname = sock.gethostname()
            all_ips = sock.gethostbyname_ex(hostname)[2]
        except Exception as e:
            _LOGGER.debug(f"Could not enumerate interfaces: {e}")

        # Priority 1: Same subnet as doorbell
        for ip in all_ips:
            if ip.startswith(doorbell_prefix):
                _LOGGER.info(f"Found local IP on same subnet as doorbell: {ip}")
                return ip

        # Priority 2: Any 192.168.x.x IP (private LAN, not Tailscale)
        for ip in all_ips:
            if ip.startswith("192.168.") and not ip.startswith("192.168.56."):  # Skip VirtualBox
                _LOGGER.info(f"Using private LAN IP: {ip}")
                return ip

        # Priority 3: Any 10.x.x.x IP (private)
        for ip in all_ips:
            if ip.startswith("10.") and not ip.startswith("100."):  # Skip Tailscale 100.x
                _LOGGER.info(f"Using private 10.x IP: {ip}")
                return ip

        # Fallback: Create a socket to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((doorbell_ip, self.config.sip_port))
            detected_ip = s.getsockname()[0]

            # Warn if detected IP is Tailscale
            if detected_ip.startswith("100."):
                _LOGGER.warning(f"Detected IP {detected_ip} appears to be Tailscale")
                _LOGGER.warning("SIP/RTP may fail - doorbell can't route to Tailscale IPs")
                _LOGGER.warning("Set TWO_WAY_LOCAL_IP in .env to your LAN IP (e.g., 192.168.1.150)")

            return detected_ip
        finally:
            s.close()

    def _compute_digest_response(self, nonce: str, method: str, uri: str) -> str:
        """Compute Digest authentication response (RFC 2617)."""
        # HA1 = MD5(username:realm:password)
        ha1 = hashlib.md5(
            f"{self.config.username}:{self.config.realm}:{self.config.password}".encode()
        ).hexdigest()

        # HA2 = MD5(method:uri)
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()

        # Response = MD5(HA1:nonce:HA2)
        response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()

        return response

    def _build_invite(self, auth_header: str = "") -> str:
        """Build SIP INVITE message."""
        local_ip = self._get_local_ip()
        doorbell_ip = self.config.doorbell_ip
        sip_port = self.config.sip_port
        rtp_port = self.config.local_rtp_port

        # SDP body for audio session
        sdp = (
            f"v=0\r\n"
            f"o=- {int(time.time())} {int(time.time())} IN IP4 {local_ip}\r\n"
            f"s=Two-Way Audio\r\n"
            f"c=IN IP4 {local_ip}\r\n"
            f"t=0 0\r\n"
            f"m=audio {rtp_port} RTP/AVP 0 8\r\n"  # PCMU (0) and PCMA (8)
            f"a=rtpmap:0 PCMU/8000\r\n"
            f"a=rtpmap:8 PCMA/8000\r\n"
            f"a=sendrecv\r\n"
        )

        # SIP headers
        invite = (
            f"INVITE sip:{self.config.username}@{doorbell_ip}:{sip_port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:5060;branch={self.branch};rport\r\n"
            f"From: <sip:user@{local_ip}>;tag={self.tag}\r\n"
            f"To: <sip:{self.config.username}@{doorbell_ip}:{sip_port}>\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: {self.cseq} INVITE\r\n"
            f"Contact: <sip:user@{local_ip}:5060>\r\n"
            f"Max-Forwards: 70\r\n"
            f"User-Agent: VivintTwoWayAudio/1.0\r\n"
            f"Content-Type: application/sdp\r\n"
        )

        if auth_header:
            invite += f"{auth_header}\r\n"

        invite += f"Content-Length: {len(sdp)}\r\n\r\n{sdp}"

        return invite

    def _build_ack(self, to_tag: str = "") -> str:
        """Build SIP ACK message."""
        local_ip = self._get_local_ip()
        doorbell_ip = self.config.doorbell_ip
        sip_port = self.config.sip_port

        to_header = f"<sip:{self.config.username}@{doorbell_ip}:{sip_port}>"
        if to_tag:
            to_header += f";tag={to_tag}"

        ack = (
            f"ACK sip:{self.config.username}@{doorbell_ip}:{sip_port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:5060;branch={self.branch};rport\r\n"
            f"From: <sip:user@{local_ip}>;tag={self.tag}\r\n"
            f"To: {to_header}\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: {self.cseq} ACK\r\n"
            f"Max-Forwards: 70\r\n"
            f"Content-Length: 0\r\n\r\n"
        )

        return ack

    def _build_bye(self, to_tag: str = "") -> str:
        """Build SIP BYE message."""
        local_ip = self._get_local_ip()
        doorbell_ip = self.config.doorbell_ip
        sip_port = self.config.sip_port

        self.cseq += 1

        to_header = f"<sip:{self.config.username}@{doorbell_ip}:{sip_port}>"
        if to_tag:
            to_header += f";tag={to_tag}"

        bye = (
            f"BYE sip:{self.config.username}@{doorbell_ip}:{sip_port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:5060;branch=z9hG4bK{random.randint(100000, 999999)};rport\r\n"
            f"From: <sip:user@{local_ip}>;tag={self.tag}\r\n"
            f"To: {to_header}\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: {self.cseq} BYE\r\n"
            f"Max-Forwards: 70\r\n"
            f"Content-Length: 0\r\n\r\n"
        )

        return bye

    def _parse_response(self, data: str) -> dict:
        """Parse SIP response."""
        result = {
            "status_code": 0,
            "status_text": "",
            "headers": {},
            "body": "",
        }

        lines = data.split("\r\n")
        if not lines:
            return result

        # Parse status line
        status_line = lines[0].split(" ", 2)
        if len(status_line) >= 2:
            result["status_code"] = int(status_line[1])
            result["status_text"] = status_line[2] if len(status_line) > 2 else ""

        # Parse headers
        body_start = 0
        for i, line in enumerate(lines[1:], 1):
            if line == "":
                body_start = i + 1
                break
            if ":" in line:
                key, value = line.split(":", 1)
                result["headers"][key.strip().lower()] = value.strip()

        # Parse body
        if body_start < len(lines):
            result["body"] = "\r\n".join(lines[body_start:])

        return result

    def _parse_sdp_rtp_port(self, sdp: str) -> Optional[int]:
        """Extract RTP port from SDP body."""
        for line in sdp.split("\r\n"):
            if line.startswith("m=audio "):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
        return None

    def _extract_nonce(self, www_auth: str) -> Optional[str]:
        """Extract nonce from WWW-Authenticate header."""
        import re
        match = re.search(r'nonce="([^"]+)"', www_auth)
        return match.group(1) if match else None

    def _extract_to_tag(self, to_header: str) -> Optional[str]:
        """Extract tag from To header."""
        import re
        match = re.search(r'tag=([^;>\s]+)', to_header)
        return match.group(1) if match else None

    def connect(self) -> bool:
        """Establish SIP connection to doorbell."""
        local_ip = self._get_local_ip()
        self.config.local_ip = local_ip

        _LOGGER.info(f"Connecting to doorbell at {self.config.doorbell_ip}:{self.config.sip_port}")
        _LOGGER.info(f"Local IP: {local_ip}, RTP port: {self.config.local_rtp_port}")

        # Create UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(5.0)
        self.socket.bind((local_ip, 5060))

        try:
            # Send initial INVITE (no auth)
            invite = self._build_invite()
            _LOGGER.debug(f"Sending INVITE:\n{invite}")
            self.socket.sendto(
                invite.encode(),
                (self.config.doorbell_ip, self.config.sip_port)
            )

            # Wait for response
            data, addr = self.socket.recvfrom(4096)
            response = self._parse_response(data.decode())
            _LOGGER.info(f"Response: {response['status_code']} {response['status_text']}")

            # Handle 401 Unauthorized (Digest auth required)
            if response["status_code"] == 401:
                www_auth = response["headers"].get("www-authenticate", "")
                nonce = self._extract_nonce(www_auth)

                if not nonce:
                    _LOGGER.error("Could not extract nonce from WWW-Authenticate header")
                    return False

                _LOGGER.info("Received 401, computing Digest auth...")

                # Compute digest response
                uri = f"sip:{self.config.username}@{self.config.doorbell_ip}:{self.config.sip_port}"
                digest_response = self._compute_digest_response(nonce, "INVITE", uri)

                # Build Authorization header
                auth_header = (
                    f'Authorization: Digest username="{self.config.username}", '
                    f'realm="{self.config.realm}", '
                    f'nonce="{nonce}", '
                    f'uri="{uri}", '
                    f'response="{digest_response}"'
                )

                # Send authenticated INVITE
                self.cseq += 1
                self.branch = f"z9hG4bK{random.randint(100000, 999999)}"
                invite = self._build_invite(auth_header)
                _LOGGER.debug(f"Sending authenticated INVITE")
                self.socket.sendto(
                    invite.encode(),
                    (self.config.doorbell_ip, self.config.sip_port)
                )

                # Wait for response
                data, addr = self.socket.recvfrom(4096)
                response = self._parse_response(data.decode())
                _LOGGER.info(f"Response: {response['status_code']} {response['status_text']}")

            # Handle provisional responses (100 Trying, 180 Ringing)
            while response["status_code"] in [100, 180, 183]:
                _LOGGER.info(f"Provisional: {response['status_code']} {response['status_text']}")
                data, addr = self.socket.recvfrom(4096)
                response = self._parse_response(data.decode())
                _LOGGER.info(f"Response: {response['status_code']} {response['status_text']}")

            # Check for 200 OK
            if response["status_code"] == 200:
                _LOGGER.info("SIP call established!")

                # Extract RTP port from SDP
                self.remote_rtp_port = self._parse_sdp_rtp_port(response["body"])
                _LOGGER.info(f"Remote RTP port: {self.remote_rtp_port}")

                # Extract To tag for ACK
                to_tag = self._extract_to_tag(response["headers"].get("to", ""))

                # Send ACK
                ack = self._build_ack(to_tag)
                self.socket.sendto(
                    ack.encode(),
                    (self.config.doorbell_ip, self.config.sip_port)
                )
                _LOGGER.info("ACK sent, call is active")

                self.session_active = True
                return True
            else:
                _LOGGER.error(f"Call failed: {response['status_code']} {response['status_text']}")
                return False

        except socket.timeout:
            _LOGGER.error("Timeout waiting for SIP response")
            return False
        except Exception as e:
            _LOGGER.error(f"SIP error: {e}")
            return False

    def disconnect(self):
        """End SIP session."""
        if self.session_active and self.socket:
            _LOGGER.info("Sending BYE to end call")
            bye = self._build_bye()
            try:
                self.socket.sendto(
                    bye.encode(),
                    (self.config.doorbell_ip, self.config.sip_port)
                )
            except Exception as e:
                _LOGGER.warning(f"Error sending BYE: {e}")

        if self.socket:
            self.socket.close()
            self.socket = None

        self.session_active = False


class RTPAudioSender:
    """Send audio to doorbell via RTP."""

    def __init__(self, target_ip: str, target_port: int, local_port: int):
        self.target_ip = target_ip
        self.target_port = target_port
        self.local_port = local_port
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.sequence = random.randint(0, 65535)
        self.timestamp = random.randint(0, 2**31)
        self.ssrc = random.randint(0, 2**31)

    def _build_rtp_packet(self, payload: bytes, payload_type: int = 0) -> bytes:
        """Build RTP packet."""
        # RTP header (12 bytes)
        # V=2, P=0, X=0, CC=0, M=0, PT=payload_type
        header = bytes([
            0x80,  # V=2, P=0, X=0, CC=0
            payload_type,  # M=0, PT
        ])
        header += struct.pack(">H", self.sequence)
        header += struct.pack(">I", self.timestamp)
        header += struct.pack(">I", self.ssrc)

        self.sequence = (self.sequence + 1) & 0xFFFF
        self.timestamp += 160  # 20ms at 8000Hz

        return header + payload

    def start(self):
        """Start RTP sender."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("0.0.0.0", self.local_port))
        self.running = True
        _LOGGER.info(f"RTP sender started on port {self.local_port} -> {self.target_ip}:{self.target_port}")

    def send_audio(self, audio_data: bytes, payload_type: int = 0):
        """Send audio data as RTP packet."""
        if not self.running or not self.socket:
            return

        packet = self._build_rtp_packet(audio_data, payload_type)
        self.socket.sendto(packet, (self.target_ip, self.target_port))

    def stop(self):
        """Stop RTP sender."""
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None


class MicrophoneCapture:
    """Capture audio from microphone."""

    def __init__(self, sample_rate: int = 8000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 160  # 20ms at 8000Hz
        self.pyaudio: Optional[pyaudio.PyAudio] = None
        self.stream = None
        self.running = False

    def start(self):
        """Start microphone capture."""
        if not HAS_PYAUDIO:
            _LOGGER.error("PyAudio not available")
            return False

        self.pyaudio = pyaudio.PyAudio()

        try:
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            self.running = True
            _LOGGER.info(f"Microphone capture started ({self.sample_rate}Hz, {self.channels}ch)")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to open microphone: {e}")
            return False

    def read(self) -> Optional[bytes]:
        """Read audio chunk from microphone."""
        if not self.running or not self.stream:
            return None

        try:
            return self.stream.read(self.chunk_size, exception_on_overflow=False)
        except Exception as e:
            _LOGGER.warning(f"Microphone read error: {e}")
            return None

    def stop(self):
        """Stop microphone capture."""
        self.running = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None


def pcm_to_ulaw(pcm_sample: int) -> int:
    """Convert 16-bit PCM sample to u-law."""
    BIAS = 0x84
    CLIP = 32635

    sign = (pcm_sample >> 8) & 0x80
    if sign:
        pcm_sample = -pcm_sample

    if pcm_sample > CLIP:
        pcm_sample = CLIP

    pcm_sample += BIAS

    # Find segment
    segment = 7
    mask = 0x4000
    while segment > 0 and not (pcm_sample & mask):
        segment -= 1
        mask >>= 1

    # Combine sign, segment, and quantization
    ulaw = sign | (segment << 4) | ((pcm_sample >> (segment + 3)) & 0x0F)
    return ulaw ^ 0xFF


def convert_pcm_to_ulaw(pcm_data: bytes) -> bytes:
    """Convert PCM audio to u-law encoding."""
    result = []
    for i in range(0, len(pcm_data), 2):
        sample = struct.unpack("<h", pcm_data[i:i+2])[0]
        result.append(pcm_to_ulaw(sample))
    return bytes(result)


class TwoWayAudio:
    """
    Two-way audio session with Vivint doorbell.

    Combines SIP signaling with RTP audio streaming.
    """

    def __init__(self, doorbell_ip: str, username: str, password: str, local_ip: str = None):
        self.config = SIPConfig(
            doorbell_ip=doorbell_ip,
            username=username,
            password=password,
            local_ip=local_ip or "",
        )
        self.sip_client: Optional[SIPClient] = None
        self.rtp_sender: Optional[RTPAudioSender] = None
        self.mic_capture: Optional[MicrophoneCapture] = None
        self.running = False
        self.audio_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start two-way audio session."""
        _LOGGER.info("Starting two-way audio session...")

        # Initialize SIP client
        self.sip_client = SIPClient(self.config)

        # Connect via SIP
        if not self.sip_client.connect():
            _LOGGER.error("SIP connection failed")
            return False

        if not self.sip_client.remote_rtp_port:
            _LOGGER.error("No remote RTP port from SDP")
            return False

        # Start RTP sender
        self.rtp_sender = RTPAudioSender(
            self.config.doorbell_ip,
            self.sip_client.remote_rtp_port,
            self.config.local_rtp_port,
        )
        self.rtp_sender.start()

        # Start microphone capture
        self.mic_capture = MicrophoneCapture()
        if not self.mic_capture.start():
            _LOGGER.warning("Microphone not available, audio output only")

        # Start audio streaming thread
        self.running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()

        _LOGGER.info("Two-way audio session active!")
        _LOGGER.info("Speak into your microphone - audio will be sent to the doorbell")
        return True

    def _audio_loop(self):
        """Audio streaming loop."""
        while self.running:
            if self.mic_capture and self.mic_capture.running:
                pcm_data = self.mic_capture.read()
                if pcm_data:
                    # Convert PCM to u-law and send
                    ulaw_data = convert_pcm_to_ulaw(pcm_data)
                    if self.rtp_sender:
                        self.rtp_sender.send_audio(ulaw_data, payload_type=0)  # PCMU
            else:
                time.sleep(0.02)  # 20ms

    def stop(self):
        """Stop two-way audio session."""
        _LOGGER.info("Stopping two-way audio session...")
        self.running = False

        if self.audio_thread:
            self.audio_thread.join(timeout=2)

        if self.mic_capture:
            self.mic_capture.stop()

        if self.rtp_sender:
            self.rtp_sender.stop()

        if self.sip_client:
            self.sip_client.disconnect()

        _LOGGER.info("Two-way audio session ended")


async def get_doorbell_credentials():
    """Get doorbell IP and credentials from Vivint."""
    import sys
    sys.path.insert(0, ".")

    from vivint_client import VivintClient
    import config

    _LOGGER.info("Connecting to Vivint to get doorbell credentials...")

    client = VivintClient()
    if not await client.connect():
        return None, None, None

    # Find doorbell
    doorbell_ip = config.VIVINT_HUB_IP
    username = None
    password = None

    # Get panel credentials (same as RTSP)
    for system in client.client.account.systems:
        panel = system.alarm_panel
        if hasattr(panel, 'credentials'):
            creds = panel.credentials
            if creds:
                username = creds.get('name', 'admin')
                password = creds.get('password', '')
                _LOGGER.info(f"Got credentials for panel")
                break

    await client.disconnect()
    return doorbell_ip, username, password


async def main():
    import sys

    test_only = "--test" in sys.argv

    # Get credentials
    doorbell_ip, username, password = await get_doorbell_credentials()

    if not doorbell_ip or not username:
        _LOGGER.error("Could not get doorbell credentials")
        return

    _LOGGER.info(f"Doorbell IP: {doorbell_ip}")
    _LOGGER.info(f"Username: {username}")

    if test_only:
        # Just test SIP connection
        config = SIPConfig(
            doorbell_ip=doorbell_ip,
            username=username,
            password=password,
        )
        sip = SIPClient(config)
        success = sip.connect()
        if success:
            _LOGGER.info("SIP connection test PASSED")
            time.sleep(2)
            sip.disconnect()
        else:
            _LOGGER.error("SIP connection test FAILED")
        return

    # Full two-way audio session
    twa = TwoWayAudio(doorbell_ip, username, password)

    if not twa.start():
        return

    print("\n" + "="*50)
    print("TWO-WAY AUDIO ACTIVE")
    print("="*50)
    print("Speak into your microphone to send audio to doorbell")
    print("Press Enter to stop...")
    print("="*50 + "\n")

    try:
        input()
    except KeyboardInterrupt:
        pass

    twa.stop()


if __name__ == "__main__":
    asyncio.run(main())
