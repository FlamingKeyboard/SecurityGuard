"""
AI Doorbell Agent - Real-time conversational AI using Gemini Live API.

This module enables the doorbell to have intelligent, context-aware conversations
with visitors using:
- Gemini Live API for real-time video/audio understanding
- ElevenLabs TTS for natural voice responses
- WebRTC for bidirectional audio with the doorbell
- Pushover notifications for alerting homeowner

The agent can:
- Decide when to speak and when to stay silent
- Have multi-turn conversations with visitors
- Handle deliveries, solicitors, guests, and security situations
- De-escalate threats and gather information
- Send notifications to homeowner via Pushover
- Trigger urgent alerts for critical situations
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any

import aiohttp

_LOGGER = logging.getLogger(__name__)

# Gemini Live API model with native audio output
# Native audio model for real-time voice conversation
# See: https://ai.google.dev/gemini-api/docs/live
GEMINI_LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

# Pushover credentials (loaded from environment or vivint_client)
_pushover_token: str | None = None
_pushover_user: str | None = None


def load_pushover_credentials():
    """Load Pushover credentials from stored credentials or environment."""
    global _pushover_token, _pushover_user

    # Try environment first
    _pushover_token = os.environ.get("PUSHOVER_TOKEN")
    _pushover_user = os.environ.get("PUSHOVER_USER")

    # If not in env, try stored credentials
    if not _pushover_token or not _pushover_user:
        try:
            from vivint_client import load_credentials
            creds = load_credentials() or {}
            _pushover_token = _pushover_token or creds.get("pushover_token")
            _pushover_user = _pushover_user or creds.get("pushover_user")
        except Exception as e:
            _LOGGER.debug(f"Could not load stored credentials: {e}")


async def send_pushover_notification(
    title: str,
    message: str,
    priority: int = 0,
    sound: str = "pushover",
    image_data: bytes = None,
) -> dict[str, Any]:
    """
    Send a push notification via Pushover with optional image.

    Args:
        title: Notification title
        message: Notification body
        priority: -2 (silent) to 2 (emergency). 0=normal, 1=high, 2=emergency
        sound: Notification sound (default "pushover")
        image_data: Optional JPEG image bytes to attach

    Returns:
        Dict with success status and message
    """
    if not _pushover_token or not _pushover_user:
        return {"success": False, "error": "Pushover not configured"}

    url = "https://api.pushover.net/1/messages.json"

    try:
        data = aiohttp.FormData()
        data.add_field("token", _pushover_token)
        data.add_field("user", _pushover_user)
        data.add_field("title", title)
        data.add_field("message", message)
        data.add_field("priority", str(priority))
        data.add_field("sound", sound)

        # Emergency priority (2) requires retry and expire parameters
        if priority == 2:
            data.add_field("retry", "60")    # Retry every 60 seconds
            data.add_field("expire", "3600")  # Expire after 1 hour

        # Attach image if provided
        if image_data:
            data.add_field(
                "attachment",
                image_data,
                filename="doorbell.jpg",
                content_type="image/jpeg"
            )
            _LOGGER.debug(f"Attaching {len(image_data)} bytes image to notification")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                if resp.status == 200:
                    _LOGGER.info(f"Pushover notification sent: {title}")
                    return {"success": True, "message": "Notification sent with image" if image_data else "Notification sent"}
                else:
                    body = await resp.text()
                    _LOGGER.error(f"Pushover failed: {resp.status} - {body}")
                    return {"success": False, "error": f"HTTP {resp.status}: {body}"}

    except Exception as e:
        _LOGGER.error(f"Failed to send Pushover notification: {e}")
        return {"success": False, "error": str(e)}


# Global to store the most recent frame for notifications
_latest_frame: bytes | None = None


def set_latest_frame(frame_data: bytes):
    """Store the most recent camera frame for use in notifications."""
    global _latest_frame
    _latest_frame = frame_data


def get_latest_frame() -> bytes | None:
    """Get the most recent camera frame."""
    return _latest_frame


def resample_audio(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """
    Resample PCM audio from one sample rate to another.

    Args:
        audio_data: 16-bit PCM audio bytes (mono)
        from_rate: Source sample rate (e.g., 24000)
        to_rate: Target sample rate (e.g., 16000)

    Returns:
        Resampled audio bytes (mono)
    """
    if from_rate == to_rate:
        return audio_data

    try:
        import numpy as np

        # Convert bytes to numpy array (16-bit signed PCM)
        samples = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate new length
        new_length = int(len(samples) * to_rate / from_rate)

        # Simple linear interpolation resampling
        old_indices = np.arange(len(samples))
        new_indices = np.linspace(0, len(samples) - 1, new_length)
        resampled = np.interp(new_indices, old_indices, samples.astype(np.float32))

        # Convert back to int16 bytes
        return resampled.astype(np.int16).tobytes()

    except ImportError:
        _LOGGER.warning("numpy not available for resampling, returning original audio")
        return audio_data


def convert_audio_for_webrtc(audio_data: bytes, from_rate: int = 24000) -> bytes:
    """
    Convert Gemini audio (24kHz mono) to WebRTC format (48kHz stereo).

    Args:
        audio_data: 16-bit PCM audio bytes from Gemini (24kHz mono)
        from_rate: Source sample rate (default 24000 for Gemini)

    Returns:
        48kHz stereo PCM audio bytes for WebRTC
    """
    try:
        import numpy as np

        # Convert bytes to numpy array (16-bit signed PCM, mono)
        samples = np.frombuffer(audio_data, dtype=np.int16)

        # Debug: Check input audio characteristics
        input_max = np.max(np.abs(samples)) if len(samples) > 0 else 0
        input_mean = np.mean(np.abs(samples)) if len(samples) > 0 else 0
        _LOGGER.info(f"Audio conversion input: {len(audio_data)} bytes, {len(samples)} samples, max={input_max}, mean={input_mean:.0f}")

        # Resample from 24kHz to 48kHz (2x)
        # For exact 2x upsampling, we can duplicate each sample
        if from_rate == 24000:
            # Simple 2x upsampling by repeating samples
            resampled = np.repeat(samples, 2)
        else:
            # General resampling
            new_length = int(len(samples) * 48000 / from_rate)
            old_indices = np.arange(len(samples))
            new_indices = np.linspace(0, len(samples) - 1, new_length)
            resampled = np.interp(new_indices, old_indices, samples.astype(np.float32)).astype(np.int16)

        # Convert mono to stereo by duplicating the channel
        stereo = np.column_stack((resampled, resampled)).flatten()

        # Debug: Check output audio characteristics
        output_max = np.max(np.abs(stereo))
        _LOGGER.info(f"Audio conversion output: {len(stereo) * 2} bytes, max={output_max}")

        return stereo.astype(np.int16).tobytes()

    except ImportError:
        _LOGGER.warning("numpy not available for audio conversion")
        return audio_data


# Tool definitions for Gemini function calling
DOORBELL_TOOLS = [
    {
        "name": "send_notification",
        "description": "Send a notification to the homeowner's phone. Use this to inform them about visitors, deliveries, or situations that need their attention but are not urgent.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for the notification (e.g., 'Delivery Arrived', 'Visitor at Door')"
                },
                "message": {
                    "type": "string",
                    "description": "Detailed message about the situation, including what you observed and any relevant context"
                },
                "category": {
                    "type": "string",
                    "enum": ["delivery", "visitor", "service", "solicitor", "other"],
                    "description": "Category of the notification"
                }
            },
            "required": ["title", "message", "category"]
        }
    },
    {
        "name": "send_urgent_alert",
        "description": "Send an URGENT high-priority alert to the homeowner. Use this ONLY for situations requiring immediate attention: suspicious behavior, potential threats, emergencies, or security concerns. This will bypass quiet hours and sound a loud alert.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short urgent title (e.g., 'Security Alert', 'Suspicious Activity')"
                },
                "message": {
                    "type": "string",
                    "description": "Detailed description of the threat or emergency, including what you observed"
                },
                "threat_level": {
                    "type": "string",
                    "enum": ["warning", "threat", "emergency"],
                    "description": "Severity level: warning (suspicious activity), threat (active concern), emergency (immediate danger)"
                }
            },
            "required": ["title", "message", "threat_level"]
        }
    }
]


async def execute_tool(tool_name: str, tool_args: dict) -> dict[str, Any]:
    """
    Execute a tool call from Gemini.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool

    Returns:
        Tool execution result
    """
    _LOGGER.info(f"Executing tool: {tool_name} with args: {tool_args}")

    # Get the latest frame for the notification
    latest_frame = get_latest_frame()

    if tool_name == "send_notification":
        title = tool_args.get("title", "Doorbell Notification")
        message = tool_args.get("message", "")
        category = tool_args.get("category", "other")

        # Add category emoji to title
        emoji_map = {
            "delivery": "📦",
            "visitor": "🚪",
            "service": "🔧",
            "solicitor": "📋",
            "other": "🔔"
        }
        emoji = emoji_map.get(category, "🔔")
        full_title = f"{emoji} {title}"

        return await send_pushover_notification(
            title=full_title,
            message=message,
            priority=0,  # Normal priority
            sound="pushover",
            image_data=latest_frame,
        )

    elif tool_name == "send_urgent_alert":
        title = tool_args.get("title", "Security Alert")
        message = tool_args.get("message", "")
        threat_level = tool_args.get("threat_level", "warning")

        # Map threat level to priority and sound
        priority_map = {
            "warning": (1, "siren"),      # High priority, bypasses quiet hours
            "threat": (2, "alien"),       # Emergency, requires acknowledgment
            "emergency": (2, "persistent") # Emergency with persistent sound
        }
        priority, sound = priority_map.get(threat_level, (1, "siren"))

        # Add threat level emoji
        emoji_map = {
            "warning": "⚠️",
            "threat": "🚨",
            "emergency": "🆘"
        }
        emoji = emoji_map.get(threat_level, "⚠️")
        full_title = f"{emoji} {title}"

        return await send_pushover_notification(
            title=full_title,
            message=message,
            priority=priority,
            sound=sound,
            image_data=latest_frame,
        )

    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}


class ConversationIntent(Enum):
    """What the agent intends to do."""
    SILENT = "silent"           # Don't speak, just observe
    GREET = "greet"             # Initial greeting
    GATHER_INFO = "gather_info" # Ask questions to understand situation
    ASSIST = "assist"           # Help with delivery, directions, etc.
    DECLINE = "decline"         # Politely decline solicitors
    WARN = "warn"               # Warn about recording/monitoring
    DE_ESCALATE = "de_escalate" # Calm a tense situation
    FAREWELL = "farewell"       # End conversation politely


@dataclass
class ConversationState:
    """Tracks the state of an ongoing conversation."""
    session_id: str
    started_at: float = field(default_factory=time.time)
    turn_count: int = 0
    last_activity: float = field(default_factory=time.time)
    person_detected: bool = False
    current_intent: ConversationIntent = ConversationIntent.SILENT
    is_active: bool = True

    # Conversation history (for logging/debugging)
    history: list[dict] = field(default_factory=list)

    def add_turn(self, role: str, content: str):
        """Add a turn to the conversation history."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        self.turn_count += 1
        self.last_activity = time.time()


# System prompt that defines the agent's persona and behavior
DOORBELL_AGENT_SYSTEM_PROMPT = """You are an AI assistant for a home doorbell camera. You can see and hear visitors in real-time through the camera and microphone. You speak through the doorbell's speaker and can send notifications to the homeowner.

## CORE IDENTITY
- You represent the homeowner in a friendly, professional manner
- You are helpful but protective of the homeowner's privacy and security
- You have a calm, warm voice (the actual voice synthesis is handled separately)

## YOUR CAPABILITIES
1. **See and Hear**: Real-time video and audio from the doorbell camera
2. **Speak**: Respond to visitors through the doorbell speaker
3. **Notify**: Send notifications to the homeowner's phone using `send_notification`
4. **Alert**: Send urgent alerts for security concerns using `send_urgent_alert`

## WHEN TO NOTIFY THE HOMEOWNER
Use `send_notification` for:
- Deliveries arriving (category: "delivery")
- Visitors at the door (category: "visitor")
- Service workers arriving (category: "service")
- Solicitors that might need follow-up (category: "solicitor")
- Any situation the homeowner should know about

Use `send_urgent_alert` ONLY for:
- Suspicious behavior or persons (threat_level: "warning")
- Someone trying doors/windows, looking into the house (threat_level: "threat")
- Aggressive or threatening behavior (threat_level: "threat")
- Active emergency or danger (threat_level: "emergency")

## WHEN TO SPEAK
SPEAK when:
- Someone rings the doorbell or knocks
- A delivery person arrives and seems unsure what to do
- Someone is clearly trying to get attention
- You detect a potential security concern that warrants verbal warning
- A visitor has been waiting and seems confused

STAY SILENT when:
- People are just walking by on the sidewalk
- Vehicles passing or parking (unless approaching the door)
- Animals, weather events, lighting changes
- The homeowner or known residents are coming/going
- Someone is leaving after completing their business

## CONVERSATION GUIDELINES

**Deliveries:**
- Thank them warmly
- If they need a signature: "I can confirm receipt. The camera is recording and the homeowner has authorized leaving the package. It's safe here."
- Offer specific placement: "You can leave it right by the door" or "The side porch is covered if you prefer"
- ALWAYS send a notification about the delivery

**Solicitors/Salespeople:**
- Be polite but brief: "Hi there! What can I help you with?"
- After hearing their pitch: "Thanks for stopping by, but we're not interested at this time. Have a great day!"
- If they persist: "I appreciate it, but please remove this address from your list. Thanks!"
- Send a notification so the homeowner knows who stopped by

**Guests/Visitors:**
- Warm greeting: "Hi! Welcome! How can I help you?"
- If homeowner unavailable: "They're not available right now. Would you like to leave a message, or should I let them know you stopped by?"
- Never confirm if homeowner is home or away
- Send a notification about the visitor

**Service Workers (utilities, maintenance):**
- "Hi! Can I help you? Are you here for a scheduled appointment?"
- Verify their purpose before offering assistance
- Send a notification with details

**Suspicious Behavior:**
- Stay calm: "Hello, can I help you with something?"
- If they're looking in windows/trying doors: "Just so you know, this property has 24/7 video monitoring and recording."
- If threatening: "I've notified the homeowner and this interaction is being recorded. I'd recommend leaving the property."
- IMMEDIATELY send an urgent alert with appropriate threat_level

**High-Severity Situations:**
- Remain calm and non-confrontational
- "The authorities have been contacted and are on their way."
- "Please step back from the property. Everything is being recorded."
- Focus on de-escalation, not confrontation
- Send emergency alert immediately

## RESPONSE FORMAT
Keep responses brief and natural - 1-3 sentences typically. This is a doorbell conversation, not a long discussion.

## NOTIFICATION GUIDELINES
When sending notifications:
- Title should be short and descriptive (e.g., "Package Delivered", "Visitor at Door")
- Message should include what you observed and any relevant context
- Include the time and any identifying details (e.g., "Amazon delivery driver", "Man in blue shirt")
- For urgent alerts, be specific about the threat and what you observed

## PRIVACY RULES
NEVER reveal:
- Whether the homeowner is home or away
- The homeowner's schedule or habits
- Names of residents
- Any personal information about the household

ALWAYS:
- Refer to "the homeowner" generically
- Keep focus on helping the visitor with their immediate need
- End conversations gracefully when the purpose is fulfilled

## CONTEXT
You are seeing real-time video and hearing real-time audio from the doorbell camera. Describe what you observe only when relevant to the conversation. Your responses will be spoken aloud through the doorbell speaker."""


class GeminiLiveSession:
    """
    Manages a real-time streaming session with Gemini Live API.

    Uses the official google-genai SDK for reliable connection handling.

    Handles:
    - Connection to Gemini Live API
    - Streaming video/audio input
    - Receiving text responses
    - Session lifecycle
    """

    def __init__(self, api_key: str = None, model: str = GEMINI_LIVE_MODEL, enable_tools: bool = True):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.enable_tools = enable_tools
        self._session = None  # google.genai.live.AsyncSession
        self._client = None   # google.genai.Client
        self._connected = False
        self._response_queue: asyncio.Queue[str] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._listener_task: Optional[asyncio.Task] = None
        self._audio_callback: Optional[Callable] = None

        # Load Pushover credentials if tools are enabled
        if enable_tools:
            load_pushover_credentials()

    @property
    def connected(self) -> bool:
        return self._connected and self._session is not None

    async def connect(self, system_instruction: str = DOORBELL_AGENT_SYSTEM_PROMPT) -> bool:
        """
        Connect to Gemini Live API and set up the session.

        Args:
            system_instruction: The system prompt defining agent behavior

        Returns:
            True if connection successful
        """
        if not self.api_key:
            _LOGGER.error("No Gemini API key found")
            return False

        try:
            from google import genai
            from google.genai import types

            _LOGGER.info(f"Connecting to Gemini Live API (model: {self.model})...")

            # Create client with API key
            self._client = genai.Client(api_key=self.api_key)

            # Configure Voice Activity Detection (VAD) for automatic speech detection
            vad_config = types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=300,
                silence_duration_ms=500,
            )

            realtime_input_config = types.RealtimeInputConfig(
                automatic_activity_detection=vad_config
            )

            # Build tool declarations for function calling
            tools = None
            if self.enable_tools:
                tool_declarations = []
                for tool_def in DOORBELL_TOOLS:
                    func_decl = types.FunctionDeclaration(
                        name=tool_def["name"],
                        description=tool_def["description"],
                        parameters=tool_def["parameters"],
                    )
                    tool_declarations.append(func_decl)

                tools = [types.Tool(function_declarations=tool_declarations)]
                _LOGGER.info(f"Enabled {len(tool_declarations)} tools for function calling")

            # Configure the live session with native audio output
            # Note: output_audio_transcription provides text logs of what the AI says
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Puck"  # Options: Puck, Charon, Kore, Fenrir, Aoede
                        )
                    )
                ),
                output_audio_transcription=types.AudioTranscriptionConfig(),
                system_instruction=system_instruction,
                realtime_input_config=realtime_input_config,
                tools=tools,
            )

            # Connect to Live API - this returns an async context manager
            self._session_context = self._client.aio.live.connect(
                model=self.model,
                config=config
            )

            # Enter the context manager
            self._session = await self._session_context.__aenter__()
            self._connected = True

            _LOGGER.info("Gemini Live session established with VAD enabled")

            # Start listener task
            self._listener_task = asyncio.create_task(self._listen_loop())
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to connect to Gemini Live: {e}")
            import traceback
            traceback.print_exc()
            self._connected = False
            return False

    async def _listen_loop(self):
        """Background task to listen for responses from Gemini."""
        _LOGGER.info("Gemini listener loop started - waiting for responses...")
        response_count = 0
        try:
            # receive() returns an AsyncIterator, so use async for
            async for response in self._session.receive():
                if not self._connected:
                    break

                response_count += 1
                _LOGGER.debug(f"Received response #{response_count} from Gemini: {type(response).__name__}")

                # Process the response
                await self._handle_response(response)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if "closed" not in str(e).lower() and "cancel" not in str(e).lower():
                _LOGGER.error(f"Listener loop error: {e}")
        finally:
            self._connected = False
            _LOGGER.info(f"Gemini listener loop ended after {response_count} responses")

    async def _handle_response(self, response):
        """Handle a response message from Gemini."""
        try:
            # Log response type for debugging
            resp_type = type(response).__name__
            has_server_content = hasattr(response, 'server_content') and response.server_content
            has_tool_call = hasattr(response, 'tool_call') and response.tool_call
            _LOGGER.info(f"Processing Gemini response: {resp_type} (server_content={has_server_content}, tool_call={has_tool_call})")

            # The response object has different attributes based on content
            # Check for server content (model's response)
            if hasattr(response, 'server_content') and response.server_content:
                content = response.server_content

                # Check if model is done with this turn
                if hasattr(content, 'turn_complete') and content.turn_complete:
                    _LOGGER.debug("Turn complete")
                    return

                # Extract text or audio from model turn
                if hasattr(content, 'model_turn') and content.model_turn:
                    model_turn = content.model_turn
                    if hasattr(model_turn, 'parts'):
                        for part in model_turn.parts:
                            # Handle text responses
                            if hasattr(part, 'text') and part.text:
                                text = part.text
                                _LOGGER.info(f"Gemini response: {text}")
                                await self._response_queue.put(text)
                            # Handle audio responses from Gemini (native audio output)
                            elif hasattr(part, 'inline_data') and part.inline_data:
                                audio_data = part.inline_data.data
                                if audio_data:
                                    _LOGGER.info(f"Received {len(audio_data)} bytes of Gemini audio")
                                    await self._audio_queue.put(audio_data)
                                    # Call audio callback if set
                                    if self._audio_callback:
                                        try:
                                            await self._audio_callback(audio_data)
                                        except Exception as e:
                                            _LOGGER.error(f"Audio callback error: {e}")

                # Handle output audio transcription (what the AI said via voice)
                if hasattr(content, 'output_transcription') and content.output_transcription:
                    transcript = content.output_transcription
                    if hasattr(transcript, 'text') and transcript.text:
                        _LOGGER.info(f"AI said (transcript): {transcript.text}")

            # Handle tool calls from the model
            elif hasattr(response, 'tool_call') and response.tool_call:
                await self._handle_tool_call(response.tool_call)

            # Also check for text attribute directly
            elif hasattr(response, 'text') and response.text:
                _LOGGER.info(f"Gemini response: {response.text}")
                await self._response_queue.put(response.text)

        except Exception as e:
            _LOGGER.error(f"Error handling response: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_tool_call(self, tool_call):
        """
        Handle a tool call from Gemini and send the result back.

        Args:
            tool_call: The tool call object from Gemini
        """
        try:
            from google.genai import types

            # Extract function calls from the tool call
            function_calls = getattr(tool_call, 'function_calls', [])
            if not function_calls:
                _LOGGER.warning("Tool call received but no function calls found")
                return

            function_responses = []

            for func_call in function_calls:
                func_name = func_call.name
                func_args = dict(func_call.args) if func_call.args else {}
                func_id = getattr(func_call, 'id', func_name)

                _LOGGER.info(f"Tool call: {func_name}({func_args})")

                # Execute the tool
                result = await execute_tool(func_name, func_args)

                # Create function response
                func_response = types.FunctionResponse(
                    name=func_name,
                    id=func_id,
                    response=result,
                )
                function_responses.append(func_response)

                _LOGGER.info(f"Tool result: {result}")

            # Send tool responses back to Gemini
            if function_responses and self._session:
                await self._session.send_tool_response(function_responses=function_responses)
                _LOGGER.debug("Sent tool responses to Gemini")

        except Exception as e:
            _LOGGER.error(f"Error handling tool call: {e}")
            import traceback
            traceback.print_exc()

    async def send_audio(self, audio_data: bytes, sample_rate: int = 16000):
        """
        Send audio data to Gemini Live API.

        Args:
            audio_data: Raw audio bytes (PCM format)
            sample_rate: Audio sample rate in Hz (default 16000)
        """
        if not self.connected:
            _LOGGER.debug("Cannot send audio - not connected")
            return

        try:
            from google.genai import types

            # Create audio blob with proper MIME type including sample rate
            audio_blob = types.Blob(
                data=audio_data,
                mime_type=f"audio/pcm;rate={sample_rate}"
            )

            # Send realtime input using the media parameter
            await self._session.send_realtime_input(media=audio_blob)

            # Log occasionally to track audio being sent to API
            if not hasattr(self, '_api_audio_count'):
                self._api_audio_count = 0
            self._api_audio_count += 1
            if self._api_audio_count % 50 == 1:
                _LOGGER.debug(f"Sent audio chunk #{self._api_audio_count} to Gemini API ({len(audio_data)} bytes)")

        except Exception as e:
            _LOGGER.error(f"Failed to send audio: {e}")

    async def send_video_frame(self, frame_data: bytes, mime_type: str = "image/jpeg"):
        """
        Send a video frame to Gemini Live API.

        Args:
            frame_data: Image bytes (JPEG or PNG)
            mime_type: Image format (default "image/jpeg")
        """
        # Store the latest frame for notifications
        set_latest_frame(frame_data)

        if not self.connected:
            return

        try:
            from google.genai import types

            # Create image blob
            image_blob = types.Blob(
                data=frame_data,
                mime_type=mime_type
            )

            # Send realtime input using the media parameter
            await self._session.send_realtime_input(media=image_blob)

        except Exception as e:
            _LOGGER.error(f"Failed to send video frame: {e}")

    async def send_text(self, text: str):
        """
        Send text input to Gemini (e.g., transcribed speech or commands).

        Args:
            text: Text to send
        """
        if not self.connected:
            _LOGGER.warning(f"Cannot send text - not connected: {text[:50]}")
            return

        try:
            from google.genai import types

            # Create proper Content object with user role
            content = types.Content(
                parts=[types.Part(text=text)],
                role="user"
            )

            # Send as client content (text turn)
            await self._session.send_client_content(
                turns=[content],
                turn_complete=True
            )
            _LOGGER.info(f"Sent text to Gemini: {text[:100]}...")

        except Exception as e:
            _LOGGER.error(f"Failed to send text: {e}")

    async def get_response(self, timeout: float = 10.0) -> Optional[str]:
        """
        Wait for and return the next text response from Gemini.

        Args:
            timeout: Maximum time to wait

        Returns:
            Response text or None if timeout
        """
        try:
            return await asyncio.wait_for(self._response_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def get_all_responses(self, timeout: float = 2.0) -> list[str]:
        """
        Collect all pending responses (for multi-part responses).

        Args:
            timeout: Time to wait for additional responses

        Returns:
            List of response texts
        """
        responses = []

        # Get first response
        first = await self.get_response(timeout=timeout)
        if first:
            responses.append(first)

            # Collect any additional parts quickly
            while True:
                try:
                    additional = await asyncio.wait_for(
                        self._response_queue.get(),
                        timeout=0.5
                    )
                    responses.append(additional)
                except asyncio.TimeoutError:
                    break

        return responses

    def set_audio_callback(self, callback: Callable):
        """Set callback for Gemini audio output."""
        self._audio_callback = callback

    async def get_audio(self, timeout: float = 10.0) -> Optional[bytes]:
        """
        Wait for and return the next audio chunk from Gemini.

        Args:
            timeout: Maximum time to wait

        Returns:
            Audio bytes or None if timeout
        """
        try:
            return await asyncio.wait_for(self._audio_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def get_all_audio(self, timeout: float = 2.0) -> bytes:
        """
        Collect all pending audio chunks.

        Args:
            timeout: Time to wait for additional audio

        Returns:
            Combined audio bytes
        """
        audio_data = b""

        # Get first chunk
        first = await self.get_audio(timeout=timeout)
        if first:
            audio_data += first

            # Collect any additional chunks quickly
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.3
                    )
                    audio_data += chunk
                except asyncio.TimeoutError:
                    break

        return audio_data

    async def close(self):
        """Close the Gemini Live session."""
        self._connected = False

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._session and self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                _LOGGER.debug(f"Error closing session: {e}")

        self._session = None
        self._client = None

        _LOGGER.info("Gemini Live session closed")


class DoorbellAgent:
    """
    AI-powered doorbell conversation agent.

    Orchestrates:
    - Real-time video/audio streaming to Gemini
    - Response generation and decision making
    - TTS output via ElevenLabs
    - WebRTC integration for doorbell communication
    """

    def __init__(
        self,
        gemini_api_key: str = None,
        eleven_labs_api_key: str = None,
        voice_id: str = None,
        on_speak: Callable[[str], None] = None,
    ):
        """
        Initialize the doorbell agent.

        Args:
            gemini_api_key: Gemini API key
            eleven_labs_api_key: ElevenLabs API key
            voice_id: ElevenLabs voice ID
            on_speak: Callback when agent speaks (receives text)
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.eleven_labs_api_key = eleven_labs_api_key or os.getenv("ELEVEN_LABS_API_KEY")
        self.voice_id = voice_id or os.getenv("ELEVEN_LABS_VOICE_ID", "c6SfcYrb2t09NHXiT80T")

        self.session: Optional[GeminiLiveSession] = None
        self.state: Optional[ConversationState] = None
        self.on_speak = on_speak

        self._running = False
        self._audio_task: Optional[asyncio.Task] = None
        self._video_task: Optional[asyncio.Task] = None
        self._response_task: Optional[asyncio.Task] = None

        # RTSP video capture settings
        self._rtsp_url: Optional[str] = None
        self._camera_name: str = "doorbell"
        self._video_capture_interval: float = 1.0  # seconds between frames
        self._video_capture_task: Optional[asyncio.Task] = None

    async def start_conversation(self, session_id: str = None) -> bool:
        """
        Start a new conversation session.

        Args:
            session_id: Optional session ID (generated if not provided)

        Returns:
            True if session started successfully
        """
        if self._running:
            _LOGGER.warning("Conversation already in progress")
            return False

        # Create session
        self.session = GeminiLiveSession(api_key=self.gemini_api_key)

        if not await self.session.connect():
            _LOGGER.error("Failed to start Gemini session")
            return False

        # Initialize state
        self.state = ConversationState(
            session_id=session_id or f"conv_{int(time.time())}"
        )

        self._running = True

        # Start response handler
        self._response_task = asyncio.create_task(self._response_loop())

        _LOGGER.info(f"Conversation started: {self.state.session_id}")
        return True

    async def _response_loop(self):
        """Handle responses from Gemini and speak them."""
        while self._running and self.session and self.session.connected:
            try:
                # Wait for responses
                responses = await self.session.get_all_responses(timeout=5.0)

                if responses:
                    # Combine multi-part responses
                    full_response = " ".join(responses)

                    if full_response.strip():
                        _LOGGER.info(f"Agent says: {full_response}")

                        # Update state
                        if self.state:
                            self.state.add_turn("assistant", full_response)

                        # Callback
                        if self.on_speak:
                            self.on_speak(full_response)

                        # Speak via TTS
                        await self._speak(full_response)

            except asyncio.CancelledError:
                break
            except Exception as e:
                _LOGGER.error(f"Response loop error: {e}")
                await asyncio.sleep(1)

    async def _speak(self, text: str):
        """
        Convert text to speech and play through doorbell.

        Args:
            text: Text to speak
        """
        if not self.eleven_labs_api_key:
            _LOGGER.warning("No ElevenLabs API key - cannot speak")
            return

        try:
            from elevenlabs import AsyncElevenLabs

            client = AsyncElevenLabs(api_key=self.eleven_labs_api_key)

            # Generate audio (returns async generator, not coroutine)
            audio_generator = client.text_to_speech.convert(
                voice_id=self.voice_id,
                model_id="eleven_turbo_v2_5",
                text=text,
                output_format="pcm_16000",  # 16kHz PCM for WebRTC
            )

            # Collect audio
            audio_data = b""
            async for chunk in audio_generator:
                audio_data += chunk

            # TODO: Send audio_data through WebRTC to doorbell
            # This will be integrated with the WebRTC module
            _LOGGER.info(f"TTS generated {len(audio_data)} bytes of audio")

            # For now, emit an event that can be handled by the WebRTC layer
            if hasattr(self, '_audio_output_callback') and self._audio_output_callback:
                await self._audio_output_callback(audio_data)

        except Exception as e:
            _LOGGER.error(f"TTS error: {e}")

    def set_audio_output_callback(self, callback: Callable[[bytes], None]):
        """Set callback for audio output (to send to WebRTC)."""
        self._audio_output_callback = callback

    async def send_audio_chunk(self, audio_data: bytes):
        """
        Send an audio chunk from the doorbell microphone to Gemini.

        Args:
            audio_data: PCM audio bytes (16kHz recommended)
        """
        if self.session and self.session.connected:
            # Log occasionally to avoid spam (every ~1 second at 16kHz with 20ms chunks)
            if not hasattr(self, '_audio_send_count'):
                self._audio_send_count = 0
            self._audio_send_count += 1
            if self._audio_send_count % 50 == 1:  # Log every ~1 second
                _LOGGER.info(f"Sending audio chunk #{self._audio_send_count} ({len(audio_data)} bytes) to Gemini")
            await self.session.send_audio(audio_data)
        else:
            if not hasattr(self, '_audio_drop_warned'):
                _LOGGER.warning("Cannot send audio - Gemini session not connected")
                self._audio_drop_warned = True

    async def send_video_frame(self, frame_data: bytes):
        """
        Send a video frame from the doorbell camera to Gemini.

        Args:
            frame_data: JPEG image bytes
        """
        if self.session and self.session.connected:
            await self.session.send_video_frame(frame_data)

    async def inject_context(self, context: str):
        """
        Inject additional context into the conversation.

        Useful for providing information the agent can't see, like:
        - "The homeowner says they'll be there in 5 minutes"
        - "This is a scheduled delivery"

        Args:
            context: Context text to inject
        """
        if self.session and self.session.connected:
            await self.session.send_text(f"[System context: {context}]")

    async def end_conversation(self):
        """End the current conversation session."""
        self._running = False

        # Stop video capture if running
        self.stop_video_capture()

        if self._response_task:
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass

        if self.session:
            await self.session.close()
            self.session = None

        if self.state:
            _LOGGER.info(
                f"Conversation ended: {self.state.session_id}, "
                f"{self.state.turn_count} turns, "
                f"duration: {time.time() - self.state.started_at:.1f}s"
            )
            self.state.is_active = False

    @property
    def is_active(self) -> bool:
        """Check if a conversation is active."""
        return self._running and self.session is not None and self.session.connected

    async def connect_to_webrtc(self, webrtc_client) -> bool:
        """
        Connect this agent to a VivintWebRTCClient for outgoing audio.

        Sets up audio piping:
        - Gemini audio -> doorbell speaker (via WebRTC)
        - Doorbell mic audio comes from RTSP stream (not WebRTC)

        Args:
            webrtc_client: VivintWebRTCClient instance (must have AI mode enabled)

        Returns:
            True if connected successfully
        """
        self._webrtc_client = webrtc_client

        # NOTE: Incoming audio from doorbell comes from RTSP stream, not WebRTC
        # Use start_audio_capture() with RTSP URL to get doorbell microphone audio

        # Set up audio output: Gemini (24kHz mono) -> convert -> doorbell speaker (48kHz stereo)
        async def on_gemini_audio(audio_data: bytes):
            """Handle Gemini audio output, convert format, and send to WebRTC."""
            _LOGGER.info(f"Received {len(audio_data)} bytes audio from Gemini, converting 24kHz mono -> 48kHz stereo")
            # Gemini outputs 24kHz mono PCM, WebRTC/Opus expects 48kHz stereo
            converted = convert_audio_for_webrtc(audio_data, from_rate=24000)
            _LOGGER.info(f"Queuing {len(converted)} bytes to WebRTC speaker")
            await webrtc_client.queue_audio(converted)

        # Connect Gemini session's audio output
        if self.session:
            self.session.set_audio_callback(on_gemini_audio)

        _LOGGER.info("DoorbellAgent connected to WebRTC (outgoing audio only)")

        # Reset TTSAudioTrack timing for fresh audio playback
        if webrtc_client.tts_track:
            webrtc_client.tts_track.reset_timing()
            _LOGGER.info("TTSAudioTrack timing reset - ready for Gemini audio")

        # Trigger initial greeting
        _LOGGER.info("Triggering initial AI greeting...")
        await self.inject_context(
            "A person has just approached the doorbell camera and pressed the button. "
            "Greet them warmly and ask how you can help."
        )

        return True

    def disconnect_from_webrtc(self):
        """Disconnect from WebRTC client."""
        if hasattr(self, '_webrtc_client') and self._webrtc_client:
            self._webrtc_client.remove_audio_callback(self.send_audio_chunk)
            self._webrtc_client = None
            self._audio_output_callback = None
            _LOGGER.info("DoorbellAgent disconnected from WebRTC")

    async def start_audio_capture(self, rtsp_url: str) -> bool:
        """
        Start capturing audio from RTSP stream and sending to Gemini.

        Uses ffmpeg to extract audio from the RTSP stream, convert to
        16kHz mono PCM, and pipe to Gemini Live for speech recognition.

        Args:
            rtsp_url: RTSP URL for the camera stream (same as video)

        Returns:
            True if capture started successfully
        """
        if hasattr(self, '_audio_capture_task') and self._audio_capture_task and not self._audio_capture_task.done():
            _LOGGER.warning("Audio capture already running")
            return False

        self._rtsp_audio_url = rtsp_url
        self._audio_capture_task = asyncio.create_task(self._audio_capture_loop())
        _LOGGER.info(f"Started RTSP audio capture -> Gemini")
        return True

    async def _audio_capture_loop(self):
        """Background loop to capture audio from RTSP and send to Gemini."""
        import subprocess

        _LOGGER.info("RTSP audio capture loop started")

        # ffmpeg command to extract audio from RTSP as 16kHz mono PCM
        # Key options for low-latency streaming:
        # -fflags nobuffer: don't buffer input
        # -flags low_delay: minimize latency
        # -flush_packets 1: flush output immediately
        cmd = [
            "ffmpeg",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-rtsp_transport", "tcp",
            "-i", self._rtsp_audio_url,
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "-flush_packets", "1",
            "-loglevel", "error",
            "pipe:1"  # Output to stdout
        ]

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _LOGGER.info("ffmpeg audio capture process started")

            # Read audio in chunks and send to Gemini
            # 16kHz * 2 bytes * 0.1s = 3200 bytes per 100ms chunk
            chunk_size = 3200

            while self._running and process.returncode is None:
                try:
                    audio_data = await asyncio.wait_for(
                        process.stdout.read(chunk_size),
                        timeout=1.0
                    )

                    if not audio_data:
                        _LOGGER.warning("RTSP audio stream ended")
                        break

                    # Send to Gemini
                    await self.send_audio_chunk(audio_data)

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    _LOGGER.error(f"Error reading RTSP audio: {e}")
                    break

        except Exception as e:
            _LOGGER.error(f"Failed to start ffmpeg audio capture: {e}")
        finally:
            if process:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
            _LOGGER.info("RTSP audio capture loop stopped")

    def stop_audio_capture(self):
        """Stop the audio capture loop."""
        if hasattr(self, '_audio_capture_task') and self._audio_capture_task:
            self._audio_capture_task.cancel()
            self._audio_capture_task = None
            _LOGGER.info("RTSP audio capture stopped")

    async def start_video_capture(
        self,
        rtsp_url: str,
        camera_name: str = "doorbell",
        interval_seconds: float = 1.0,
    ) -> bool:
        """
        Start capturing video frames from RTSP and sending to Gemini.

        Also starts audio capture from the same RTSP stream.

        Args:
            rtsp_url: RTSP URL for the camera stream
            camera_name: Camera name for logging
            interval_seconds: Time between frame captures (default 1s)

        Returns:
            True if capture started successfully
        """
        if self._video_capture_task and not self._video_capture_task.done():
            _LOGGER.warning("Video capture already running")
            return False

        self._rtsp_url = rtsp_url
        self._camera_name = camera_name
        self._video_capture_interval = interval_seconds

        # Start video capture
        self._video_capture_task = asyncio.create_task(self._video_capture_loop())
        _LOGGER.info(f"Started video capture from {camera_name} (interval: {interval_seconds}s)")

        # Also start audio capture from the same RTSP stream
        await self.start_audio_capture(rtsp_url)

        return True

    async def _video_capture_loop(self):
        """Background loop to capture and send video frames to Gemini."""
        from frame_capture import capture_single_frame

        _LOGGER.info(f"Video capture loop started for {self._camera_name}")

        try:
            while self._running and self._rtsp_url:
                try:
                    # Capture a single frame
                    frame_path = await capture_single_frame(self._rtsp_url, self._camera_name)

                    if frame_path and frame_path.exists():
                        # Read frame data
                        frame_data = frame_path.read_bytes()

                        # Send to Gemini
                        await self.send_video_frame(frame_data)
                        _LOGGER.debug(f"Sent video frame ({len(frame_data)} bytes) to Gemini")

                        # Clean up frame file
                        try:
                            frame_path.unlink()
                        except Exception:
                            pass
                    else:
                        _LOGGER.warning("Frame capture returned no frame")

                except Exception as e:
                    _LOGGER.error(f"Error capturing video frame: {e}")

                # Wait for next capture
                await asyncio.sleep(self._video_capture_interval)

        except asyncio.CancelledError:
            _LOGGER.info("Video capture loop cancelled")
        finally:
            _LOGGER.info("Video capture loop stopped")

    def stop_video_capture(self):
        """Stop the video and audio capture loops."""
        if self._video_capture_task:
            self._video_capture_task.cancel()
            self._video_capture_task = None
            _LOGGER.info("Video capture stopped")
        # Also stop audio capture
        self.stop_audio_capture()


async def run_doorbell_conversation(
    oauth_token: str,
    camera_uuid: str,
    rtsp_url: str = None,
    duration_seconds: float = 60.0,
    video_interval: float = 1.0,
) -> None:
    """
    Run an AI-powered doorbell conversation session.

    This is a high-level convenience function that sets up everything needed
    for an AI doorbell conversation:
    - WebRTC connection to doorbell camera (two-way audio)
    - RTSP video frame capture for visual context
    - Gemini Live session for AI understanding
    - Audio resampling and piping between components

    Args:
        oauth_token: Vivint OAuth token
        camera_uuid: Camera/doorbell UUID
        rtsp_url: RTSP URL for video capture (optional, for visual context)
        duration_seconds: How long to run the conversation (default 60s)
        video_interval: Seconds between video frame captures (default 1s)
    """
    from vivint_webrtc import VivintWebRTCClient, VivintWebRTCConfig

    _LOGGER.info("Starting AI doorbell conversation session...")

    # Create WebRTC config
    config = VivintWebRTCConfig(
        oauth_token=oauth_token,
        camera_uuid=camera_uuid,
    )

    # Create and configure WebRTC client
    webrtc = VivintWebRTCClient(config)
    webrtc.enable_ai_conversation_mode()

    # Create doorbell agent
    agent = DoorbellAgent()

    try:
        # Connect WebRTC
        _LOGGER.info("Connecting WebRTC...")
        if not await webrtc.connect():
            _LOGGER.error("Failed to connect WebRTC")
            return

        # Start Gemini Live session
        _LOGGER.info("Starting Gemini Live session...")
        if not await agent.start_conversation():
            _LOGGER.error("Failed to start Gemini session")
            await webrtc.disconnect()
            return

        # Connect agent to WebRTC (audio piping)
        await agent.connect_to_webrtc(webrtc)

        # Start video capture if RTSP URL provided
        if rtsp_url:
            _LOGGER.info("Starting video capture...")
            await agent.start_video_capture(
                rtsp_url=rtsp_url,
                camera_name="doorbell",
                interval_seconds=video_interval,
            )

        # Start two-way talk
        _LOGGER.info("Starting two-way talk...")
        if not await webrtc.start_two_way_talk():
            _LOGGER.error("Failed to start two-way talk")
            await agent.end_conversation()
            await webrtc.disconnect()
            return

        _LOGGER.info(f"AI doorbell conversation active for {duration_seconds}s")
        _LOGGER.info("The agent can now see, hear, and speak with visitors!")

        # Run for specified duration
        await asyncio.sleep(duration_seconds)

    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user")
    except Exception as e:
        _LOGGER.error(f"Error during conversation: {e}")
    finally:
        _LOGGER.info("Cleaning up...")
        agent.disconnect_from_webrtc()
        await agent.end_conversation()
        await webrtc.stop_two_way_talk()
        await webrtc.disconnect()
        _LOGGER.info("AI doorbell conversation ended")


async def test_agent():
    """Test the doorbell agent with simulated input."""
    logging.basicConfig(level=logging.INFO)

    def on_speak(text: str):
        print(f"\n🔊 Agent says: {text}\n")

    agent = DoorbellAgent(on_speak=on_speak)

    if not await agent.start_conversation():
        print("Failed to start conversation")
        return

    print("Conversation started! Type messages to simulate visitor speech.")
    print("Type 'quit' to end.\n")

    try:
        while agent.is_active:
            # Simulate visitor input
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, input, "Visitor: "
            )

            if user_input.lower() == 'quit':
                break

            # Send as text (in real use, this would be transcribed audio)
            await agent.session.send_text(user_input)

            # Wait a moment for response
            await asyncio.sleep(2)

    except KeyboardInterrupt:
        pass
    finally:
        await agent.end_conversation()
        print("\nConversation ended.")


if __name__ == "__main__":
    asyncio.run(test_agent())
