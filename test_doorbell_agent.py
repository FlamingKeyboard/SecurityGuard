"""
Test script for the AI Doorbell Agent.

Tests the conversation system locally without connecting to the actual doorbell.
You can either:
1. Type messages to simulate visitor speech
2. Use your microphone for real speech input (requires sounddevice)

Requirements:
    pip install sounddevice numpy elevenlabs google-genai

Usage:
    python test_doorbell_agent.py          # Text-only mode
    python test_doorbell_agent.py --audio  # With microphone/speaker
    python test_doorbell_agent.py --mode tts    # Test TTS only
    python test_doorbell_agent.py --mode gemini # Test Gemini only
"""

import asyncio
import logging
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("\\", 1)[0])

# Load stored credentials
def load_stored_credentials():
    """Load API keys from encrypted storage."""
    try:
        from vivint_client import load_credentials
        creds = load_credentials() or {}

        if creds.get('gemini_api_key') and not os.getenv('GEMINI_API_KEY'):
            os.environ['GEMINI_API_KEY'] = creds['gemini_api_key']

        if creds.get('eleven_labs_api_key') and not os.getenv('ELEVEN_LABS_API_KEY'):
            os.environ['ELEVEN_LABS_API_KEY'] = creds['eleven_labs_api_key']

    except Exception as e:
        print(f"Note: Could not load stored credentials: {e}")

load_stored_credentials()

from doorbell_agent import DoorbellAgent, GeminiLiveSession

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
_LOGGER = logging.getLogger(__name__)


async def play_audio_local(audio_data: bytes, sample_rate: int = 16000):
    """Play PCM audio through local speakers."""
    try:
        import sounddevice as sd
        import numpy as np

        # Convert bytes to numpy array (16-bit signed PCM)
        samples = np.frombuffer(audio_data, dtype=np.int16)
        # Normalize to float32 for sounddevice
        samples_float = samples.astype(np.float32) / 32768.0

        # Play audio (blocking)
        sd.play(samples_float, sample_rate)
        sd.wait()

    except ImportError:
        _LOGGER.warning("sounddevice not installed - cannot play audio")
        _LOGGER.info("Install with: pip install sounddevice")
    except Exception as e:
        _LOGGER.error(f"Error playing audio: {e}")


async def test_text_mode():
    """Test the agent with text input (simulated speech)."""
    print("=" * 60)
    print("AI Doorbell Agent - Text Test Mode")
    print("=" * 60)
    print()
    print("This simulates conversations with the doorbell agent.")
    print("Type messages as if you were a visitor at the door.")
    print("The agent will respond with text (and audio if configured).")
    print()
    print("Commands:")
    print("  quit     - Exit the test")
    print("  context: - Inject system context (e.g., 'context: delivery scheduled')")
    print()

    # Check API keys
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        print("Run: set GEMINI_API_KEY=your_key_here")
        return

    eleven_labs_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not eleven_labs_key:
        print("WARNING: ELEVEN_LABS_API_KEY not set - TTS disabled")
        print("Run: set ELEVEN_LABS_API_KEY=your_key_here")
    print()

    # Track responses for audio playback
    pending_audio = []

    async def on_tts_audio(audio_data: bytes):
        """Play audio immediately when generated."""
        print("🔈 Playing audio...")
        await play_audio_local(audio_data)
        print("🔈 Audio complete.")

    def on_speak(text: str):
        """Called when agent wants to speak."""
        print(f"\n🔊 Agent: {text}\n")

    # Create agent
    agent = DoorbellAgent(on_speak=on_speak)
    agent.set_audio_output_callback(on_tts_audio)

    print("Connecting to Gemini Live API...")
    if not await agent.start_conversation():
        print("Failed to start conversation - check your API key")
        return

    print("[OK] Connected! Agent is ready.")
    print()
    print("-" * 60)
    print("Scenario: Someone approaches the doorbell")
    print("-" * 60)
    print()

    # Send initial context
    await agent.inject_context("A person has just approached the doorbell camera.")

    try:
        while agent.is_active:
            # Get user input
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "Visitor: "
                )
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            # Handle context injection
            if user_input.lower().startswith('context:'):
                context = user_input[8:].strip()
                await agent.inject_context(context)
                print(f"[Context injected: {context}]")
                continue

            # Send visitor message
            if agent.session and agent.session.connected:
                # Record as user turn
                if agent.state:
                    agent.state.add_turn("user", user_input)

                # Send to Gemini
                await agent.session.send_text(f"[Visitor says]: {user_input}")

                # Wait for response and play audio
                await asyncio.sleep(2)  # Give time for response

                # Play any pending audio
                if pending_audio:
                    print("🔈 Playing audio response...")
                    for audio in pending_audio:
                        await play_audio_local(audio)
                    pending_audio.clear()

    except KeyboardInterrupt:
        print("\n\nInterrupted")
    finally:
        await agent.end_conversation()
        print("\nConversation ended.")

        if agent.state:
            print(f"\nStats: {agent.state.turn_count} turns, "
                  f"{len(agent.state.history)} messages in history")


async def test_audio_mode():
    """Test with real microphone input and speaker output."""
    print("=" * 60)
    print("AI Doorbell Agent - Audio Test Mode")
    print("=" * 60)
    print()

    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("ERROR: sounddevice not installed")
        print("Install with: pip install sounddevice numpy")
        return

    # Check API keys
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        return

    if not os.getenv("ELEVEN_LABS_API_KEY"):
        print("WARNING: ELEVEN_LABS_API_KEY not set - TTS disabled")
    print()

    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_DURATION = 0.1  # 100ms chunks
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

    # List audio devices
    print("Available audio devices:")
    print(sd.query_devices())
    print()

    pending_audio = []
    recording = True

    async def on_tts_audio(audio_data: bytes):
        pending_audio.append(audio_data)

    def on_speak(text: str):
        print(f"\n🔊 Agent: {text}")

    # Create agent
    agent = DoorbellAgent(on_speak=on_speak)
    agent.set_audio_output_callback(on_tts_audio)

    print("Connecting to Gemini Live API...")
    if not await agent.start_conversation():
        print("Failed to start conversation")
        return

    print("[OK] Connected!")
    print()
    print("Speak into your microphone. The agent will respond.")
    print("Press Ctrl+C to stop.")
    print()

    # Send initial context
    await agent.inject_context("A person has just approached the doorbell camera and may speak.")

    # Audio callback for microphone input
    audio_queue = asyncio.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            _LOGGER.warning(f"Audio status: {status}")
        if recording:
            # Convert to bytes
            audio_bytes = (indata * 32768).astype(np.int16).tobytes()
            try:
                audio_queue.put_nowait(audio_bytes)
            except asyncio.QueueFull:
                pass

    async def send_audio_loop():
        """Send audio chunks to Gemini."""
        while recording and agent.is_active:
            try:
                audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                await agent.send_audio_chunk(audio_data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                _LOGGER.error(f"Error sending audio: {e}")

    async def play_audio_loop():
        """Play TTS audio responses."""
        while recording and agent.is_active:
            if pending_audio:
                audio = pending_audio.pop(0)
                await play_audio_local(audio, SAMPLE_RATE)
            else:
                await asyncio.sleep(0.1)

    try:
        # Start audio input stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        ):
            print("🎤 Microphone active - speak now!")

            # Run audio loops
            audio_task = asyncio.create_task(send_audio_loop())
            play_task = asyncio.create_task(play_audio_loop())

            # Wait for interrupt
            while agent.is_active:
                await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        recording = False
        await agent.end_conversation()
        print("Test ended.")


async def test_tts_only():
    """Test just the TTS system without Gemini."""
    print("=" * 60)
    print("TTS Test Mode - Testing ElevenLabs")
    print("=" * 60)
    print()

    api_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVEN_LABS_API_KEY not set")
        print("Run: set ELEVEN_LABS_API_KEY=your_key_here")
        return

    voice_id = os.getenv("ELEVEN_LABS_VOICE_ID", "c6SfcYrb2t09NHXiT80T")

    print(f"API Key: {api_key[:12]}...")
    print(f"Voice ID: {voice_id}")
    print()

    test_phrases = [
        "Hi there! Welcome! How can I help you?",
        "You can leave the package right by the door, I'll make sure it's safe.",
        "Thanks for stopping by, but we're not interested at this time. Have a great day!",
    ]

    try:
        from elevenlabs import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=api_key)

        for i, phrase in enumerate(test_phrases, 1):
            print(f"\n[{i}/{len(test_phrases)}] Generating: \"{phrase[:40]}...\"")

            # Returns async generator, not coroutine
            audio_generator = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id="eleven_turbo_v2_5",
                text=phrase,
                output_format="pcm_16000",
            )

            # Collect audio
            audio_data = b""
            async for chunk in audio_generator:
                audio_data += chunk

            print(f"    Generated {len(audio_data)} bytes")
            print("    Playing...")

            await play_audio_local(audio_data)

            await asyncio.sleep(0.5)

        print("\n[OK] TTS test complete!")

    except ImportError:
        print("ERROR: elevenlabs not installed")
        print("Install with: pip install elevenlabs")
    except Exception as e:
        print(f"ERROR: {e}")


async def test_gemini_live_only():
    """Test just the Gemini Live connection."""
    print("=" * 60)
    print("Gemini Live Test Mode")
    print("=" * 60)
    print()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        print("Set it with: set GEMINI_API_KEY=your_key")
        return

    print(f"API Key: {api_key[:15]}...")
    print()

    # Use a simple system instruction for testing
    test_system_prompt = """You are a helpful AI assistant being tested.
    Keep your responses brief (1-2 sentences).
    If asked to roleplay as a doorbell, do so."""

    session = GeminiLiveSession(api_key=api_key)

    print("Connecting to Gemini Live API...")
    print(f"Model: {session.model}")

    if not await session.connect(system_instruction=test_system_prompt):
        print("Failed to connect!")
        print("Check your API key and internet connection.")
        return

    print("[OK] Connected to Gemini Live!")
    print()
    print("Type messages to test. Type 'quit' to exit.")
    print()

    try:
        while session.connected:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            await session.send_text(user_input)

            # Wait for response
            responses = await session.get_all_responses(timeout=10.0)
            if responses:
                print(f"Gemini: {' '.join(responses)}")
            else:
                print("(no response - waiting longer...)")
                # Try waiting a bit more
                more = await session.get_all_responses(timeout=5.0)
                if more:
                    print(f"Gemini: {' '.join(more)}")
                else:
                    print("(still no response)")
            print()

    except KeyboardInterrupt:
        print("\n\nInterrupted")
    finally:
        await session.close()
        print("\nSession closed.")


async def test_native_voice_mode():
    """Test with Gemini's native voice output (no ElevenLabs)."""
    print("=" * 60)
    print("AI Doorbell Agent - Native Gemini Voice Test")
    print("=" * 60)
    print()
    print("Using Gemini's built-in voice (Puck) instead of ElevenLabs.")
    print("Type messages to chat. Type 'quit' to exit.")
    print()

    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        return

    # Create session (uses native Gemini audio by default)
    session = GeminiLiveSession()

    # Collect audio chunks for playback
    audio_buffer = []

    async def on_audio(audio_data: bytes):
        """Handle incoming audio from Gemini."""
        audio_buffer.append(audio_data)

    session.set_audio_callback(on_audio)

    print("Connecting to Gemini Live API with native voice...")
    if not await session.connect():
        print("Failed to connect")
        return

    print("[OK] Connected! Gemini will respond with voice.")
    print()

    try:
        while session.connected:
            # Get user input
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break

            # Clear audio buffer
            audio_buffer.clear()

            # Send message
            await session.send_text(user_input)

            # Wait for response
            print("🔊 Gemini is speaking...")
            await asyncio.sleep(3)  # Wait for audio to arrive

            # Collect and play audio
            while not session._audio_queue.empty():
                try:
                    chunk = session._audio_queue.get_nowait()
                    audio_buffer.append(chunk)
                except:
                    break

            if audio_buffer:
                # Combine all audio chunks
                all_audio = b"".join(audio_buffer)
                print(f"   Received {len(all_audio)} bytes of audio")

                # Play the audio (Gemini outputs 24kHz PCM)
                await play_audio_local(all_audio, sample_rate=24000)
                print("   ✓ Audio played")
            else:
                print("   (No audio received)")

            print()

    except KeyboardInterrupt:
        print("\n\nInterrupted")
    finally:
        await session.close()
        print("Session closed.")


def main():
    parser = argparse.ArgumentParser(description="Test the AI Doorbell Agent")
    parser.add_argument(
        '--mode', '-m',
        choices=['text', 'audio', 'tts', 'gemini', 'native'],
        default='text',
        help='Test mode: text (default), audio (mic+speaker), tts (TTS only), gemini (Gemini only), native (Gemini voice)'
    )

    args = parser.parse_args()

    if args.mode == 'text':
        asyncio.run(test_text_mode())
    elif args.mode == 'audio':
        asyncio.run(test_audio_mode())
    elif args.mode == 'tts':
        asyncio.run(test_tts_only())
    elif args.mode == 'gemini':
        asyncio.run(test_gemini_live_only())
    elif args.mode == 'native':
        asyncio.run(test_native_voice_mode())


if __name__ == "__main__":
    main()
