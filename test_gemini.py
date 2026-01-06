"""Test script for Gemini API integration."""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from vivint_client import get_stored_credential


def test_api_key_format():
    """Test that we have a valid API key."""
    api_key = get_stored_credential("gemini_api_key")

    print("=== API Key Test ===")
    if not api_key:
        print("FAIL: No API key stored")
        print("  Run: python setup_credentials.py")
        return False

    print(f"Key length: {len(api_key)}")
    print(f"Key prefix: {api_key[:10]}...")

    # Standard Gemini API keys from AI Studio start with "AIza" and are 39 chars
    if api_key.startswith("AIza") and len(api_key) == 39:
        print("PASS: Valid AI Studio API key format (AIza...)")
        return True
    elif api_key.startswith("AIza"):
        print("WARN: Key starts with AIza but unexpected length")
        return True
    else:
        print("FAIL: Invalid API key format!")
        print()
        print("  Your key does NOT appear to be a Gemini API key.")
        print("  Gemini API keys start with 'AIza' and are 39 characters.")
        print()
        print("  To get a valid key:")
        print("  1. Go to: https://aistudio.google.com/apikey")
        print("  2. Click 'Create API Key'")
        print("  3. Copy the key (should start with AIzaSy...)")
        print("  4. Run: python setup_credentials.py")
        print("     and enter the new key when prompted")
        return False


def test_sdk_import():
    """Test that the new SDK is available."""
    print("\n=== SDK Import Test ===")
    try:
        from google import genai
        from google.genai import types
        print("SUCCESS: google-genai SDK imported")
        return True
    except ImportError as e:
        print(f"FAIL: Could not import google-genai: {e}")
        return False


def test_client_creation():
    """Test creating a Gemini client."""
    print("\n=== Client Creation Test ===")
    api_key = get_stored_credential("gemini_api_key")
    if not api_key:
        print("SKIP: No API key")
        return False

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        print("SUCCESS: Client created")
        return client
    except Exception as e:
        print(f"FAIL: Could not create client: {e}")
        return False


def test_simple_request(client):
    """Test a simple text request."""
    print("\n=== Simple Text Request Test ===")
    if not client:
        print("SKIP: No client")
        return False

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='Say "Hello, test successful!" and nothing else.',
        )
        print(f"SUCCESS: Response received")
        print(f"Response: {response.text}")
        return True
    except Exception as e:
        print(f"FAIL: Request failed: {e}")
        return False


def test_image_request(client, image_path: str = None):
    """Test an image analysis request."""
    print("\n=== Image Analysis Test ===")
    if not client:
        print("SKIP: No client")
        return False

    # Find a test image
    if image_path:
        path = Path(image_path)
    else:
        # Look for captured frames
        frames_dir = Path(__file__).parent / "data" / "frames"
        frames = list(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
        if frames:
            path = frames[0]
            print(f"Using captured frame: {path.name}")
        else:
            print("SKIP: No test image available")
            print("  Run setup_credentials.py with frame capture first")
            return False

    if not path.exists():
        print(f"SKIP: Image not found: {path}")
        return False

    try:
        from google.genai import types

        image_bytes = path.read_bytes()
        print(f"Image size: {len(image_bytes)} bytes")

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                'Describe what you see in this image in one sentence.',
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            ],
        )
        print(f"SUCCESS: Image analyzed")
        print(f"Response: {response.text}")
        return True
    except Exception as e:
        print(f"FAIL: Image request failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_security_analysis(client):
    """Test the full security analysis prompt."""
    print("\n=== Security Analysis Test ===")
    if not client:
        print("SKIP: No client")
        return False

    # Find a test image
    frames_dir = Path(__file__).parent / "data" / "frames"
    frames = list(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if not frames:
        print("SKIP: No test image available")
        return False

    path = frames[0]
    print(f"Using: {path.name}")

    prompt = """Analyze this security camera image. Respond with JSON only:
{
    "risk_tier": "low" | "medium" | "high",
    "person_detected": true | false,
    "person_count": <number>,
    "summary": "one sentence description"
}"""

    try:
        from google.genai import types

        image_bytes = path.read_bytes()

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            ],
        )
        print(f"SUCCESS: Security analysis complete")
        print(f"Response:\n{response.text}")
        return True
    except Exception as e:
        print(f"FAIL: Security analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Gemini API Test Suite")
    print("=" * 60)

    results = {}

    results["api_key"] = test_api_key_format()
    results["sdk_import"] = test_sdk_import()

    client = test_client_creation()
    results["client"] = bool(client)

    if client:
        results["simple_request"] = test_simple_request(client)
        results["image_request"] = test_image_request(client)
        results["security_analysis"] = test_security_analysis(client)

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
