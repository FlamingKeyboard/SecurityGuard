"""
First-time setup script for Vivint Security Guard.

This script helps you:
1. Set up Vivint credentials and perform initial authentication
2. Save refresh token for future use (avoiding MFA each time)
3. Test RTSP camera access
4. Optionally set up Gemini API key

All credentials are stored encrypted using Windows DPAPI.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add vivintpy to path
sys.path.insert(0, str(Path(__file__).parent / "vivintpy"))

# Reload config module to pick up env changes
import importlib
import config
importlib.reload(config)

from vivint_client import save_credentials, load_credentials, get_stored_credential


async def main():
    print("=" * 60)
    print("Vivint Security Guard - Setup")
    print("=" * 60)
    print()

    # Load any existing stored credentials
    stored_creds = load_credentials() or {}

    # Step 1: Vivint credentials
    print("Step 1: Vivint Credentials")
    print("-" * 40)

    # Username - check env, then stored, then prompt
    username = os.environ.get("VIVINT_USERNAME", "")
    if not username:
        username = stored_creds.get("username", "")

    if username:
        print(f"Username: {username}")
    else:
        username = input("Enter your Vivint username (email): ").strip()

    # Password - check env, then stored, then prompt
    password = os.environ.get("VIVINT_PASSWORD", "")
    if not password:
        password = stored_creds.get("password", "")

    if password:
        print("Password: (stored securely)")
    else:
        import getpass
        password = getpass.getpass("Enter your Vivint password: ")

    # Set in environment for this session
    os.environ["VIVINT_USERNAME"] = username
    os.environ["VIVINT_PASSWORD"] = password
    importlib.reload(config)

    print()

    # Step 2: Test Vivint connection
    print("Step 2: Testing Vivint Connection")
    print("-" * 40)

    from vivint_client import VivintClient

    client = VivintClient()
    connected = False
    rtsp_urls = {}

    try:
        connected = await client.connect()
        if connected:
            print("\nVivint connection successful!")
            print(f"Found {len(client.cameras)} cameras:")
            for cam in client.cameras:
                print(f"  - {cam.name} (ID: {cam.id}, Online: {cam.is_online})")
                url = client.get_rtsp_url(cam.id)
                if url:
                    rtsp_urls[cam.id] = url
                    # Mask credentials
                    import re
                    masked = re.sub(r':([^:@]+)@', ':****@', url)
                    print(f"    RTSP URL: {masked}")

            # Save credentials (including username, password, refresh token)
            save_credentials({
                "username": username,
                "password": password,
            })
            print("\nCredentials saved securely.")
        else:
            print("Failed to connect to Vivint. Please check your credentials.")
            return
    except Exception as e:
        print(f"Connection error: {e}")
        return

    print()

    # Step 3: Gemini API key
    print("Step 3: Gemini API Key")
    print("-" * 40)

    def is_valid_gemini_key(key):
        """Check if key has valid format (starts with AIza)."""
        return key and key.startswith("AIza")

    # Check env, then stored, then prompt
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        gemini_key = stored_creds.get("gemini_api_key", "")

    if gemini_key and is_valid_gemini_key(gemini_key):
        print(f"Gemini API key: {gemini_key[:15]}... (stored, valid format)")
    elif gemini_key:
        print(f"Stored key has invalid format: {gemini_key[:10]}...")
        print("Valid Gemini API keys start with 'AIza'")
        gemini_key = ""  # Force re-entry

    if not gemini_key:
        print("To use AI-powered security analysis, you need a Gemini API key.")
        print()
        print("Get one at: https://aistudio.google.com/apikey")
        print("  1. Click 'Create API Key'")
        print("  2. The key should start with 'AIzaSy...'")
        print()
        while True:
            gemini_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
            if not gemini_key:
                break
            if is_valid_gemini_key(gemini_key):
                break
            print(f"Invalid key format. Keys should start with 'AIza', got: {gemini_key[:10]}...")
            print("Please get a key from https://aistudio.google.com/apikey")
            gemini_key = ""

    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        save_credentials({"gemini_api_key": gemini_key})
        print("Gemini API key saved securely.")
    else:
        print("Skipping Gemini setup. You can run this script again later.")

    print()

    # Step 4: Pushover credentials
    print("Step 4: Pushover Push Notifications")
    print("-" * 40)

    pushover_token = stored_creds.get("pushover_token", "")
    pushover_user = stored_creds.get("pushover_user", "")

    if pushover_token and pushover_user:
        print(f"Pushover App Token: {pushover_token[:8]}... (stored)")
        print(f"Pushover User Key: {pushover_user[:8]}... (stored)")
    else:
        print("Pushover sends push notifications to your phone with images.")
        print()
        print("Setup (one-time, $5 for the app):")
        print("  1. Download Pushover app on iOS/Android")
        print("  2. Create account at https://pushover.net")
        print("  3. Your User Key is shown on the dashboard")
        print("  4. Create an Application at https://pushover.net/apps/build")
        print("     - Name it 'Vivint Security Guard' or similar")
        print("     - Copy the API Token/Key")
        print()

        pushover_user = input("Enter your Pushover User Key (or Enter to skip): ").strip()
        if pushover_user:
            pushover_token = input("Enter your Pushover App Token: ").strip()

    if pushover_token and pushover_user:
        save_credentials({
            "pushover_token": pushover_token,
            "pushover_user": pushover_user,
        })
        print("Pushover credentials saved securely.")
    else:
        print("Skipping Pushover setup. Notifications will only appear in console.")

    print()

    # Step 5: Test RTSP capture (optional)
    print("Step 5: Test RTSP Frame Capture")
    print("-" * 40)

    test_rtsp = input("Test frame capture? (y/n): ").strip().lower()
    if test_rtsp == 'y' and client.cameras and rtsp_urls:
        from frame_capture import capture_single_frame

        cam = client.cameras[0]
        url = rtsp_urls.get(cam.id)
        if url:
            print(f"\nCapturing frame from {cam.name}...")
            frame = await capture_single_frame(url, cam.name)
            if frame:
                print(f"Success! Frame saved to: {frame}")
                print(f"File size: {frame.stat().st_size} bytes")
            else:
                print("Frame capture failed.")
                print("\nTroubleshooting:")
                print("  1. Make sure ffmpeg is installed and in PATH")
                print("  2. Check if your PC can reach the Vivint hub")
        else:
            print("No RTSP URL available for first camera.")

    # Disconnect
    if connected:
        await client.disconnect()

    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("To run the security guard service, simply run:")
    print()
    print("    python security_guard.py")
    print()
    print("All credentials are stored securely - no environment variables needed!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
