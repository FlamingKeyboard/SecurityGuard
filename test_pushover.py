"""Test script for Pushover push notifications."""

import asyncio
import sys
from pathlib import Path

import aiohttp

from vivint_client import load_credentials


async def send_test_notification(critical: bool = True, with_image: bool = True):
    """Send a test push notification via Pushover."""

    creds = load_credentials() or {}
    token = creds.get("pushover_token")
    user = creds.get("pushover_user")

    if not token or not user:
        print("ERROR: Pushover credentials not configured")
        print("Run: python setup_credentials.py")
        return False

    print(f"Pushover Token: {token[:8]}...")
    print(f"Pushover User: {user[:8]}...")

    # Find a test image
    image_path = None
    if with_image:
        frames_dir = Path(__file__).parent / "data" / "frames"
        frames = list(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
        if frames:
            image_path = frames[0]
            print(f"Using test image: {image_path.name}")
        else:
            print("No test image found (will send without image)")

    # Build the request
    url = "https://api.pushover.net/1/messages.json"

    data = aiohttp.FormData()
    data.add_field("token", token)
    data.add_field("user", user)
    data.add_field("title", "🚨 Test Alert - Vivint Security")
    data.add_field("message", "This is a test notification from Vivint Security Guard.\n\nIf you see this, Pushover is working correctly!")

    # Priority 2 = emergency (critical), requires retry/expire params
    # Priority 1 = high priority, bypasses quiet hours
    if critical:
        data.add_field("priority", "2")
        data.add_field("retry", "30")  # Retry every 30 seconds
        data.add_field("expire", "60")  # Stop retrying after 60 seconds
        print("Priority: CRITICAL (emergency, requires acknowledgment)")
    else:
        data.add_field("priority", "1")
        print("Priority: HIGH (bypasses quiet hours)")

    # Attach image if available
    if image_path and image_path.exists():
        image_bytes = image_path.read_bytes()
        data.add_field(
            "attachment",
            image_bytes,
            filename=image_path.name,
            content_type="image/jpeg",
        )
        print(f"Image attached: {len(image_bytes)} bytes")

    print("\nSending notification...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                body = await resp.json()

                if resp.status == 200 and body.get("status") == 1:
                    print("SUCCESS! Notification sent.")
                    print(f"Request ID: {body.get('request')}")
                    if critical:
                        print("\nThis is a CRITICAL alert - you must acknowledge it in the Pushover app!")
                    return True
                else:
                    print(f"FAILED: {resp.status}")
                    print(f"Response: {body}")

                    # Common errors
                    errors = body.get("errors", [])
                    if "application token is invalid" in str(errors):
                        print("\nYour App Token is invalid. Create a new app at:")
                        print("  https://pushover.net/apps/build")
                    elif "user identifier is invalid" in str(errors):
                        print("\nYour User Key is invalid. Find it at:")
                        print("  https://pushover.net")

                    return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False


async def main():
    print("=" * 50)
    print("Pushover Test")
    print("=" * 50)

    # Parse args
    critical = "--normal" not in sys.argv
    no_image = "--no-image" in sys.argv

    if not critical:
        print("Mode: Normal (high priority)")
    else:
        print("Mode: CRITICAL (emergency)")

    print()

    success = await send_test_notification(critical=critical, with_image=not no_image)

    print()
    if success:
        print("Check your phone for the notification!")
    else:
        print("Test failed. Check your credentials.")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
