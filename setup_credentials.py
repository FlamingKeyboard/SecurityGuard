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

    # Step 5: GCP Integration
    print("Step 5: Google Cloud Platform Integration")
    print("-" * 40)

    gcp_project = stored_creds.get("gcp_project_id", "") or os.environ.get("GCP_PROJECT_ID", "")
    gcp_service_account = stored_creds.get("gcp_service_account_file", "") or os.environ.get("GCP_SERVICE_ACCOUNT_FILE", "")

    if gcp_project:
        print(f"GCP Project ID: {gcp_project} (stored)")
    else:
        print("GCP integration enables:")
        print("  - Image archival to Google Cloud Storage")
        print("  - Gemini API logging to BigQuery")
        print()
        print("To set up GCP:")
        print("  1. Create a GCP project at https://console.cloud.google.com")
        print("  2. Enable BigQuery and Cloud Storage APIs")
        print("  3. For local dev: create a service account with Storage Admin + BigQuery Admin roles")
        print("     Download the JSON key file")
        print("  4. For production: use Application Default Credentials (ADC)")
        print()
        gcp_project = input("Enter your GCP Project ID (or Enter to skip): ").strip()

    if gcp_project and not gcp_service_account:
        print()
        print("For local development, you can use a service account key file.")
        print("For production (GCE/Cloud Run), leave empty to use ADC.")
        gcp_service_account = input("Path to service account JSON file (or Enter to skip): ").strip()

    if gcp_project:
        os.environ["GCP_PROJECT_ID"] = gcp_project
        save_data = {"gcp_project_id": gcp_project}
        if gcp_service_account:
            os.environ["GCP_SERVICE_ACCOUNT_FILE"] = gcp_service_account
            save_data["gcp_service_account_file"] = gcp_service_account
        save_credentials(save_data)
        print("GCP settings saved.")
    else:
        print("Skipping GCP setup. Images will only be stored locally.")

    print()

    # Step 6: Test RTSP capture (optional)
    print("Step 6: Test RTSP Frame Capture")
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

    # Disconnect Vivint
    if connected:
        await client.disconnect()

    print()

    # Step 7: Optional diagnostics
    print("Step 7: Run Diagnostics (Optional)")
    print("-" * 40)
    print("These tests verify your setup is working correctly.")
    print()

    run_diag = input("Run diagnostic tests? (y/n): ").strip().lower()
    if run_diag == 'y':
        await run_diagnostics(pushover_token, pushover_user, gemini_key, gcp_project)

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


async def run_diagnostics(pushover_token: str, pushover_user: str, gemini_key: str, gcp_project: str = ""):
    """Run optional diagnostic tests."""
    import socket
    import aiohttp

    print()
    print("Running diagnostics...")
    print()

    # Test 1: Hub connectivity
    print("[1/4] Hub Connectivity")
    hub_ip = config.VIVINT_HUB_IP
    hub_port = config.VIVINT_HUB_RTSP_PORT

    if not hub_ip:
        print("  [SKIP] VIVINT_HUB_IP not configured")
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            result = sock.connect_ex((hub_ip, hub_port))
            if result == 0:
                print(f"  [OK] Hub reachable at {hub_ip}:{hub_port}")
            else:
                print(f"  [FAIL] Cannot connect to {hub_ip}:{hub_port}")
                print("    - Check hub IP is correct")
                print("    - For cloud: verify Tailscale subnet routing")
        except socket.timeout:
            print(f"  [FAIL] Connection timed out to {hub_ip}:{hub_port}")
        except Exception as e:
            print(f"  [FAIL] {e}")
        finally:
            sock.close()

    print()

    # Test 2: Gemini API
    print("[2/4] Gemini API")
    if not gemini_key:
        print("  [SKIP] Gemini API key not configured")
    else:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents='Reply with exactly: OK',
            )
            if 'OK' in response.text:
                print("  [OK] Gemini API responding")
            else:
                print(f"  [OK] Gemini API responding (got: {response.text[:50]})")
        except Exception as e:
            print(f"  [FAIL] Gemini API error: {e}")

    print()

    # Test 3: Pushover
    print("[3/4] Pushover Notification")
    if not pushover_token or not pushover_user:
        print("  [SKIP] Pushover not configured")
    else:
        send_test = input("  Send test notification to your phone? (y/n): ").strip().lower()
        if send_test == 'y':
            try:
                url = "https://api.pushover.net/1/messages.json"
                data = aiohttp.FormData()
                data.add_field("token", pushover_token)
                data.add_field("user", pushover_user)
                data.add_field("title", "Security Guard - Test")
                data.add_field("message", "Setup diagnostic test successful!")
                data.add_field("priority", "0")  # Normal priority

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=data) as resp:
                        if resp.status == 200:
                            print("  [OK] Test notification sent - check your phone!")
                        else:
                            body = await resp.text()
                            print(f"  [FAIL] Pushover error: {resp.status} - {body}")
            except Exception as e:
                print(f"  [FAIL] Pushover error: {e}")
        else:
            print("  [SKIP] Skipped by user")

    print()

    # Test 4: GCP Integration (GCS + BigQuery)
    print("[4/4] Google Cloud Platform")
    if not gcp_project:
        print("  [SKIP] GCP not configured")
    else:
        try:
            from gcp_storage import test_gcs_connection, get_bucket_name
            from gcp_logging import test_bigquery_connection

            # Test GCS
            gcs_ok, gcs_msg = test_gcs_connection()
            if gcs_ok:
                print(f"  [OK] GCS: {gcs_msg}")
            else:
                print(f"  [FAIL] GCS: {gcs_msg}")

            # Test BigQuery
            bq_ok, bq_msg = test_bigquery_connection()
            if bq_ok:
                print(f"  [OK] BigQuery: {bq_msg}")
            else:
                print(f"  [FAIL] BigQuery: {bq_msg}")

        except ImportError as e:
            print(f"  [FAIL] Missing GCP dependencies: {e}")
            print("  Install with: pip install google-cloud-storage google-cloud-bigquery")
        except Exception as e:
            print(f"  [FAIL] GCP error: {e}")

    print()
    print("Diagnostics complete.")


if __name__ == "__main__":
    asyncio.run(main())
