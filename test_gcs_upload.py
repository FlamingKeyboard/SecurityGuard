#!/usr/bin/env python3
"""Test GCS upload with a real image."""
import sys
sys.path.insert(0, "/app")

from pathlib import Path
from datetime import datetime, timezone
from gcp_storage import upload_image, get_bucket_name

frames_dir = Path("/app/data/frames")
images = list(frames_dir.glob("*.jpg"))

print("=== GCS Upload Test ===")
print(f"Bucket: {get_bucket_name()}")
print(f"Found {len(images)} local images")
print()

if images:
    # Upload the first image as a test
    img = images[0]
    print(f"Uploading: {img.name}")

    # Parse filename: Doorbell_20260107_035847_00.jpg
    parts = img.stem.split("_")
    camera_name = parts[0]

    uri = upload_image(
        local_path=img,
        camera_name=camera_name,
        event_id="test_manual_upload",
        timestamp=datetime.now(timezone.utc),
        frame_index=0
    )

    if uri:
        print(f"SUCCESS: {uri}")
    else:
        print("FAILED: Upload returned None")
else:
    print("No images found to upload")

print()
print("=== Done ===")
