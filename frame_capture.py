"""Frame capture from RTSP streams using ffmpeg."""

import asyncio
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import config

_LOGGER = logging.getLogger(__name__)


async def capture_frames(
    rtsp_url: str,
    camera_name: str,
    count: int = config.FRAME_BURST_COUNT,
    interval_ms: int = config.FRAME_BURST_INTERVAL_MS,
) -> list[Path]:
    """
    Capture multiple frames from an RTSP stream.

    Args:
        rtsp_url: The RTSP URL to capture from
        camera_name: Name of the camera (for filename)
        count: Number of frames to capture
        interval_ms: Interval between frames in milliseconds

    Returns:
        List of paths to captured frame files
    """
    captured_frames = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in camera_name)

    for i in range(count):
        output_path = config.FRAME_CAPTURE_DIR / f"{safe_name}_{timestamp}_{i:02d}.jpg"

        try:
            # Use ffmpeg to capture a single frame
            # -rtsp_transport tcp: Use TCP for reliability
            # -y: Overwrite output file
            # -frames:v 1: Capture only 1 frame
            # -q:v 2: High quality JPEG (2 is high quality, 31 is lowest)
            cmd = [
                "ffmpeg",
                "-rtsp_transport", "tcp",
                "-i", rtsp_url,
                "-frames:v", "1",
                "-q:v", "2",
                "-y",
                str(output_path),
            ]

            # Run ffmpeg with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.RTSP_TIMEOUT_SECONDS
                )
                if process.returncode == 0 and output_path.exists():
                    captured_frames.append(output_path)
                    _LOGGER.debug("Captured frame %d for %s: %s", i, camera_name, output_path)
                else:
                    _LOGGER.warning("Frame capture failed for %s: %s",
                                    camera_name, stderr.decode() if stderr else "unknown error")
            except asyncio.TimeoutError:
                process.kill()
                _LOGGER.warning("Frame capture timed out for %s", camera_name)

        except Exception as e:
            _LOGGER.error("Error capturing frame %d for %s: %s", i, camera_name, e)

        # Wait between frames (if not the last frame)
        if i < count - 1:
            await asyncio.sleep(interval_ms / 1000)

    return captured_frames


async def capture_single_frame(rtsp_url: str, camera_name: str) -> Path | None:
    """Capture a single frame from an RTSP stream."""
    frames = await capture_frames(rtsp_url, camera_name, count=1)
    return frames[0] if frames else None


def cleanup_old_frames(max_age_hours: int = 1) -> int:
    """
    Remove frames older than max_age_hours.

    Returns:
        Number of files deleted
    """
    deleted = 0
    cutoff = time.time() - (max_age_hours * 3600)

    for frame_file in config.FRAME_CAPTURE_DIR.glob("*.jpg"):
        try:
            if frame_file.stat().st_mtime < cutoff:
                frame_file.unlink()
                deleted += 1
        except Exception as e:
            _LOGGER.warning("Error deleting old frame %s: %s", frame_file, e)

    if deleted:
        _LOGGER.info("Cleaned up %d old frame files", deleted)
    return deleted


async def test_capture(rtsp_url: str):
    """Test frame capture from an RTSP URL."""
    logging.basicConfig(level=logging.DEBUG)

    print(f"Testing capture from: {rtsp_url[:50]}...")
    frames = await capture_frames(rtsp_url, "test_camera", count=3)

    if frames:
        print(f"Successfully captured {len(frames)} frames:")
        for f in frames:
            print(f"  - {f} ({f.stat().st_size} bytes)")
    else:
        print("No frames captured")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        asyncio.run(test_capture(sys.argv[1]))
    else:
        print("Usage: python frame_capture.py <rtsp_url>")
