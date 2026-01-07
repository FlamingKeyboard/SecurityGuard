"""Frame and video capture from RTSP streams using ffmpeg."""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

_LOGGER = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    """Result of a capture operation."""
    success: bool
    video_path: Optional[Path] = None  # Path to video file if video capture succeeded
    frame_paths: list[Path] = None  # Paths to frame files (fallback or primary)
    is_video: bool = False  # True if video capture was used
    error: Optional[str] = None  # Error message if capture failed

    def __post_init__(self):
        if self.frame_paths is None:
            self.frame_paths = []


async def capture_video(
    rtsp_url: str,
    camera_name: str,
    duration_seconds: int = config.VIDEO_CAPTURE_DURATION_SECONDS,
) -> CaptureResult:
    """
    Capture a short video clip from an RTSP stream using remux (no re-encoding).

    This uses -c:v copy to avoid CPU-intensive transcoding, making it suitable
    for resource-constrained environments like e2-micro VMs.

    Args:
        rtsp_url: The RTSP URL to capture from
        camera_name: Name of the camera (for filename)
        duration_seconds: Duration of clip to capture (default: 3 seconds)

    Returns:
        CaptureResult with video_path if successful
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in camera_name)
    output_path = config.FRAME_CAPTURE_DIR / f"{safe_name}_{timestamp}.mp4"

    try:
        # Use remux (-c:v copy) to avoid CPU-intensive transcoding
        # -t: duration in seconds
        # -c:v copy: copy video stream without re-encoding (remux)
        # -c:a copy: copy audio stream (if present) for voice/sound context
        # -movflags +faststart: optimize for streaming/quick playback
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-t", str(duration_seconds),
            "-c:v", "copy",  # Remux video - no transcoding!
            "-c:a", "copy",  # Remux audio - captures voices, sounds
            "-movflags", "+faststart",
            "-y",
            str(output_path),
        ]

        _LOGGER.debug("Capturing video: %s", " ".join(cmd[:6]) + " ...")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Allow extra time for connection + capture
            timeout = duration_seconds + config.RTSP_TIMEOUT_SECONDS
            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            if process.returncode == 0 and output_path.exists():
                file_size = output_path.stat().st_size
                if file_size > 1000:  # At least 1KB - sanity check
                    _LOGGER.info(
                        "Captured video for %s: %s (%d bytes)",
                        camera_name, output_path.name, file_size
                    )
                    return CaptureResult(
                        success=True,
                        video_path=output_path,
                        is_video=True,
                    )
                else:
                    _LOGGER.warning("Video file too small (%d bytes), likely failed", file_size)
                    output_path.unlink(missing_ok=True)
                    return CaptureResult(
                        success=False,
                        error=f"Video file too small: {file_size} bytes"
                    )
            else:
                error_msg = stderr.decode() if stderr else "unknown error"
                _LOGGER.warning("Video capture failed for %s: %s", camera_name, error_msg[:200])
                return CaptureResult(success=False, error=error_msg[:200])

        except asyncio.TimeoutError:
            process.kill()
            _LOGGER.warning("Video capture timed out for %s", camera_name)
            output_path.unlink(missing_ok=True)
            return CaptureResult(success=False, error="Timeout")

    except Exception as e:
        _LOGGER.error("Error capturing video for %s: %s", camera_name, e)
        return CaptureResult(success=False, error=str(e))


@dataclass
class MultiCameraCapture:
    """Result of multi-camera capture operation."""
    primary_camera: str
    videos: dict[str, Path]  # camera_name -> video_path
    success: bool
    error: Optional[str] = None


async def capture_multiple_cameras(
    camera_urls: dict[str, str],  # camera_name -> rtsp_url
    primary_camera: str,
    duration_seconds: int = config.VIDEO_CAPTURE_DURATION_SECONDS,
) -> MultiCameraCapture:
    """
    Capture video from multiple cameras simultaneously.

    Captures from all provided cameras in parallel for synchronized context.
    Used for multi-camera correlation when motion is detected.

    Args:
        camera_urls: Dict mapping camera names to RTSP URLs
        primary_camera: Name of the camera that triggered the event
        duration_seconds: Duration of each clip

    Returns:
        MultiCameraCapture with paths to all captured videos
    """
    if not camera_urls:
        return MultiCameraCapture(
            primary_camera=primary_camera,
            videos={},
            success=False,
            error="No camera URLs provided"
        )

    _LOGGER.info("Capturing from %d cameras: %s", len(camera_urls), list(camera_urls.keys()))

    # Capture from all cameras in parallel
    async def capture_one(name: str, url: str) -> tuple[str, CaptureResult]:
        result = await capture_video(url, name, duration_seconds)
        return (name, result)

    tasks = [capture_one(name, url) for name, url in camera_urls.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful captures
    videos = {}
    errors = []

    for result in results:
        if isinstance(result, Exception):
            errors.append(str(result))
            continue

        camera_name, capture_result = result
        if capture_result.success and capture_result.video_path:
            videos[camera_name] = capture_result.video_path
            _LOGGER.info("  Captured %s: %s", camera_name, capture_result.video_path.name)
        else:
            errors.append(f"{camera_name}: {capture_result.error}")

    # Ensure primary camera was captured
    if primary_camera not in videos:
        return MultiCameraCapture(
            primary_camera=primary_camera,
            videos=videos,
            success=False,
            error=f"Primary camera {primary_camera} capture failed"
        )

    return MultiCameraCapture(
        primary_camera=primary_camera,
        videos=videos,
        success=True,
        error="; ".join(errors) if errors else None
    )


async def capture_with_fallback(
    rtsp_url: str,
    camera_name: str,
) -> CaptureResult:
    """
    Capture video if enabled, falling back to frames if video fails.

    This is the main entry point for capture operations. It respects the
    VIDEO_CAPTURE_ENABLED config setting and automatically falls back to
    frame capture if video capture fails or is disabled.

    Args:
        rtsp_url: The RTSP URL to capture from
        camera_name: Name of the camera

    Returns:
        CaptureResult with either video_path or frame_paths populated
    """
    # Try video capture if enabled
    if config.VIDEO_CAPTURE_ENABLED:
        _LOGGER.info("Attempting video capture for %s...", camera_name)
        result = await capture_video(rtsp_url, camera_name)

        if result.success:
            return result

        # Video failed - fall back to frames if configured
        if config.VIDEO_FALLBACK_TO_FRAMES:
            _LOGGER.info("Video capture failed, falling back to frames for %s", camera_name)
        else:
            return result  # Return the failure

    # Capture frames (either as fallback or primary method)
    frames = await capture_frames(rtsp_url, camera_name)

    if frames:
        return CaptureResult(
            success=True,
            frame_paths=frames,
            is_video=False,
        )
    else:
        return CaptureResult(
            success=False,
            error="Frame capture failed"
        )


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
    Remove frames and videos older than max_age_hours.

    Returns:
        Number of files deleted
    """
    deleted = 0
    cutoff = time.time() - (max_age_hours * 3600)

    # Clean up both JPG frames and MP4 videos
    for pattern in ["*.jpg", "*.mp4"]:
        for media_file in config.FRAME_CAPTURE_DIR.glob(pattern):
            try:
                if media_file.stat().st_mtime < cutoff:
                    media_file.unlink()
                    deleted += 1
            except Exception as e:
                _LOGGER.warning("Error deleting old file %s: %s", media_file, e)

    if deleted:
        _LOGGER.info("Cleaned up %d old media files", deleted)
    return deleted


async def test_capture(rtsp_url: str, test_video: bool = False):
    """Test frame or video capture from an RTSP URL."""
    logging.basicConfig(level=logging.DEBUG)

    print(f"Testing capture from: {rtsp_url[:50]}...")

    if test_video:
        print("Testing VIDEO capture (remux mode)...")
        result = await capture_video(rtsp_url, "test_camera", duration_seconds=3)
        if result.success:
            print(f"Successfully captured video:")
            print(f"  - {result.video_path} ({result.video_path.stat().st_size} bytes)")
        else:
            print(f"Video capture failed: {result.error}")
    else:
        print("Testing FRAME capture...")
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
        test_video = "--video" in sys.argv
        url = [arg for arg in sys.argv[1:] if not arg.startswith("--")][0]
        asyncio.run(test_capture(url, test_video=test_video))
    else:
        print("Usage: python frame_capture.py <rtsp_url> [--video]")
        print("  --video: Test video capture instead of frame capture")
