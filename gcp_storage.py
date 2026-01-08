"""Google Cloud Storage integration for media (image and video) archival."""

import logging
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import config

# Timeout for GCP API calls (seconds)
GCP_API_TIMEOUT = 15

_LOGGER = logging.getLogger(__name__)

# Thread lock for global state (RLock allows same thread to re-acquire)
_client_lock = threading.RLock()

# GCS client (initialized lazily, protected by _client_lock)
_gcs_client = None
_bucket = None
_gcs_init_failed = False  # Set to True only for permanent failures (credentials/config)


def _get_gcs_client():
    """Get or create GCS client with appropriate credentials (thread-safe)."""
    global _gcs_client, _gcs_init_failed

    # Fast path without lock
    if _gcs_client is not None:
        return _gcs_client

    # If we've already determined initialization will fail permanently, don't retry
    if _gcs_init_failed:
        return None

    with _client_lock:
        # Double-check after acquiring lock
        if _gcs_client is not None:
            return _gcs_client

        if _gcs_init_failed:
            return None

        try:
            from google.cloud import storage

            # Use service account file if specified (local development)
            if config.GCP_SERVICE_ACCOUNT_FILE and os.path.exists(config.GCP_SERVICE_ACCOUNT_FILE):
                _gcs_client = storage.Client.from_service_account_json(
                    config.GCP_SERVICE_ACCOUNT_FILE
                )
                _LOGGER.info("GCS client initialized with service account file")
            else:
                # Use Application Default Credentials (production/GCE)
                _gcs_client = storage.Client(project=config.GCP_PROJECT_ID or None)
                _LOGGER.info("GCS client initialized with ADC")

            return _gcs_client

        except Exception as e:
            # Determine if this is a permanent or transient failure
            error_str = str(e).lower()
            error_type = type(e).__name__

            # Permanent failures: credentials, authentication, missing config
            permanent_error_indicators = [
                'credential',
                'authentication',
                'permission denied',
                'unauthorized',
                'invalid',
                'service account',
                'project',
                'defaultcredentialserror',
            ]

            # Transient failures: network, timeout, connection issues
            transient_error_indicators = [
                'timeout',
                'connection',
                'network',
                'temporary',
                'unavailable',
                'deadline exceeded',
            ]

            is_permanent = any(indicator in error_str or indicator in error_type.lower()
                             for indicator in permanent_error_indicators)
            is_transient = any(indicator in error_str or indicator in error_type.lower()
                             for indicator in transient_error_indicators)

            if is_permanent and not is_transient:
                # This is a permanent failure - mark it and don't retry
                _gcs_init_failed = True
                _LOGGER.error("GCS client initialization failed permanently (credentials/config): %s", e)
            else:
                # This might be transient - allow retry on next call
                _LOGGER.warning("GCS client initialization failed (will retry): %s", e)

            return None


def _get_bucket_name() -> str:
    """Get or generate bucket name."""
    if config.GCS_BUCKET_NAME:
        return config.GCS_BUCKET_NAME

    # Auto-generate bucket name from project ID
    project_id = config.GCP_PROJECT_ID
    if not project_id:
        client = _get_gcs_client()
        if client:
            project_id = client.project

    if project_id:
        return f"securityguard-{project_id}"

    return "securityguard-images"


def _get_bucket():
    """Get or create the GCS bucket (thread-safe)."""
    global _bucket

    # Fast path without lock
    if _bucket is not None:
        return _bucket

    with _client_lock:
        # Double-check after acquiring lock
        if _bucket is not None:
            return _bucket

        client = _get_gcs_client()
        if not client:
            return None

        bucket_name = _get_bucket_name()

        try:
            _bucket = client.bucket(bucket_name)

            # Check if bucket exists, create if not (with timeout)
            if not _bucket.exists(timeout=GCP_API_TIMEOUT):
                _LOGGER.info("Creating GCS bucket: %s", bucket_name)
                try:
                    _bucket = client.create_bucket(bucket_name, location="US", timeout=GCP_API_TIMEOUT)
                    _LOGGER.info("GCS bucket created: %s", bucket_name)
                except Exception as create_err:
                    # Bucket may have been created by another process
                    if "already exists" in str(create_err).lower():
                        _LOGGER.debug("Bucket already exists (race condition): %s", bucket_name)
                        _bucket = client.bucket(bucket_name)
                    else:
                        raise
            else:
                _LOGGER.debug("Using existing GCS bucket: %s", bucket_name)

            return _bucket

        except Exception as e:
            _LOGGER.error("Failed to get/create GCS bucket: %s", e)
            return None


def generate_media_path(
    camera_name: str,
    event_id: str,
    timestamp: datetime,
    frame_index: int = 0,
    extension: str = "jpg",
) -> str:
    """
    Generate a GCS path for a media file (image or video) with relevant metadata.

    Format: images/YYYY/MM/DD/camera_name/event_id/HHMMSS_index.ext

    Args:
        camera_name: Name of the camera
        event_id: Event ID for grouping
        timestamp: Timestamp of the capture
        frame_index: Frame/clip index (0 for single captures)
        extension: File extension without dot (jpg, mp4)

    Returns:
        GCS path string
    """
    date_path = timestamp.strftime("%Y/%m/%d")
    time_str = timestamp.strftime("%H%M%S")

    # Sanitize camera name for path
    safe_camera = camera_name.replace(" ", "_").replace("/", "_")

    return (
        f"{config.GCS_IMAGE_PREFIX}/{date_path}/{safe_camera}/"
        f"{event_id}/{time_str}_{frame_index:02d}.{extension}"
    )


# Backward compatibility alias
def generate_image_path(
    camera_name: str,
    event_id: str,
    timestamp: datetime,
    frame_index: int = 0,
) -> str:
    """Generate a GCS path for an image. (Deprecated: use generate_media_path)"""
    return generate_media_path(camera_name, event_id, timestamp, frame_index, "jpg")


# MIME type mapping for supported media files
_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


def upload_media(
    local_path: Path,
    camera_name: str,
    event_id: str,
    timestamp: datetime,
    frame_index: int = 0,
) -> Optional[str]:
    """
    Upload a media file (image or video) to GCS.

    Automatically detects file type from extension and sets appropriate
    content type for the upload.

    Args:
        local_path: Path to the local file
        camera_name: Name of the camera
        event_id: Event ID for grouping
        timestamp: Timestamp of the capture
        frame_index: Frame/clip index

    Returns:
        GCS URI (gs://bucket/path) or None if failed
    """
    bucket = _get_bucket()
    if not bucket:
        _LOGGER.warning("GCS not available, skipping media upload")
        return None

    if not local_path.exists():
        _LOGGER.warning("Media file not found: %s", local_path)
        return None

    # Determine extension and content type
    extension = local_path.suffix.lower()
    content_type = _MIME_TYPES.get(extension, "application/octet-stream")

    # Generate GCS path with correct extension (without the dot)
    gcs_path = generate_media_path(
        camera_name, event_id, timestamp, frame_index, extension.lstrip(".")
    )

    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path), content_type=content_type)

        uri = f"gs://{bucket.name}/{gcs_path}"
        _LOGGER.debug("Uploaded %s to %s", local_path.name, uri)
        return uri

    except Exception as e:
        _LOGGER.error("Failed to upload media to GCS: %s", e)
        return None


# Backward compatibility alias
def upload_image(
    local_path: Path,
    camera_name: str,
    event_id: str,
    timestamp: datetime,
    frame_index: int = 0,
) -> Optional[str]:
    """Upload an image to GCS. (Deprecated: use upload_media)"""
    return upload_media(local_path, camera_name, event_id, timestamp, frame_index)


def upload_image_bytes(
    image_bytes: bytes,
    camera_name: str,
    event_id: str,
    timestamp: datetime,
    frame_index: int = 0,
) -> Optional[str]:
    """
    Upload image bytes directly to GCS.

    Returns the GCS URI or None if failed.
    """
    bucket = _get_bucket()
    if not bucket:
        _LOGGER.warning("GCS not available, skipping image upload")
        return None

    gcs_path = generate_image_path(camera_name, event_id, timestamp, frame_index)

    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(image_bytes, content_type="image/jpeg")

        uri = f"gs://{bucket.name}/{gcs_path}"
        _LOGGER.debug("Uploaded image bytes to %s", uri)
        return uri

    except Exception as e:
        _LOGGER.error("Failed to upload image bytes to GCS: %s", e)
        return None


def get_disk_space_gb() -> float:
    """Get available disk space in GB."""
    try:
        total, used, free = shutil.disk_usage(config.DATA_DIR)
        return free / (1024 ** 3)
    except Exception as e:
        _LOGGER.error("Failed to get disk space: %s", e)
        # Return 0 to trigger archival on error (safer default than assuming infinite space)
        return 0.0


def archive_old_media(
    max_age_days: int = None,
    force_if_low_disk: bool = True,
) -> tuple[int, int]:
    """
    Archive old media files (images and videos) to GCS and delete local copies.

    Archives files when either:
    - Files are older than max_age_days (default: 30 days)
    - Disk space is below DISK_SPACE_THRESHOLD_GB (default: 10GB)

    Args:
        max_age_days: Archive media older than this (default: config.IMAGE_RETENTION_DAYS)
        force_if_low_disk: Force archive if disk space is low

    Returns:
        Tuple of (uploaded_count, deleted_count)
    """
    if max_age_days is None:
        max_age_days = config.IMAGE_RETENTION_DAYS

    # Check if we should proceed
    disk_space = get_disk_space_gb()
    low_disk = disk_space < config.DISK_SPACE_THRESHOLD_GB

    if low_disk:
        _LOGGER.warning(
            "Low disk space (%.1f GB < %.1f GB threshold), forcing media archival",
            disk_space, config.DISK_SPACE_THRESHOLD_GB
        )
        # Archive everything older than 1 day if low disk
        max_age_days = min(max_age_days, 1)

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    frames_dir = config.FRAME_CAPTURE_DIR

    if not frames_dir.exists():
        return 0, 0

    uploaded = 0
    deleted = 0
    uploaded_images = 0
    uploaded_videos = 0

    # Process both images (.jpg) and videos (.mp4)
    media_patterns = ["*.jpg", "*.mp4"]

    for pattern in media_patterns:
        for media_path in frames_dir.glob(pattern):
            try:
                # Get file modification time
                mtime = datetime.fromtimestamp(media_path.stat().st_mtime, tz=timezone.utc)

                if mtime < cutoff:
                    # Parse camera name and timestamp from filename
                    # Format: {camera_name}_{YYYYMMDD}_{HHMMSS}[_{NN}].ext
                    # Camera name may contain underscores, so parse from right side
                    # Videos may not have frame index, images do
                    parts = media_path.stem.split("_")

                    camera_name = "Unknown"
                    timestamp = mtime
                    frame_index = 0

                    if len(parts) >= 3:
                        try:
                            # Check if last part is a frame index (2 digits)
                            if len(parts[-1]) == 2 and parts[-1].isdigit():
                                # Has frame index: camera_YYYYMMDD_HHMMSS_NN
                                frame_index = int(parts[-1])
                                time_part = parts[-2]
                                date_part = parts[-3]
                                camera_name = "_".join(parts[:-3])
                            else:
                                # No frame index: camera_YYYYMMDD_HHMMSS
                                time_part = parts[-1]
                                date_part = parts[-2]
                                camera_name = "_".join(parts[:-2])
                                frame_index = 0

                            timestamp = datetime.strptime(
                                f"{date_part}_{time_part}", "%Y%m%d_%H%M%S"
                            ).replace(tzinfo=timezone.utc)
                        except (ValueError, IndexError):
                            # Parsing failed, use defaults
                            pass

                    # Generate event ID from timestamp (for old files without event_id)
                    event_id = f"archive_{timestamp.strftime('%Y%m%d_%H%M%S')}"

                    # Upload to GCS (upload_media handles both images and videos)
                    uri = upload_media(media_path, camera_name, event_id, timestamp, frame_index)

                    if uri:
                        uploaded += 1
                        if media_path.suffix.lower() == ".mp4":
                            uploaded_videos += 1
                        else:
                            uploaded_images += 1

                        # Delete local file
                        try:
                            media_path.unlink()
                            deleted += 1
                            _LOGGER.debug("Archived and deleted: %s -> %s", media_path.name, uri)
                        except OSError as unlink_err:
                            _LOGGER.warning("Failed to delete %s after upload: %s", media_path.name, unlink_err)
                    else:
                        _LOGGER.warning("Failed to upload %s, keeping local copy", media_path.name)

            except Exception as e:
                _LOGGER.error("Error processing %s: %s", media_path.name, e)

    if uploaded > 0:
        _LOGGER.info(
            "Archived %d media files to GCS (%d images, %d videos), deleted %d local files",
            uploaded, uploaded_images, uploaded_videos, deleted
        )

    return uploaded, deleted


# Backward compatibility alias
def archive_old_images(
    max_age_days: int = None,
    force_if_low_disk: bool = True,
) -> tuple[int, int]:
    """Archive old media to GCS. (Deprecated: use archive_old_media)"""
    return archive_old_media(max_age_days, force_if_low_disk)


def test_gcs_connection() -> tuple[bool, str]:
    """
    Test GCS connectivity with timeout.

    Returns (success, message).
    """
    if not config.GCP_PROJECT_ID and not config.GCP_SERVICE_ACCOUNT_FILE:
        return False, "GCP_PROJECT_ID or GCP_SERVICE_ACCOUNT_FILE not configured"

    try:
        bucket = _get_bucket()
        if bucket is None:
            return False, "Failed to get/create bucket"

        # Try to list blobs (limited to 1) to verify access (with timeout)
        list(bucket.list_blobs(max_results=1, timeout=GCP_API_TIMEOUT))

        return True, f"Connected to bucket: {bucket.name}"

    except Exception as e:
        return False, f"GCS error: {e}"


def get_bucket_name() -> str:
    """Get the bucket name being used."""
    return _get_bucket_name()
