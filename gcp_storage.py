"""Google Cloud Storage integration for image archival."""

import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import config

# Timeout for GCP API calls (seconds)
GCP_API_TIMEOUT = 15

_LOGGER = logging.getLogger(__name__)

# Thread lock for global state
_client_lock = threading.Lock()

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

            # Check if bucket exists, create if not
            if not _bucket.exists():
                _LOGGER.info("Creating GCS bucket: %s", bucket_name)
                try:
                    _bucket = client.create_bucket(bucket_name, location="US")
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


def generate_image_path(
    camera_name: str,
    event_id: str,
    timestamp: datetime,
    frame_index: int = 0,
) -> str:
    """
    Generate a GCS path for an image with relevant metadata.

    Format: images/YYYY/MM/DD/camera_name/event_id/HHMMSS_frame.jpg
    """
    date_path = timestamp.strftime("%Y/%m/%d")
    time_str = timestamp.strftime("%H%M%S")

    # Sanitize camera name for path
    safe_camera = camera_name.replace(" ", "_").replace("/", "_")

    return (
        f"{config.GCS_IMAGE_PREFIX}/{date_path}/{safe_camera}/"
        f"{event_id}/{time_str}_{frame_index:02d}.jpg"
    )


def upload_image(
    local_path: Path,
    camera_name: str,
    event_id: str,
    timestamp: datetime,
    frame_index: int = 0,
) -> Optional[str]:
    """
    Upload an image to GCS.

    Returns the GCS URI (gs://bucket/path) or None if failed.
    """
    bucket = _get_bucket()
    if not bucket:
        _LOGGER.warning("GCS not available, skipping image upload")
        return None

    if not local_path.exists():
        _LOGGER.warning("Image file not found: %s", local_path)
        return None

    gcs_path = generate_image_path(camera_name, event_id, timestamp, frame_index)

    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path), content_type="image/jpeg")

        uri = f"gs://{bucket.name}/{gcs_path}"
        _LOGGER.debug("Uploaded image to %s", uri)
        return uri

    except Exception as e:
        _LOGGER.error("Failed to upload image to GCS: %s", e)
        return None


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


def archive_old_images(
    max_age_days: int = None,
    force_if_low_disk: bool = True,
) -> tuple[int, int]:
    """
    Archive old images to GCS and delete local copies.

    Args:
        max_age_days: Archive images older than this (default: config.IMAGE_RETENTION_DAYS)
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
        _LOGGER.warning("Low disk space (%.1f GB), forcing image archival", disk_space)
        # Archive everything older than 1 day if low disk
        max_age_days = min(max_age_days, 1)

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    frames_dir = config.FRAME_CAPTURE_DIR

    if not frames_dir.exists():
        return 0, 0

    uploaded = 0
    deleted = 0

    for image_path in frames_dir.glob("*.jpg"):
        try:
            # Get file modification time
            mtime = datetime.fromtimestamp(image_path.stat().st_mtime, tz=timezone.utc)

            if mtime < cutoff:
                # Parse camera name and timestamp from filename
                # Format: {camera_name}_{YYYYMMDD}_{HHMMSS}_{NN}.jpg
                # Camera name may contain underscores, so parse from right side
                parts = image_path.stem.split("_")
                if len(parts) >= 4:
                    try:
                        # Last part is frame index (NN)
                        frame_index = int(parts[-1])
                        # Second-to-last is time (HHMMSS)
                        time_part = parts[-2]
                        # Third-to-last is date (YYYYMMDD)
                        date_part = parts[-3]
                        # Everything before is camera name (may have underscores)
                        camera_name = "_".join(parts[:-3])

                        timestamp = datetime.strptime(
                            f"{date_part}_{time_part}", "%Y%m%d_%H%M%S"
                        ).replace(tzinfo=timezone.utc)
                    except (ValueError, IndexError):
                        camera_name = "Unknown"
                        timestamp = mtime
                        frame_index = 0
                else:
                    camera_name = "Unknown"
                    timestamp = mtime
                    frame_index = 0

                # Generate event ID from timestamp (for old images without event_id)
                event_id = f"archive_{timestamp.strftime('%Y%m%d_%H%M%S')}"

                # Upload to GCS
                uri = upload_image(image_path, camera_name, event_id, timestamp, frame_index)

                if uri:
                    uploaded += 1
                    # Delete local file
                    try:
                        image_path.unlink()
                        deleted += 1
                        _LOGGER.debug("Archived and deleted: %s -> %s", image_path.name, uri)
                    except OSError as unlink_err:
                        _LOGGER.warning("Failed to delete %s after upload: %s", image_path.name, unlink_err)
                else:
                    _LOGGER.warning("Failed to upload %s, keeping local copy", image_path.name)

        except Exception as e:
            _LOGGER.error("Error processing %s: %s", image_path.name, e)

    if uploaded > 0:
        _LOGGER.info("Archived %d images to GCS, deleted %d local files", uploaded, deleted)

    return uploaded, deleted


def _test_gcs_connection_impl() -> tuple[bool, str]:
    """Internal implementation of GCS connection test."""
    if not config.GCP_PROJECT_ID and not config.GCP_SERVICE_ACCOUNT_FILE:
        return False, "GCP_PROJECT_ID or GCP_SERVICE_ACCOUNT_FILE not configured"

    try:
        bucket = _get_bucket()
        if bucket is None:
            return False, "Failed to get/create bucket"

        # Try to list blobs (limited to 1) to verify access
        list(bucket.list_blobs(max_results=1))

        return True, f"Connected to bucket: {bucket.name}"

    except Exception as e:
        return False, f"GCS error: {e}"


def test_gcs_connection() -> tuple[bool, str]:
    """
    Test GCS connectivity with timeout.

    Returns (success, message).
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_test_gcs_connection_impl)
            return future.result(timeout=GCP_API_TIMEOUT)
    except FuturesTimeoutError:
        return False, f"GCS connection timed out after {GCP_API_TIMEOUT}s"
    except Exception as e:
        return False, f"GCS error: {e}"


def get_bucket_name() -> str:
    """Get the bucket name being used."""
    return _get_bucket_name()
