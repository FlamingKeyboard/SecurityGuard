"""Configuration for the Vivint Security Guard service."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Vivint credentials - set via environment variables
VIVINT_USERNAME = os.getenv("VIVINT_USERNAME", "")
VIVINT_PASSWORD = os.getenv("VIVINT_PASSWORD", "")

# Token persistence file (encrypted with Windows DPAPI)
TOKEN_FILE = DATA_DIR / "vivint_tokens.enc"

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3-flash-preview"  # Latest fast vision model

# Frame capture settings
FRAME_CAPTURE_DIR = DATA_DIR / "frames"
FRAME_CAPTURE_DIR.mkdir(exist_ok=True)
FRAME_BURST_COUNT = 3  # Number of frames to capture per event
FRAME_BURST_INTERVAL_MS = 500  # Interval between burst frames

# RTSP settings
RTSP_TIMEOUT_SECONDS = 30  # Increased for slower connections
RTSP_PREFER_HD = False  # SD is faster and more reliable for analysis

# Event handling
MOTION_COOLDOWN_SECONDS = 30  # Don't re-alert for same camera within this window
KEEPALIVE_INTERVAL_SECONDS = 300  # 5 minutes - refresh session to keep PubNub alive

# Risk classification thresholds
# priority: 0=normal, 1=high (bypass quiet hours), 2=emergency (requires ack)
RISK_LEVELS = {
    "low": {"notify": False, "log": True, "priority": -1},
    "medium": {"notify": True, "log": True, "priority": 0},
    "high": {"notify": True, "log": True, "urgent": True, "priority": 1},
    "critical": {"notify": True, "log": True, "urgent": True, "emergency": True, "priority": 2},
}

# Hub IP - set via environment variable for container deployments
# Find your hub IP by checking your router's DHCP leases or running a network scan
VIVINT_HUB_IP = os.getenv("VIVINT_HUB_IP", "192.168.8.132")
VIVINT_HUB_RTSP_PORT = int(os.getenv("VIVINT_HUB_RTSP_PORT", "8554"))

# =============================================================================
# Google Cloud Platform Settings
# =============================================================================

# GCP Project ID (required for BigQuery and GCS)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")

# Service account key file (for local development; production uses ADC)
GCP_SERVICE_ACCOUNT_FILE = os.getenv("GCP_SERVICE_ACCOUNT_FILE", "")

# Google Cloud Storage settings
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")  # Auto-generated if empty
GCS_IMAGE_PREFIX = "images"  # Folder prefix for images in bucket

# BigQuery settings
BQ_DATASET = os.getenv("BQ_DATASET", "security_guard")
BQ_TABLE = os.getenv("BQ_TABLE", "security_logs")

# Archival settings
IMAGE_RETENTION_DAYS = 30  # Archive images older than this to GCS
LOG_RETENTION_DAYS = 30    # Archive logs older than this to BigQuery
DISK_SPACE_THRESHOLD_GB = 5  # Trigger archival if disk space below this

# Sync settings
SYNC_INTERVAL_SECONDS = 3600  # Hourly sync to GCP

# Conversation grouping
CONVERSATION_TIMEOUT_SECONDS = 300  # 5 minutes - new conversation if gap exceeds this
EVENT_GROUPING_SECONDS = 5  # Cameras triggering within this window share event_id

# Local SQLite buffer
SQLITE_BUFFER_FILE = DATA_DIR / "log_buffer.db"
