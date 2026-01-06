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
GEMINI_MODEL = "gemini-2.0-flash"  # Fast vision model, good for real-time analysis

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
RISK_LEVELS = {
    "low": {"notify": False, "log": True},
    "medium": {"notify": True, "log": True},
    "high": {"notify": True, "log": True, "urgent": True},
}

# Hub IP (from your port scan)
VIVINT_HUB_IP = "192.168.8.132"
VIVINT_HUB_RTSP_PORT = 8554
