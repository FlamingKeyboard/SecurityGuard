FROM python:3.12-slim

# Install ffmpeg, git, and curl for RTSP frame capture, cloning vivintpy, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash secguard
WORKDIR /app

# Clone vivintpy library
RUN git clone --depth 1 https://github.com/natekspencer/vivintpy.git

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Create data directory
RUN mkdir -p /app/data && chown -R secguard:secguard /app

# Switch to non-root user
USER secguard

# Environment variables (set these when running the container)
# VIVINT_USERNAME - Vivint account email
# VIVINT_PASSWORD - Vivint account password
# GEMINI_API_KEY - Google AI Studio API key
# PUSHOVER_TOKEN - Pushover application token
# PUSHOVER_USER - Pushover user key
# VIVINT_HUB_IP - Tailscale IP of Vivint hub (default: 192.168.8.132)
#
# GCP Integration (optional, for cloud logging and image archival):
# GCP_PROJECT_ID - Google Cloud project ID
# GCP_SERVICE_ACCOUNT_FILE - Path to service account JSON (local dev only)
# GCS_BUCKET_NAME - GCS bucket name (auto-generated if empty)
# BQ_DATASET - BigQuery dataset name (default: security_guard)
# BQ_TABLE - BigQuery table name (default: security_logs)
#
# Note: On GCE with ADC, only GCP_PROJECT_ID is needed

# Health check - uses the /health/live endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health/live || exit 1

CMD ["python", "-u", "security_guard.py"]
