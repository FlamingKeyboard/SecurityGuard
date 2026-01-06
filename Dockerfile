FROM python:3.12-slim

# Install ffmpeg for RTSP frame capture
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash secguard
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY vivintpy ./vivintpy/

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

# Health check - verify Python can import the app
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import security_guard" || exit 1

CMD ["python", "-u", "security_guard.py"]
