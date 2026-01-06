#!/bin/bash
#
# Auto-update script for Vivint Security Guard
#
# This script:
#   1. Pulls the latest code from git
#   2. Rebuilds the container if changes detected
#   3. Restarts the service
#
# Setup (on your GCE VM):
#   1. Clone the repo: git clone <your-repo-url> /opt/security-guard
#   2. Copy env.example to .env and configure
#   3. Add to crontab: */15 * * * * /opt/security-guard/scripts/update.sh
#
# Or use systemd timer for more reliable scheduling

set -e

# Configuration
REPO_DIR="${REPO_DIR:-/opt/security-guard}"
COMPOSE_CMD="${COMPOSE_CMD:-podman-compose}"  # or docker-compose
LOG_FILE="${LOG_FILE:-/var/log/security-guard-update.log}"

# Change to repo directory
cd "$REPO_DIR"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Fetch latest changes
log "Checking for updates..."
git fetch origin main --quiet

# Check if there are new commits
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    log "Already up to date ($LOCAL)"
    exit 0
fi

log "Updates available: $LOCAL -> $REMOTE"

# Pull changes
log "Pulling latest changes..."
git pull origin main --quiet

# Rebuild and restart
log "Rebuilding container..."
$COMPOSE_CMD build --no-cache

log "Restarting service..."
$COMPOSE_CMD down
$COMPOSE_CMD up -d

log "Update complete! Now running: $(git rev-parse --short HEAD)"

# Show recent logs
log "Recent container logs:"
$COMPOSE_CMD logs --tail=20
