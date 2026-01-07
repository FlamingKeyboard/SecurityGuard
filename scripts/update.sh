#!/bin/bash
#
# Auto-update script for Vivint Security Guard
#
# This script:
#   1. Pulls the latest code from git
#   2. Rebuilds the container if changes detected
#   3. Restarts the service
#   4. Rolls back on failure
#
# Setup (on your GCE VM):
#   1. Clone the repo: git clone <your-repo-url> /opt/security-guard
#   2. Copy env.example to .env and configure
#   3. Add to crontab: */15 * * * * /opt/security-guard/scripts/update.sh
#
# Or use systemd timer for more reliable scheduling

set -e

# PATH export for cron environment (cron has minimal PATH)
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Configuration
REPO_DIR="${REPO_DIR:-/opt/security-guard}"
COMPOSE_CMD="${COMPOSE_CMD:-podman-compose}"  # or docker-compose
LOG_FILE="${LOG_FILE:-/var/log/security-guard-update.log}"
LOCK_FILE="${LOCK_FILE:-/var/run/security-guard-update.lock}"

# Ensure lock directory exists
mkdir -p "$(dirname "$LOCK_FILE")"

# Set secure permissions on lock file
touch "$LOCK_FILE"
chmod 600 "$LOCK_FILE"

# Lock file mechanism - prevent concurrent runs
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Update already running, exiting."
    exit 0
fi

# Change to repo directory
cd "$REPO_DIR"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Save current commit for rollback
PREVIOUS_COMMIT=$(git rev-parse HEAD)

# Rollback function - called on failure
rollback() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log "ERROR: Update failed with exit code $exit_code"
        log "ROLLBACK: Reverting to previous commit $PREVIOUS_COMMIT"
        git reset --hard "$PREVIOUS_COMMIT"
        $COMPOSE_CMD up -d || true
        log "ROLLBACK: Complete"
    fi
    # Release lock
    flock -u 200
    exit $exit_code
}

# Set trap for rollback on error
trap rollback EXIT

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

# Verify service is running
sleep 5
if $COMPOSE_CMD ps | grep -q "Up"; then
    log "Service verified running"
else
    log "ERROR: Service failed to start correctly"
    $COMPOSE_CMD ps || true
    exit 1  # Trigger rollback
fi

# Success - disable trap's rollback behavior before diagnostic commands
trap - EXIT

# Show recent logs (protected from triggering rollback)
log "Recent container logs:"
$COMPOSE_CMD logs --tail=20 || true

# Release lock
flock -u 200
