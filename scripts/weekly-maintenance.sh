#!/bin/bash
# Weekly VM maintenance for Security Guard
# Runs every Sunday at 8 PM

set -e
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

echo "$LOG_PREFIX Starting weekly maintenance..."

# Update system packages
echo "$LOG_PREFIX Updating system packages..."
dnf update -y --refresh

# Update container
echo "$LOG_PREFIX Checking for container updates..."
/opt/security-guard/scripts/update-container.sh

# Check if reboot is required (kernel update)
CURRENT_KERNEL=$(uname -r)
LATEST_KERNEL=$(rpm -q kernel --last | head -1 | awk '{print $1}' | sed 's/kernel-//')

if [ "$CURRENT_KERNEL" != "$LATEST_KERNEL" ]; then
    echo "$LOG_PREFIX Kernel updated from $CURRENT_KERNEL to $LATEST_KERNEL, scheduling reboot..."
    # Send notification before reboot
    source /opt/security-guard/.env 2>/dev/null || true
    if [ -n "$PUSHOVER_TOKEN" ] && [ -n "$PUSHOVER_USER" ]; then
        curl -s -F "token=$PUSHOVER_TOKEN" -F "user=$PUSHOVER_USER" \
            -F "title=Security Guard VM Rebooting" \
            -F "message=Kernel updated, rebooting now. Will be back shortly." \
            https://api.pushover.net/1/messages.json
    fi
    # Reboot in 1 minute
    shutdown -r +1 "Weekly maintenance: kernel update requires reboot"
else
    echo "$LOG_PREFIX No kernel update, skipping reboot"
fi

echo "$LOG_PREFIX Weekly maintenance complete"
