#!/bin/bash
# Install Security Guard systemd services and timers

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"
LOG_DIR="/var/log/security-guard"

echo "Installing Security Guard systemd services..."

# Create log directory
mkdir -p $LOG_DIR

# Copy service and timer files
for file in $SCRIPT_DIR/systemd/*.service $SCRIPT_DIR/systemd/*.timer; do
    if [ -f "$file" ]; then
        cp "$file" "$SYSTEMD_DIR/"
        echo "  Installed: $(basename $file)"
    fi
done

# Copy scripts
cp "$SCRIPT_DIR/update-container.sh" /opt/security-guard/scripts/
cp "$SCRIPT_DIR/weekly-maintenance.sh" /opt/security-guard/scripts/
cp "$SCRIPT_DIR/vm_monitor.py" /opt/security-guard/scripts/
chmod +x /opt/security-guard/scripts/*.sh

# Reload systemd
systemctl daemon-reload

# Enable timers
systemctl enable --now security-guard-sync.timer
systemctl enable --now security-guard-update.timer
systemctl enable --now security-guard-weekly.timer
systemctl enable --now security-guard-monitor.timer

echo "Done! Timers enabled:"
systemctl list-timers | grep security-guard
