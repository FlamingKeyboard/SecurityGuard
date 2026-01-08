#!/bin/bash
#
# Security Guard Setup Helper
# Run this to configure credentials and re-authenticate with Vivint MFA
#
# Usage: security-guard-setup
#

set -e

INSTALL_DIR="/opt/security-guard"
COMPOSE_FILE="compose-rocky.yaml"

echo "============================================================"
echo "Security Guard Setup"
echo "============================================================"
echo ""

cd "$INSTALL_DIR"

# Check if container is running
if podman ps --format "{{.Names}}" | grep -q "vivint-security-guard"; then
    echo "Stopping container for setup..."
    podman-compose -f "$COMPOSE_FILE" stop
    echo ""
fi

# Run the setup script
echo "Running credential setup..."
echo ""
python3 setup_credentials.py

# Copy credentials to the bind-mounted data directory
if [[ -f "data/credentials.enc" ]]; then
    chmod 777 data/credentials.enc 2>/dev/null || true
    echo ""
    echo "Credentials saved to data/credentials.enc"
fi

# Restart container
echo ""
echo "Starting container..."
podman-compose -f "$COMPOSE_FILE" up -d

# Wait and show status
sleep 10
echo ""
echo "============================================================"
echo "Container Status:"
echo "============================================================"
podman ps --filter "name=vivint-security-guard" --format "table {{.Names}}\t{{.Status}}"
echo ""

# Show recent logs
echo "Recent logs:"
echo "------------------------------------------------------------"
podman logs vivint-security-guard --tail=15 2>&1
echo "============================================================"
echo ""
echo "Setup complete! Run 'podman logs -f vivint-security-guard' to follow logs."
