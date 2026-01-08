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
    echo "Stopping container for credential setup..."
    podman-compose -f "$COMPOSE_FILE" stop 2>/dev/null || true
    echo ""
fi

# Run the setup script
echo "Running credential setup..."
echo ""
python3 setup_credentials.py

# Ensure correct permissions on credentials
if [[ -f "data/credentials.enc" ]]; then
    chmod 666 data/credentials.enc 2>/dev/null || true
    echo ""
    echo "Credentials saved to data/credentials.enc"
fi

# Restart container
echo ""
echo "Starting container..."
podman-compose -f "$COMPOSE_FILE" up -d

# Wait for container to be healthy
echo "Waiting for container to become healthy..."
for i in {1..30}; do
    STATUS=$(podman ps --filter "name=vivint-security-guard" --format "{{.Status}}" 2>/dev/null || echo "")
    if [[ "$STATUS" == *"healthy"* ]]; then
        echo "Container is healthy!"
        break
    elif [[ "$STATUS" == *"Up"* ]]; then
        echo -n "."
        sleep 1
    else
        sleep 1
    fi
done
echo ""

echo ""
echo "============================================================"
echo "Container Status:"
echo "============================================================"
podman ps --filter "name=vivint-security-guard" --format "table {{.Names}}\t{{.Status}}"
echo ""

# Show recent logs
echo "Recent logs:"
echo "------------------------------------------------------------"
podman logs vivint-security-guard --tail=20 2>&1 | tail -20
echo "------------------------------------------------------------"
echo ""

# Check if service is running properly
if podman ps --format "{{.Status}}" --filter "name=vivint-security-guard" | grep -q "Up"; then
    echo "Setup complete! Service is running."
    echo ""
    echo "Useful commands:"
    echo "  podman logs -f vivint-security-guard  # Follow logs"
    echo "  curl localhost:8080/health            # Check health"
else
    echo "WARNING: Container may not be running properly. Check logs above."
fi
