#!/bin/bash
#
# GCE VM Setup Script for Vivint Security Guard
#
# Run this on a fresh e2-micro instance (Debian/Ubuntu):
#   curl -sSL https://raw.githubusercontent.com/<user>/<repo>/main/scripts/setup-gce.sh | bash
#
# Or manually:
#   1. SSH into your VM
#   2. Run: bash setup-gce.sh

set -e

echo "=== Vivint Security Guard - GCE Setup ==="

# Install Podman
echo "Installing Podman..."
sudo apt-get update
sudo apt-get install -y podman podman-compose git

# Install Tailscale (if not already installed)
if ! command -v tailscale &> /dev/null; then
    echo "Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
    echo ""
    echo "Run 'sudo tailscale up' to connect to your tailnet"
fi

# Clone repository
REPO_URL="${REPO_URL:-https://github.com/FlamingKeyboard/SecurityGuard.git}"
INSTALL_DIR="/opt/security-guard"

if [ -d "$INSTALL_DIR" ]; then
    echo "Directory exists, pulling latest..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "Cloning repository..."
    sudo git clone "$REPO_URL" "$INSTALL_DIR"
    sudo chown -R $USER:$USER "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Clone vivintpy if not present (needed for local dev, Docker clones its own)
if [ ! -d "vivintpy" ]; then
    echo "Cloning vivintpy library..."
    git clone --depth 1 https://github.com/natekspencer/vivintpy.git
fi

# Create .env file
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp env.example .env
    echo ""
    echo "=== IMPORTANT ==="
    echo "Edit .env with your credentials:"
    echo "  nano $INSTALL_DIR/.env"
    echo ""
    echo "Required values:"
    echo "  - VIVINT_USERNAME"
    echo "  - VIVINT_PASSWORD"
    echo "  - GEMINI_API_KEY"
    echo "  - PUSHOVER_TOKEN"
    echo "  - PUSHOVER_USER"
    echo "  - VIVINT_HUB_IP (your hub's Tailscale IP)"
fi

# Create log directory
sudo mkdir -p /var/log
sudo touch /var/log/security-guard-update.log
sudo chown $USER:$USER /var/log/security-guard-update.log

# Make scripts executable
chmod +x scripts/*.sh

# Setup cron for auto-updates (every 15 minutes)
CRON_CMD="*/15 * * * * $INSTALL_DIR/scripts/update.sh >> /var/log/security-guard-update.log 2>&1"
(crontab -l 2>/dev/null | grep -v "security-guard"; echo "$CRON_CMD") | crontab -

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Connect Tailscale: sudo tailscale up"
echo "  2. Edit credentials: nano $INSTALL_DIR/.env"
echo "  3. Start the service: cd $INSTALL_DIR && podman-compose up -d"
echo "  4. View logs: podman-compose logs -f"
echo ""
echo "Auto-updates are enabled (every 15 minutes via cron)"
