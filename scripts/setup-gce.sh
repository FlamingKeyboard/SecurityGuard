#!/bin/bash
#
# GCE VM Setup Script for Vivint Security Guard
#
# Supports:
#   - Rocky Linux (uses dnf)
#   - Debian/Ubuntu (uses apt-get)
#
# Run this on a fresh e2-micro instance:
#   curl -sSL https://raw.githubusercontent.com/<user>/<repo>/main/scripts/setup-gce.sh | bash
#
# Or manually:
#   1. SSH into your VM
#   2. Run: bash setup-gce.sh

set -e

echo "=== Vivint Security Guard - GCE Setup ==="

# Detect OS and install packages accordingly
install_packages() {
    if [ -f /etc/rocky-release ] || [ -f /etc/redhat-release ]; then
        echo "Detected Rocky Linux / RHEL..."
        echo "Installing Podman and dependencies via dnf..."
        sudo dnf install -y podman git python3-pip curl
        # podman-compose is not in Rocky Linux repos, install via pip
        echo "Installing podman-compose via pip3..."
        sudo pip3 install podman-compose
    elif [ -f /etc/debian_version ]; then
        echo "Detected Debian/Ubuntu..."
        echo "Installing Podman and dependencies via apt..."
        sudo apt-get update
        sudo apt-get install -y podman podman-compose git curl
    else
        echo "ERROR: Unsupported OS. This script supports Rocky Linux and Debian/Ubuntu."
        echo "Please install podman, podman-compose, and git manually."
        exit 1
    fi
}

echo "Installing Podman..."
install_packages

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

# Create .env file with secure permissions
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp env.example .env
    # Secure the .env file - contains credentials
    chmod 600 .env
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
else
    # Ensure existing .env has secure permissions
    chmod 600 .env
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
