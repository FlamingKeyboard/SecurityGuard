#!/usr/bin/env python3
"""
VM Resource Monitor for Security Guard

Monitors VM health metrics and sends alerts via Pushover when thresholds are exceeded.
Designed to run via systemd timer every 5 minutes.

Usage: python vm_monitor.py [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

# Configuration
DISK_THRESHOLD_GB = 5  # Alert if free space below this
CPU_THRESHOLD_PCT = 90  # Alert if CPU above this
MEMORY_THRESHOLD_PCT = 90  # Alert if memory above this
HEALTH_ENDPOINT = "http://localhost:8080/health"
CONTAINER_NAME = "vivint-security-guard"
ALERT_COOLDOWN_SECONDS = 3600  # 1 hour between same alerts
STATE_FILE = Path("/var/lib/security-guard/monitor_state.json")
ENV_FILE = Path("/opt/security-guard/.env")


def load_env():
    """Load environment variables from .env file."""
    env_vars = {}
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value.strip('"').strip("'")
    return env_vars


def load_state():
    """Load alert state to implement cooldown."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"last_alerts": {}}


def save_state(state):
    """Save alert state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def should_alert(state, alert_type):
    """Check if we should send an alert (respecting cooldown)."""
    last_alerts = state.get("last_alerts", {})
    last_time = last_alerts.get(alert_type, 0)
    return time.time() - last_time > ALERT_COOLDOWN_SECONDS


def record_alert(state, alert_type):
    """Record that an alert was sent."""
    if "last_alerts" not in state:
        state["last_alerts"] = {}
    state["last_alerts"][alert_type] = time.time()


def send_pushover(title, message, env_vars, priority=0):
    """Send a Pushover notification."""
    token = env_vars.get("PUSHOVER_TOKEN")
    user = env_vars.get("PUSHOVER_USER")

    if not token or not user:
        print(f"  Pushover not configured, would send: {title}")
        return False

    data = urllib.parse.urlencode({
        "token": token,
        "user": user,
        "title": title,
        "message": message,
        "priority": str(priority),
    }).encode()

    try:
        req = urllib.request.Request(
            "https://api.pushover.net/1/messages.json",
            data=data,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  Failed to send Pushover: {e}")
        return False


def get_hostname():
    """Get the VM hostname."""
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return "unknown"


def check_disk_space():
    """Check available disk space."""
    try:
        result = subprocess.run(
            ["df", "-BG", "/"],
            capture_output=True,
            text=True,
            timeout=10
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            available_gb = int(parts[3].rstrip('G'))
            total_gb = int(parts[1].rstrip('G'))
            used_pct = int(parts[4].rstrip('%'))
            return {
                "available_gb": available_gb,
                "total_gb": total_gb,
                "used_pct": used_pct,
                "ok": available_gb >= DISK_THRESHOLD_GB
            }
    except Exception as e:
        return {"error": str(e), "ok": False}
    return {"error": "Unknown", "ok": False}


def check_cpu_usage():
    """Check CPU usage (1-minute load average as percentage)."""
    try:
        with open("/proc/loadavg") as f:
            load1 = float(f.read().split()[0])

        # Get number of CPUs
        cpu_count = os.cpu_count() or 1

        # Convert load average to percentage
        cpu_pct = (load1 / cpu_count) * 100

        return {
            "load_avg": load1,
            "cpu_count": cpu_count,
            "usage_pct": round(cpu_pct, 1),
            "ok": cpu_pct < CPU_THRESHOLD_PCT
        }
    except Exception as e:
        return {"error": str(e), "ok": False}


def check_memory_usage():
    """Check memory usage."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    value = int(parts[1])  # in kB
                    meminfo[key] = value

        total = meminfo.get("MemTotal", 1)
        available = meminfo.get("MemAvailable", 0)
        used = total - available
        used_pct = (used / total) * 100

        return {
            "total_mb": round(total / 1024),
            "available_mb": round(available / 1024),
            "used_pct": round(used_pct, 1),
            "ok": used_pct < MEMORY_THRESHOLD_PCT
        }
    except Exception as e:
        return {"error": str(e), "ok": False}


def check_container_running():
    """Check if the container is running."""
    try:
        result = subprocess.run(
            ["podman", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        status = result.stdout.strip()
        running = "Up" in status
        return {
            "status": status if status else "Not found",
            "ok": running
        }
    except Exception as e:
        return {"error": str(e), "ok": False}


def check_container_health():
    """Check container health endpoint."""
    try:
        req = urllib.request.Request(HEALTH_ENDPOINT, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                status = data.get("status", "unknown")
                return {
                    "status": status,
                    "ok": status in ["healthy", "degraded"]
                }
            return {"status": f"HTTP {resp.status}", "ok": False}
    except urllib.error.URLError as e:
        return {"status": f"Unreachable: {e.reason}", "ok": False}
    except Exception as e:
        return {"status": str(e), "ok": False}


def main():
    parser = argparse.ArgumentParser(description="VM Resource Monitor")
    parser.add_argument("--dry-run", action="store_true", help="Don't send alerts")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hostname = get_hostname()
    print(f"[{timestamp}] VM Monitor Check on {hostname}")
    print("=" * 60)

    env_vars = load_env()
    state = load_state()
    alerts = []
    all_ok = True

    # Check disk space
    print("\n[Disk Space]")
    disk = check_disk_space()
    if "error" in disk:
        print(f"  ERROR: {disk['error']}")
        all_ok = False
    else:
        status = "OK" if disk["ok"] else "ALERT"
        print(f"  Status: {status}")
        print(f"  Available: {disk['available_gb']}GB / {disk['total_gb']}GB ({disk['used_pct']}% used)")
        if not disk["ok"]:
            all_ok = False
            alerts.append(("disk_low", f"Low disk space: {disk['available_gb']}GB available ({disk['used_pct']}% used)"))

    # Check CPU
    print("\n[CPU Usage]")
    cpu = check_cpu_usage()
    if "error" in cpu:
        print(f"  ERROR: {cpu['error']}")
        all_ok = False
    else:
        status = "OK" if cpu["ok"] else "ALERT"
        print(f"  Status: {status}")
        print(f"  Load: {cpu['load_avg']} ({cpu['usage_pct']}% of {cpu['cpu_count']} CPUs)")
        if not cpu["ok"]:
            all_ok = False
            alerts.append(("cpu_high", f"High CPU usage: {cpu['usage_pct']}% (load: {cpu['load_avg']})"))

    # Check memory
    print("\n[Memory Usage]")
    memory = check_memory_usage()
    if "error" in memory:
        print(f"  ERROR: {memory['error']}")
        all_ok = False
    else:
        status = "OK" if memory["ok"] else "ALERT"
        print(f"  Status: {status}")
        print(f"  Used: {memory['used_pct']}% ({memory['available_mb']}MB available of {memory['total_mb']}MB)")
        if not memory["ok"]:
            all_ok = False
            alerts.append(("memory_high", f"High memory usage: {memory['used_pct']}% ({memory['available_mb']}MB available)"))

    # Check container running
    print("\n[Container Status]")
    container = check_container_running()
    if "error" in container:
        print(f"  ERROR: {container['error']}")
        all_ok = False
    else:
        status = "OK" if container["ok"] else "ALERT"
        print(f"  Status: {status}")
        print(f"  Container: {container['status']}")
        if not container["ok"]:
            all_ok = False
            alerts.append(("container_down", f"Container not running: {container['status']}"))

    # Check container health (only if container is running)
    print("\n[Container Health]")
    if container.get("ok"):
        health = check_container_health()
        status = "OK" if health["ok"] else "ALERT"
        print(f"  Status: {status}")
        print(f"  Health: {health['status']}")
        if not health["ok"]:
            all_ok = False
            alerts.append(("health_failed", f"Container unhealthy: {health['status']}"))
    else:
        print("  Skipped (container not running)")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print(f"[{timestamp}] All checks passed")
    else:
        print(f"[{timestamp}] ALERTS DETECTED: {len(alerts)}")

    # Send alerts
    if alerts and not args.dry_run:
        for alert_type, message in alerts:
            if should_alert(state, alert_type):
                title = f"Security Guard VM Alert: {hostname}"
                full_message = f"{message}\n\nTime: {timestamp}\nHost: {hostname}"

                print(f"\nSending alert: {alert_type}")
                if send_pushover(title, full_message, env_vars, priority=1):
                    record_alert(state, alert_type)
                    print("  Alert sent successfully")
                else:
                    print("  Failed to send alert")
            else:
                print(f"\nSkipping alert (cooldown): {alert_type}")

        save_state(state)
    elif alerts and args.dry_run:
        print("\n[DRY RUN] Would send the following alerts:")
        for alert_type, message in alerts:
            print(f"  - {alert_type}: {message}")

    # Exit code
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
