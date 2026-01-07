#!/usr/bin/env python3
"""
Host-level heartbeat for Cloud Logging.

Sends host OS and container metrics to Cloud Logging every 5 minutes.
Run via systemd timer on the VM host (not inside container).

Metrics included:
- Disk space (total, used, free, percent)
- Container status (running, uptime, health)
- Host memory usage
- Host CPU load
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# GCP Project ID - read from environment or hardcode for VM
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "gen-lang-client-0027906266")

def get_disk_space():
    """Get disk space for root partition."""
    try:
        usage = shutil.disk_usage("/")
        return {
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent_used": round((usage.used / usage.total) * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def get_memory_usage():
    """Get host memory usage."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value = int(parts[1])  # in KB
                    meminfo[key] = value

        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable", 0)
        used = total - available

        return {
            "total_mb": round(total / 1024, 0),
            "used_mb": round(used / 1024, 0),
            "available_mb": round(available / 1024, 0),
            "percent_used": round((used / total) * 100, 1) if total > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def get_cpu_load():
    """Get CPU load averages."""
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            return {
                "load_1m": float(parts[0]),
                "load_5m": float(parts[1]),
                "load_15m": float(parts[2]),
            }
    except Exception as e:
        return {"error": str(e)}


def get_container_status():
    """Get podman container status for security-guard."""
    container_name = "vivint-security-guard"

    try:
        # Check if container is running
        result = subprocess.run(
            ["podman", "inspect", container_name, "--format",
             "{{.State.Status}}|{{.State.Running}}|{{.State.StartedAt}}|{{.State.Health.Status}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return {
                "name": container_name,
                "exists": False,
                "error": result.stderr.strip()
            }

        parts = result.stdout.strip().split("|")
        status = parts[0] if len(parts) > 0 else "unknown"
        running = parts[1].lower() == "true" if len(parts) > 1 else False
        started_at = parts[2] if len(parts) > 2 else ""
        health = parts[3] if len(parts) > 3 else "unknown"

        # Calculate uptime
        uptime_seconds = 0
        if running and started_at:
            try:
                # Parse ISO format datetime
                started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                uptime_seconds = int((datetime.now(timezone.utc) - started).total_seconds())
            except Exception:
                pass

        # Format uptime as human readable
        if uptime_seconds > 0:
            days = uptime_seconds // 86400
            hours = (uptime_seconds % 86400) // 3600
            minutes = (uptime_seconds % 3600) // 60
            if days > 0:
                uptime_str = f"{days}d {hours}h {minutes}m"
            elif hours > 0:
                uptime_str = f"{hours}h {minutes}m"
            else:
                uptime_str = f"{minutes}m"
        else:
            uptime_str = "0m"

        return {
            "name": container_name,
            "exists": True,
            "status": status,
            "running": running,
            "health": health,
            "uptime_seconds": uptime_seconds,
            "uptime": uptime_str,
            "started_at": started_at,
        }

    except subprocess.TimeoutExpired:
        return {
            "name": container_name,
            "exists": False,
            "error": "podman command timed out"
        }
    except Exception as e:
        return {
            "name": container_name,
            "exists": False,
            "error": str(e)
        }


def get_container_health_endpoint():
    """Query the container's health endpoint."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            "http://localhost:8080/health",
            headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return {
                "reachable": True,
                "status": data.get("status", "unknown"),
                "checks": {
                    k: v.get("status") if isinstance(v, dict) else "ok"
                    for k, v in data.get("checks", {}).items()
                    if isinstance(v, dict)
                },
                "errors": data.get("errors", {}),
            }
    except urllib.error.URLError as e:
        return {"reachable": False, "error": str(e.reason)}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


def send_to_cloud_logging(payload):
    """Send payload to Cloud Logging using gcloud."""
    try:
        # Use gcloud logging write command
        log_name = "security-guard-host-heartbeat"

        # Create the log entry
        result = subprocess.run(
            [
                "gcloud", "logging", "write", log_name,
                json.dumps(payload),
                "--project", GCP_PROJECT_ID,
                "--payload-type=json",
                "--severity=INFO",
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"Error sending to Cloud Logging: {result.stderr}", file=sys.stderr)
            return False

        return True

    except Exception as e:
        print(f"Exception sending to Cloud Logging: {e}", file=sys.stderr)
        return False


def main():
    """Collect metrics and send heartbeat."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Collect all metrics
    payload = {
        "service": "security-guard-host",
        "type": "heartbeat",
        "timestamp": timestamp,
        "host": {
            "disk": get_disk_space(),
            "memory": get_memory_usage(),
            "cpu": get_cpu_load(),
        },
        "container": get_container_status(),
        "health_endpoint": get_container_health_endpoint(),
    }

    # Determine overall status
    container = payload["container"]
    health = payload["health_endpoint"]
    disk = payload["host"]["disk"]

    if not container.get("running"):
        payload["overall_status"] = "critical"
        payload["issue"] = "Container not running"
    elif health.get("status") == "unhealthy":
        payload["overall_status"] = "unhealthy"
        payload["issue"] = "Container health check failed"
    elif disk.get("percent_used", 0) > 90:
        payload["overall_status"] = "warning"
        payload["issue"] = f"Disk space low: {disk.get('percent_used')}% used"
    elif health.get("status") == "degraded":
        payload["overall_status"] = "degraded"
        payload["issue"] = "Some services degraded"
    else:
        payload["overall_status"] = "healthy"

    # Print summary
    print(f"[{timestamp}] Host heartbeat:")
    print(f"  Disk: {disk.get('free_gb', 'N/A')} GB free ({disk.get('percent_used', 'N/A')}% used)")
    print(f"  Memory: {payload['host']['memory'].get('percent_used', 'N/A')}% used")
    print(f"  Container: {container.get('status', 'unknown')} (uptime: {container.get('uptime', 'N/A')})")
    print(f"  Health: {health.get('status', 'unknown')}")
    print(f"  Overall: {payload['overall_status']}")

    # Send to Cloud Logging
    if send_to_cloud_logging(payload):
        print("  Sent to Cloud Logging: OK")
    else:
        print("  Sent to Cloud Logging: FAILED")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
