"""
Health Check Module for Security Guard

Provides HTTP endpoints for container health monitoring:
- /health - Full health status of all services
- /health/ready - Readiness check (can process events)
- /health/live - Liveness check (process is running)

Usage:
    from health_check import start_health_server, increment_error

    # Start server in background
    await start_health_server()

    # Track errors from other modules
    increment_error("http_error")
"""

import asyncio
import json
import os
import socket
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aiohttp import web

import config

# Configuration
HEALTH_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8080"))

# Error counters (thread-safe via GIL for simple increments)
_error_counts = {
    "http_errors": 0,
    "auth_errors": 0,
    "exceptions": 0,
    "rate_limits": 0,
}

# Health server instance
_health_app: Optional[web.Application] = None
_health_runner: Optional[web.AppRunner] = None


def increment_error(error_type: str) -> None:
    """Increment an error counter. Call from other modules to track errors."""
    if error_type in _error_counts:
        _error_counts[error_type] += 1


def reset_errors() -> None:
    """Reset all error counters."""
    for key in _error_counts:
        _error_counts[key] = 0


def get_error_counts() -> dict:
    """Get current error counts."""
    return _error_counts.copy()


def _check_tcp_connectivity(host: str, port: int, timeout: float = 5.0) -> tuple[bool, float]:
    """
    Check TCP connectivity to a host:port.
    Returns (success, latency_ms).
    """
    start = time.time()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        latency = (time.time() - start) * 1000
        sock.close()
        return result == 0, latency
    except Exception:
        return False, 0


def _check_https_connectivity(host: str, timeout: float = 5.0) -> tuple[bool, float]:
    """
    Check HTTPS connectivity to a host.
    Returns (success, latency_ms).
    """
    start = time.time()
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                latency = (time.time() - start) * 1000
                return True, latency
    except Exception:
        return False, 0




def _check_gemini_api() -> dict:
    """Check Gemini API connectivity."""
    ok, latency = _check_https_connectivity("generativelanguage.googleapis.com")
    return {
        "status": "ok" if ok else "error",
        "latency_ms": round(latency) if ok else None,
    }


def _check_pushover_api() -> dict:
    """Check Pushover API connectivity."""
    ok, latency = _check_https_connectivity("api.pushover.net")
    return {
        "status": "ok" if ok else "error",
        "latency_ms": round(latency) if ok else None,
    }


def _check_vivint_hub() -> dict:
    """Check Vivint hub connectivity via RTSP port."""
    hub_ip = config.VIVINT_HUB_IP
    hub_port = config.VIVINT_HUB_RTSP_PORT

    if not hub_ip:
        return {"status": "not_configured", "ip": None}

    ok, latency = _check_tcp_connectivity(hub_ip, hub_port)
    return {
        "status": "ok" if ok else "error",
        "ip": hub_ip,
        "port": hub_port,
        "latency_ms": round(latency) if ok else None,
    }


def _check_gcs() -> dict:
    """Check GCS connectivity."""
    try:
        from gcp_storage import test_gcs_connection
        ok, message = test_gcs_connection()
        return {
            "status": "ok" if ok else "error",
            "bucket": config.GCS_BUCKET_NAME,
            "message": message if not ok else None,
        }
    except Exception as e:
        return {
            "status": "error",
            "bucket": config.GCS_BUCKET_NAME,
            "message": str(e),
        }


def _check_bigquery() -> dict:
    """Check BigQuery connectivity."""
    try:
        from gcp_logging import test_bigquery_connection
        ok, message = test_bigquery_connection()
        return {
            "status": "ok" if ok else "error",
            "dataset": f"{config.BQ_DATASET}.{config.BQ_TABLE}",
            "message": message if not ok else None,
        }
    except Exception as e:
        return {
            "status": "error",
            "dataset": f"{config.BQ_DATASET}.{config.BQ_TABLE}",
            "message": str(e),
        }


def _check_sqlite() -> dict:
    """Check SQLite buffer accessibility."""
    try:
        db_path = config.SQLITE_BUFFER_FILE
        exists = db_path.exists()
        readable = os.access(db_path, os.R_OK) if exists else False
        return {
            "status": "ok" if exists and readable else "error",
            "path": str(db_path),
            "exists": exists,
            "readable": readable,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


def _check_frame_dir() -> dict:
    """Check frame capture directory."""
    try:
        frame_dir = config.FRAME_CAPTURE_DIR
        exists = frame_dir.exists()
        writable = os.access(frame_dir, os.W_OK) if exists else False
        return {
            "status": "ok" if exists and writable else "error",
            "path": str(frame_dir),
            "exists": exists,
            "writable": writable,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


def _check_credentials_file() -> dict:
    """Check if credentials file exists with refresh token."""
    try:
        from vivint_client import CREDENTIALS_FILE, load_credentials
        exists = CREDENTIALS_FILE.exists()
        has_token = False
        if exists:
            creds = load_credentials()
            has_token = bool(creds and creds.get("refresh_token"))
        return {
            "status": "ok" if has_token else "warning",
            "path": str(CREDENTIALS_FILE),
            "exists": exists,
            "has_refresh_token": has_token,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


def _get_memory_usage() -> int:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return round(usage.ru_maxrss / 1024)  # Convert KB to MB
    except Exception:
        try:
            # Fallback: read from /proc
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        return round(kb / 1024)
        except Exception:
            pass
    return 0


def _determine_overall_status(checks: dict) -> str:
    """
    Determine overall health status based on individual checks.

    - healthy: All critical services OK
    - degraded: Some non-critical services down
    - unhealthy: Critical services down
    """
    critical_services = ["vivint_hub", "gcs", "bigquery"]
    non_critical_services = ["pushover", "gemini"]

    critical_failures = 0
    non_critical_failures = 0

    for service in critical_services:
        if service in checks and checks[service].get("status") != "ok":
            # Skip GCS/BigQuery if not configured
            if service in ["gcs", "bigquery"] and not config.GCP_PROJECT_ID:
                continue
            critical_failures += 1

    for service in non_critical_services:
        if service in checks and checks[service].get("status") != "ok":
            non_critical_failures += 1

    if critical_failures > 0:
        return "unhealthy"
    elif non_critical_failures > 0:
        return "degraded"
    return "healthy"


async def health_handler(request: web.Request) -> web.Response:
    """Full health check endpoint."""
    checks = {
        "gemini": _check_gemini_api(),
        "pushover": _check_pushover_api(),
        "vivint_hub": _check_vivint_hub(),
        "sqlite": _check_sqlite(),
        "frame_dir": _check_frame_dir(),
        "credentials": _check_credentials_file(),
        "memory_mb": _get_memory_usage(),
    }

    # Only check GCP if configured
    if config.GCP_PROJECT_ID or config.GCP_SERVICE_ACCOUNT_FILE:
        checks["gcs"] = _check_gcs()
        checks["bigquery"] = _check_bigquery()

    status = _determine_overall_status(checks)

    response = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "errors": get_error_counts(),
    }

    http_status = 200 if status in ["healthy", "degraded"] else 503
    return web.json_response(response, status=http_status)


async def ready_handler(request: web.Request) -> web.Response:
    """Readiness check - can process events."""
    hub_ok = _check_vivint_hub().get("status") == "ok"
    gemini_ok = _check_gemini_api().get("status") == "ok"

    ready = hub_ok and gemini_ok

    response = {
        "ready": ready,
        "vivint_hub": "ok" if hub_ok else "error",
        "gemini": "ok" if gemini_ok else "error",
    }

    return web.json_response(response, status=200 if ready else 503)


async def live_handler(request: web.Request) -> web.Response:
    """Liveness check - process is running."""
    return web.json_response({"alive": True}, status=200)


async def start_health_server() -> None:
    """Start the health check HTTP server."""
    global _health_app, _health_runner

    _health_app = web.Application()
    _health_app.router.add_get("/health", health_handler)
    _health_app.router.add_get("/health/ready", ready_handler)
    _health_app.router.add_get("/health/live", live_handler)

    _health_runner = web.AppRunner(_health_app)
    await _health_runner.setup()

    site = web.TCPSite(_health_runner, "0.0.0.0", HEALTH_PORT)
    await site.start()

    print(f"Health check server started on port {HEALTH_PORT}")


async def stop_health_server() -> None:
    """Stop the health check HTTP server."""
    global _health_runner
    if _health_runner:
        await _health_runner.cleanup()
        _health_runner = None


if __name__ == "__main__":
    # Test the health check locally
    async def test():
        await start_health_server()
        print(f"Health server running on http://localhost:{HEALTH_PORT}/health")
        print("Press Ctrl+C to stop")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await stop_health_server()

    asyncio.run(test())
