#!/usr/bin/env python3
"""Manual GCP sync script - triggers log sync and image archival."""
import asyncio
import sys
sys.path.insert(0, "/app")

async def main():
    print("=== Manual GCP Sync ===")
    print()

    from gcp_logging import run_sync, get_buffer_stats
    from gcp_storage import archive_old_images

    # Check buffer stats
    print("1. SQLite Buffer Stats:")
    stats = get_buffer_stats()
    print(f"   Total: {stats['total']}, Synced: {stats['synced']}, Unsynced: {stats['unsynced']}")
    print()

    # Sync to BigQuery
    print("2. Syncing to BigQuery...")
    synced, failed, cleaned = await run_sync()
    print(f"   Synced: {synced}, Failed: {failed}, Cleaned: {cleaned}")
    print()

    # Archive to GCS
    print("3. Archiving images to GCS (30-day retention)...")
    uploaded, deleted = archive_old_images()
    print(f"   Uploaded: {uploaded}, Deleted locally: {deleted}")
    print()

    print("=== Sync Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
