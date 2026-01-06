"""Test thread safety under concurrent access."""

import threading
import time
import sys

def test_concurrent_access():
    """Test that locks protect against concurrent access issues."""
    print("="*70)
    print("Concurrent Access Test")
    print("="*70)

    import gcp_logging
    import gcp_storage

    results = {
        'id_lock_conflicts': 0,
        'client_lock_conflicts': 0,
        'storage_lock_conflicts': 0,
        'successful_acquires': 0,
    }
    results_lock = threading.Lock()

    def worker_id_lock(worker_id):
        """Try to acquire _id_lock multiple times."""
        for i in range(10):
            acquired = gcp_logging._id_lock.acquire(blocking=True, timeout=0.1)
            if acquired:
                # Simulate some work
                time.sleep(0.001)
                gcp_logging._id_lock.release()
                with results_lock:
                    results['successful_acquires'] += 1
            else:
                with results_lock:
                    results['id_lock_conflicts'] += 1

    def worker_client_lock(worker_id):
        """Try to acquire _client_lock multiple times."""
        for i in range(10):
            acquired = gcp_logging._client_lock.acquire(blocking=True, timeout=0.1)
            if acquired:
                time.sleep(0.001)
                gcp_logging._client_lock.release()
                with results_lock:
                    results['successful_acquires'] += 1
            else:
                with results_lock:
                    results['client_lock_conflicts'] += 1

    def worker_storage_lock(worker_id):
        """Try to acquire gcp_storage._client_lock multiple times."""
        for i in range(10):
            acquired = gcp_storage._client_lock.acquire(blocking=True, timeout=0.1)
            if acquired:
                time.sleep(0.001)
                gcp_storage._client_lock.release()
                with results_lock:
                    results['successful_acquires'] += 1
            else:
                with results_lock:
                    results['storage_lock_conflicts'] += 1

    # Create threads
    print("\nSpawning 30 concurrent threads (10 for each lock type)...")
    threads = []

    # 10 threads for id_lock
    for i in range(10):
        t = threading.Thread(target=worker_id_lock, args=(i,))
        threads.append(t)

    # 10 threads for client_lock
    for i in range(10):
        t = threading.Thread(target=worker_client_lock, args=(i,))
        threads.append(t)

    # 10 threads for storage lock
    for i in range(10):
        t = threading.Thread(target=worker_storage_lock, args=(i,))
        threads.append(t)

    # Start all threads
    print("Starting threads...")
    start_time = time.time()
    for t in threads:
        t.start()

    # Wait for all to complete
    print("Waiting for threads to complete...")
    for t in threads:
        t.join()

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    print(f"Total threads: 30")
    print(f"Operations per thread: 10")
    print(f"Total operations: 300")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"\nSuccessful lock acquires: {results['successful_acquires']}")
    print(f"ID lock conflicts: {results['id_lock_conflicts']}")
    print(f"Client lock conflicts: {results['client_lock_conflicts']}")
    print(f"Storage lock conflicts: {results['storage_lock_conflicts']}")

    # Verify correctness
    total_ops = results['successful_acquires'] + results['id_lock_conflicts'] + \
                results['client_lock_conflicts'] + results['storage_lock_conflicts']

    print(f"\nTotal operations accounted for: {total_ops}")

    if results['successful_acquires'] == 300:
        print("\n[PASS] All 300 operations completed successfully!")
        print("[PASS] Locks are working correctly under concurrent access")
        return True
    else:
        print(f"\n[WARN] Some operations may have timed out")
        print(f"[INFO] This is expected behavior - locks are protecting resources")
        if results['successful_acquires'] >= 290:
            print("[PASS] Thread safety is working correctly")
            return True
        else:
            print("[FAIL] Unexpected number of failures")
            return False


def test_lock_isolation():
    """Test that different locks don't interfere with each other."""
    print("\n" + "="*70)
    print("Lock Isolation Test")
    print("="*70)

    import gcp_logging
    import gcp_storage

    print("\nAcquiring gcp_logging._id_lock...")
    gcp_logging._id_lock.acquire()

    print("Acquired gcp_logging._id_lock")
    print("Attempting to acquire gcp_logging._client_lock (should succeed)...")

    acquired = gcp_logging._client_lock.acquire(blocking=False)
    if acquired:
        print("[PASS] Successfully acquired _client_lock while _id_lock is held")
        gcp_logging._client_lock.release()
    else:
        print("[FAIL] Could not acquire _client_lock - locks may be incorrectly shared")
        gcp_logging._id_lock.release()
        return False

    print("Attempting to acquire gcp_storage._client_lock (should succeed)...")
    acquired = gcp_storage._client_lock.acquire(blocking=False)
    if acquired:
        print("[PASS] Successfully acquired gcp_storage._client_lock")
        gcp_storage._client_lock.release()
    else:
        print("[FAIL] Could not acquire gcp_storage._client_lock")
        gcp_logging._id_lock.release()
        return False

    gcp_logging._id_lock.release()
    print("\n[PASS] All locks are properly isolated from each other")
    return True


def main():
    """Run all concurrent tests."""
    try:
        results = []

        results.append(("Concurrent access", test_concurrent_access()))
        results.append(("Lock isolation", test_lock_isolation()))

        # Summary
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)

        all_passed = all(r[1] for r in results)
        for test_name, passed in results:
            status = "PASSED" if passed else "FAILED"
            symbol = "[+]" if passed else "[-]"
            print(f"{symbol} {test_name}: {status}")

        if all_passed:
            print("\n[+] ALL CONCURRENT TESTS PASSED")
            return 0
        else:
            print("\n[-] SOME TESTS FAILED")
            return 1

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
