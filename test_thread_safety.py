"""Test thread safety locks in gcp_logging and gcp_storage modules."""

import threading
import sys

def test_gcp_logging_locks():
    """Test that gcp_logging has the required locks."""
    print("\n=== Testing gcp_logging.py ===")

    try:
        import gcp_logging

        # Test 1: Check _id_lock exists and is a threading.Lock
        if hasattr(gcp_logging, '_id_lock'):
            id_lock = gcp_logging._id_lock
            # Check if it's a Lock by checking for the lock's type
            if type(id_lock).__name__ == 'lock':
                print("[PASS] gcp_logging._id_lock exists and is a threading.Lock")
            else:
                print(f"[FAIL] gcp_logging._id_lock exists but is NOT a threading.Lock (type: {type(id_lock)})")
                return False
        else:
            print("[FAIL] gcp_logging._id_lock does NOT exist")
            return False

        # Test 2: Check _client_lock exists and is a threading.Lock
        if hasattr(gcp_logging, '_client_lock'):
            client_lock = gcp_logging._client_lock
            if type(client_lock).__name__ == 'lock':
                print("[PASS] gcp_logging._client_lock exists and is a threading.Lock")
            else:
                print(f"[FAIL] gcp_logging._client_lock exists but is NOT a threading.Lock (type: {type(client_lock)})")
                return False
        else:
            print("[FAIL] gcp_logging._client_lock does NOT exist")
            return False

        # Additional check: Verify locks are different instances
        if id_lock is client_lock:
            print("[WARN] _id_lock and _client_lock are the SAME object (should be separate)")
        else:
            print("[PASS] _id_lock and _client_lock are separate Lock instances")

        return True

    except ImportError as e:
        print(f"[FAIL] Cannot import gcp_logging: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error testing gcp_logging: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gcp_storage_locks():
    """Test that gcp_storage has the required locks and flags."""
    print("\n=== Testing gcp_storage.py ===")

    try:
        import gcp_storage

        # Test 1: Check _client_lock exists and is a threading.Lock
        if hasattr(gcp_storage, '_client_lock'):
            client_lock = gcp_storage._client_lock
            if type(client_lock).__name__ == 'lock':
                print("[PASS] gcp_storage._client_lock exists and is a threading.Lock")
            else:
                print(f"[FAIL] gcp_storage._client_lock exists but is NOT a threading.Lock (type: {type(client_lock)})")
                return False
        else:
            print("[FAIL] gcp_storage._client_lock does NOT exist")
            return False

        # Test 2: Check _gcs_init_failed flag exists
        if hasattr(gcp_storage, '_gcs_init_failed'):
            init_failed = gcp_storage._gcs_init_failed
            if isinstance(init_failed, bool):
                print(f"[PASS] gcp_storage._gcs_init_failed exists and is a bool (value: {init_failed})")
            else:
                print(f"[WARN] gcp_storage._gcs_init_failed exists but is NOT a bool (type: {type(init_failed)}, value: {init_failed})")
            # This is a warning, not a failure
        else:
            print("[FAIL] gcp_storage._gcs_init_failed does NOT exist")
            return False

        return True

    except ImportError as e:
        print(f"[FAIL] Cannot import gcp_storage: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error testing gcp_storage: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lock_functionality():
    """Test that the locks can actually be acquired and released."""
    print("\n=== Testing Lock Functionality ===")

    try:
        import gcp_logging
        import gcp_storage

        # Test gcp_logging locks
        print("\nTesting gcp_logging._id_lock acquire/release:")
        acquired = gcp_logging._id_lock.acquire(blocking=False)
        if acquired:
            print("[PASS] Successfully acquired _id_lock")
            gcp_logging._id_lock.release()
            print("[PASS] Successfully released _id_lock")
        else:
            print("[FAIL] Could not acquire _id_lock")
            return False

        print("\nTesting gcp_logging._client_lock acquire/release:")
        acquired = gcp_logging._client_lock.acquire(blocking=False)
        if acquired:
            print("[PASS] Successfully acquired _client_lock")
            gcp_logging._client_lock.release()
            print("[PASS] Successfully released _client_lock")
        else:
            print("[FAIL] Could not acquire _client_lock")
            return False

        # Test gcp_storage lock
        print("\nTesting gcp_storage._client_lock acquire/release:")
        acquired = gcp_storage._client_lock.acquire(blocking=False)
        if acquired:
            print("[PASS] Successfully acquired _client_lock")
            gcp_storage._client_lock.release()
            print("[PASS] Successfully released _client_lock")
        else:
            print("[FAIL] Could not acquire _client_lock")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Error testing lock functionality: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Thread Safety Lock Tests")
    print("="*60)

    results = []

    # Run all tests
    results.append(("gcp_logging locks", test_gcp_logging_locks()))
    results.append(("gcp_storage locks", test_gcp_storage_locks()))
    results.append(("lock functionality", test_lock_functionality()))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "[+]" if passed else "[-]"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n[+] ALL TESTS PASSED")
        return 0
    else:
        print("\n[-] SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
