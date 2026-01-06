"""Test script for ID generation functions in gcp_logging.py"""

import time
from datetime import datetime, timezone
import sys

# Add the parent directory to path to import modules
sys.path.insert(0, r'C:\Users\Gavin Fuller\Documents\code\SecurityGuardVivint')

import gcp_logging

def test_event_id_format():
    """Test 1: get_or_create_event_id() returns a string starting with "evt_" """
    print("Test 1: Checking event ID format...")
    event_id = gcp_logging.get_or_create_event_id()
    print(f"  Generated event ID: {event_id}")

    assert isinstance(event_id, str), f"Expected string, got {type(event_id)}"
    assert event_id.startswith("evt_"), f"Expected event ID to start with 'evt_', got: {event_id}"
    assert len(event_id) > 4, f"Event ID seems too short: {event_id}"

    print("  [PASS] Event ID format is correct\n")
    return True


def test_conversation_id_format():
    """Test 2: get_or_create_conversation_id() returns a string starting with "conv_" """
    print("Test 2: Checking conversation ID format...")
    conv_id = gcp_logging.get_or_create_conversation_id()
    print(f"  Generated conversation ID: {conv_id}")

    assert isinstance(conv_id, str), f"Expected string, got {type(conv_id)}"
    assert conv_id.startswith("conv_"), f"Expected conversation ID to start with 'conv_', got: {conv_id}"
    assert len(conv_id) > 5, f"Conversation ID seems too short: {conv_id}"

    print("  [PASS] Conversation ID format is correct\n")
    return True


def test_event_id_grouping():
    """Test 3: Calling get_or_create_event_id() twice quickly returns the same ID"""
    print("Test 3: Testing event ID grouping (within EVENT_GROUPING_SECONDS)...")

    # First call
    event_id_1 = gcp_logging.get_or_create_event_id()
    print(f"  First event ID: {event_id_1}")

    # Second call immediately after (within EVENT_GROUPING_SECONDS = 5 seconds)
    time.sleep(0.1)  # Small delay to simulate realistic timing
    event_id_2 = gcp_logging.get_or_create_event_id()
    print(f"  Second event ID (0.1s later): {event_id_2}")

    assert event_id_1 == event_id_2, f"Expected same event ID within grouping window, got different IDs: {event_id_1} vs {event_id_2}"

    print("  [PASS] Event IDs are grouped correctly within time window\n")
    return True


def test_functions_no_errors():
    """Test 4: Both functions work without errors"""
    print("Test 4: Testing functions execute without errors...")

    try:
        # Test event ID creation multiple times
        for i in range(3):
            event_id = gcp_logging.get_or_create_event_id()
            print(f"  Event ID call {i+1}: {event_id}")

        # Test conversation ID creation multiple times
        for i in range(3):
            conv_id = gcp_logging.get_or_create_conversation_id()
            print(f"  Conversation ID call {i+1}: {conv_id}")

        # Test with explicit timestamps
        now = datetime.now(timezone.utc)
        event_id_with_ts = gcp_logging.get_or_create_event_id(timestamp=now)
        conv_id_with_ts = gcp_logging.get_or_create_conversation_id(timestamp=now)
        print(f"  Event ID with timestamp: {event_id_with_ts}")
        print(f"  Conversation ID with timestamp: {conv_id_with_ts}")

        print("  [PASS] All functions executed without errors\n")
        return True

    except Exception as e:
        print(f"  [FAIL] Exception occurred: {e}\n")
        return False


def test_event_id_new_after_timeout():
    """Bonus Test: Verify new event ID is created after EVENT_GROUPING_SECONDS"""
    print("Bonus Test: Testing event ID creates new ID after timeout...")
    print("  (This test will take ~6 seconds to complete)")

    # Get first event ID
    event_id_1 = gcp_logging.get_or_create_event_id()
    print(f"  Initial event ID: {event_id_1}")

    # Wait for EVENT_GROUPING_SECONDS + 1 (5 + 1 = 6 seconds)
    print("  Waiting 6 seconds (EVENT_GROUPING_SECONDS = 5)...")
    time.sleep(6)

    # Get second event ID - should be different
    event_id_2 = gcp_logging.get_or_create_event_id()
    print(f"  Event ID after timeout: {event_id_2}")

    assert event_id_1 != event_id_2, f"Expected different event ID after timeout, got same ID: {event_id_1}"

    print("  [PASS] New event ID created after timeout\n")
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("Testing ID Generation Functions in gcp_logging.py")
    print("="*70)
    print()

    tests = [
        ("Event ID Format", test_event_id_format),
        ("Conversation ID Format", test_conversation_id_format),
        ("Event ID Grouping", test_event_id_grouping),
        ("Functions Execute Without Errors", test_functions_no_errors),
        ("Event ID New After Timeout", test_event_id_new_after_timeout),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except AssertionError as e:
            print(f"  [FAIL] {e}\n")
            results.append((test_name, False))
        except Exception as e:
            print(f"  [ERROR] {e}\n")
            results.append((test_name, False))

    print("="*70)
    print("Test Results Summary")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed successfully!")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
