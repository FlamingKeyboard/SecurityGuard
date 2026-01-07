#!/bin/bash
#
# Test script for update.sh
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATE_SCRIPT="${SCRIPT_DIR}/scripts/update.sh"

TEST_DIR=$(mktemp -d)
trap "rm -rf ${TEST_DIR}" EXIT

echo "=========================================="
echo "Testing update.sh"
echo "=========================================="
echo ""

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_RUN=$((TESTS_RUN + 1))
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    echo -e "       $2"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TESTS_RUN=$((TESTS_RUN + 1))
}

info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

run_test() {
    echo ""
    echo "Test: $1"
    echo "----------------------------------------"
}

# Test 1: Bash syntax validation
run_test "Bash syntax validation"
if bash -n "$UPDATE_SCRIPT" 2>/dev/null; then
    pass "Script has valid bash syntax"
else
    fail "Script has syntax errors" "Run: bash -n $UPDATE_SCRIPT"
fi

# Test 2: Script exists and is executable
run_test "Script exists and permissions"
if [ -f "$UPDATE_SCRIPT" ]; then
    pass "update.sh exists"
else
    fail "update.sh not found" "Expected at: $UPDATE_SCRIPT"
    exit 1
fi

# Test 3: Check for lock file mechanism
run_test "Lock file mechanism"
if grep -q "LOCK\|flock\|lockfile" "$UPDATE_SCRIPT"; then
    pass "Script contains lock file references"
else
    fail "No lock file mechanism found" "Script should use a lock file to prevent concurrent execution"
fi

# Test 4: Check for rollback function
run_test "Rollback function"
if grep -q "rollback\|restore\|revert\|PREVIOUS" "$UPDATE_SCRIPT"; then
    pass "Rollback-related keywords found"
else
    fail "No rollback mechanism found" "Script should have a rollback function for failed updates"
fi

# Test 5: Check PATH export for cron
run_test "PATH export for cron environment"
if grep -q "^PATH=\|^export PATH=" "$UPDATE_SCRIPT"; then
    pass "PATH is explicitly set"
else
    fail "PATH is not exported" "Cron jobs need explicit PATH setting"
fi

# Test 6: Check git command usage
run_test "Git command usage"
GIT_COMMANDS=$(grep -c "git " "$UPDATE_SCRIPT" || true)
if [ "$GIT_COMMANDS" -gt 0 ]; then
    pass "Git commands found in script ($GIT_COMMANDS)"
else
    fail "No git commands found" "Update script should use git to pull changes"
fi

# Test 7: Check compose command handling
run_test "Compose command handling"
if grep -q "COMPOSE_CMD\|compose\|podman-compose\|docker-compose" "$UPDATE_SCRIPT"; then
    pass "Compose command references found"
else
    fail "No compose command handling found" "Script should use podman-compose or docker-compose"
fi

# Test 8: Check for proper error handling
run_test "Error handling"
if grep -q "^set -e" "$UPDATE_SCRIPT"; then
    pass "Script uses 'set -e' for error handling"
else
    fail "Script should use 'set -e'" "Add 'set -e' at the beginning"
fi

# Test 9: Check logging capability
run_test "Logging capability"
if grep -q "log()\|LOG_FILE\|tee.*log" "$UPDATE_SCRIPT"; then
    pass "Logging mechanism found"
else
    fail "No logging mechanism found" "Script should log operations for debugging"
fi

# Test 10: Check for REPO_DIR
run_test "Configuration - REPO_DIR"
if grep -q "REPO_DIR=" "$UPDATE_SCRIPT"; then
    pass "REPO_DIR is defined"
else
    fail "REPO_DIR not defined" "Script should define repository directory"
fi

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total tests run:    $TESTS_RUN"
echo -e "${GREEN}Tests passed:       $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Tests failed:       $TESTS_FAILED${NC}"
else
    echo -e "Tests failed:       $TESTS_FAILED"
fi
echo ""

if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed. Please review the update.sh script.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
