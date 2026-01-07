#!/bin/bash
#
# Test Script for setup-gce.sh
#
# Usage: bash test_setup_script.sh
#

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Script path
SCRIPT_PATH="${SCRIPT_PATH:-./scripts/setup-gce.sh}"
TEST_DIR="/tmp/setup-gce-test-$$"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

pass_test() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    log_info "PASS: $1"
}

fail_test() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    log_error "FAIL: $1"
}

# Cleanup function
cleanup() {
    if [ -d "$TEST_DIR" ]; then
        rm -rf "$TEST_DIR"
    fi
}

trap cleanup EXIT

echo "========================================="
echo "setup-gce.sh Test Suite"
echo "========================================="
echo ""

# Test 1: Check if script exists
echo "=== Test 1: Script Existence ==="
if [ -f "$SCRIPT_PATH" ]; then
    pass_test "Script exists at $SCRIPT_PATH"
else
    fail_test "Script not found at $SCRIPT_PATH"
    echo "Cannot continue without script"
    exit 1
fi
echo ""

# Test 2: Validate bash syntax
echo "=== Test 2: Bash Syntax Validation ==="
if bash -n "$SCRIPT_PATH" 2>/dev/null; then
    pass_test "Script has valid bash syntax"
else
    fail_test "Script has bash syntax errors"
    bash -n "$SCRIPT_PATH"
fi
echo ""

# Test 3: Check shebang
echo "=== Test 3: Shebang Check ==="
FIRST_LINE=$(head -n 1 "$SCRIPT_PATH")
if [[ "$FIRST_LINE" == "#!/bin/bash"* ]] || [[ "$FIRST_LINE" == "#!/usr/bin/env bash"* ]]; then
    pass_test "Script has correct shebang"
else
    fail_test "Script has incorrect or missing shebang: $FIRST_LINE"
fi
echo ""

# Test 4: Check OS detection (Rocky Linux vs Debian)
echo "=== Test 4: OS Package Manager Detection ==="

# Check for OS detection logic
if grep -qE "/etc/rocky-release|/etc/redhat-release|/etc/debian_version|/etc/os-release" "$SCRIPT_PATH"; then
    pass_test "Script includes OS detection logic"
else
    fail_test "Script missing OS detection logic - cannot determine dnf vs apt-get"
fi

# Rocky Linux is the primary target - must have dnf support
if grep -qE "dnf|yum" "$SCRIPT_PATH"; then
    pass_test "Script contains Rocky Linux (dnf/yum) support"
else
    fail_test "Script missing Rocky Linux (dnf) support - REQUIRED for target platform"
fi

# Debian/Ubuntu support is also good to have
if grep -q "apt-get" "$SCRIPT_PATH"; then
    pass_test "Script contains Debian/Ubuntu (apt-get) support"
else
    log_warn "Script missing Debian/Ubuntu support (optional)"
fi
echo ""

# Test 5: Check for required commands
echo "=== Test 5: Required Commands Check ==="
REQUIRED_CMDS=("git" "curl" "podman")
for cmd in "${REQUIRED_CMDS[@]}"; do
    if grep -q "$cmd" "$SCRIPT_PATH"; then
        pass_test "Script references '$cmd' command"
    else
        fail_test "Script does not reference '$cmd' command"
    fi
done
echo ""

# Test 6: Check script structure
echo "=== Test 6: Script Structure ==="
if grep -q "echo" "$SCRIPT_PATH"; then
    pass_test "Script provides user feedback with echo statements"
else
    fail_test "Script lacks user feedback"
fi

if grep -q "git clone" "$SCRIPT_PATH"; then
    pass_test "Script includes repository cloning logic"
else
    fail_test "Script missing repository cloning logic"
fi

if grep -q "\.env" "$SCRIPT_PATH"; then
    pass_test "Script handles .env file"
else
    fail_test "Script does not handle .env file"
fi
echo ""

# Test 7: Lock file mechanism
echo "=== Test 7: Lock File Mechanism ==="
if grep -q "lock\|LOCK" "$SCRIPT_PATH"; then
    pass_test "Script contains lock file mechanism"
else
    log_warn "Script does not implement lock file mechanism (may allow concurrent runs)"
fi
echo ""

# Test 8: .env file permissions
echo "=== Test 8: .env File Permissions ==="
if grep -qE "chmod.*600.*\.env|chmod.*\.env.*600" "$SCRIPT_PATH"; then
    pass_test "Script sets .env permissions to 600"
else
    fail_test "Script does not set .env permissions to 600 (SECURITY RISK - credentials exposed)"
fi
echo ""

# Test 9: Variable definitions
echo "=== Test 9: Variable Definitions ==="
if grep -q "INSTALL_DIR=" "$SCRIPT_PATH"; then
    pass_test "Script defines INSTALL_DIR variable"
else
    fail_test "Script missing INSTALL_DIR definition"
fi

if grep -q 'REPO_URL=.*:-' "$SCRIPT_PATH"; then
    pass_test "Script defines REPO_URL with default value"
else
    log_warn "REPO_URL may not have a default value"
fi
echo ""

# Test 10: Error handling
echo "=== Test 10: Error Handling ==="
CMD_CHECKS=$(grep -c "command -v\|which" "$SCRIPT_PATH" || true)
if [ "$CMD_CHECKS" -gt 0 ]; then
    pass_test "Script checks for command existence ($CMD_CHECKS checks found)"
else
    log_warn "Script does not check for command existence"
fi

DIR_CHECKS=$(grep -c "\[ -d\|-d \]" "$SCRIPT_PATH" || true)
if [ "$DIR_CHECKS" -gt 0 ]; then
    pass_test "Script checks for directory existence ($DIR_CHECKS checks found)"
else
    fail_test "Script does not check for directory existence"
fi

FILE_CHECKS=$(grep -c "\[ -f\|-f \]" "$SCRIPT_PATH" || true)
if [ "$FILE_CHECKS" -gt 0 ]; then
    pass_test "Script checks for file existence ($FILE_CHECKS checks found)"
else
    fail_test "Script does not check for file existence"
fi
echo ""

# Final summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total Tests: $TESTS_TOTAL"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please review the output above.${NC}"
    exit 1
fi
