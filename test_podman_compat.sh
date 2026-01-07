#!/bin/bash
#
# Podman Compatibility Test Script
# Tests Podman installation and compatibility for Vivint Security Guard
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#

# Note: NOT using set -e because test scripts should continue on individual test failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNED=0

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((TESTS_WARNED++))
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Test 1: Check if Podman is installed
test_podman_installed() {
    print_test "Checking if Podman is installed..."

    if command -v podman &> /dev/null; then
        PODMAN_VERSION=$(podman --version)
        print_pass "Podman is installed: $PODMAN_VERSION"

        # Extract version number (using portable regex)
        VERSION_NUM=$(echo "$PODMAN_VERSION" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        print_info "Version: $VERSION_NUM"

        # Check minimum version (4.0.0+)
        MAJOR_VERSION=$(echo "$VERSION_NUM" | cut -d. -f1)
        if [ "$MAJOR_VERSION" -ge 4 ]; then
            print_pass "Podman version meets minimum requirement (4.0.0+)"
        else
            print_warn "Podman version is below 4.0.0, some features may not work"
        fi
        return 0
    else
        print_fail "Podman is not installed or not in PATH"
        return 1
    fi
}

# Test 2: Check Podman Compose availability
test_podman_compose() {
    print_test "Checking for Podman Compose compatibility..."

    COMPOSE_CMD=""

    # Check for podman-compose (standalone)
    if command -v podman-compose &> /dev/null; then
        COMPOSE_VERSION=$(podman-compose --version 2>&1)
        print_pass "podman-compose is installed: $COMPOSE_VERSION"
        COMPOSE_CMD="podman-compose"
    # Check for podman compose (built-in)
    elif podman compose version &> /dev/null; then
        COMPOSE_VERSION=$(podman compose version 2>&1)
        print_pass "podman compose (built-in) is available: $COMPOSE_VERSION"
        COMPOSE_CMD="podman compose"
    else
        print_fail "Neither 'podman-compose' nor 'podman compose' is available"
        print_info "Install with: pip3 install podman-compose"
        print_info "Or use Podman 4.0+ which includes compose support"
        return 1
    fi

    # Export for later tests
    export COMPOSE_CMD
    return 0
}

# Test 3: Validate compose.yaml files
test_compose_config() {
    print_test "Validating compose.yaml configuration..."

    if [ -z "$COMPOSE_CMD" ]; then
        print_fail "Compose command not available, skipping validation"
        return 1
    fi

    # Test compose.yaml
    if [ -f "$SCRIPT_DIR/compose.yaml" ]; then
        print_info "Validating compose.yaml..."
        if $COMPOSE_CMD -f "$SCRIPT_DIR/compose.yaml" config &> /dev/null; then
            print_pass "compose.yaml is valid"
        else
            print_fail "compose.yaml validation failed"
            $COMPOSE_CMD -f "$SCRIPT_DIR/compose.yaml" config 2>&1 | head -20
            return 1
        fi
    else
        print_warn "compose.yaml not found at $SCRIPT_DIR/compose.yaml"
    fi

    # Test compose-rocky.yaml
    if [ -f "$SCRIPT_DIR/compose-rocky.yaml" ]; then
        print_info "Validating compose-rocky.yaml..."
        if $COMPOSE_CMD -f "$SCRIPT_DIR/compose-rocky.yaml" config &> /dev/null; then
            print_pass "compose-rocky.yaml is valid"
        else
            print_fail "compose-rocky.yaml validation failed"
            $COMPOSE_CMD -f "$SCRIPT_DIR/compose-rocky.yaml" config 2>&1 | head -20
            return 1
        fi
    else
        print_warn "compose-rocky.yaml not found at $SCRIPT_DIR/compose-rocky.yaml"
    fi

    return 0
}

# Test 4: Check for SELinux and volume labels
test_selinux_labels() {
    print_test "Checking SELinux configuration..."

    # Check if SELinux is enabled
    if command -v getenforce &> /dev/null; then
        SELINUX_STATUS=$(getenforce 2>&1)
        print_info "SELinux status: $SELINUX_STATUS"

        if [ "$SELINUX_STATUS" = "Enforcing" ] || [ "$SELINUX_STATUS" = "Permissive" ]; then
            print_pass "SELinux is active ($SELINUX_STATUS)"

            # Check if Rocky Linux
            if [ -f /etc/rocky-release ]; then
                print_info "Rocky Linux detected: $(cat /etc/rocky-release)"

                # Check for :Z labels in compose-rocky.yaml
                if [ -f "$SCRIPT_DIR/compose-rocky.yaml" ]; then
                    if grep -q ':Z' "$SCRIPT_DIR/compose-rocky.yaml"; then
                        print_pass "SELinux volume labels (:Z) found in compose-rocky.yaml"
                    else
                        print_fail "SELinux volume labels (:Z) not found in compose-rocky.yaml"
                        print_info "Rocky Linux with SELinux should use :Z or :z volume labels"
                        return 1
                    fi
                fi

                # Check Podman SELinux support (more specific check)
                SELINUX_ENABLED=$(podman info --format '{{.Host.Security.SELinuxEnabled}}' 2>/dev/null || echo "false")
                if [ "$SELINUX_ENABLED" = "true" ]; then
                    print_pass "Podman has SELinux support enabled"
                else
                    print_warn "Podman SELinux support not detected"
                fi
            else
                print_info "Not running on Rocky Linux"
            fi
        else
            print_info "SELinux is disabled"
        fi
    else
        print_info "SELinux not available on this system"
    fi

    return 0
}

# Test 5: Test container build (dry-run)
test_container_build() {
    print_test "Testing container build capability..."

    if [ ! -f "$SCRIPT_DIR/Dockerfile" ]; then
        print_warn "Dockerfile not found, skipping build test"
        return 0
    fi

    print_info "Attempting dry-run build test..."

    # Try to build with --layers to cache layers
    if podman build --help | grep -q '\-\-dry-run'; then
        # Podman supports --dry-run
        if podman build --dry-run -t vivint-security-guard-test -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR" &> /dev/null; then
            print_pass "Container build dry-run successful"
        else
            print_fail "Container build dry-run failed"
            return 1
        fi
    else
        # Fallback: Just check if Dockerfile is parseable
        print_info "Dry-run not supported, checking Dockerfile syntax..."

        # Basic Dockerfile validation
        if grep -q '^FROM ' "$SCRIPT_DIR/Dockerfile"; then
            print_pass "Dockerfile appears valid (has FROM instruction)"
        else
            print_fail "Dockerfile missing FROM instruction"
            return 1
        fi

        print_info "Run 'podman build -t vivint-security-guard .' to test full build"
    fi

    return 0
}

# Test 6: Check rootless Podman configuration
test_rootless_config() {
    print_test "Checking rootless Podman configuration..."

    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        print_info "Running as root (rootful mode)"
        return 0
    fi

    print_info "Running as non-root user (rootless mode)"

    # Check user namespaces
    if [ -f /proc/sys/user/max_user_namespaces ]; then
        MAX_USER_NS=$(cat /proc/sys/user/max_user_namespaces)
        if [ "$MAX_USER_NS" -gt 0 ]; then
            print_pass "User namespaces enabled (max_user_namespaces=$MAX_USER_NS)"
        else
            print_fail "User namespaces disabled"
            print_info "Enable with: sysctl -w user.max_user_namespaces=15000"
            return 1
        fi
    fi

    # Check subuid/subgid
    if [ -f /etc/subuid ] && grep -q "^$(whoami):" /etc/subuid; then
        SUBUID_RANGE=$(grep "^$(whoami):" /etc/subuid | cut -d: -f2-3)
        print_pass "subuid configured: $SUBUID_RANGE"
    else
        print_fail "subuid not configured for user $(whoami)"
        print_info "Configure with: sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 $(whoami)"
        return 1
    fi

    if [ -f /etc/subgid ] && grep -q "^$(whoami):" /etc/subgid; then
        SUBGID_RANGE=$(grep "^$(whoami):" /etc/subgid | cut -d: -f2-3)
        print_pass "subgid configured: $SUBGID_RANGE"
    else
        print_fail "subgid not configured for user $(whoami)"
        print_info "Configure with: sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 $(whoami)"
        return 1
    fi

    # Check cgroup v2
    if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
        print_pass "cgroup v2 detected"
    else
        print_warn "cgroup v2 not detected, some features may be limited"
    fi

    # Check Podman socket (for compose)
    if podman info --format '{{.Host.RemoteSocket.Path}}' &> /dev/null; then
        SOCKET_PATH=$(podman info --format '{{.Host.RemoteSocket.Path}}' 2>/dev/null || echo "N/A")
        if [ "$SOCKET_PATH" != "N/A" ] && [ -S "$SOCKET_PATH" ]; then
            print_pass "Podman socket available at $SOCKET_PATH"
        else
            print_info "Podman socket: $SOCKET_PATH"
        fi
    fi

    return 0
}

# Test 7: Check Podman storage configuration
test_storage_config() {
    print_test "Checking Podman storage configuration..."

    # Get storage driver
    STORAGE_DRIVER=$(podman info --format '{{.Store.GraphDriverName}}' 2>/dev/null || echo "unknown")
    print_info "Storage driver: $STORAGE_DRIVER"

    if [ "$STORAGE_DRIVER" = "overlay" ]; then
        print_pass "Using overlay storage driver (recommended)"
    elif [ "$STORAGE_DRIVER" = "vfs" ]; then
        print_warn "Using vfs storage driver (slow, not recommended for production)"
    else
        print_info "Using $STORAGE_DRIVER storage driver"
    fi

    # Check storage location
    GRAPH_ROOT=$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo "unknown")
    print_info "Storage location: $GRAPH_ROOT"

    # Check available space
    if [ -d "$GRAPH_ROOT" ]; then
        AVAILABLE_SPACE=$(df -h "$GRAPH_ROOT" | tail -1 | awk '{print $4}')
        print_info "Available storage space: $AVAILABLE_SPACE"
    fi

    return 0
}

# Test 8: Verify Podman can run a simple container
test_simple_container() {
    print_test "Testing simple container execution..."

    print_info "Attempting to run 'podman run --rm alpine echo test'..."

    if timeout 30 podman run --rm alpine echo "Podman container test" &> /dev/null; then
        print_pass "Successfully ran test container"
    else
        print_fail "Failed to run test container"
        print_info "This may indicate networking, storage, or configuration issues"
        return 1
    fi

    return 0
}

# Main execution
main() {
    print_header "Podman Compatibility Test Suite"
    echo ""

    # Run all tests
    test_podman_installed || true
    echo ""

    test_podman_compose || true
    echo ""

    test_compose_config || true
    echo ""

    test_selinux_labels || true
    echo ""

    test_container_build || true
    echo ""

    test_rootless_config || true
    echo ""

    test_storage_config || true
    echo ""

    test_simple_container || true
    echo ""

    # Print summary
    print_header "Test Summary"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo -e "${YELLOW}Warnings: $TESTS_WARNED${NC}"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}All critical tests passed!${NC}"
        echo -e "${GREEN}Podman is configured correctly for Vivint Security Guard${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed. Please review the output above.${NC}"
        exit 1
    fi
}

# Run main function
main
