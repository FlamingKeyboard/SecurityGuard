#!/bin/bash
#
# Master Test Runner for Deployment Tests
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VERBOSE=false
SKIP_GCP=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXIT_CODE=0

declare -A TEST_RESULTS
declare -A TEST_DURATIONS

TESTS=(
    "test_setup_script.sh"
    "test_update_script.sh"
    "test_container_config.py"
    "test_podman_compat.sh"
    "test_gcp_config.sh"
)

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --verbose     Enable verbose output
    --skip-gcp    Skip GCP-related tests
    -h, --help    Show this help message
EOF
}

run_test() {
    local test_script=$1
    local test_path="${SCRIPT_DIR}/${test_script}"

    print_info "Running: $test_script"

    local start_time=$(date +%s)

    if [ ! -f "$test_path" ]; then
        echo -e "${YELLOW}[SKIP]${NC} Test script not found: $test_path"
        TEST_RESULTS[$test_script]="SKIP"
        TEST_DURATIONS[$test_script]="0"
        return 0
    fi

    local output_file=$(mktemp)
    local result=0

    if [[ "$test_script" == *.py ]]; then
        python3 "$test_path" > "$output_file" 2>&1 || result=$?
    else
        chmod +x "$test_path" 2>/dev/null || true
        bash "$test_path" > "$output_file" 2>&1 || result=$?
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $result -eq 0 ]; then
        TEST_RESULTS[$test_script]="PASS"
        print_pass "$test_script completed in ${duration}s"
    else
        TEST_RESULTS[$test_script]="FAIL"
        print_fail "$test_script failed after ${duration}s"
        EXIT_CODE=1
    fi

    TEST_DURATIONS[$test_script]="$duration"

    if [ "$VERBOSE" = true ]; then
        echo "--- Output ---"
        cat "$output_file"
        echo "--------------"
    fi

    rm -f "$output_file"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --skip-gcp)
            SKIP_GCP=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

print_header "DEPLOYMENT TEST RUNNER"
echo "Start time: $(date)"
echo "Working directory: $SCRIPT_DIR"
echo "Verbose: $VERBOSE"
echo "Skip GCP: $SKIP_GCP"
echo ""

# Run tests
for test in "${TESTS[@]}"; do
    if [ "$SKIP_GCP" = true ] && [[ "$test" == *"gcp"* ]]; then
        print_info "Skipping GCP test: $test"
        TEST_RESULTS[$test]="SKIP"
        TEST_DURATIONS[$test]="0"
        echo ""
        continue
    fi

    run_test "$test"
done

# Summary
print_header "TEST SUMMARY"

printf "%-30s %-10s %-10s\n" "TEST NAME" "STATUS" "DURATION"
printf "%-30s %-10s %-10s\n" "----------" "------" "--------"

total=0
passed=0
failed=0
skipped=0

for test in "${TESTS[@]}"; do
    status="${TEST_RESULTS[$test]:-UNKNOWN}"
    duration="${TEST_DURATIONS[$test]:-0}"

    total=$((total + 1))

    case "$status" in
        "PASS")
            printf "%-30s ${GREEN}%-10s${NC} %-10s\n" "$test" "$status" "${duration}s"
            passed=$((passed + 1))
            ;;
        "FAIL")
            printf "%-30s ${RED}%-10s${NC} %-10s\n" "$test" "$status" "${duration}s"
            failed=$((failed + 1))
            ;;
        "SKIP")
            printf "%-30s ${YELLOW}%-10s${NC} %-10s\n" "$test" "$status" "N/A"
            skipped=$((skipped + 1))
            ;;
        *)
            printf "%-30s ${YELLOW}%-10s${NC} %-10s\n" "$test" "UNKNOWN" "N/A"
            skipped=$((skipped + 1))
            ;;
    esac
done

echo ""
echo -e "${BLUE}========================================${NC}"
printf "Total:   %d\n" "$total"
printf "${GREEN}Passed:  %d${NC}\n" "$passed"
printf "${RED}Failed:  %d${NC}\n" "$failed"
printf "${YELLOW}Skipped: %d${NC}\n" "$skipped"
echo -e "${BLUE}========================================${NC}"

if [ $failed -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
else
    echo -e "\n${RED}Some tests failed!${NC}"
fi

exit $EXIT_CODE
