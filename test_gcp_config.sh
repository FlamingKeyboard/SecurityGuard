#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASS++))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAIL++))
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_info() {
    echo -e "[INFO] $1"
}

echo "========================================"
echo "GCP/ADC Configuration Validation"
echo "========================================"
echo ""

# Test 1: Check if running on GCE
echo "Test 1: Checking if running on GCE..."
# Try both hostname and direct IP (169.254.169.254 is the metadata server IP)
if curl -s -f -H "Metadata-Flavor: Google" --connect-timeout 2 "http://metadata.google.internal/computeMetadata/v1/instance/id" > /dev/null 2>&1; then
    print_pass "Running on GCE - metadata server accessible via hostname"
    ON_GCE=true
elif curl -s -f -H "Metadata-Flavor: Google" --connect-timeout 2 "http://169.254.169.254/computeMetadata/v1/instance/id" > /dev/null 2>&1; then
    print_pass "Running on GCE - metadata server accessible via IP (169.254.169.254)"
    ON_GCE=true
else
    print_fail "Not running on GCE or metadata server not accessible"
    print_info "Note: Metadata server is at 169.254.169.254 - requires network_mode: host in containers"
    ON_GCE=false
fi
echo ""

# Test 2: Verify GCP_PROJECT_ID is set
echo "Test 2: Checking GCP_PROJECT_ID environment variable..."
if [ -z "$GCP_PROJECT_ID" ]; then
    print_fail "GCP_PROJECT_ID is not set"
else
    print_pass "GCP_PROJECT_ID is set: $GCP_PROJECT_ID"
fi
echo ""

# Test 3: Test metadata server connectivity
echo "Test 3: Testing metadata server connectivity..."
if $ON_GCE; then
    PROJECT_ID=$(curl -s -f -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/project/project-id" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$PROJECT_ID" ]; then
        print_pass "Metadata server accessible, Project ID: $PROJECT_ID"

        if [ -n "$GCP_PROJECT_ID" ] && [ "$GCP_PROJECT_ID" != "$PROJECT_ID" ]; then
            print_warn "GCP_PROJECT_ID ($GCP_PROJECT_ID) does not match metadata project ID ($PROJECT_ID)"
        fi
    else
        print_fail "Failed to retrieve project ID from metadata server"
    fi
else
    print_warn "Skipping metadata server test (not on GCE)"
fi
echo ""

# Test 4: Check if service account is attached to VM
echo "Test 4: Checking service account attachment..."
if $ON_GCE; then
    SERVICE_ACCOUNT=$(curl -s -f -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$SERVICE_ACCOUNT" ]; then
        print_pass "Service account attached: $SERVICE_ACCOUNT"

        # Get service account scopes
        SCOPES=$(curl -s -f -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/scopes" 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$SCOPES" ]; then
            print_info "Service account scopes:"
            echo "$SCOPES" | while read -r scope; do
                echo "  - $scope"
            done
        fi
    else
        print_fail "No service account attached to VM"
    fi
else
    print_warn "Skipping service account check (not on GCE)"
fi
echo ""

# Test 5: Verify required IAM permissions
echo "Test 5: Listing required IAM permissions for this application..."
print_info "The service account should have the following permissions:"
echo "  - vertexai.endpoints.predict (for Vertex AI predictions)"
echo "  - vertexai.publishers.predict (for Vertex AI model predictions)"
echo "  - aiplatform.endpoints.predict (for AI Platform endpoints)"
echo "  - storage.objects.get (if accessing GCS buckets)"
echo "  - storage.objects.list (if accessing GCS buckets)"
echo ""

if $ON_GCE && [ -n "$SERVICE_ACCOUNT" ]; then
    print_info "To verify permissions, run:"
    echo "  gcloud projects get-iam-policy $PROJECT_ID --flatten=\"bindings[].members\" --filter=\"bindings.members:serviceAccount:$SERVICE_ACCOUNT\""
    echo ""

    # Test if we can get an access token
    TOKEN=$(curl -s -f -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$TOKEN" ]; then
        print_pass "Successfully retrieved access token from metadata server"
    else
        print_fail "Failed to retrieve access token from metadata server"
    fi
else
    print_warn "Skipping permission verification (not on GCE or no service account)"
fi
echo ""

# Test 6: Check compose.yaml for network_mode: host
echo "Test 6: Checking compose.yaml for network_mode: host..."
COMPOSE_FILE="compose.yaml"
if [ -f "$COMPOSE_FILE" ]; then
    if grep -q "network_mode:.*host" "$COMPOSE_FILE"; then
        print_pass "compose.yaml contains 'network_mode: host'"
    else
        print_fail "compose.yaml does not contain 'network_mode: host' (required for metadata server access)"
        print_info "Add the following to your service definition in compose.yaml:"
        echo "  network_mode: host"
    fi
else
    # Try docker-compose.yaml as alternative
    COMPOSE_FILE="docker-compose.yaml"
    if [ -f "$COMPOSE_FILE" ]; then
        if grep -q "network_mode:.*host" "$COMPOSE_FILE"; then
            print_pass "docker-compose.yaml contains 'network_mode: host'"
        else
            print_fail "docker-compose.yaml does not contain 'network_mode: host' (required for metadata server access)"
            print_info "Add the following to your service definition in docker-compose.yaml:"
            echo "  network_mode: host"
        fi
    else
        print_fail "compose.yaml or docker-compose.yaml not found"
    fi
fi
echo ""

# Summary
echo "========================================"
echo "Summary"
echo "========================================"
echo -e "Tests passed: ${GREEN}$PASS${NC}"
echo -e "Tests failed: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please review the output above.${NC}"
    exit 1
fi
