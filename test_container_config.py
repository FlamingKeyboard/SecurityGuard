#!/usr/bin/env python3
"""
Test script for validating compose.yaml and Dockerfile configuration.

Requirements:
1. Validate YAML syntax of compose.yaml
2. Check that deploy.resources is NOT used (should use mem_limit instead)
3. Verify mem_limit and mem_reservation are present
4. Check for network_mode: host (for GCP ADC)
5. Validate Dockerfile syntax (basic checks)
6. Verify HEALTHCHECK is present
7. Check that USER directive exists (non-root)
"""

import re
import sys
import yaml
from pathlib import Path


class TestResults:
    """Simple test result tracker."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, test_name):
        self.passed.append(test_name)
        print(f"[PASS] {test_name}")

    def add_fail(self, test_name, reason):
        self.failed.append((test_name, reason))
        print(f"[FAIL] {test_name}")
        print(f"  Reason: {reason}")

    def add_warning(self, test_name, reason):
        self.warnings.append((test_name, reason))
        print(f"[WARN] {test_name}")
        print(f"  Reason: {reason}")

    def print_summary(self):
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Passed:   {len(self.passed)}")
        print(f"Failed:   {len(self.failed)}")
        print(f"Warnings: {len(self.warnings)}")

        if self.failed:
            print("\nFailed tests:")
            for test_name, reason in self.failed:
                print(f"  - {test_name}: {reason}")

        if self.warnings:
            print("\nWarnings:")
            for test_name, reason in self.warnings:
                print(f"  - {test_name}: {reason}")

        print("=" * 70)
        return len(self.failed) == 0


def test_compose_yaml(compose_path, results):
    """Test compose.yaml configuration."""
    print("\n" + "=" * 70)
    print("Testing compose.yaml")
    print("=" * 70)

    if not compose_path.exists():
        results.add_fail("compose.yaml exists", f"File not found: {compose_path}")
        return None

    results.add_pass("compose.yaml exists")

    # Test 1: Validate YAML syntax
    try:
        with open(compose_path, 'r') as f:
            compose_data = yaml.safe_load(f)
        results.add_pass("YAML syntax is valid")
    except yaml.YAMLError as e:
        results.add_fail("YAML syntax is valid", f"YAML parsing error: {e}")
        return None

    # Check if services exist
    if not compose_data or 'services' not in compose_data:
        results.add_fail("services section exists", "No 'services' section found")
        return None

    results.add_pass("services section exists")

    # Get the first service (assuming security-guard)
    services = compose_data['services']
    if not services:
        results.add_fail("service defined", "No services defined in compose.yaml")
        return None

    service_name = list(services.keys())[0]
    service = services[service_name]

    # Test 2: Check that deploy.resources is NOT used
    if 'deploy' in service and 'resources' in service.get('deploy', {}):
        results.add_fail(
            "deploy.resources NOT used",
            "Found 'deploy.resources' - should use 'mem_limit' and 'mem_reservation' instead"
        )
    else:
        results.add_pass("deploy.resources NOT used")

    # Test 3: Verify mem_limit and mem_reservation are present
    has_mem_limit = 'mem_limit' in service
    has_mem_reservation = 'mem_reservation' in service

    if has_mem_limit:
        results.add_pass("mem_limit is present")
    else:
        results.add_fail("mem_limit is present", "Missing 'mem_limit' directive")

    if has_mem_reservation:
        results.add_pass("mem_reservation is present")
    else:
        results.add_fail("mem_reservation is present", "Missing 'mem_reservation' directive")

    # Test 4: Check for network_mode: host (REQUIRED for GCP ADC)
    if 'network_mode' in service and service['network_mode'] == 'host':
        results.add_pass("network_mode: host is set")
    else:
        results.add_fail(
            "network_mode: host is set",
            "network_mode: host not found - REQUIRED for GCP ADC metadata server access (169.254.169.254)"
        )

    return compose_data


def test_dockerfile(dockerfile_path, results):
    """Test Dockerfile configuration."""
    print("\n" + "=" * 70)
    print("Testing Dockerfile")
    print("=" * 70)

    if not dockerfile_path.exists():
        results.add_fail("Dockerfile exists", f"File not found: {dockerfile_path}")
        return

    results.add_pass("Dockerfile exists")

    # Read Dockerfile
    try:
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
    except Exception as e:
        results.add_fail("Dockerfile readable", f"Error reading file: {e}")
        return

    results.add_pass("Dockerfile readable")

    # Test 5: Basic Dockerfile syntax validation
    lines = dockerfile_content.split('\n')
    has_from = False
    has_healthcheck = False
    has_user = False
    user_directive = None

    for line in lines:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue

        # Check for FROM directive
        if line.upper().startswith('FROM '):
            has_from = True

        # Check for HEALTHCHECK directive
        if line.upper().startswith('HEALTHCHECK '):
            has_healthcheck = True

        # Check for USER directive
        if line.upper().startswith('USER '):
            has_user = True
            user_directive = line.split()[1] if len(line.split()) > 1 else None

    # Test: FROM directive exists
    if has_from:
        results.add_pass("FROM directive present")
    else:
        results.add_fail("FROM directive present", "No FROM directive found")

    # Test 6: Verify HEALTHCHECK is present
    if has_healthcheck:
        results.add_pass("HEALTHCHECK is present")
    else:
        results.add_fail("HEALTHCHECK is present", "No HEALTHCHECK directive found")

    # Test 7: Check that USER directive exists (non-root)
    if has_user:
        results.add_pass("USER directive exists")

        # Additional check: ensure it's not root
        if user_directive and user_directive.lower() not in ['root', '0']:
            results.add_pass("USER is non-root")
        elif user_directive and user_directive.lower() in ['root', '0']:
            results.add_fail("USER is non-root", f"USER is set to root: {user_directive}")
        else:
            results.add_warning("USER is non-root", "Could not parse USER directive value")
    else:
        results.add_fail("USER directive exists", "No USER directive found - container will run as root")

    # Additional validation: Check for basic Dockerfile structure
    if 'CMD' not in dockerfile_content.upper() and 'ENTRYPOINT' not in dockerfile_content.upper():
        results.add_warning(
            "CMD or ENTRYPOINT present",
            "No CMD or ENTRYPOINT directive found - container may not start properly"
        )
    else:
        results.add_pass("CMD or ENTRYPOINT present")


def main():
    """Main test runner."""
    print("\n" + "=" * 70)
    print("Container Configuration Validation Tests")
    print("=" * 70)

    # Setup paths
    script_dir = Path(__file__).parent
    compose_path = script_dir / 'compose.yaml'
    dockerfile_path = script_dir / 'Dockerfile'

    # Initialize test results
    results = TestResults()

    # Run tests
    test_compose_yaml(compose_path, results)
    test_dockerfile(dockerfile_path, results)

    # Print summary and exit
    success = results.print_summary()

    if success:
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
