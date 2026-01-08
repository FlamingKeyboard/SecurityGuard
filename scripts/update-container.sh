#!/bin/bash

################################################################################
# Security Guard Container Auto-Update Script
#
# This script automatically updates the Security Guard container when changes
# are detected in the git repository, with minimal downtime and comprehensive
# error handling.
################################################################################

set -euo pipefail

# PATH export for cron/systemd environment (minimal PATH by default)
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Configuration
INSTALL_DIR="/opt/security-guard"
COMPOSE_FILE="compose-rocky.yaml"
LOG_DIR="/var/log/security-guard"
LOG_FILE="${LOG_DIR}/update.log"
HEALTH_ENDPOINT="http://localhost:8080/health"
HEALTH_CHECK_RETRIES=3
HEALTH_CHECK_WAIT=30
CONTAINER_STARTUP_WAIT=30

# Container runs rootless under this user
PODMAN_USER="gavinfullertx"

# Function to run podman-compose as the correct user
run_podman_compose() {
    if [[ "$(id -un)" == "root" ]]; then
        sudo -u "${PODMAN_USER}" podman-compose "$@"
    else
        podman-compose "$@"
    fi
}

# Colors for output (optional, for terminal visibility)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

################################################################################
# Logging Functions
################################################################################

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() {
    log "INFO" "$@"
}

log_error() {
    log "ERROR" "$@"
}

log_success() {
    log "SUCCESS" "$@"
}

log_warning() {
    log "WARNING" "$@"
}

################################################################################
# Pushover Notification Function
################################################################################

send_pushover_notification() {
    local title="$1"
    local message="$2"
    local priority="${3:-0}"  # Default priority 0 (normal)

    # Source environment variables from .env file
    if [[ -f "${INSTALL_DIR}/.env" ]]; then
        # Export only PUSHOVER variables to avoid conflicts
        export $(grep -E '^PUSHOVER_' "${INSTALL_DIR}/.env" | xargs)
    else
        log_warning "No .env file found at ${INSTALL_DIR}/.env, skipping Pushover notification"
        return 1
    fi

    # Check if Pushover credentials are set
    if [[ -z "${PUSHOVER_TOKEN:-}" ]] || [[ -z "${PUSHOVER_USER:-}" ]]; then
        log_warning "PUSHOVER_TOKEN or PUSHOVER_USER not set, skipping notification"
        return 1
    fi

    # Send Pushover notification
    local response
    response=$(curl -s \
        --form-string "token=${PUSHOVER_TOKEN}" \
        --form-string "user=${PUSHOVER_USER}" \
        --form-string "title=${title}" \
        --form-string "message=${message}" \
        --form-string "priority=${priority}" \
        https://api.pushover.net/1/messages.json)

    if echo "${response}" | grep -q '"status":1'; then
        log_info "Pushover notification sent successfully"
        return 0
    else
        log_error "Failed to send Pushover notification: ${response}"
        return 1
    fi
}

################################################################################
# Health Check Function
################################################################################

check_container_health() {
    local retries="$1"
    local wait_time="$2"

    log_info "Waiting ${wait_time} seconds for container to start..."
    sleep "${wait_time}"

    for ((i=1; i<=retries; i++)); do
        log_info "Health check attempt ${i}/${retries}..."

        if curl -sf "${HEALTH_ENDPOINT}" > /dev/null 2>&1; then
            log_success "Container health check passed"
            return 0
        else
            log_warning "Health check failed (attempt ${i}/${retries})"
            if [[ ${i} -lt ${retries} ]]; then
                log_info "Waiting 10 seconds before retry..."
                sleep 10
            fi
        fi
    done

    log_error "Container health check failed after ${retries} attempts"
    return 1
}

################################################################################
# Rollback Function
################################################################################

rollback_to_commit() {
    local previous_commit="$1"

    log_warning "Initiating rollback to commit ${previous_commit}..."

    if git checkout "${previous_commit}" >> "${LOG_FILE}" 2>&1; then
        log_info "Git checkout to ${previous_commit} successful"

        log_info "Rebuilding container with previous version..."
        if run_podman_compose -f "${COMPOSE_FILE}" build >> "${LOG_FILE}" 2>&1; then
            log_info "Build successful"

            log_info "Stopping existing container..."
            run_podman_compose -f "${COMPOSE_FILE}" down >> "${LOG_FILE}" 2>&1 || true

            log_info "Starting container with rolled-back version..."
            if run_podman_compose -f "${COMPOSE_FILE}" up -d >> "${LOG_FILE}" 2>&1; then
                log_info "Container started"

                if check_container_health "${HEALTH_CHECK_RETRIES}" "${CONTAINER_STARTUP_WAIT}"; then
                    log_success "Rollback successful - container is healthy"
                    return 0
                else
                    log_error "Rollback failed - container is still unhealthy"
                    return 1
                fi
            else
                log_error "Failed to restart container during rollback"
                return 1
            fi
        else
            log_error "Failed to build container during rollback"
            return 1
        fi
    else
        log_error "Failed to checkout previous commit during rollback"
        return 1
    fi
}

################################################################################
# Main Update Logic
################################################################################

main() {
    log_info "========== Security Guard Update Check Started =========="

    # Create log directory if it doesn't exist
    if [[ ! -d "${LOG_DIR}" ]]; then
        mkdir -p "${LOG_DIR}"
        log_info "Created log directory: ${LOG_DIR}"
    fi

    # Change to installation directory
    if [[ ! -d "${INSTALL_DIR}" ]]; then
        log_error "Installation directory not found: ${INSTALL_DIR}"
        exit 1
    fi

    cd "${INSTALL_DIR}"
    log_info "Changed to directory: ${INSTALL_DIR}"

    # Ensure data directory exists and has correct permissions for container
    if [[ ! -d "${INSTALL_DIR}/data" ]]; then
        mkdir -p "${INSTALL_DIR}/data"
        log_info "Created data directory"
    fi
    chmod -R 777 "${INSTALL_DIR}/data" 2>/dev/null || true

    # Fetch latest changes from remote
    log_info "Fetching latest changes from origin/main..."
    if ! git fetch origin main >> "${LOG_FILE}" 2>&1; then
        log_error "Failed to fetch from remote repository"
        send_pushover_notification \
            "Security Guard Update Failed" \
            "Failed to fetch from remote repository at $(date '+%Y-%m-%d %H:%M:%S')" \
            1
        exit 1
    fi

    # Get local and remote commit hashes
    LOCAL_COMMIT=$(git rev-parse HEAD)
    REMOTE_COMMIT=$(git rev-parse origin/main)

    log_info "Local commit:  ${LOCAL_COMMIT}"
    log_info "Remote commit: ${REMOTE_COMMIT}"

    # Check if update is needed
    if [[ "${LOCAL_COMMIT}" == "${REMOTE_COMMIT}" ]]; then
        log_info "No updates available - already at latest version"
        log_info "========== Update Check Completed (No Changes) =========="
        exit 0
    fi

    log_info "Update available - proceeding with update..."

    # Save previous commit for potential rollback
    PREVIOUS_COMMIT="${LOCAL_COMMIT}"

    # Pull latest changes
    log_info "Pulling latest changes from origin/main..."
    if ! git pull origin main >> "${LOG_FILE}" 2>&1; then
        log_error "Failed to pull latest changes"
        send_pushover_notification \
            "Security Guard Update Failed" \
            "Failed to pull changes from repository at $(date '+%Y-%m-%d %H:%M:%S')" \
            1
        exit 1
    fi

    NEW_COMMIT=$(git rev-parse HEAD)
    log_info "Successfully pulled changes - now at commit ${NEW_COMMIT}"

    # Build new container image
    log_info "Building new container image..."
    if ! run_podman_compose -f "${COMPOSE_FILE}" build >> "${LOG_FILE}" 2>&1; then
        log_error "Failed to build container image"
        send_pushover_notification \
            "Security Guard Update Failed" \
            "Failed to build container image at $(date '+%Y-%m-%d %H:%M:%S')" \
            1

        # Rollback
        rollback_to_commit "${PREVIOUS_COMMIT}"
        exit 1
    fi

    log_success "Container image built successfully"

    # Stop existing container before deploying new one
    log_info "Stopping existing container..."
    run_podman_compose -f "${COMPOSE_FILE}" down >> "${LOG_FILE}" 2>&1 || true

    # Deploy updated container with new image
    log_info "Deploying updated container..."
    if ! run_podman_compose -f "${COMPOSE_FILE}" up -d >> "${LOG_FILE}" 2>&1; then
        log_error "Failed to deploy updated container"
        send_pushover_notification \
            "Security Guard Update Failed" \
            "Failed to deploy container at $(date '+%Y-%m-%d %H:%M:%S')" \
            1

        # Rollback
        rollback_to_commit "${PREVIOUS_COMMIT}"
        exit 1
    fi

    log_info "Container deployment initiated"

    # Perform health check
    if check_container_health "${HEALTH_CHECK_RETRIES}" "${CONTAINER_STARTUP_WAIT}"; then
        log_success "========== Update Completed Successfully =========="
        log_success "Updated from ${PREVIOUS_COMMIT} to ${NEW_COMMIT}"
        # No notification on success - only alert on failures
        exit 0
    else
        log_error "========== Update Failed - Health Check Failed =========="

        # Send failure notification
        send_pushover_notification \
            "Security Guard Update Failed" \
            "Container health check failed after update at $(date '+%Y-%m-%d %H:%M:%S')

Attempted update: ${PREVIOUS_COMMIT:0:8} -> ${NEW_COMMIT:0:8}

Initiating rollback..." \
            1

        # Rollback to previous version
        if rollback_to_commit "${PREVIOUS_COMMIT}"; then
            log_warning "Rollback completed successfully - running on previous version"
            send_pushover_notification \
                "Security Guard Rolled Back" \
                "Update failed, successfully rolled back to ${PREVIOUS_COMMIT:0:8} at $(date '+%Y-%m-%d %H:%M:%S')" \
                0
            exit 1
        else
            log_error "Rollback failed - manual intervention required"
            send_pushover_notification \
                "Security Guard Critical Failure" \
                "Update and rollback both failed at $(date '+%Y-%m-%d %H:%M:%S') - MANUAL INTERVENTION REQUIRED" \
                2
            exit 1
        fi
    fi
}

# Execute main function
main
