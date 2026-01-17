#!/bin/bash
# lib/functions.sh - Common functions for ACTIVATE RAG-vLLM
# Source this file in other scripts: source "$(dirname "$0")/lib/functions.sh"

set -o pipefail

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_DIR="${LOG_DIR:-./logs}"
LOG_FILE="${LOG_DIR}/service-$(date +%Y%m%d-%H%M%S).log"
DEBUG="${DEBUG:-0}"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR" 2>/dev/null || true

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Color codes for terminal output
    local color_reset="\033[0m"
    local color=""
    case "$level" in
        INFO)  color="\033[0;32m" ;;  # Green
        WARN)  color="\033[0;33m" ;;  # Yellow
        ERROR) color="\033[0;31m" ;;  # Red
        DEBUG) color="\033[0;36m" ;;  # Cyan
    esac
    
    # Log to file (without colors)
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE" 2>/dev/null || true
    
    # Log to terminal (with colors if interactive)
    if [[ -t 1 ]]; then
        echo -e "${color}[$timestamp] [$level]${color_reset} $message"
    else
        echo "[$timestamp] [$level] $message"
    fi
}

info()  { log "INFO" "$@"; }
warn()  { log "WARN" "$@" >&2; }
error() { log "ERROR" "$@" >&2; }
debug() { [[ "${DEBUG}" == "1" ]] && log "DEBUG" "$@"; }

# ============================================================================
# Environment Detection
# ============================================================================

detect_container_runtime() {
    # Detect available container runtime
    if command -v singularity >/dev/null 2>&1; then
        echo "singularity"
    elif command -v docker >/dev/null 2>&1; then
        echo "docker"
    else
        echo "none"
    fi
}

detect_scheduler() {
    # Detect job scheduler on the system
    if command -v sbatch >/dev/null 2>&1; then
        echo "slurm"
    elif command -v qsub >/dev/null 2>&1; then
        echo "pbs"
    else
        echo "ssh"
    fi
}

detect_gpu_count() {
    # Detect number of available GPUs
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# ============================================================================
# Port Management
# ============================================================================

find_available_port() {
    local start_port="${1:-8000}"
    local max_attempts="${2:-100}"
    local port=$start_port
    
    for ((i=0; i<max_attempts; i++)); do
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
        ((port++))
    done
    
    error "Could not find available port after $max_attempts attempts"
    return 1
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout="${3:-60}"
    local interval="${4:-2}"
    
    local elapsed=0
    while (( elapsed < timeout )); do
        if nc -z "$host" "$port" 2>/dev/null; then
            info "Port $port is now available on $host"
            return 0
        fi
        sleep "$interval"
        ((elapsed += interval))
        debug "Waiting for port $port on $host... (${elapsed}s/${timeout}s)"
    done
    
    error "Timeout waiting for port $port on $host"
    return 1
}

# ============================================================================
# Service Health Checks
# ============================================================================

check_vllm_health() {
    local host="${1:-localhost}"
    local port="${2:-8000}"
    local timeout="${3:-300}"
    local interval="${4:-5}"
    
    info "Waiting for vLLM server to be ready at $host:$port..."
    
    local elapsed=0
    while (( elapsed < timeout )); do
        local response
        response=$(curl -s -o /dev/null -w "%{http_code}" "http://$host:$port/health" 2>/dev/null || echo "000")
        
        if [[ "$response" == "200" ]]; then
            info "vLLM server is ready"
            return 0
        fi
        
        sleep "$interval"
        ((elapsed += interval))
        debug "vLLM health check... (${elapsed}s/${timeout}s, response: $response)"
    done
    
    error "Timeout waiting for vLLM server"
    return 1
}

check_rag_health() {
    local host="${1:-localhost}"
    local port="${2:-8080}"
    local timeout="${3:-120}"
    local interval="${4:-5}"
    
    info "Waiting for RAG server to be ready at $host:$port..."
    
    local elapsed=0
    while (( elapsed < timeout )); do
        local response
        response=$(curl -s -o /dev/null -w "%{http_code}" "http://$host:$port/health" 2>/dev/null || echo "000")
        
        if [[ "$response" == "200" ]]; then
            info "RAG server is ready"
            return 0
        fi
        
        sleep "$interval"
        ((elapsed += interval))
        debug "RAG health check... (${elapsed}s/${timeout}s, response: $response)"
    done
    
    error "Timeout waiting for RAG server"
    return 1
}

# ============================================================================
# Cleanup Functions
# ============================================================================

cleanup_on_exit() {
    local exit_code=$?
    
    if (( exit_code != 0 )); then
        error "Script failed with exit code: $exit_code"
        
        # Capture logs for debugging
        if [[ -d "./logs" ]]; then
            local debug_archive="debug-logs-$(date +%Y%m%d-%H%M%S).tar.gz"
            tar -czf "$debug_archive" ./logs/ 2>/dev/null && \
                info "Debug logs saved to $debug_archive"
        fi
    fi
    
    # Run cancel script if it exists
    if [[ -x "./cancel.sh" ]]; then
        debug "Running cleanup via cancel.sh"
        ./cancel.sh 2>/dev/null || true
    fi
}

# ============================================================================
# Utility Functions
# ============================================================================

require_command() {
    local cmd="$1"
    local install_hint="${2:-}"
    
    if ! command -v "$cmd" >/dev/null 2>&1; then
        error "Required command not found: $cmd"
        [[ -n "$install_hint" ]] && error "Install hint: $install_hint"
        return 1
    fi
    return 0
}

create_directory_structure() {
    local base_dir="${1:-.}"
    
    local dirs=(
        "$base_dir/logs"
        "$base_dir/cache"
        "$base_dir/cache/chroma"
        "$base_dir/cache/huggingface"
        "$base_dir/cache/sagemaker_sessions"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        debug "Created directory: $dir"
    done
    
    # Fix sagemaker sessions permissions
    chmod 700 "$base_dir/cache/sagemaker_sessions" 2>/dev/null || true
    
    # Create /dev/shm directory if accessible
    if [[ -d "/dev/shm" && -w "/dev/shm" ]]; then
        mkdir -p /dev/shm/sagemaker_sessions 2>/dev/null || true
        chmod 700 /dev/shm/sagemaker_sessions 2>/dev/null || true
    fi
}

export_session_port() {
    local port="$1"
    local file="${2:-SESSION_PORT}"
    
    echo "${port}" > "$file"
    info "Session port exported: $port"
}

# ============================================================================
# Configuration Helpers
# ============================================================================

load_env_file() {
    local env_file="$1"
    
    if [[ -f "$env_file" ]]; then
        # shellcheck source=/dev/null
        source "$env_file"
        debug "Loaded environment from: $env_file"
    else
        warn "Environment file not found: $env_file"
    fi
}

validate_required_vars() {
    local missing=()
    
    for var in "$@"; do
        if [[ -z "${!var:-}" ]]; then
            missing+=("$var")
        fi
    done
    
    if (( ${#missing[@]} > 0 )); then
        error "Missing required environment variables: ${missing[*]}"
        return 1
    fi
    return 0
}
