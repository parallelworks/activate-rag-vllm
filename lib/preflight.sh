#!/bin/bash
# lib/preflight.sh - Pre-flight checks for ACTIVATE RAG-vLLM deployment
# Source this file: source "$(dirname "$0")/lib/preflight.sh"

# Requires lib/functions.sh to be sourced first

# ============================================================================
# Pre-flight Check Entry Point
# ============================================================================

run_preflight_checks() {
    local errors=0
    local warnings=0
    
    info "=========================================="
    info "Running pre-flight checks..."
    info "=========================================="
    
    # Core checks
    check_container_runtime || ((errors++))
    check_gpu_access || ((warnings++))  # Warning only, might be running CPU mode
    check_network_access || ((warnings++))
    check_disk_space || ((warnings++))
    
    # Model checks
    if [[ "${MODEL_SOURCE:-local}" == "local" ]]; then
        check_local_model || ((errors++))
    fi
    
    # Port checks
    check_port_availability || ((errors++))
    
    info "=========================================="
    if (( errors > 0 )); then
        error "Pre-flight checks FAILED with $errors error(s) and $warnings warning(s)"
        return 1
    elif (( warnings > 0 )); then
        warn "Pre-flight checks PASSED with $warnings warning(s)"
        return 0
    else
        info "Pre-flight checks PASSED ✓"
        return 0
    fi
}

# ============================================================================
# Individual Check Functions
# ============================================================================

check_container_runtime() {
    local runmode="${RUNMODE:-singularity}"
    
    info "Checking container runtime: $runmode"
    
    case "$runmode" in
        singularity)
            if ! command -v singularity >/dev/null 2>&1; then
                error "Singularity not found in PATH"
                error "Please load the singularity module or install singularity"
                return 1
            fi
            local version
            version=$(singularity --version 2>/dev/null || echo "unknown")
            info "  Singularity version: $version"
            
            # Check for singularity-compose
            if ! command -v singularity-compose >/dev/null 2>&1; then
                # Try to source from common location
                if [[ -f ~/pw/software/singularity-compose/bin/activate ]]; then
                    source ~/pw/software/singularity-compose/bin/activate
                fi
            fi
            
            if ! command -v singularity-compose >/dev/null 2>&1; then
                warn "singularity-compose not found - will attempt to install"
            else
                info "  singularity-compose: available"
            fi
            ;;
        docker)
            if ! command -v docker >/dev/null 2>&1; then
                error "Docker not found in PATH"
                return 1
            fi
            
            if ! docker info >/dev/null 2>&1; then
                warn "Docker daemon not accessible - may need to start docker service"
            else
                info "  Docker: available and running"
            fi
            ;;
        *)
            error "Unknown run mode: $runmode"
            return 1
            ;;
    esac
    
    return 0
}

check_gpu_access() {
    info "Checking GPU access..."
    
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi not available - GPU may not be accessible"
        warn "If running on CPU-only node, this is expected"
        return 1
    fi
    
    local gpu_info
    gpu_info=$(nvidia-smi -L 2>/dev/null)
    
    if [[ -z "$gpu_info" ]]; then
        warn "No GPUs detected by nvidia-smi"
        return 1
    fi
    
    local gpu_count
    gpu_count=$(echo "$gpu_info" | wc -l)
    info "  GPUs detected: $gpu_count"
    
    # Show GPU details
    while IFS= read -r line; do
        debug "    $line"
    done <<< "$gpu_info"
    
    # Check CUDA visibility
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        info "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    fi
    
    return 0
}

check_network_access() {
    local model_source="${MODEL_SOURCE:-local}"
    
    info "Checking network access..."
    
    # Only critical if we need to download from HuggingFace
    if [[ "$model_source" == "huggingface" ]]; then
        if ! curl -s --connect-timeout 5 https://huggingface.co >/dev/null 2>&1; then
            error "Cannot reach HuggingFace Hub - check network/proxy settings"
            error "Set MODEL_SOURCE=local to use pre-downloaded models"
            return 1
        fi
        info "  HuggingFace Hub: reachable"
    else
        # Just informational for local mode
        if curl -s --connect-timeout 3 https://huggingface.co >/dev/null 2>&1; then
            info "  Network: available (HuggingFace reachable)"
        else
            info "  Network: limited or offline mode"
        fi
    fi
    
    return 0
}

check_disk_space() {
    info "Checking disk space..."
    
    local cache_dir="${MODEL_CACHE_BASE:-$HOME/.cache}"
    local work_dir="${PWD}"
    
    # Check cache directory space
    if [[ -d "$cache_dir" ]] || mkdir -p "$cache_dir" 2>/dev/null; then
        local free_gb
        free_gb=$(df -BG "$cache_dir" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
        
        if [[ -n "$free_gb" ]]; then
            if (( free_gb < 20 )); then
                warn "Low disk space in cache directory: ${free_gb}GB free"
                warn "Model downloads may fail. Consider freeing space or changing MODEL_CACHE_BASE"
            elif (( free_gb < 50 )); then
                info "  Cache directory ($cache_dir): ${free_gb}GB free (adequate)"
            else
                info "  Cache directory ($cache_dir): ${free_gb}GB free"
            fi
        fi
    fi
    
    # Check working directory space
    local work_free_gb
    work_free_gb=$(df -BG "$work_dir" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
    
    if [[ -n "$work_free_gb" ]]; then
        if (( work_free_gb < 10 )); then
            warn "Low disk space in working directory: ${work_free_gb}GB free"
        else
            info "  Working directory: ${work_free_gb}GB free"
        fi
    fi
    
    return 0
}

check_local_model() {
    local model_path="${MODEL_PATH:-${MODEL_NAME:-}}"
    
    info "Checking local model..."
    
    if [[ -z "$model_path" ]]; then
        error "MODEL_PATH or MODEL_NAME not specified"
        return 1
    fi
    
    if [[ ! -d "$model_path" ]]; then
        error "Model directory not found: $model_path"
        error "Please download model weights to this location first"
        return 1
    fi
    
    # Check for required files
    if [[ ! -f "$model_path/config.json" ]]; then
        error "Missing config.json in model directory"
        return 1
    fi
    
    # Check for tokenizer
    local has_tokenizer=false
    for f in "tokenizer.json" "tokenizer_config.json" "tokenizer.model"; do
        if [[ -f "$model_path/$f" ]]; then
            has_tokenizer=true
            break
        fi
    done
    
    if [[ "$has_tokenizer" == "false" ]]; then
        warn "No tokenizer file found in model directory"
    fi
    
    # Check for weights
    local has_weights=false
    for pattern in "*.safetensors" "*.bin" "*.pt"; do
        if compgen -G "$model_path/$pattern" >/dev/null 2>&1; then
            has_weights=true
            break
        fi
    done
    
    if [[ "$has_weights" == "false" ]]; then
        error "No model weight files found in: $model_path"
        return 1
    fi
    
    # Calculate approximate model size
    local model_size
    model_size=$(du -sh "$model_path" 2>/dev/null | cut -f1)
    info "  Model directory: $model_path"
    info "  Model size: ${model_size:-unknown}"
    
    return 0
}

check_port_availability() {
    info "Checking port availability..."
    
    local ports_to_check=(
        "${VLLM_SERVER_PORT:-8000}"
        "${RAG_PORT:-8080}"
        "${PROXY_PORT:-8081}"
        "${CHROMA_PORT:-8001}"
    )
    
    local conflicts=0
    for port in "${ports_to_check[@]}"; do
        if ss -tuln 2>/dev/null | grep -q ":$port "; then
            warn "Port $port is already in use"
            ((conflicts++))
        else
            debug "  Port $port: available"
        fi
    done
    
    if (( conflicts > 0 )); then
        warn "$conflicts port(s) already in use - will attempt to find alternatives"
    else
        info "  All default ports available"
    fi
    
    return 0
}

# ============================================================================
# Environment Validation
# ============================================================================

validate_environment() {
    info "Validating environment configuration..."
    
    local required_vars=()
    local warnings=()
    
    # Check RUNMODE
    case "${RUNMODE:-}" in
        docker|singularity) ;;
        "") required_vars+=("RUNMODE") ;;
        *) warnings+=("RUNMODE has unexpected value: $RUNMODE") ;;
    esac
    
    # Check RUNTYPE
    case "${RUNTYPE:-}" in
        all|vllm) ;;
        "") required_vars+=("RUNTYPE") ;;
        *) warnings+=("RUNTYPE has unexpected value: $RUNTYPE") ;;
    esac
    
    # Check model configuration
    if [[ -z "${MODEL_NAME:-}" && -z "${MODEL_PATH:-}" ]]; then
        required_vars+=("MODEL_NAME or MODEL_PATH")
    fi
    
    # Report issues
    if (( ${#required_vars[@]} > 0 )); then
        error "Missing required environment variables:"
        for var in "${required_vars[@]}"; do
            error "  - $var"
        done
        return 1
    fi
    
    if (( ${#warnings[@]} > 0 )); then
        for w in "${warnings[@]}"; do
            warn "$w"
        done
    fi
    
    info "Environment validation passed"
    return 0
}
