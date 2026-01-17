#!/bin/bash
# lib/model_manager.sh - Model download and validation for ACTIVATE RAG-vLLM
# Source this file: source "$(dirname "$0")/lib/model_manager.sh"

# Requires lib/functions.sh to be sourced first

# ============================================================================
# Configuration
# ============================================================================

MODEL_CACHE_BASE="${MODEL_CACHE_BASE:-$HOME/pw/models}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# ============================================================================
# Model Setup Entry Point
# ============================================================================

setup_model() {
    local source="$1"      # local | huggingface
    local model_id="$2"    # path or HF model ID
    local hf_token="${3:-}"  # optional HF token
    
    info "Setting up model: source=$source, model_id=$model_id"
    
    case "$source" in
        local)
            validate_local_model "$model_id"
            MODEL_PATH="$model_id"
            ;;
        huggingface)
            download_hf_model "$model_id" "$hf_token"
            # Model path is set inside download_hf_model
            ;;
        *)
            error "Unknown model source: $source"
            return 1
            ;;
    esac
    
    export MODEL_PATH
    export MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
    
    info "Model path set to: $MODEL_PATH"
    return 0
}

# ============================================================================
# Local Model Validation
# ============================================================================

validate_local_model() {
    local path="$1"
    
    if [[ -z "$path" ]]; then
        error "Model path not specified"
        return 1
    fi
    
    if [[ ! -d "$path" ]]; then
        error "Model directory not found: $path"
        error "Please ensure the model weights are downloaded to this location"
        return 1
    fi
    
    # Check for required model files
    local required_files=("config.json")
    local optional_files=("tokenizer.json" "tokenizer_config.json" "tokenizer.model")
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$path/$file" ]]; then
            error "Missing required file: $path/$file"
            return 1
        fi
    done
    
    # Check for at least one tokenizer file
    local has_tokenizer=false
    for file in "${optional_files[@]}"; do
        if [[ -f "$path/$file" ]]; then
            has_tokenizer=true
            break
        fi
    done
    
    if [[ "$has_tokenizer" == "false" ]]; then
        warn "No tokenizer file found in $path (checked: ${optional_files[*]})"
    fi
    
    # Check for model weights
    local has_weights=false
    for pattern in "*.safetensors" "*.bin" "*.pt" "pytorch_model*.bin"; do
        if compgen -G "$path/$pattern" >/dev/null 2>&1; then
            has_weights=true
            break
        fi
    done
    
    if [[ "$has_weights" == "false" ]]; then
        warn "No model weight files found (*.safetensors, *.bin, *.pt)"
    fi
    
    info "Local model validated: $path"
    return 0
}

# ============================================================================
# HuggingFace Model Download
# ============================================================================

download_hf_model() {
    local model_id="$1"
    local hf_token="${2:-}"
    
    if [[ -z "$model_id" ]]; then
        error "HuggingFace model ID not specified"
        return 1
    fi
    
    # Sanitize model ID for directory path (replace / with --)
    local safe_model_id="${model_id//\//__}"
    local target_dir="$MODEL_CACHE_BASE/$safe_model_id"
    
    # Check if model is already cached and complete
    if [[ -d "$target_dir" ]] && model_is_complete "$target_dir"; then
        info "Model already cached: $target_dir"
        MODEL_PATH="$target_dir"
        return 0
    fi
    
    info "Downloading model from HuggingFace: $model_id"
    mkdir -p "$target_dir"
    
    # Try git-lfs first (preferred for HPC environments)
    if download_with_git_lfs "$model_id" "$target_dir" "$hf_token"; then
        MODEL_PATH="$target_dir"
        return 0
    fi
    
    # Fallback to huggingface-cli if available
    if command -v huggingface-cli >/dev/null 2>&1; then
        warn "Git LFS download failed, trying huggingface-cli..."
        if download_with_hf_cli "$model_id" "$target_dir" "$hf_token"; then
            MODEL_PATH="$target_dir"
            return 0
        fi
    fi
    
    error "Failed to download model: $model_id"
    return 1
}

download_with_git_lfs() {
    local model_id="$1"
    local target_dir="$2"
    local hf_token="${3:-}"
    
    # Check for git and git-lfs
    if ! command -v git >/dev/null 2>&1; then
        warn "git not found, cannot use git-lfs download method"
        return 1
    fi
    
    # Check if git-lfs is installed
    if ! git lfs version >/dev/null 2>&1; then
        warn "git-lfs not installed, attempting to install..."
        if ! install_git_lfs; then
            return 1
        fi
    fi
    
    local repo_url="https://huggingface.co/$model_id"
    
    # Add authentication if token provided
    if [[ -n "$hf_token" ]]; then
        repo_url="https://user:${hf_token}@huggingface.co/$model_id"
        debug "Using authenticated HuggingFace URL"
    fi
    
    info "Cloning model repository via git-lfs..."
    
    # Clone with LFS enabled
    if git clone --depth 1 "$repo_url" "$target_dir" 2>&1; then
        info "Model downloaded successfully"
        
        # Verify download
        if model_is_complete "$target_dir"; then
            return 0
        else
            error "Model download incomplete - missing required files"
            return 1
        fi
    else
        error "git clone failed"
        return 1
    fi
}

download_with_hf_cli() {
    local model_id="$1"
    local target_dir="$2"
    local hf_token="${3:-}"
    
    info "Downloading model via huggingface-cli..."
    
    local hf_cmd="huggingface-cli download $model_id --local-dir $target_dir"
    
    if [[ -n "$hf_token" ]]; then
        hf_cmd="$hf_cmd --token $hf_token"
    fi
    
    if eval "$hf_cmd" 2>&1; then
        info "Model downloaded successfully via huggingface-cli"
        return 0
    else
        error "huggingface-cli download failed"
        return 1
    fi
}

install_git_lfs() {
    info "Attempting to install git-lfs..."
    
    # Try common installation methods
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y git-lfs
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y git-lfs
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y git-lfs
    elif command -v brew >/dev/null 2>&1; then
        brew install git-lfs
    else
        error "Cannot install git-lfs: no supported package manager found"
        return 1
    fi
    
    # Initialize git-lfs
    git lfs install
    
    if git lfs version >/dev/null 2>&1; then
        info "git-lfs installed successfully"
        return 0
    else
        error "git-lfs installation failed"
        return 1
    fi
}

# ============================================================================
# Model Verification
# ============================================================================

model_is_complete() {
    local path="$1"
    
    # Must have config.json
    [[ -f "$path/config.json" ]] || return 1
    
    # Must have either tokenizer.json or tokenizer_config.json
    [[ -f "$path/tokenizer.json" || -f "$path/tokenizer_config.json" || -f "$path/tokenizer.model" ]] || return 1
    
    # Must have at least one weight file
    local has_weights=false
    for pattern in "*.safetensors" "*.bin" "*.pt"; do
        if compgen -G "$path/$pattern" >/dev/null 2>&1; then
            has_weights=true
            break
        fi
    done
    
    [[ "$has_weights" == "true" ]] || return 1
    
    return 0
}

get_model_info() {
    local path="$1"
    
    if [[ ! -f "$path/config.json" ]]; then
        echo "Model info unavailable"
        return 1
    fi
    
    # Extract basic model info from config.json
    local model_type arch hidden_size num_layers
    model_type=$(grep -o '"model_type"[[:space:]]*:[[:space:]]*"[^"]*"' "$path/config.json" | cut -d'"' -f4)
    arch=$(grep -o '"architectures"[[:space:]]*:[[:space:]]*\["[^"]*"' "$path/config.json" | cut -d'"' -f4)
    hidden_size=$(grep -o '"hidden_size"[[:space:]]*:[[:space:]]*[0-9]*' "$path/config.json" | grep -o '[0-9]*$')
    num_layers=$(grep -o '"num_hidden_layers"[[:space:]]*:[[:space:]]*[0-9]*' "$path/config.json" | grep -o '[0-9]*$')
    
    echo "Model Type: ${model_type:-unknown}"
    echo "Architecture: ${arch:-unknown}"
    echo "Hidden Size: ${hidden_size:-unknown}"
    echo "Layers: ${num_layers:-unknown}"
}

# ============================================================================
# Tiktoken Encodings (for offline mode)
# ============================================================================

download_tiktoken_encodings() {
    local cache_dir="${1:-$HF_CACHE/tiktoken}"
    
    info "Downloading tiktoken encodings for offline use..."
    
    mkdir -p "$cache_dir"
    
    local encodings=(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    )
    
    for url in "${encodings[@]}"; do
        local filename
        filename=$(basename "$url")
        local target="$cache_dir/$filename"
        
        if [[ -f "$target" ]]; then
            debug "Encoding already cached: $filename"
            continue
        fi
        
        info "Downloading: $filename"
        if curl -sL "$url" -o "$target"; then
            debug "Downloaded: $target"
        else
            warn "Failed to download: $url"
        fi
    done
    
    # Set environment variable for tiktoken
    export TIKTOKEN_CACHE_DIR="$cache_dir"
    info "Tiktoken encodings cached in: $cache_dir"
}
