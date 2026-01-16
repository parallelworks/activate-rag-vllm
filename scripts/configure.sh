#!/bin/bash
# ==============================================================================
# ACTIVATE RAG-vLLM Configuration Wizard
# ==============================================================================
# Interactive configuration script for local development and testing.
# Run: ./scripts/configure.sh
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

print_header() {
    echo -e "${CYAN}"
    echo "============================================================"
    echo "       ACTIVATE RAG-vLLM Configuration Wizard"
    echo "============================================================"
    echo -e "${NC}"
}

print_section() {
    echo ""
    echo -e "${BLUE}━━━ $1 ━━━${NC}"
}

info() {
    echo -e "${GREEN}✓${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_section "Checking Prerequisites"
    
    local errors=0
    
    # Check for GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_count
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        info "NVIDIA GPUs detected: $gpu_count"
    else
        warn "nvidia-smi not found - GPU may not be available"
    fi
    
    # Check container runtime
    if command -v singularity >/dev/null 2>&1; then
        info "Singularity: $(singularity --version 2>/dev/null || echo 'available')"
        CONTAINER_RUNTIME="singularity"
    elif command -v docker >/dev/null 2>&1; then
        info "Docker: $(docker --version 2>/dev/null | head -1 || echo 'available')"
        CONTAINER_RUNTIME="docker"
    else
        error "No container runtime found (need Docker or Singularity)"
        ((errors++))
    fi
    
    # Check for git-lfs (optional but recommended)
    if command -v git >/dev/null 2>&1 && git lfs version >/dev/null 2>&1; then
        info "git-lfs: available"
    else
        warn "git-lfs not installed - HuggingFace downloads may be slower"
    fi
    
    if ((errors > 0)); then
        error "Prerequisites check failed"
        exit 1
    fi
}

# Model configuration
configure_model() {
    print_section "Model Configuration"
    
    echo ""
    echo "How will you provide the model?"
    echo "  1) Local Path - Use pre-downloaded model weights"
    echo "  2) HuggingFace Hub - Download from HuggingFace"
    echo ""
    
    read -p "Select option [1-2]: " model_choice
    
    case "$model_choice" in
        1)
            MODEL_SOURCE="local"
            echo ""
            read -p "Enter model path: " MODEL_PATH
            
            if [[ ! -d "$MODEL_PATH" ]]; then
                warn "Directory does not exist: $MODEL_PATH"
                read -p "Continue anyway? [y/N]: " continue_choice
                if [[ "$continue_choice" != "y" && "$continue_choice" != "Y" ]]; then
                    exit 1
                fi
            else
                if [[ -f "$MODEL_PATH/config.json" ]]; then
                    info "Model config found at $MODEL_PATH"
                else
                    warn "No config.json found - may not be a valid model directory"
                fi
            fi
            
            MODEL_NAME="$MODEL_PATH"
            ;;
        2)
            MODEL_SOURCE="huggingface"
            echo ""
            echo "Popular models:"
            echo "  - meta-llama/Llama-3.1-8B-Instruct (8B, recommended for testing)"
            echo "  - meta-llama/Llama-3.1-70B-Instruct (70B, requires 4+ GPUs)"
            echo "  - mistralai/Mistral-7B-Instruct-v0.3 (7B, fast)"
            echo ""
            
            read -p "Enter HuggingFace model ID: " HF_MODEL_ID
            MODEL_NAME="$HF_MODEL_ID"
            
            read -p "Enter model cache directory [~/.cache/activate-models]: " MODEL_CACHE
            MODEL_CACHE="${MODEL_CACHE:-$HOME/.cache/activate-models}"
            
            # HuggingFace token for gated models
            echo ""
            echo "Some models (like Llama) require a HuggingFace token."
            read -sp "Enter HuggingFace token (leave empty if not needed): " HF_TOKEN
            echo ""
            ;;
        *)
            error "Invalid selection"
            exit 1
            ;;
    esac
}

# Deployment configuration
configure_deployment() {
    print_section "Deployment Configuration"
    
    echo ""
    echo "What do you want to deploy?"
    echo "  1) vLLM + RAG (Full Stack) - Inference server with RAG capabilities"
    echo "  2) vLLM Only - Just the inference server"
    echo ""
    
    read -p "Select option [1-2]: " deploy_choice
    
    case "$deploy_choice" in
        1)
            RUNTYPE="all"
            info "Will deploy full vLLM + RAG stack"
            
            read -p "Enter documents directory for RAG [./docs]: " DOCS_DIR
            DOCS_DIR="${DOCS_DIR:-./docs}"
            ;;
        2)
            RUNTYPE="vllm"
            info "Will deploy vLLM only"
            DOCS_DIR="./docs"
            ;;
        *)
            error "Invalid selection"
            exit 1
            ;;
    esac
    
    echo ""
    echo "Container runtime: $CONTAINER_RUNTIME"
    RUNMODE="$CONTAINER_RUNTIME"
}

# vLLM configuration
configure_vllm() {
    print_section "vLLM Configuration"
    
    echo ""
    echo "Configure vLLM server options"
    echo ""
    
    # Tensor parallel size
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        GPU_COUNT=1
    fi
    
    read -p "Tensor parallel size (GPUs to use) [${GPU_COUNT}]: " TP_SIZE
    TP_SIZE="${TP_SIZE:-$GPU_COUNT}"
    
    # Data type
    echo ""
    echo "Data type options:"
    echo "  1) bfloat16 (recommended for A100/H100)"
    echo "  2) float16 (better compatibility)"
    echo "  3) auto (let vLLM decide)"
    echo ""
    
    read -p "Select data type [1]: " dtype_choice
    case "${dtype_choice:-1}" in
        1) DTYPE="bfloat16" ;;
        2) DTYPE="float16" ;;
        3) DTYPE="auto" ;;
        *) DTYPE="bfloat16" ;;
    esac
    
    # Max model length
    read -p "Max model length (context size) [8192]: " MAX_MODEL_LEN
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    
    # GPU memory utilization
    read -p "GPU memory utilization (0.5-0.95) [0.85]: " GPU_MEM
    GPU_MEM="${GPU_MEM:-0.85}"
    
    # Build vLLM extra args
    VLLM_EXTRA_ARGS="--dtype ${DTYPE} --tensor-parallel-size ${TP_SIZE} --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization ${GPU_MEM} --trust_remote_code"
    
    # API key
    echo ""
    read -p "Set vLLM API key? (required for Cline/code assist) [y/N]: " set_api_key
    if [[ "$set_api_key" == "y" || "$set_api_key" == "Y" ]]; then
        read -sp "Enter API key: " API_KEY
        echo ""
    else
        API_KEY=""
    fi
}

# Generate configuration
generate_config() {
    print_section "Generating Configuration"
    
    # Create env.sh for singularity or .env for docker
    if [[ "$RUNMODE" == "singularity" ]]; then
        CONFIG_FILE="$ROOT_DIR/singularity/env.sh"
        cp "$ROOT_DIR/singularity/env.sh.example" "$CONFIG_FILE"
    else
        CONFIG_FILE="$ROOT_DIR/docker/.env"
        cp "$ROOT_DIR/docker/env.example" "$CONFIG_FILE" 2>/dev/null || touch "$CONFIG_FILE"
    fi
    
    # Also create .run.env for the service script
    RUN_ENV="$ROOT_DIR/.run.env"
    
    cat > "$RUN_ENV" << EOF
# Generated by configure.sh on $(date)
export RUNMODE=${RUNMODE}
export BUILD=false
export RUNTYPE=${RUNTYPE}
export MODEL_SOURCE=${MODEL_SOURCE}
export MODEL_NAME=${MODEL_NAME}
export MODEL_PATH=${MODEL_PATH:-$MODEL_NAME}
export DOCS_DIR=${DOCS_DIR}
export HF_TOKEN=${HF_TOKEN:-}
export API_KEY=${API_KEY:-}
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS}"
EOF

    if [[ "$MODEL_SOURCE" == "huggingface" ]]; then
        echo "export MODEL_CACHE_BASE=${MODEL_CACHE}" >> "$RUN_ENV"
        echo "export HF_MODEL_ID=${HF_MODEL_ID}" >> "$RUN_ENV"
    else
        echo "export TRANSFORMERS_OFFLINE=1" >> "$RUN_ENV"
    fi
    
    info "Configuration saved to: $RUN_ENV"
}

# Summary and next steps
print_summary() {
    print_section "Configuration Summary"
    
    echo ""
    echo "  Container Runtime: $RUNMODE"
    echo "  Deployment Type:   $RUNTYPE"
    echo "  Model Source:      $MODEL_SOURCE"
    echo "  Model:             $MODEL_NAME"
    if [[ "$RUNTYPE" == "all" ]]; then
        echo "  Documents Dir:     $DOCS_DIR"
    fi
    echo "  Tensor Parallel:   $TP_SIZE"
    echo "  Data Type:         $DTYPE"
    echo "  Max Context:       $MAX_MODEL_LEN"
    
    print_section "Next Steps"
    
    echo ""
    if [[ "$MODEL_SOURCE" == "huggingface" && ! -d "$MODEL_CACHE/${HF_MODEL_ID//\//__}" ]]; then
        echo "  1. Download the model (will happen on first run):"
        echo "     The model will be cached at: $MODEL_CACHE"
        echo ""
    fi
    
    echo "  To start the service:"
    echo -e "     ${GREEN}cd $ROOT_DIR && ./start_service.sh${NC}"
    echo ""
    
    if [[ "$RUNTYPE" == "all" ]]; then
        echo "  To add documents for RAG:"
        echo "     cp your_documents.pdf $DOCS_DIR/"
        echo ""
    fi
    
    echo "  To clean up and stop services:"
    echo "     ./clean.sh"
    echo ""
}

# Main
main() {
    print_header
    
    cd "$ROOT_DIR"
    
    check_prerequisites
    configure_model
    configure_deployment
    configure_vllm
    generate_config
    print_summary
    
    echo -e "${GREEN}Configuration complete!${NC}"
}

main "$@"
