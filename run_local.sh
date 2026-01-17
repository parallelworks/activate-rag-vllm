#!/bin/bash
# ==============================================================================
# Local Development Runner for activate-rag-vllm
# ==============================================================================
# This script allows running the entire workflow locally for debugging.
# It mimics what the ParallelWorks workflow does without needing PW infrastructure.
#
# Usage:
#   ./run_local.sh                    # Interactive mode with prompts
#   ./run_local.sh --config local.env # Use config file
#   ./run_local.sh --singularity      # Force Singularity mode
#   ./run_local.sh --docker           # Force Docker mode
#   ./run_local.sh --vllm-only        # Run vLLM without RAG
#   ./run_local.sh --help             # Show help
#
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# Default Configuration
# ==============================================================================
RUNMODE="${RUNMODE:-auto}"          # auto, singularity, docker
RUNTYPE="${RUNTYPE:-all}"           # all (vLLM+RAG) or vllm
BUILD="${BUILD:-false}"             # Build containers from source
MODEL_SOURCE="${MODEL_SOURCE:-local}"  # local or huggingface
MODEL_PATH="${MODEL_PATH:-}"        # Path to local model weights
HF_MODEL_ID="${HF_MODEL_ID:-}"      # HuggingFace model ID
HF_TOKEN="${HF_TOKEN:-}"            # HuggingFace token
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/activate-models}"
DOCS_DIR="${DOCS_DIR:-./docs}"
API_KEY="${API_KEY:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---dtype bfloat16 --trust_remote_code}"

# Port configuration (0 = auto-assign)
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-0}"
PROXY_PORT="${PROXY_PORT:-0}"
RAG_PORT="${RAG_PORT:-0}"
CHROMA_PORT="${CHROMA_PORT:-0}"

# ==============================================================================
# Color Output
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_debug() { [[ "${DEBUG:-0}" == "1" ]] && echo -e "${BLUE}[DEBUG]${NC} $*" || true; }

# ==============================================================================
# Help
# ==============================================================================
show_help() {
    cat << 'EOF'
Local Development Runner for activate-rag-vllm

USAGE:
    ./run_local.sh [OPTIONS]

OPTIONS:
    --config FILE       Load configuration from file
    --singularity       Force Singularity runtime
    --docker            Force Docker runtime
    --vllm-only         Run vLLM only (no RAG)
    --build             Build containers from source
    --model PATH        Path to local model weights
    --hf-model ID       HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)
    --hf-token TOKEN    HuggingFace API token
    --docs-dir DIR      Directory for RAG documents
    --api-key KEY       API key for vLLM server
    --debug             Enable debug output
    --dry-run           Show what would be done without executing
    --help              Show this help message

EXAMPLES:
    # Run with local model
    ./run_local.sh --model /models/Llama-3.1-8B-Instruct

    # Run with HuggingFace model (will download if not cached)
    ./run_local.sh --hf-model meta-llama/Llama-3.1-8B-Instruct --hf-token $HF_TOKEN

    # Run vLLM only with Docker
    ./run_local.sh --docker --vllm-only --model /models/Llama-3.1-8B-Instruct

    # Use configuration file
    ./run_local.sh --config my-config.env

CONFIGURATION FILE FORMAT (my-config.env):
    MODEL_SOURCE=local
    MODEL_PATH=/models/Llama-3.1-8B-Instruct
    RUNMODE=singularity
    RUNTYPE=all
    VLLM_EXTRA_ARGS="--dtype bfloat16 --tensor-parallel-size 4"

EOF
    exit 0
}

# ==============================================================================
# Parse Arguments
# ==============================================================================
DRY_RUN=0
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --singularity)
            RUNMODE="singularity"
            shift
            ;;
        --docker)
            RUNMODE="docker"
            shift
            ;;
        --vllm-only)
            RUNTYPE="vllm"
            shift
            ;;
        --build)
            BUILD="true"
            shift
            ;;
        --model)
            MODEL_SOURCE="local"
            MODEL_PATH="$2"
            shift 2
            ;;
        --hf-model)
            MODEL_SOURCE="huggingface"
            HF_MODEL_ID="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --docs-dir)
            DOCS_DIR="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --debug)
            DEBUG=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Load Configuration File
# ==============================================================================
if [[ -n "$CONFIG_FILE" ]]; then
    if [[ -f "$CONFIG_FILE" ]]; then
        log_info "Loading configuration from $CONFIG_FILE"
        source "$CONFIG_FILE"
    else
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
fi

# ==============================================================================
# Detect Container Runtime
# ==============================================================================
detect_runtime() {
    if [[ "$RUNMODE" != "auto" ]]; then
        log_debug "Using specified runtime: $RUNMODE"
        return
    fi

    # Prefer Singularity if available (typical for HPC)
    if command -v singularity &>/dev/null; then
        RUNMODE="singularity"
        log_info "Detected Singularity runtime"
    elif command -v docker &>/dev/null; then
        RUNMODE="docker"
        log_info "Detected Docker runtime"
    else
        log_error "No container runtime found. Please install Docker or Singularity."
        exit 1
    fi
}

# ==============================================================================
# Find Available Port
# ==============================================================================
find_available_port() {
    local start_port=${1:-8000}
    local port=$start_port
    
    while [[ $port -lt 65535 ]]; do
        if ! ss -tuln 2>/dev/null | grep -q ":$port " && \
           ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
            echo $port
            return 0
        fi
        ((port++))
    done
    
    log_error "No available ports found starting from $start_port"
    return 1
}

# ==============================================================================
# Setup Ports
# ==============================================================================
setup_ports() {
    log_info "Configuring service ports..."
    
    [[ "$VLLM_SERVER_PORT" == "0" ]] && VLLM_SERVER_PORT=$(find_available_port 8000)
    [[ "$PROXY_PORT" == "0" ]] && PROXY_PORT=$(find_available_port $((VLLM_SERVER_PORT + 1)))
    [[ "$RAG_PORT" == "0" ]] && RAG_PORT=$(find_available_port $((PROXY_PORT + 1)))
    [[ "$CHROMA_PORT" == "0" ]] && CHROMA_PORT=$(find_available_port $((RAG_PORT + 1)))
    
    log_info "  vLLM Server: $VLLM_SERVER_PORT"
    log_info "  Proxy Port:  $PROXY_PORT"
    log_info "  RAG Port:    $RAG_PORT"
    log_info "  Chroma Port: $CHROMA_PORT"
}

# ==============================================================================
# Validate Model Configuration
# ==============================================================================
validate_model() {
    if [[ "$MODEL_SOURCE" == "local" ]]; then
        if [[ -z "$MODEL_PATH" ]]; then
            log_error "Local model path not specified. Use --model PATH"
            exit 1
        fi
        if [[ ! -d "$MODEL_PATH" ]]; then
            log_error "Model path does not exist: $MODEL_PATH"
            exit 1
        fi
        if [[ ! -f "$MODEL_PATH/config.json" ]]; then
            log_warn "config.json not found in model directory - model may be incomplete"
        fi
        MODEL_NAME="$MODEL_PATH"
        log_info "Using local model: $MODEL_PATH"
    elif [[ "$MODEL_SOURCE" == "huggingface" ]]; then
        if [[ -z "$HF_MODEL_ID" ]]; then
            log_error "HuggingFace model ID not specified. Use --hf-model ID"
            exit 1
        fi
        MODEL_NAME="$HF_MODEL_ID"
        log_info "Using HuggingFace model: $HF_MODEL_ID"
        
        # Check if model is already cached
        SAFE_MODEL_ID="${HF_MODEL_ID//\//__}"
        CACHED_PATH="$MODEL_CACHE_DIR/$SAFE_MODEL_ID"
        
        if [[ -d "$CACHED_PATH" && -f "$CACHED_PATH/config.json" ]]; then
            log_info "Model already cached at: $CACHED_PATH"
            MODEL_PATH="$CACHED_PATH"
        else
            download_hf_model
        fi
    else
        log_error "Invalid MODEL_SOURCE: $MODEL_SOURCE (must be 'local' or 'huggingface')"
        exit 1
    fi
}

# ==============================================================================
# Download HuggingFace Model
# ==============================================================================
download_hf_model() {
    log_info "Downloading model from HuggingFace: $HF_MODEL_ID"
    
    mkdir -p "$MODEL_CACHE_DIR"
    
    SAFE_MODEL_ID="${HF_MODEL_ID//\//__}"
    TARGET_DIR="$MODEL_CACHE_DIR/$SAFE_MODEL_ID"
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_info "[DRY RUN] Would download $HF_MODEL_ID to $TARGET_DIR"
        MODEL_PATH="$TARGET_DIR"
        return
    fi
    
    # Prefer git-lfs for HPC compatibility
    if command -v git-lfs &>/dev/null; then
        log_info "Using git-lfs for download..."
        
        if [[ -n "$HF_TOKEN" ]]; then
            REPO_URL="https://user:${HF_TOKEN}@huggingface.co/$HF_MODEL_ID"
        else
            REPO_URL="https://huggingface.co/$HF_MODEL_ID"
        fi
        
        GIT_LFS_SKIP_SMUDGE=0 git clone "$REPO_URL" "$TARGET_DIR"
        
    elif command -v huggingface-cli &>/dev/null; then
        log_info "Using huggingface-cli for download..."
        
        HF_CMD="huggingface-cli download $HF_MODEL_ID --local-dir $TARGET_DIR"
        [[ -n "$HF_TOKEN" ]] && HF_CMD="$HF_CMD --token $HF_TOKEN"
        
        eval $HF_CMD
    else
        log_error "Neither git-lfs nor huggingface-cli available for model download"
        log_info "Install with: pip install huggingface_hub"
        exit 1
    fi
    
    MODEL_PATH="$TARGET_DIR"
    log_info "Model downloaded to: $MODEL_PATH"
}

# ==============================================================================
# Install Singularity Compose
# ==============================================================================
install_singularity_compose() {
    if command -v singularity-compose &>/dev/null; then
        return 0
    fi
    
    # Check common install locations
    if [[ -f ~/pw/software/singularity-compose/bin/activate ]]; then
        source ~/pw/software/singularity-compose/bin/activate
        return 0
    fi
    
    log_info "Installing singularity-compose..."
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_info "[DRY RUN] Would install singularity-compose"
        return 0
    fi
    
    mkdir -p ~/pw/software
    python3 -m venv ~/pw/software/singularity-compose
    source ~/pw/software/singularity-compose/bin/activate
    pip install --upgrade pip
    pip install singularity-compose
    
    log_info "singularity-compose installed successfully"
}

# ==============================================================================
# Check Container Images
# ==============================================================================
check_containers() {
    log_info "Checking container images..."
    
    if [[ "$RUNMODE" == "singularity" ]]; then
        if [[ ! -f "vllm.sif" ]]; then
            log_warn "vllm.sif not found in current directory"
            log_info "You can build it with: sudo singularity build vllm.sif singularity/Singularity.vllm"
            log_info "Or download from your PW bucket"
            
            if [[ $DRY_RUN -eq 0 ]]; then
                read -p "Continue anyway? [y/N] " -n 1 -r
                echo
                [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
            fi
        fi
        
        if [[ "$RUNTYPE" == "all" && ! -f "rag.sif" ]]; then
            log_warn "rag.sif not found in current directory"
            log_info "You can build it with: sudo singularity build rag.sif singularity/Singularity.rag"
        fi
    fi
}

# ==============================================================================
# Create Environment File
# ==============================================================================
create_env_file() {
    log_info "Creating environment configuration..."
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_info "[DRY RUN] Would create .run.env"
        return 0
    fi
    
    cat > .run.env << EOF
# Generated by run_local.sh on $(date)
export RUNMODE=$RUNMODE
export BUILD=$BUILD
export RUNTYPE=$RUNTYPE
export MODEL_NAME=$MODEL_NAME
export MODEL_PATH=$MODEL_PATH
export MODEL_SOURCE=$MODEL_SOURCE
export HF_TOKEN=$HF_TOKEN
export API_KEY=$API_KEY
export DOCS_DIR=$DOCS_DIR
export VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS"
export VLLM_SERVER_PORT=$VLLM_SERVER_PORT
export PROXY_PORT=$PROXY_PORT
export RAG_PORT=$RAG_PORT
export CHROMA_PORT=$CHROMA_PORT
EOF
    
    # Add offline mode for local models
    if [[ "$MODEL_SOURCE" == "local" ]]; then
        echo "export TRANSFORMERS_OFFLINE=1" >> .run.env
    fi
    
    log_debug "Environment file created: .run.env"
}

# ==============================================================================
# Create Directories
# ==============================================================================
create_directories() {
    log_info "Creating required directories..."
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_info "[DRY RUN] Would create directories"
        return 0
    fi
    
    mkdir -p logs cache cache/chroma tmp
    mkdir -p tmp/cuda_cache tmp/torch_extensions tmp/flashinfer_jit
    mkdir -p cache/sagemaker_sessions
    chmod 700 cache/sagemaker_sessions
    
    if [[ -d /dev/shm ]]; then
        mkdir -p /dev/shm/sagemaker_sessions 2>/dev/null || true
        chmod 700 /dev/shm/sagemaker_sessions 2>/dev/null || true
    fi
    
    # Create docs directory for RAG
    if [[ "$RUNTYPE" == "all" ]]; then
        mkdir -p "$DOCS_DIR"
    fi
}

# ==============================================================================
# Prepare Singularity
# ==============================================================================
prepare_singularity() {
    log_info "Preparing Singularity environment..."
    
    install_singularity_compose
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_info "[DRY RUN] Would copy Singularity configs"
        return 0
    fi
    
    # Copy singularity configs
    cp singularity/singularity-compose.yml ./
    cp singularity/env.sh.example ./env.sh
    
    # Update env.sh with our configuration
    sed -i "s/^export VLLM_SERVER_PORT=.*/export VLLM_SERVER_PORT=${VLLM_SERVER_PORT}/" env.sh
    sed -i "s/^export RAG_PORT=.*/export RAG_PORT=${RAG_PORT}/" env.sh
    sed -i "s/^export PROXY_PORT=.*/export PROXY_PORT=${PROXY_PORT}/" env.sh
    sed -i "s/^export CHROMA_PORT=.*/export CHROMA_PORT=${CHROMA_PORT}/" env.sh
    sed -i "s/\(.*HF_TOKEN=\"\)[^\"]*\(\".*\)/\1$HF_TOKEN\2/" env.sh
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|export MODEL_NAME=\"$MODEL_NAME\"|" env.sh
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?DOCS_DIR=.*|export DOCS_DIR=\"$DOCS_DIR\"|" env.sh
    sed -i "s|__VLLM_EXTRA_ARGS__|${VLLM_EXTRA_ARGS}|" env.sh
    
    # Update singularity-compose.yml with model paths
    MODEL_BASE=$(basename "$MODEL_PATH")
    sed -i "s|__MODEL_PATH__|${MODEL_PATH}|g" singularity-compose.yml
    sed -i "s|__MODEL_BASE__|${MODEL_BASE}|g" singularity-compose.yml
    
    # Handle docs directory symlink if needed
    if [[ "$DOCS_DIR" != "./docs" && "$RUNTYPE" == "all" ]]; then
        ln -sf "$DOCS_DIR" ./docs 2>/dev/null || true
    fi
    
    log_info "Singularity environment prepared"
}

# ==============================================================================
# Prepare Docker
# ==============================================================================
prepare_docker() {
    log_info "Preparing Docker environment..."
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_info "[DRY RUN] Would copy Docker configs"
        return 0
    fi
    
    # Copy docker configs
    cp docker/docker-compose.yml ./
    cp docker/env.example ./.env
    
    # Update .env with our configuration
    sed -i "s/^VLLM_SERVER_PORT=.*/VLLM_SERVER_PORT=${VLLM_SERVER_PORT}/" .env
    sed -i "s/^PROXY_PORT=.*/PROXY_PORT=${PROXY_PORT}/" .env
    sed -i "s/^[#[:space:]]*HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|MODEL_NAME=$MODEL_NAME|" .env
    sed -i "s|__VLLM_EXTRA_ARGS__|${VLLM_EXTRA_ARGS}|" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?DOCS_DIR=.*|DOCS_DIR=$DOCS_DIR|" .env
    
    if [[ -n "$API_KEY" ]]; then
        echo "" >> .env
        echo "VLLM_API_KEY=$API_KEY" >> .env
    fi
    
    log_info "Docker environment prepared"
}

# ==============================================================================
# Start Services
# ==============================================================================
start_services() {
    log_info "Starting services..."
    
    # Create cancel script
    echo '#!/bin/bash' > cancel.sh
    chmod +x cancel.sh
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_info "[DRY RUN] Would start services with:"
        log_info "  Runtime: $RUNMODE"
        log_info "  Type: $RUNTYPE"
        log_info "  Model: $MODEL_NAME"
        return 0
    fi
    
    if [[ "$RUNMODE" == "singularity" ]]; then
        start_singularity
    else
        start_docker
    fi
}

start_singularity() {
    source env.sh
    
    # Create cache directories
    mkdir -p ${CUDA_CACHE_PATH:-tmp/cuda_cache} ${TORCH_EXTENSIONS_DIR:-tmp/torch_extensions} ${FLASHINFER_JIT_DIR:-tmp/flashinfer_jit}
    chmod -R 777 ${TMPDIR:-tmp}
    
    echo "singularity-compose down" >> cancel.sh
    
    if [[ "$RUNTYPE" == "all" ]]; then
        [[ "$BUILD" == "true" ]] && singularity-compose build
        DOCS_DIR=$DOCS_DIR singularity-compose up
    else
        [[ "$BUILD" == "true" ]] && singularity-compose build vllm1
        singularity-compose up vllm1
    fi
    
    # Follow logs
    sleep 2
    if ls logs/* 1>/dev/null 2>&1; then
        tail -F logs/* &
        TAIL_PID=$!
        trap "kill $TAIL_PID 2>/dev/null; echo 'Cleaning up...'; ./cancel.sh" EXIT INT TERM
        wait $TAIL_PID
    else
        log_warn "No logs found, waiting for services..."
        wait
    fi
}

start_docker() {
    source .env
    
    # Determine docker compose command
    if docker compose version &>/dev/null; then
        docker_compose_cmd="docker compose"
    else
        log_error "docker compose not available"
        exit 1
    fi
    
    # Check if we need sudo
    if ! docker ps &>/dev/null; then
        if sudo -n docker ps &>/dev/null; then
            docker_compose_cmd="sudo $docker_compose_cmd"
        else
            log_error "Cannot access Docker. Try running with sudo or add user to docker group."
            exit 1
        fi
    fi
    
    # Create unique stack name
    stack_name=$(echo "ragvllm${PWD}" | tr '/' '-' | tr '.' '-' | tr '[:upper:]' '[:lower:]' | tail -c 50)
    docker_compose_cmd="$docker_compose_cmd -p $stack_name"
    
    echo "$docker_compose_cmd down" >> cancel.sh
    
    mkdir -p logs cache cache/chroma
    
    if [[ "$RUNTYPE" == "all" ]]; then
        [[ "$BUILD" == "true" ]] && $docker_compose_cmd build
        $docker_compose_cmd up -d --remove-orphans
    else
        [[ "$BUILD" == "true" ]] && $docker_compose_cmd build vllm
        $docker_compose_cmd up vllm -d --remove-orphans
    fi
    
    log_info "Following logs (Ctrl+C to stop)..."
    trap "echo 'Stopping...'; ./cancel.sh" EXIT INT TERM
    $docker_compose_cmd logs -f
}

# ==============================================================================
# Print Summary
# ==============================================================================
print_summary() {
    echo ""
    echo "=============================================================================="
    echo -e "${GREEN}ACTIVATE RAG-vLLM Local Runner${NC}"
    echo "=============================================================================="
    echo "Runtime:      $RUNMODE"
    echo "Deploy Type:  $RUNTYPE"
    echo "Model Source: $MODEL_SOURCE"
    echo "Model:        $MODEL_NAME"
    echo ""
    echo "Service Endpoints:"
    echo "  vLLM API:   http://localhost:$VLLM_SERVER_PORT/v1"
    if [[ "$RUNTYPE" == "all" ]]; then
        echo "  RAG Proxy:  http://localhost:$PROXY_PORT/v1"
        echo "  RAG Server: http://localhost:$RAG_PORT"
        echo "  ChromaDB:   http://localhost:$CHROMA_PORT"
    fi
    echo ""
    echo "Test with:"
    echo "  curl http://localhost:$VLLM_SERVER_PORT/v1/models"
    if [[ "$RUNTYPE" == "all" ]]; then
        echo "  curl http://localhost:$PROXY_PORT/v1/models"
    fi
    echo "=============================================================================="
    echo ""
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    log_info "ACTIVATE RAG-vLLM Local Development Runner"
    log_info "Working directory: $SCRIPT_DIR"
    echo ""
    
    # Detect and validate environment
    detect_runtime
    setup_ports
    validate_model
    check_containers
    
    # Print configuration summary
    print_summary
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_info "[DRY RUN] Would execute the above configuration"
        exit 0
    fi
    
    # Prepare environment
    create_env_file
    create_directories
    
    if [[ "$RUNMODE" == "singularity" ]]; then
        prepare_singularity
    else
        prepare_docker
    fi
    
    # Start services
    start_services
}

main "$@"
