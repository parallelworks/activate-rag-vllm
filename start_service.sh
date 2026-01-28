#!/bin/bash
set -x
# Note: job.started and HOSTNAME markers are injected by script_submitter v4.0
# when inject_markers=true (default)
touch job.started
hostname | cut -d'.' -f1 > HOSTNAME

source .run.env > /dev/null 2>&1
#rm .run.env > /dev/null 2>&1

# set these if running manually
export RUNMODE=${RUNMODE:-docker} # docker or singularity
export BUILD=${BUILD:-false} # true or false
export RUNTYPE=${RUNTYPE:-all} # all or vllm
export MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
export DOCS_DIR=${DOCS_DIR:-./docs}
export API_KEY=${API_KEY:-undefined}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

echo ""
echo "Running workflow with the below inputs:"
echo "  RUNMODE=$RUNMODE"
echo "  BUILD=$BUILD"
echo "  RUNTYPE=$RUNTYPE"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  DOCS_DIR=$DOCS_DIR"
echo "  API_KEY=$API_KEY"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo ""

install_docker_compose(){
    echo "$(date) Downloading docker-compose v2.39.1"
    curl -L "https://github.com/docker/compose/releases/download/v2.39.1/docker-compose-$(uname -s)-$(uname -m)" -o docker-compose
    chmod +x docker-compose
}

start_rootless_docker() {
    local MAX_RETRIES=20
    local RETRY_INTERVAL=2
    local ATTEMPT=1

    export XDG_RUNTIME_DIR=/run/user/$(id -u)
    dockerd-rootless-setuptool.sh install

    # Run Docker rootless daemon — use screen if available, otherwise run in background
    if command -v screen >/dev/null 2>&1; then
        echo "$(date): Starting Docker rootless daemon in a screen session..."
        screen -dmS docker-rootless bash -c "PATH=/usr/bin:/sbin:/usr/sbin:\$PATH dockerd-rootless.sh --exec-opt native.cgroupdriver=cgroupfs > ~/docker-rootless.log 2>&1"
    else
        echo "$(date): 'screen' not found, starting Docker rootless daemon in background..."
        PATH=/usr/bin:/sbin:/usr/sbin:$PATH dockerd-rootless.sh --exec-opt native.cgroupdriver=cgroupfs > ~/docker-rootless.log 2>&1 &
    fi

    # Wait for Docker daemon to be ready
    until docker info > /dev/null 2>&1; do
        if [ $ATTEMPT -le $MAX_RETRIES ]; then
            echo "$(date) Attempt $ATTEMPT of $MAX_RETRIES: Waiting for Docker daemon to start..."
            sleep $RETRY_INTERVAL
            ((ATTEMPT++))
        else
            echo "$(date) ERROR: Docker daemon failed to start after $MAX_RETRIES attempts."
            return 1
        fi
    done

    echo "$(date): Docker daemon is ready!"
    return 0
}

# Create cleanup script
echo '#!/bin/bash' > cancel.sh
chmod +x cancel.sh

if [ "$RUNMODE" == "docker" ];then

    # Ensure docker service is installed
    which docker >/dev/null 2>&1 || { 
        echo "$(date) ERROR: Docker is not installed."
        exit 1
    }

    # Ensure docker compose is installed and meets requirements
    major_version=$(docker compose version --short | cut -d'.' -f1)
    minor_version=$(docker compose version --short | cut -d'.' -f2)
    if [ -z "$major_version" ] || [ -z "$minor_version" ] || [ "$major_version" -lt 2 ] || { [ "$major_version" -eq 2 ] && [ "$minor_version" -lt 39 ]; }; then
        install_docker_compose
        docker_compose_cmd="./docker-compose"
    else
        docker_compose_cmd="docker compose"
    fi
    
    # Ensure docker service is started and set docker_compose_cmd
    docker ps >/dev/null 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date) User has docker access" 
    elif ! sudo -n true 2>/dev/null; then
        start_rootless_docker
    else
        if command -v nvidia-ctk >/dev/null 2>&1; then
            sudo systemctl start docker
        else
            # FIXME: This is not robust!
            sudo dnf install -y nvidia-container-toolkit
            sudo systemctl restart docker
        fi
        docker_compose_cmd="sudo ${docker_compose_cmd}"
    fi

    cp docker/* ./ -Rf
    cp env.example .env

    if [ "$RUNTYPE" == "all" ];then
        VLLM_SERVER_PORT=$(pw agent open-port)
        PROXY_PORT=${service_port}
        # TRANSITION CODE
        echo "${VLLM_SERVER_PORT}" > SESSION_PORT
    else
        PROXY_PORT=$(pw agent open-port)
        VLLM_SERVER_PORT=${service_port}
        echo "${VLLM_SERVER_PORT}" > SESSION_PORT
    fi
    
    sed -i "s/^VLLM_SERVER_PORT=.*/VLLM_SERVER_PORT=${VLLM_SERVER_PORT}/" .env
    sed -i "s/^PROXY_PORT=.*/PROXY_PORT=${PROXY_PORT}/" .env

    sed -i "s/^[#[:space:]]*HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|export MODEL_NAME=$MODEL_NAME|" .env
    sed -i "s|__VLLM_EXTRA_ARGS__|${VLLM_EXTRA_ARGS}|" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?DOCS_DIR=.*|export DOCS_DIR=$DOCS_DIR|" .env
    sed -i "s|^EMBEDDING_MODEL=.*|EMBEDDING_MODEL=${EMBEDDING_MODEL}|" .env
    [[ -n "$VLLM_ATTENTION_BACKEND" ]] && echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}" >> .env

    # Tiktoken encodings for offline tokenizer support (standard tiktoken + openai_harmony for GPT-OSS)
    if [[ -n "$TIKTOKEN_ENCODINGS_BASE" ]]; then
        echo "TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE}" >> .env
        echo "TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_ENCODINGS_BASE}" >> .env
    fi

    if [[ "$DOCS_DIR" != "undefined" ]]; then
        sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?DOCS_DIR=.*|DOCS_DIR=$DOCS_DIR|" .env
        mkdir -p $DOCS_DIR
    fi
    
    if [[ "$API_KEY" != "undefined" ]]; then
        echo "" >> .env
        echo "VLLM_API_KEY=$API_KEY" >> .env
    fi

    # Disable weight download
    # Check if cache/huggingface directory exists
    # TODO - only disable online if the actual weight doesn't exist because this fails when changing models
    # if [ -d "cache/huggingface" ]; then
    #     sed -i 's/#TRANSFORMERS_OFFLINE=1/TRANSFORMERS_OFFLINE=1/' .env
    #     sed -i '/HF_HOME: \/root\/.cache\/huggingface/a\      TRANSFORMERS_OFFLINE: 1' docker-compose.yml
    #     echo "$(date) Disabled model weight download"
    # fi

    source .env

    mkdir -p logs cache cache/chroma

    stack_name=$(echo "ragvllm${PWD}" | tr '/' '-' | tr '.' '-' | tr '[:upper:]' '[:lower:]')

    if [ ${#stack_name} -gt 50 ]; then
        stack_name=${stack_name: -50}
    fi
    docker_compose_cmd="${docker_compose_cmd} -p ${stack_name}"

    # Check if any containers are running in the project
    if [ "$(${docker_compose_cmd} ps -q)" ]; then
        echo "$(date) ERROR: Stack ${stack_name} is already running. Choose a different run directory or delete stack."
        exit 1
    fi

    echo "${docker_compose_cmd} down" >> cancel.sh
    if [ "$RUNTYPE" == "all" ];then
        if [ "$BUILD" = "true" ];then
            ${docker_compose_cmd} build
            else
            docker pull parallelworks/activate-rag-vllm:latest
        fi
        ${docker_compose_cmd} up -d --remove-orphans
    else
        [ "$BUILD" = "true" ] && ${docker_compose_cmd} build $RUNTYPE
        ${docker_compose_cmd} up $RUNTYPE -d --remove-orphans
    fi
    ${docker_compose_cmd} logs -f

elif [ "$RUNMODE" == "singularity" ]; then

    # Check if singularity is installed
    if ! command -v singularity >/dev/null 2>&1; then
        echo "$(date) ERROR: singularity is not installed"
        exit 1
    fi

    cp singularity/env.sh.example env.sh
    cp singularity/Singularity.* ./

    # Allocate ports for all services
    VLLM_SERVER_PORT=$(pw agent open-port)
    RAG_PORT=$(pw agent open-port)
    CHROMA_PORT=$(pw agent open-port)
    PROXY_PORT=$(pw agent open-port)

    # Set SESSION_PORT based on deployment type
    # - vLLM only: users connect directly to vLLM
    # - all (RAG): users connect to RAG proxy which forwards to vLLM
    if [ "$RUNTYPE" == "all" ]; then
        echo "${PROXY_PORT}" > SESSION_PORT 
    else
        echo "${VLLM_SERVER_PORT}" > SESSION_PORT 
    fi

    sed -i "s/^export VLLM_SERVER_PORT=.*/export VLLM_SERVER_PORT=${VLLM_SERVER_PORT}/" env.sh
    sed -i "s/^export RAG_PORT=.*/export RAG_PORT=${RAG_PORT}/" env.sh
    sed -i "s/^export PROXY_PORT=.*/export PROXY_PORT=${PROXY_PORT}/" env.sh
    sed -i "s/^export CHROMA_PORT=.*/export CHROMA_PORT=${CHROMA_PORT}/" env.sh

    sed -i "s/\(.*HF_TOKEN=\"\)[^\"]*\(\".*\)/\1$HF_TOKEN\2/" env.sh
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|export MODEL_NAME=$MODEL_NAME|" env.sh
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?DOCS_DIR=.*|export DOCS_DIR=$DOCS_DIR|" env.sh
    sed -i "s|__VLLM_EXTRA_ARGS__|${VLLM_EXTRA_ARGS}|" env.sh
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?EMBEDDING_MODEL=.*|export EMBEDDING_MODEL=$EMBEDDING_MODEL|" env.sh
    [[ -n "$VLLM_ATTENTION_BACKEND" ]] && sed -i "s|^export VLLM_ATTENTION_BACKEND=.*|export VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND|" env.sh

    # Tiktoken encodings for offline tokenizer support (standard tiktoken + openai_harmony for GPT-OSS)
    if [[ -n "$TIKTOKEN_ENCODINGS_BASE" ]]; then
        sed -i "s|^#*export TIKTOKEN_ENCODINGS_BASE=.*|export TIKTOKEN_ENCODINGS_BASE=$TIKTOKEN_ENCODINGS_BASE|" env.sh
        sed -i "s|^#*export TIKTOKEN_RS_CACHE_DIR=.*|export TIKTOKEN_RS_CACHE_DIR=$TIKTOKEN_ENCODINGS_BASE|" env.sh
    fi

    # Get model path and basename for bind mounts
    MODEL_PATH="${MODEL_NAME}"
    MODEL_BASE=$(basename "$MODEL_NAME")

    # Disable weight download if cache exists
    if [ -d "cache/huggingface" ]; then
        sed -i 's/#export TRANSFORMERS_OFFLINE=1/export TRANSFORMERS_OFFLINE=1/' env.sh
        echo "$(date) Disabled model weight download"
    fi

    source env.sh
    mkdir -p ${CUDA_CACHE_PATH} ${TORCH_EXTENSIONS_DIR} ${FLASHINFER_JIT_DIR}
    chmod -R 777 ${TMPDIR}

    mkdir -p logs cache cache/chroma cache/tiktoken_encodings cache/sagemaker_sessions
    chmod 700 cache/sagemaker_sessions
    
    mkdir -p /dev/shm/sagemaker_sessions 2>/dev/null || true
    chmod 700 /dev/shm/sagemaker_sessions 2>/dev/null || true

    # Create docs directory / symlink
    if [ "$DOCS_DIR" != "./docs" ] && [ -n "$DOCS_DIR" ]; then
        mkdir -p "$DOCS_DIR"
        ln -sf "$DOCS_DIR" ./docs
    else
        mkdir -p ./docs
    fi

    # Resolve container paths (from workflow input or default)
    VLLM_SIF="${VLLM_CONTAINER_PATH:-./vllm.sif}"
    VLLM_SIF="${VLLM_SIF/#\~/$HOME}"
    RAG_SIF="${RAG_CONTAINER_PATH:-./rag.sif}"
    RAG_SIF="${RAG_SIF/#\~/$HOME}"

    # Verify containers exist
    [[ ! -f "$VLLM_SIF" ]] && { echo "$(date) ERROR: vLLM container not found at $VLLM_SIF"; exit 1; }
    [[ "$RUNTYPE" == "all" && ! -f "$RAG_SIF" ]] && { echo "$(date) ERROR: RAG container not found at $RAG_SIF"; exit 1; }

    # Define common bind mounts
    COMMON_BINDS="--bind ./logs:/logs --bind ./cache:/root/.cache --bind ./env.sh:/.singularity.d/env/env.sh"

    # If embedding model is a local path, bind it into the RAG container.
    RAG_EMBED_BIND=""
    RAG_INDEXER_BIND=""
    RAG_APP_BINDS=""
    EMBEDDING_MODEL_CONTAINER="${EMBEDDING_MODEL:-}"
    if [[ -n "${EMBEDDING_MODEL_CONTAINER}" && "${EMBEDDING_MODEL_CONTAINER}" != /* ]]; then
        CACHE_BASE="${EMBEDDING_CACHE_DIR:-${MODEL_CACHE_BASE:-}}"
        if [[ -z "${CACHE_BASE}" || "${CACHE_BASE}" == "undefined" ]]; then
            CACHE_BASE="~/pw/models"
        fi
        CACHE_BASE="${CACHE_BASE/#\~/$HOME}"
        SAFE_ID="${EMBEDDING_MODEL_CONTAINER//\//__}"
        CANDIDATE="${CACHE_BASE}/${SAFE_ID}"
        if [[ -d "$CANDIDATE" ]]; then
            EMBEDDING_MODEL_CONTAINER="$CANDIDATE"
        fi
    fi
    if [[ -n "${EMBEDDING_MODEL_CONTAINER}" && "${EMBEDDING_MODEL_CONTAINER}" == /* ]]; then
        EMB_MODEL_PATH="${EMBEDDING_MODEL_CONTAINER/#\~/$HOME}"
        EMB_MODEL_BASE=$(basename "$EMB_MODEL_PATH")
        EMBEDDING_MODEL_CONTAINER="/${EMB_MODEL_BASE}"
        sed -i "s|^[#[:space:]]*\\(export[[:space:]]\\+\\)\\?EMBEDDING_MODEL=.*|export EMBEDDING_MODEL=$EMBEDDING_MODEL_CONTAINER|" env.sh
        RAG_EMBED_BIND="--bind ${EMB_MODEL_PATH}:${EMBEDDING_MODEL_CONTAINER}"
    fi
    if [[ -n "${EMBEDDING_MODEL_CONTAINER}" ]]; then
        INDEXER_CFG="./indexer_config.runtime.yaml"
        cp -f indexer_config.yaml "$INDEXER_CFG"
        sed -i "s|^embedding_model:.*|embedding_model: ${EMBEDDING_MODEL_CONTAINER}|" "$INDEXER_CFG"
        RAG_INDEXER_BIND="--bind ${INDEXER_CFG}:/app/indexer_config.yaml"
    fi
    for app_file in rag_server.py rag_proxy.py indexer.py; do
        if [[ -f "${app_file}" ]]; then
            RAG_APP_BINDS="${RAG_APP_BINDS} --bind ${PWD}/${app_file}:/app/${app_file}"
        fi
    done

    # Cleanup function
    cleanup() {
        echo "$(date) Cleaning up singularity instances..."
        singularity instance stop vllm 2>/dev/null || true
        singularity instance stop rag 2>/dev/null || true
    }
    trap cleanup EXIT

    # Create cancel script
    cat > cancel.sh << EOF
#!/bin/bash
set -x
singularity instance stop vllm 2>/dev/null || true
singularity instance stop rag 2>/dev/null || true
if [ -d "${TMPDIR}" ]; then
    tmp_dir="tmp-$(date +%s)"
    mv ${TMPDIR} ${tmp_dir} 
    rm -Rf ${tmp_dir}
fi
EOF
    chmod +x cancel.sh

    echo "$(date) Starting vLLM instance..."
    
    # Start vLLM instance with GPU support
    singularity instance start --nv \
        $COMMON_BINDS \
        --bind ./cache/sagemaker_sessions:/dev/shm/sagemaker_sessions \
        --bind ./cache/tiktoken_encodings:/root/.cache/tiktoken_encodings \
        --bind "${MODEL_PATH}:/${MODEL_BASE}" \
        "$VLLM_SIF" vllm

    # Run vLLM server inside the instance
    singularity exec instance://vllm bash -c "
        source /.singularity.d/env/env.sh
        nohup python3 -m vllm.entrypoints.openai.api_server \
            --model '/${MODEL_BASE}' \
            --tokenizer '/${MODEL_BASE}' \
            --host 0.0.0.0 \
            --port ${VLLM_SERVER_PORT} \
            ${VLLM_EXTRA_ARGS} \
            > /logs/vllm.out 2>&1 &
        echo \$! > /logs/vllm.pid
    "

    echo "$(date) vLLM server starting on port ${VLLM_SERVER_PORT}"

    if [ "$RUNTYPE" == "all" ]; then
        echo "$(date) Starting RAG instance..."
        
        # Start RAG instance
        singularity instance start \
            $COMMON_BINDS \
            --bind ./cache/chroma:/chroma_data \
            --bind ./docs:/docs \
            --bind ${DOCS_DIR}:/docs \
            $RAG_EMBED_BIND \
            $RAG_INDEXER_BIND \
            $RAG_APP_BINDS \
            "$RAG_SIF" rag

        # Run RAG services inside the instance
        singularity exec instance://rag bash -c "
            source /.singularity.d/env/env.sh
            
            # Start ChromaDB
            nohup chroma run --host 127.0.0.1 --port ${CHROMA_PORT} --path /chroma_data > /logs/chroma.out 2>&1 &
            echo \$! > /logs/chroma.pid
            
            # Wait for ChromaDB
            sleep 3
            
            # Start RAG server
            nohup python3 /app/rag_server.py --embedding_model \"${EMBEDDING_MODEL_CONTAINER}\" --port ${RAG_PORT} > /logs/rag_server.out 2>&1 &
            echo \$! > /logs/rag_server.pid
            
            # Start RAG proxy
            nohup python3 /app/rag_proxy.py > /logs/rag_proxy.out 2>&1 &
            echo \$! > /logs/rag_proxy.pid
            
            # Start indexer
            nohup python3 /app/indexer.py --config /app/indexer_config.yaml > /logs/indexer.out 2>&1 &
            echo \$! > /logs/indexer.pid
        "
        
        echo "$(date) RAG services starting (ChromaDB: ${CHROMA_PORT}, RAG: ${RAG_PORT}, Proxy: ${PROXY_PORT})"
    fi

    echo "$(date) All services started. Tailing logs..."
    
    # Wait a moment for logs to be created
    sleep 2
    
    # Tail logs (will keep script running)
    shopt -s nullglob
    logs=(logs/*.out)
    if ((${#logs[@]} > 0)); then
        tail -F "${logs[@]}" &
        tail_pid=$!
        wait "$tail_pid"
    else
        echo "No logs found. Waiting..."
        # Keep running so job doesn't end
        while true; do
            sleep 60
            # Check if vLLM is still running
            if ! singularity instance list | grep -q vllm; then
                echo "$(date) vLLM instance stopped. Exiting."
                exit 1
            fi
        done
    fi

fi
