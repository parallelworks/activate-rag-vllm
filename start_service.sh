#!/bin/bash
set -x

source .run.env > /dev/null 2>&1
#rm .run.env > /dev/null 2>&1

# set these if running manually
export RUNMODE=${RUNMODE:-docker} # docker or singularity
export BUILD=${BUILD:-false} # true or false
export RUNTYPE=${RUNTYPE:-all} # all or vllm
export MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
export DOCS_DIR=${DOCS_DIR:-./docs}

echo ""
echo "Running workflow with the below inputs:"
echo "  RUNMODE=$RUNMODE"
echo "  BUILD=$BUILD"
echo "  RUNTYPE=$RUNTYPE"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  DOCS_DIR=$DOCS_DIR"
echo ""

install_docker_compose(){
    echo "$(date) Downloading docker-compose v2.39.1"
    curl -L "https://github.com/docker/compose/releases/download/v2.39.1/docker-compose-$(uname -s)-$(uname -m)" -o docker-compose
    chmod +x docker-compose
}

findAvailablePort() {
    availablePort=$(pw agent open-port)
    echo ${availablePort}
    if [ -z "${availablePort}" ]; then
        echo "$(date) ERROR: No port found. Exiting job"
        exit 1
    fi
}

start_rootless_docker() {
    local MAX_RETRIES=20
    local RETRY_INTERVAL=2
    local ATTEMPT=1

    dockerd-rootless-setuptool.sh install
    PATH=/usr/bin:/sbin:/usr/sbin:$PATH dockerd-rootless.sh --exec-opt native.cgroupdriver=cgroupfs > docker-rootless.log 2>&1 & #--data-root /docker-rootless/docker-rootless/

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

    echo  "$(date): Docker daemon is ready!"
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
    VLLM_SERVER_PORT=$(findAvailablePort)
    PROXY_PORT=$(findAvailablePort)
    echo "PROXY_PORT=${PROXY_PORT}" > PROXY_PORT
    sed -i "s/^VLLM_SERVER_PORT=.*/VLLM_SERVER_PORT=${VLLM_SERVER_PORT}/" .env
    sed -i "s/^PROXY_PORT=.*/PROXY_PORT=${PROXY_PORT}/" .env

    sed -i "s/^[#[:space:]]*HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|export MODEL_NAME=$MODEL_NAME|" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?DOCS_DIR=.*|export DOCS_DIR=$DOCS_DIR|" .env

    # Disable weight download
    # Check if cache/huggingface directory exists
    if [ -d "cache/huggingface" ]; then
        sed -i 's/#TRANSFORMERS_OFFLINE=1/TRANSFORMERS_OFFLINE=1/' .env
        echo "$(date) Disabled model weight download"
    fi

    source .env

    mkdir -p logs cache cache/chroma $DOCS_DIR

    stack_name=$(echo ragvllm${PWD} | tr '/' '-')
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
        echo "$(date) Error: singularity is not installed"
        exit 1
    fi

    # Check if singularity-compose is installed
    source ~/pw/software/singularity-compose/bin/activate
    if ! command -v singularity-compose >/dev/null 2>&1; then
        source ~/pw/software/singularity-compose/bin/activate
    fi
    if ! command -v singularity-compose >/dev/null 2>&1; then
        echo "$(date) Error: Failed to install singularity-compose"
        exit 1
    fi

    cp singularity/* ./ -Rf
    cp env.sh.example env.sh
    
    VLLM_SERVER_PORT=$(findAvailablePort)
    RAG_PORT=$(findAvailablePort)
    PROXY_PORT=$(findAvailablePort)
    CHROMA_PORT=$(findAvailablePort)
    echo "PROXY_PORT=${PROXY_PORT}" > PROXY_PORT
    sed -i "s/^export VLLM_SERVER_PORT=.*/export VLLM_SERVER_PORT=${VLLM_SERVER_PORT}/" env.sh
    sed -i "s/^export RAG_PORT=.*/export RAG_PORT=${RAG_PORT}/" env.sh
    sed -i "s/^export PROXY_PORT=.*/export PROXY_PORT=${PROXY_PORT}/" env.sh
    sed -i "s/^export CHROMA_PORT=.*/export CHROMA_PORT=${CHROMA_PORT}/" env.sh

    sed -i "s/\(.*HF_TOKEN=\"\)[^\"]*\(\".*\)/\1$HF_TOKEN\2/" env.sh
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|export MODEL_NAME=$MODEL_NAME|" env.sh
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?DOCS_DIR=.*|export DOCS_DIR=$DOCS_DIR|" env.sh

    # Disable weight download
    # Check if cache/huggingface directory exists
    if [ -d "cache/huggingface" ]; then
        sed -i 's/#export TRANSFORMERS_OFFLINE=1/export TRANSFORMERS_OFFLINE=1/' env.sh
        echo "$(date) Disabled model weight download"
    fi

    source env.sh

    mkdir -p logs cache cache/chroma $DOCS_DIR

    # singularity-compose does not support env variables in the yml config file
    if [ "$DOCS_DIR" != "./docs" ];then
        ln -s $DOCS_DIR ./docs
    fi

    # If build is true check that user has root access
    #if [ "$BUILD" = "true" ] && ! sudo -n true 2>/dev/null; then
    #    echo "$(date) ERROR: User needs root access to build singularity containers"
    #    exit 1
    #fi
    echo "singularity-compose down" >> cancel.sh
    if [ "$RUNTYPE" == "all" ];then
        [ "$BUILD" = "true" ] && singularity-compose build
        DOCS_DIR=$DOCS_DIR singularity-compose up
    else
        [ "$BUILD" = "true" ] && singularity-compose build "${RUNTYPE}1"
        singularity-compose up "${RUNTYPE}1"
    fi
    # Follow the logs
    tail -f logs/*

fi
