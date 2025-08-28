#!/bin/bash
set -x

export RUNMODE=${RUNMODE:-docker} # docker or singularity
export BUILD=${BUILD:-true} # true or false
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
    which docker-compose >/dev/null 2>&1 || install_docker_compose

    major_version=$(docker compose version --short | cut -d'.' -f1)
    minor_version=$(docker compose version --short | cut -d'.' -f2)
    if [ -z "$major_version" ] || [ -z "$minor_version" ] || [ "$major_version" -lt 2 ] || { [ "$major_version" -eq 2 ] && [ "$minor_version" -lt 39 ]; }; then
        install_docker_compose
        docker_compose_cmd="./docker-compose"
    else
        docker_compose_cmd="docker-compose"
    fi
    
    # Ensure docker service is started and set docker_compose_cmd
    docker ps >/dev/null 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date) User has docker access" 
    elif ! sudo -n true 2>/dev/null; then
        echo "$(date) ERROR: User cannot run docker and has no root access to run sudo docker"
        exit 1
    else
        sudo dnf install -y nvidia-container-toolkit
        sudo systemctl start docker
        docker_compose_cmd="sudo ${docker_compose_cmd}"
    fi

    cp docker/* ./ -Rf
    cp env.example .env
    sed -i "s/^[#[:space:]]*HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|MODEL_NAME=$MODEL_NAME|" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?DOCS_DIR=.*|DOCS_DIR=$DOCS_DIR|" .env
    source .env

    mkdir -p logs cache cache/chroma $DOCS_DIR

    echo "${docker_compose_cmd} down" >> cancel.sh
    if [ "$RUNTYPE" == "all" ];then
        [ "$BUILD" = "true" ] && ${docker_compose_cmd} build
        ${docker_compose_cmd} up -d
    else
        [ "$BUILD" = "true" ] && ${docker_compose_cmd} build $RUNTYPE
        ${docker_compose_cmd} up $RUNTYPE -d
    fi

    ${docker_compose_cmd} logs -f

elif [ "$RUNMODE" == "singularity" ]; then

    cp singularity/* ./ -Rf
    cp env.sh.example env.sh
    sed -i "s/\(.*HF_TOKEN=\"\)[^\"]*\(\".*\)/\1$HF_TOKEN\2/" env.sh
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|MODEL_NAME=$MODEL_NAME|" env.sh
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?DOCS_DIR=.*|DOCS_DIR=$DOCS_DIR|" env.sh
    source env.sh

    mkdir -p logs cache cache/chroma $DOCS_DIR

    # singularity-compose does not support env variables in the yml config file
    if [ "$DOCS_DIR" != "./docs" ];then
        ln -s $DOCS_DIR ./docs
    fi

    # pip3 install singularity-compose 
    if [ "$RUNTYPE" == "all" ];then
        [ "$BUILD" = "true" ] && singularity-compose build
        DOCS_DIR=$DOCS_DIR singularity-compose up
    else
        [ "$BUILD" = "true" ] && singularity-compose build "${RUNTYPE}1"
        singularity-compose up "${RUNTYPE}1"
    fi

fi
