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

# Create cleanup script
echo '#!/bin/bash' > cancel.sh
chmod +x cancel.sh

if [ "$RUNMODE" == "docker" ];then

    # Ensure docker service is started and set docker_cmd
    which docker >/dev/null 2>&1 || { 
        echo "$(date) ERROR: Docker is not installed."
        exit 1
    }

    docker ps >/dev/null 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        docker_cmd="docker"
    elif ! sudo -n true 2>/dev/null; then
        echo " $(date) ERROR: User cannot run docker and has no root access to run sudo docker"
        exit 1
    else
        sudo systemctl start docker
        docker_cmd="sudo docker"
    fi

    cp docker/* ./ -Rf
    cp env.example .env
    sed -i "s/^[#[:space:]]*HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|MODEL_NAME=$MODEL_NAME|" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?DOCS_DIR=.*|DOCS_DIR=$DOCS_DIR|" .env
    source .env

    mkdir -p logs cache cache/chroma $DOCS_DIR

    major_version=$(docker compose version --short | cut -d'.' -f1)
    minor_version=$(docker compose version --short | cut -d'.' -f2)
    if [ "${major_version}" -ge 3 ]; then
        cp docker-compose-v2.39.1.yml docker-compose.yml 
    elif [ "${major_version}" -le 1 ]; then
        cp docker-compose-v2.27.0.yml docker-compose.yml
    else
        if [ "${minor_version}" -ge 39 ]; then
            cp docker-compose-v2.39.1.yml docker-compose.yml
        else
            cp docker-compose-v2.27.0.yml docker-compose.yml
        fi
    fi

    echo "${docker_cmd} compose down" >> cancel.sh
    if [ "$RUNTYPE" == "all" ];then
        [ "$BUILD" = "true" ] && ${docker_cmd} compose build
        ${docker_cmd} compose up -d
    else
        [ "$BUILD" = "true" ] && ${docker_cmd} compose build $RUNTYPE
        ${docker_cmd} compose up $RUNTYPE -d
    fi

    ${docker_cmd} compose logs

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
