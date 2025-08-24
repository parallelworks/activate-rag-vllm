#!/bin/bash

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

if [ "$RUNMODE" == "docker" ];then

    cp docker/* ./ -Rf
    cp env.example .env
    sed -i "s/^[#[:space:]]*HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|MODEL_NAME=$MODEL_NAME|" .env
    source .env

    mkdir -p logs cache cache/chroma $DOCS_DIR

    if [ "$RUNTYPE" == "all" ];then
        [ "$BUILD" = "true" ] && docker compose build
        docker compose up -d
    else
        [ "$BUILD" = "true" ] && docker compose build $RUNTYPE
        docker compose up $RUNTYPE -d
    fi

    docker compose logs

elif [ "$RUNMODE" == "singularity" ]; then

    cp singularity/* ./ -Rf
    cp env.sh.example env.sh
    sed -i "s/\(.*HF_TOKEN=\"\)[^\"]*\(\".*\)/\1$HF_TOKEN\2/" env.sh
    sed -i "s|^[#[:space:]]*\(export[[:space:]]\+\)\?MODEL_NAME=.*|MODEL_NAME=$MODEL_NAME|" env.sh
    source env.sh

    mkdir -p logs cache cache/chroma $DOCS_DIR

    # singularity-compose does not support env variables in the yml config file
    if [ "$DOCS_DIR" != "./docs" ];then
        ln -s $DOCS_DIR ./docs
    fi

    # needed to set an explicit tmp and cache location to avoid errors on the PW lab server
    mkdir -p /tmp/singularity-tmp /tmp/singularity-cache
    export SINGULARITY_TMPDIR=/tmp/singularity-tmp
    export SINGULARITY_CACHEDIR=/tmp/singularity-cache

    # pip3 install singularity-compose 
    if [ "$RUNTYPE" == "all" ];then
        [ "$BUILD" = "true" ] && singularity-compose build
        DOCS_DIR=$DOCS_DIR singularity-compose up
    else
        [ "$BUILD" = "true" ] && singularity-compose build "${RUNTYPE}1"
        singularity-compose up "${RUNTYPE}1"
    fi

fi
