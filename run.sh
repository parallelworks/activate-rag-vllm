#!/bin/bash

export RUNMODE=${RUNMODE:-docker} # docker or singularity
export BUILD=${BUILD:-true} # true or false
export RUNTYPE=${RUNTYPE:-all} # all or vllm

if [ "$RUNMODE" == "docker" ];then

    cp docker/* ./ -Rf
    cp env.example .env
    sed -i "s/^[#[:space:]]*HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env
    source .env

    mkdir -p logs cache cache/chroma docs

    [ "$BUILD" = "true" ] && docker compose build
    
    if [ "$RUNTYPE" == "all" ];then
        docker compose up -d
    else
        docker compose up $RUNTYPE -d
    fi

elif [ "$RUNMODE" == "singularity" ]; then

    cp singularity/* ./ -Rf
    cp env.sh.example env.sh
    sed -i "s/\(.*HF_TOKEN=\"\)[^\"]*\(\".*\)/\1$HF_TOKEN\2/" env.sh
    source env.sh

    mkdir -p logs cache cache/chroma docs

    # pip3 install singularity-compose 

    [ "$BUILD" = "true" ] && singularity-compose build

    if [ "$RUNTYPE" == "all" ];then
        singularity-compose up
    else
        singularity-compose up "${RUNTYPE}1"
    fi

fi
