

if [[ "${service_runmode}" == "singularity" ]];then
    # Check if singularity-compose is installed globally
    if ! command -v singularity-compose &> /dev/null; then
        # Check if virtual environment exists and activate it
        if [ -d ~/pw/software/singularity-compose ]; then
            source ~/pw/software/singularity-compose/bin/activate
        fi
        # Check again if singularity-compose is available after activation
        if ! command -v singularity-compose &> /dev/null; then
            echo "$(date) singularity-compose not found, installing..."      
            # Create directory for Python environment
            mkdir -p ~/pw/software
                  
            # Create virtual environment named singularity-compose and install singularity-compose
            python3 -m venv ~/pw/software/singularity-compose
            source ~/pw/software/singularity-compose/bin/activate
            pip install --upgrade pip
            pip install singularity-compose
        fi
    fi
    if ! command -v singularity-compose >/dev/null 2>&1; then
        echo "$(date) Error: Failed to install singularity-compose"
        exit 1
    fi
fi

if ! [ -z "${service_container_bucket}" ]; then
    # vllm container
    if [[ ! -f "vllm.sif" ]]; then
        echo "$(date) vllm.sif not found, pulling from ${service_container_bucket}"
        pw bucket cp "${service_container_bucket}/vllm.sif" ./
    else
        echo "$(date) vllm.sif already exists, skipping pull"
    fi
    # rag container (only for runmode=all)
    if [[ "${service_runmode}" == "all" ]]; then
        if [[ ! -f "rag.sif" ]]; then
            echo "$(date) rag.sif not found, pulling from ${service_container_bucket}"
            pw bucket cp "${service_container_bucket}/rag.sif" ./
        else
            echo "$(date) rag.sif already exists, skipping pull"
        fi
    fi
    echo "$(date) Singularity container pull step complete."
fi