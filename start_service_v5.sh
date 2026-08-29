# start_service_v5.sh — start template for the v5 endpoint workflows (yamls/*_v5.yaml).
# Runs on the execution node with inputs.sh prepended (see the Create Service Script
# step). Launches the service, waits for it to answer, and registers a pw endpoint.

set -x

echo "::group::vLLM Service Starting"

if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
fi
# Some systems (Fran) install apptainer under a prefix that is only added to
# PATH in interactive shells and ship no module for it; probe the usual
# install prefixes before giving up
if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
    for dir in /opt/apptainer/bin /opt/singularity/bin /usr/local/apptainer/bin /usr/local/singularity/bin; do
        if [ -x "${dir}/apptainer" ] || [ -x "${dir}/singularity" ]; then
            export PATH="${dir}:${PATH}"
            break
        fi
    done
fi
singularity_bin=$(command -v singularity || command -v apptainer)
if [ -z "${singularity_bin}" ]; then
    echo "::error title=Error::singularity/apptainer not found on the execution node"
    exit 1
fi

if [ ! -f "${container_sif}" ]; then
    echo "::error title=Error::Missing container image ${container_sif}"
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi -L; then
        echo "::error title=Error::no GPU visible on $(hostname); request GPUs with a scheduler directive such as '#SBATCH --gres=gpu:1'"
        exit 1
    fi
else
    echo "::warning::nvidia-smi not found on $(hostname)"
fi

if [ "${model_source}" = "local" ]; then
    model_dir="${model_local_path/#\~/$HOME}"
    served_model_name=$(basename "${model_dir}")
elif [ "${model_source}" = "cached_model" ]; then
    model_dir="${model_cache_dir/#\~/$HOME}/${cached_model_id##*/}"
    served_model_name="${cached_model_id}"
else
    model_dir="${model_cache_dir/#\~/$HOME}/${hf_model_id##*/}"
    served_model_name="${hf_model_id}"
fi
if [ ! -s "${model_dir}/config.json" ]; then
    echo "::error title=Error::model directory ${model_dir} is missing or incomplete"
    exit 1
fi

# Prefer running a SIF directly; some nodes cannot mount it (no squashfs
# kernel/FUSE support), in which case unpack it once into a sandbox directory
resolve_ref() {
    local sif=$1 sandbox=$2
    if "${singularity_bin}" exec "${sif}" /bin/true > /dev/null 2>&1; then
        echo "${sif}"
        return 0
    fi
    echo "::notice::Cannot mount ${sif} on this node; using a sandbox directory" >&2
    export SINGULARITY_TMPDIR=${HOME}/.singularity_tmp
    export SINGULARITY_CACHEDIR=${HOME}/.singularity_cache
    mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"
    if [ ! -d "${sandbox}" ]; then
        "${singularity_bin}" build --fakeroot --force --sandbox "${sandbox}" "${sif}" >&2 || return 1
    fi
    echo "${sandbox}"
}

container_ref=$(resolve_ref "${container_sif}" "${sandbox_dir}")
if [ -z "${container_ref}" ]; then
    echo "::error title=Error::cannot run or unpack ${container_sif}"
    exit 1
fi

echo "::endgroup::"

if [ "${runtype}" = "all" ]; then
    echo "::group::Starting vLLM + RAG"

    rag_ref=$(resolve_ref "${rag_sif}" "${rag_sandbox_dir}")
    if [ -z "${rag_ref}" ]; then
        echo "::error title=Error::cannot run or unpack ${rag_sif}"
        exit 1
    fi

    # start_service.sh calls 'singularity' by name; shim it when only apptainer exists
    if ! command -v singularity >/dev/null 2>&1; then
        mkdir -p "${PWD}/bin"
        ln -sf "${singularity_bin}" "${PWD}/bin/singularity"
        export PATH="${PWD}/bin:${PATH}"
    fi

    endpoint_port=$(pw agent open-port)
    svc_log="${PWD}/rag-vllm-${PW_JOB_ID}.out"

    # The repository's start_service.sh launches vLLM + ChromaDB + RAG
    # server/proxy/indexer; the endpoint targets the RAG proxy (the
    # user-facing OpenAI-compatible API)
    export VLLM_CONTAINER_PATH="${container_ref}"
    export RAG_CONTAINER_PATH="${rag_ref}"
    export service_port=${endpoint_port}
    ( cd "${rag_rundir}" && exec bash start_service.sh ) > "${svc_log}" 2>&1 &
    svc_pid=$!

    cat > cancel.sh <<CANCELEOF
#!/bin/bash
[ -f "${rag_rundir}/cancel.sh" ] && bash "${rag_rundir}/cancel.sh"
kill ${svc_pid} 2>/dev/null || true
CANCELEOF
    chmod +x cancel.sh

    # The proxy's /v1/models forwards to vLLM, so one gate covers both
    gate_url="localhost:${endpoint_port}/v1/models"
else
    echo "::group::Starting vLLM"

    # Per-job /tmp: the host /tmp is often small and shared, and the job dir
    # path is too long for ZMQ IPC sockets, so tmp/cache paths must be the
    # short in-container /tmp backed by this bind
    mkdir -p "${PWD}/container_tmp"
    mkdir -p "${tiktoken_dir}" 2>/dev/null || true

    extra_env=""
    if [ -n "${vllm_attention_backend}" ] && [ "${vllm_attention_backend}" != "undefined" ]; then
        extra_env="--env VLLM_ATTENTION_BACKEND=${vllm_attention_backend}"
    fi

    endpoint_port=$(pw agent open-port)
    svc_log="${PWD}/vllm-${PW_JOB_ID}.out"

    # CC/CXX must be pinned to the container's compilers: singularity passes
    # the host env through by default, and some systems (Jean) export CC=icc,
    # which Triton uses to build its CUDA driver stub but does not exist in
    # the container. TRITON_CACHE_DIR joins the other JIT caches in the
    # per-job /tmp so runs never share ~/.triton on NFS home.
    cat > launch-vllm-${PW_JOB_ID}.sh <<LAUNCHEOF
#!/bin/bash
exec "${singularity_bin}" exec --nv --writable-tmpfs \\
    --bind "${model_dir}:${model_dir}" \\
    --bind "${PWD}/container_tmp:/tmp" \\
    --bind "${tiktoken_dir}:${tiktoken_dir}" \\
    --env PYTHONNOUSERSITE=1 \\
    --env TRANSFORMERS_OFFLINE=1 \\
    --env HF_HUB_OFFLINE=1 \\
    --env TIKTOKEN_ENCODINGS_BASE="${tiktoken_dir}" \\
    --env TIKTOKEN_RS_CACHE_DIR="${tiktoken_dir}" \\
    --env TMPDIR=/tmp \\
    --env CUDA_CACHE_PATH=/tmp/cuda_cache \\
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \\
    --env FLASHINFER_JIT_DIR=/tmp/flashinfer_jit \\
    --env TRITON_CACHE_DIR=/tmp/triton_cache \\
    --env CC=gcc \\
    --env CXX=g++ \\
    --env CUDA_DEVICE_ORDER=PCI_BUS_ID \\
    ${extra_env} \\
    "${container_ref}" \\
    python3 -m vllm.entrypoints.openai.api_server \\
        --model "${model_dir}" \\
        --served-model-name "${served_model_name}" \\
        --host 127.0.0.1 \\
        --port ${endpoint_port} \\
        ${vllm_args}
LAUNCHEOF
    chmod +x launch-vllm-${PW_JOB_ID}.sh
    cat launch-vllm-${PW_JOB_ID}.sh

    ./launch-vllm-${PW_JOB_ID}.sh > "${svc_log}" 2>&1 &
    svc_pid=$!

    # Both the generic cleanup trap and script_submitter's cleanup run
    # cancel.sh: vLLM must die on endpoint delete and on cancel-before-ready
    cat > cancel.sh <<CANCELEOF
#!/bin/bash
kill ${svc_pid} 2>/dev/null || true
CANCELEOF
    chmod +x cancel.sh

    gate_url="localhost:${endpoint_port}/health"
fi

# Register the endpoint only once the service is answering: the platform
# chat reports the session as unreachable while the model is still loading
echo "Waiting for the service at ${gate_url} (log: ${svc_log})"

# Stream the service log into this script's output while the service starts,
# so model-loading progress shows up in the workflow logs without having to
# tail ${svc_log} on the cluster; stopped once the service answers so the
# runtime log is not duplicated for the lifetime of the endpoint
tail -n +1 -F "${svc_log}" 2>/dev/null &
tail_pid=$!
stop_log_stream() {
    kill ${tail_pid} 2>/dev/null
    wait ${tail_pid} 2>/dev/null
}

elapsed=0
until curl -sf -o /dev/null --max-time 5 "${gate_url}"; do
    if ! kill -0 ${svc_pid} 2>/dev/null; then
        stop_log_stream
        echo "::error title=Error::service exited during startup; last log lines:"
        tail -50 "${svc_log}"
        pw workflows runs cancel ${PW_RUN_SLUG}
        exit 1
    fi
    if [ ${elapsed} -ge 3600 ]; then
        stop_log_stream
        echo "::error title=Error::service did not start within 60 minutes; last log lines:"
        tail -50 "${svc_log}"
        pw workflows runs cancel ${PW_RUN_SLUG}
        exit 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
    if [ $((elapsed % 60)) -eq 0 ]; then
        echo "$(date) service still starting (${elapsed}s elapsed)"
    fi
done
stop_log_stream
echo "Service is answering at ${gate_url}; runtime log continues at ${svc_log}; registering endpoint"

# --link ties the port's server process to the endpoint: deleting the
# endpoint stops it even if the surrounding script is gone
pw endpoints http --link ${pw_endpoints_args} -o text ${endpoint_port}

if [ $? -ne 0 ]; then
    echo "::error title=Error::pw endpoints command failed"
    # Fail loud: without this, wait_for_endpoint polls forever for an
    # endpoint that will never register
    pw workflows runs cancel ${PW_RUN_SLUG}
    exit 1
fi
echo "::endgroup::"
