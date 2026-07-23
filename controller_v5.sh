# controller_v5.sh — login-node setup for the v5 endpoint workflows (yamls/*_v5.yaml).
# Runs with inputs.sh prepended and appends the resolved paths back to ./inputs.sh
# for start_service_v5.sh.

set -o pipefail
set -x

if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
fi
if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    if command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y epel-release || true
        sudo dnf install -y apptainer
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y epel-release || true
        sudo yum install -y apptainer
    elif command -v apt-get >/dev/null 2>&1; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y apptainer
    fi
fi
singularity_bin=$(command -v singularity || command -v apptainer)
if [ -z "${singularity_bin}" ]; then
    echo "::error title=Error::singularity/apptainer not found and could not be installed"
    exit 1
fi
"${singularity_bin}" --version

if [ -z "${service_parent_install_dir}" ] || [ "${service_parent_install_dir}" = "undefined" ]; then
    service_parent_install_dir=${HOME}/pw/software
fi
service_parent_install_dir="${service_parent_install_dir/#\~/$HOME}"
mkdir -p "${service_parent_install_dir}/containers" "${service_parent_install_dir}/tools" 2>/dev/null || true
if [ ! -w "${service_parent_install_dir}/containers" ]; then
    echo "::warning::${service_parent_install_dir} is not writable; using ${HOME}/pw/software"
    service_parent_install_dir=${HOME}/pw/software
    mkdir -p "${service_parent_install_dir}/containers" "${service_parent_install_dir}/tools"
fi

oras_bin="${service_parent_install_dir}/tools/oras/oras"
if [ ! -x "${oras_bin}" ]; then
    echo "::group::oras Install"
    VER=1.2.0
    case "$(uname -m)" in
        x86_64) ARCH=amd64 ;;
        aarch64) ARCH=arm64 ;;
        *) echo "::error title=Error::unsupported architecture $(uname -m)"; exit 1 ;;
    esac
    curl -fsSL --connect-timeout 15 --max-time 300 -o oras.tar.gz \
        "https://github.com/oras-project/oras/releases/download/v${VER}/oras_${VER}_linux_${ARCH}.tar.gz"
    mkdir -p "${service_parent_install_dir}/tools/oras"
    tar -xzf oras.tar.gz -C "${service_parent_install_dir}/tools/oras" oras
    rm -f oras.tar.gz
    chmod -R a+rX "${service_parent_install_dir}/tools/oras"
    echo "::endgroup::"
fi

# SIF paths are derived from the tag so changing a container URI re-pulls
sif_path_for() {
    echo "${service_parent_install_dir}/containers/$(echo "${1##*/}" | tr ':' '-').sif"
}
pull_sif() {
    local uri=$1 sif=$2 pull_dir pulled_sif
    [ -f "${sif}" ] && return 0
    echo "::group::SIF Download ${uri}"
    pull_dir=$(mktemp -d -p "${service_parent_install_dir}/containers")
    if ! "${oras_bin}" pull "${uri}" -o "${pull_dir}"; then
        echo "::error title=Error::oras pull failed for ${uri}"
        exit 1
    fi
    pulled_sif=$(find "${pull_dir}" -name '*.sif' -type f | head -1)
    if [ ! -s "${pulled_sif}" ]; then
        echo "::error title=Error::no SIF file found in ${uri}"
        exit 1
    fi
    mv "${pulled_sif}" "${sif}"
    rm -rf "${pull_dir}"
    chmod a+r "${sif}"
    echo "::endgroup::"
}

container_sif=$(sif_path_for "${container_uri}")
pull_sif "${container_uri}" "${container_sif}"
if [ "${runtype}" = "all" ]; then
    rag_sif=$(sif_path_for "${rag_container_uri}")
    pull_sif "${rag_container_uri}" "${rag_sif}"
fi

# Tokenizer encodings for offline use (gpt-oss and tiktoken-based models)
tiktoken_dir="${service_parent_install_dir}/cache/tiktoken_encodings"
mkdir -p "${tiktoken_dir}"
for enc in o200k_base cl100k_base; do
    [ -s "${tiktoken_dir}/${enc}.tiktoken" ] && continue
    curl -fsSL --connect-timeout 15 --max-time 300 -o "${tiktoken_dir}/${enc}.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/${enc}.tiktoken" || {
        rm -f "${tiktoken_dir}/${enc}.tiktoken"
        echo "::warning::could not pre-download ${enc}.tiktoken"
    }
done

if [ -z "${model_cache_dir}" ] || [ "${model_cache_dir}" = "undefined" ]; then
    model_cache_dir=${HOME}/pw/models
fi

if [ "${runtype}" = "all" ]; then
    # The RAG stack (start_service.sh, rag_server.py, rag_proxy.py, indexer.py)
    # runs from a clone of this repository
    rag_rundir="${rag_rundir/#\~/$HOME}"
    mkdir -p "$(dirname "${rag_rundir}")"
    if [ -d "${rag_rundir}/.git" ]; then
        git -C "${rag_rundir}" fetch origin
        git -C "${rag_rundir}" checkout "${repository_branch}"
        git -C "${rag_rundir}" reset --hard "origin/${repository_branch}"
    else
        git clone -b "${repository_branch}" "${repository}" "${rag_rundir}"
    fi
    rm -f "${rag_rundir}"/{jobid,SESSION_PORT,job.started,job.ended,run.out,HOSTNAME,cancel.sh}
    rm -rf "${rag_rundir}/logs"

    # start_service.sh binds ./cache:/root/.cache; seed the shared encodings
    mkdir -p "${rag_rundir}/cache/tiktoken_encodings"
    cp -n "${tiktoken_dir}"/*.tiktoken "${rag_rundir}/cache/tiktoken_encodings/" 2>/dev/null || true

    if [ "${model_source}" = "local" ]; then
        model_dir="${model_local_path/#\~/$HOME}"
    elif [ "${model_source}" = "cached_model" ]; then
        model_dir="${model_cache_dir/#\~/$HOME}/${cached_model_id##*/}"
    else
        model_dir="${model_cache_dir/#\~/$HOME}/${hf_model_id##*/}"
    fi

    emb_cache="${rag_embedding_cache_dir:-${model_cache_dir}}"
    [ "${emb_cache}" = "undefined" ] && emb_cache="${model_cache_dir}"
    emb_cache="${emb_cache/#\~/$HOME}"

    {
        echo "export RUNMODE=singularity"
        echo "export RUNTYPE=all"
        echo "export SYSTEM_PROMPT=\"${rag_systemprompt}\""
        echo "export HF_TOKEN=${hf_token}"
        echo "export DOCS_DIR=${rag_docsdir:-./docs}"
        echo "export MAX_TOKENS=${rag_max_tokens}"
        echo "export TIKTOKEN_ENCODINGS_BASE=/root/.cache/tiktoken_encodings"
        echo "export TIKTOKEN_RS_CACHE_DIR=/root/.cache/tiktoken_encodings"
        echo "export VLLM_EXTRA_ARGS=\"${vllm_args}\""
        echo "export MODEL_SOURCE=${model_source}"
        echo "export MODEL_NAME=${model_dir}"
        echo "export MODEL_PATH=${model_dir}"
        echo "export TRANSFORMERS_OFFLINE=1"
        if [ "${rag_embedding_source}" = "local" ]; then
            echo "export EMBEDDING_MODEL=${rag_embedding_path}"
        else
            echo "export EMBEDDING_MODEL=${emb_cache}/${rag_embedding_id##*/}"
            echo "export EMBEDDING_CACHE_DIR=${emb_cache}"
        fi
        if [ -n "${vllm_attention_backend}" ] && [ "${vllm_attention_backend}" != "undefined" ]; then
            echo "export VLLM_ATTENTION_BACKEND=${vllm_attention_backend}"
        fi
    } > "${rag_rundir}/.run.env"
fi

# Pass the resolved paths to the start script
{
    echo "export service_parent_install_dir=\"${service_parent_install_dir}\""
    echo "export model_cache_dir=\"${model_cache_dir}\""
    echo "export container_sif=\"${container_sif}\""
    echo "export sandbox_dir=\"${container_sif%.sif}-sandbox\""
    echo "export tiktoken_dir=\"${tiktoken_dir}\""
    if [ "${runtype}" = "all" ]; then
        echo "export rag_sif=\"${rag_sif}\""
        echo "export rag_sandbox_dir=\"${rag_sif%.sif}-sandbox\""
        echo "export rag_rundir=\"${rag_rundir}\""
    fi
} >> ./inputs.sh
