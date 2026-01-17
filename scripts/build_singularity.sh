#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/build_singularity.sh [options]

Build Apptainer/Singularity containers from local recipes and optionally push to a PW bucket.

Options:
  --runtype {all|vllm}   Build both containers or only vLLM (default: all)
  --vllm-out PATH        Output path for vLLM container (default: ./vllm.sif)
  --rag-out PATH         Output path for RAG container (default: ./rag.sif)
  --bucket URI           PW bucket URI (e.g., pw://user/bucket)
  --push                 Push containers to the bucket
  --push-only            Skip build; only push existing container files
  --push-vllm            Push only vLLM container (use with --push/--push-only)
  --push-rag             Push only RAG container (use with --push/--push-only)
  --force                Rebuild even if output files exist
  -h, --help             Show this help message

Examples:
  scripts/build_singularity.sh --runtype all
  scripts/build_singularity.sh --runtype vllm --vllm-out /tmp/vllm.sif
  scripts/build_singularity.sh --push --bucket pw://mshaxted/codeassist
  scripts/build_singularity.sh --push-only --bucket pw://mshaxted/codeassist
  scripts/build_singularity.sh --push-only --push-rag --rag-out /path/rag.sif --bucket pw://mshaxted/codeassist
USAGE
}

runtype="all"
vllm_out="./vllm.sif"
rag_out="./rag.sif"
bucket=""
push="false"
push_only="false"
push_vllm="false"
push_rag="false"
force="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtype)
      runtype="$2"
      shift 2
      ;;
    --vllm-out)
      vllm_out="$2"
      shift 2
      ;;
    --rag-out)
      rag_out="$2"
      shift 2
      ;;
    --bucket)
      bucket="$2"
      shift 2
      ;;
    --push)
      push="true"
      shift
      ;;
    --push-only)
      push_only="true"
      push="true"
      shift
      ;;
    --push-vllm)
      push_vllm="true"
      shift
      ;;
    --push-rag)
      push_rag="true"
      shift
      ;;
    --force)
      force="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
 done

if [[ "$runtype" != "all" && "$runtype" != "vllm" ]]; then
  echo "ERROR: --runtype must be 'all' or 'vllm'." >&2
  exit 1
fi

expand_path() {
  local p="$1"
  if [[ "$p" == ~* ]]; then
    echo "${p/#\~/$HOME}"
  else
    echo "$p"
  fi
}

vllm_out="$(expand_path "$vllm_out")"
rag_out="$(expand_path "$rag_out")"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

should_build_vllm="false"
should_build_rag="false"
if [[ "$push_only" != "true" ]]; then
  if [[ ! -f "$vllm_out" || "$force" == "true" ]]; then
    should_build_vllm="true"
  fi
  if [[ "$runtype" == "all" ]]; then
    if [[ ! -f "$rag_out" || "$force" == "true" ]]; then
      should_build_rag="true"
    fi
  fi
fi

build_tool=""
if [[ "$should_build_vllm" == "true" || "$should_build_rag" == "true" ]]; then
  if [[ ! -f "${repo_root}/singularity/Singularity.vllm" ]]; then
    echo "ERROR: Missing ${repo_root}/singularity/Singularity.vllm" >&2
    exit 1
  fi
  if command -v apptainer >/dev/null 2>&1; then
    build_tool="apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    build_tool="singularity"
  else
    echo "ERROR: apptainer or singularity not found in PATH." >&2
    exit 1
  fi
fi

build_container() {
  local target="$1"
  local def_file="$2"

  if [[ -f "$target" && "$force" != "true" ]]; then
    echo "${target} exists, skipping (use --force to rebuild)."
    return 0
  fi

  mkdir -p "$(dirname "$target")"

  if sudo -n true 2>/dev/null; then
    sudo "$build_tool" build "$target" "$def_file"
  else
    "$build_tool" build --fakeroot "$target" "$def_file"
  fi
}

cd "$repo_root"

if [[ "$should_build_vllm" == "true" ]]; then
  echo "Using build tool: ${build_tool}"
  echo "Building vLLM container -> ${vllm_out}"
  build_container "$vllm_out" "${repo_root}/singularity/Singularity.vllm"
fi

if [[ "$should_build_rag" == "true" ]]; then
  if [[ ! -f "${repo_root}/singularity/Singularity.rag" ]]; then
    echo "ERROR: Missing ${repo_root}/singularity/Singularity.rag" >&2
    exit 1
  fi
  echo "Building RAG container -> ${rag_out}"
  build_container "$rag_out" "${repo_root}/singularity/Singularity.rag"
fi

if [[ "$push" == "true" ]]; then
  if [[ -z "$bucket" ]]; then
    echo "ERROR: --bucket is required when using --push." >&2
    exit 1
  fi
  if ! command -v pw >/dev/null 2>&1; then
    echo "ERROR: pw CLI not found in PATH." >&2
    exit 1
  fi

  bucket="${bucket%/}"
  echo "Pushing containers to ${bucket}"
  do_push_vllm="true"
  do_push_rag="false"
  if [[ "$push_vllm" == "true" || "$push_rag" == "true" ]]; then
    do_push_vllm="$push_vllm"
    do_push_rag="$push_rag"
  elif [[ "$runtype" == "all" ]]; then
    do_push_rag="true"
  fi

  if [[ "$do_push_vllm" == "true" ]]; then
    [[ -f "$vllm_out" ]] || { echo "ERROR: vLLM container not found at $vllm_out"; exit 1; }
    pw bucket cp "$vllm_out" "${bucket}/vllm.sif"
  fi
  if [[ "$do_push_rag" == "true" ]]; then
    [[ -f "$rag_out" ]] || { echo "ERROR: RAG container not found at $rag_out"; exit 1; }
    pw bucket cp "$rag_out" "${bucket}/rag.sif"
  fi
fi

echo "Done."
