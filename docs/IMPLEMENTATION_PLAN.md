# ACTIVATE RAG-vLLM Implementation Plan

**Date**: January 16, 2026  
**Repository**: parallelworks/activate-rag-vllm  
**Objective**: Improve, refactor, and consolidate the repository for long-term supportability, ease of use, and multi-environment deployment (Singularity-focused for HPC).

---

## Executive Summary

This plan outlines the steps to:
1. Merge the `nemotron` branch improvements into `main`
2. Consolidate duplicate code and configurations
3. Add flexible model sourcing (local path or HuggingFace pull)
4. Improve Singularity deployment for HPC environments
5. Create a unified, user-friendly workflow experience

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User / Open WebUI                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Proxy (Port 8081)                         │
│   - OpenAI-compatible endpoints                                  │
│   - Injects RAG context into prompts                            │
│   - Citation handling                                            │
└───────────────┬─────────────────────────────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌───────────────┐  ┌─────────────────────────────────────────────┐
│ RAG Server    │  │           vLLM Server (8000)                 │
│ (8080)        │  │   - OpenAI-compatible inference API          │
│ - ChromaDB    │  │   - GPU acceleration                         │
└───────┬───────┘  └─────────────────────────────────────────────┘
        │
        ▼
┌───────────────┐  ┌─────────────────────────────────────────────┐
│ ChromaDB      │◄─│         Indexer (background)                 │
│ (8001)        │  │   - File watcher for docs                    │
└───────────────┘  └─────────────────────────────────────────────┘
```

---

## Phase 1: Branch Consolidation & Core Cleanup

### 1.1 Merge Nemotron Branch Improvements

**Priority**: High | **Effort**: Medium

The `nemotron` branch contains valuable improvements that should be merged:

| Feature | Description | Action |
|---------|-------------|--------|
| `controller.sh` | Extracted preprocessing logic | ✅ Adopt |
| `parallelworks/checkout` action | Cleaner git clone | ✅ Adopt |
| PBS scheduler support | Extended HPC compatibility | ✅ Adopt |
| VLLM attention backend options | 20+ backend choices | ✅ Adopt |
| Offline mode defaults | `TRANSFORMERS_OFFLINE=1` | ✅ Adopt |
| Container pull options | `pull` boolean + bucket source | ✅ Adopt |
| Tiktoken encodings download | Offline tokenizer support | ✅ Adopt |

**Implementation Steps**:
```bash
# Create integration branch
git checkout main
git checkout -b feature/nemotron-integration
git merge origin/nemotron --no-commit

# Resolve conflicts, keeping best of both
# Test thoroughly before merging to main
```

### 1.2 Consolidate Workflow YAML Files

**Priority**: High | **Effort**: Medium

**Current State**: 4 similar workflow files with 70%+ code duplication
- `workflow.yaml` (main)
- `workflow-vllm.yaml` (vLLM-only mode)
- `yamls/hsp.yaml` (HPC-specific)
- `yamls/emed.yaml` (medical domain)

**Target State**: Single `workflow.yaml` with conditional sections

**Implementation**:

```yaml
# Proposed unified workflow.yaml structure
name: activate-rag-vllm
description: Deploy vLLM + RAG stack on HPC or cloud

inputs:
  # === Mode Selection ===
  deployment_mode:
    type: dropdown
    label: Deployment Mode
    options:
      - label: "vLLM + RAG (Full Stack)"
        value: all
      - label: "vLLM Only"
        value: vllm
    default: all

  # === Model Configuration ===
  model_source:
    type: dropdown
    label: Model Source
    options:
      - label: "Local Path (pre-downloaded)"
        value: local
      - label: "HuggingFace Hub (auto-download)"
        value: huggingface
    default: local

  model_path:
    type: text
    label: Local Model Path
    description: "Full path to model weights directory"
    hidden: inputs.model_source != 'local'

  hf_model_id:
    type: text
    label: HuggingFace Model ID
    placeholder: "meta-llama/Llama-3.1-8B-Instruct"
    hidden: inputs.model_source != 'huggingface'

  # === Scheduler Selection ===
  scheduler:
    type: dropdown
    label: Job Scheduler
    options:
      - { label: SSH (direct), value: ssh }
      - { label: SLURM, value: slurm }
      - { label: PBS, value: pbs }
    default: slurm

  # Conditional scheduler options shown based on selection
  slurm_partition:
    hidden: inputs.scheduler != 'slurm'
  pbs_queue:
    hidden: inputs.scheduler != 'pbs'
```

### 1.3 Unify Entrypoint Scripts

**Priority**: High | **Effort**: Low

**Current State**: Logic split between `start_service.sh` and `controller.sh`

**Target State**: Single `start_service.sh` with modular functions

**Proposed Structure**:
```bash
#!/bin/bash
# start_service.sh - Unified entrypoint

set -euo pipefail

# Source common functions
source "$(dirname "$0")/lib/functions.sh"

# Main execution
main() {
    parse_arguments "$@"
    detect_environment    # Docker vs Singularity vs local
    validate_config
    setup_model           # New: handles local vs HF download
    configure_ports
    launch_services
    wait_for_ready
    export_session_port
}

main "$@"
```

---

## Phase 2: Model Management Enhancement

### 2.1 Flexible Model Sourcing

**Priority**: High | **Effort**: Medium

Create a model management system that supports:
1. Local pre-downloaded models
2. HuggingFace Hub downloads (git-lfs preferred for HPC)
3. Cached model reuse across runs

**New File**: `lib/model_manager.sh`

```bash
#!/bin/bash
# lib/model_manager.sh - Model download and validation

MODEL_CACHE_BASE="${MODEL_CACHE_BASE:-$HOME/.cache/activate-models}"

setup_model() {
    local source="$1"      # local | huggingface
    local model_id="$2"    # path or HF model ID
    local hf_token="$3"    # optional HF token

    case "$source" in
        local)
            validate_local_model "$model_id"
            MODEL_PATH="$model_id"
            ;;
        huggingface)
            download_hf_model "$model_id" "$hf_token"
            MODEL_PATH="$MODEL_CACHE_BASE/$model_id"
            ;;
    esac
    
    export MODEL_PATH
}

validate_local_model() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        error "Model directory not found: $path"
        exit 1
    fi
    
    # Check for required files
    local required_files=("config.json" "tokenizer.json")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$path/$file" ]]; then
            warn "Missing expected file: $path/$file"
        fi
    done
    
    info "Local model validated: $path"
}

download_hf_model() {
    local model_id="$1"
    local hf_token="$2"
    local target_dir="$MODEL_CACHE_BASE/$model_id"
    
    if [[ -d "$target_dir" ]] && model_is_complete "$target_dir"; then
        info "Model already cached: $target_dir"
        return 0
    fi
    
    mkdir -p "$target_dir"
    
    # Prefer git-lfs for HPC (more reliable than hf_hub_download)
    info "Downloading model via git-lfs: $model_id"
    
    local repo_url="https://huggingface.co/$model_id"
    if [[ -n "$hf_token" ]]; then
        repo_url="https://user:${hf_token}@huggingface.co/$model_id"
    fi
    
    GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 "$repo_url" "$target_dir"
    
    # Verify download
    if ! model_is_complete "$target_dir"; then
        error "Model download incomplete"
        exit 1
    fi
    
    info "Model downloaded successfully: $target_dir"
}

model_is_complete() {
    local path="$1"
    [[ -f "$path/config.json" ]] && \
    [[ -f "$path/tokenizer.json" || -f "$path/tokenizer_config.json" ]]
}
```

### 2.2 Workflow Form with Conditional Elements

**Priority**: High | **Effort**: Medium

Update `workflow.yaml` to show/hide form elements based on model source:

```yaml
inputs:
  model:
    type: section
    label: Model Configuration
    
    source:
      type: dropdown
      label: Model Source
      options:
        - label: "📁 Local Path (recommended for HPC)"
          value: local
          description: "Use pre-downloaded model weights"
        - label: "🤗 HuggingFace Hub"
          value: huggingface
          description: "Download from HuggingFace (requires network)"
      default: local
    
    # Shown when source=local
    local_path:
      type: text
      label: Model Path
      placeholder: /path/to/model/weights
      description: "Full path to directory containing model weights"
      required: true
      hidden:
        source: '!= local'
    
    # Shown when source=huggingface
    hf_model_id:
      type: text
      label: HuggingFace Model ID
      placeholder: meta-llama/Llama-3.1-8B-Instruct
      hidden:
        source: '!= huggingface'
    
    hf_token:
      type: secret
      label: HuggingFace Token
      description: "Required for gated models (Llama, etc.)"
      hidden:
        source: '!= huggingface'
    
    cache_dir:
      type: text
      label: Model Cache Directory
      default: ~/pw/models
      description: "Where to store downloaded models"
      hidden:
        source: '!= huggingface'
```

---

## Phase 3: Singularity Optimization for HPC

### 3.1 Improved Singularity Compose Configuration

**Priority**: High | **Effort**: Medium

**Issues to Address**:
- Manual `__MODEL_PATH__` substitution
- No native env var interpolation
- Port management complexity

**Proposed `singularity/singularity-compose.yml`**:

```yaml
version: "1.0"

instances:
  vllm:
    build:
      context: .
      recipe: Singularity.vllm
    ports:
      - "${VLLM_PORT:-8000}:8000"
    volumes:
      - "${MODEL_PATH}:/models/active:ro"
      - "${HF_CACHE:-./cache}:/root/.cache/huggingface"
    environment:
      - MODEL_NAME=/models/active
      - VLLM_API_KEY=${VLLM_API_KEY:-}
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}
    runtime:
      options: "--nv"  # GPU support
    start:
      options: "--env-file env.sh"

  rag:
    build:
      context: .
      recipe: Singularity.rag
    depends_on:
      - vllm
    ports:
      - "${RAG_PROXY_PORT:-8081}:8081"
      - "${RAG_SERVER_PORT:-8080}:8080"
      - "${CHROMA_PORT:-8001}:8001"
    volumes:
      - "${DOCS_DIR:-./docs}:/docs:rw"
      - "${CHROMA_DATA:-./chroma_data}:/chroma_data"
    environment:
      - VLLM_URL=http://127.0.0.1:${VLLM_PORT:-8000}/v1
      - VLLM_API_KEY=${VLLM_API_KEY:-}
```

### 3.2 HPC-Specific Configuration Templates

**Priority**: Medium | **Effort**: Low

Create configuration presets for common HPC environments:

**New File**: `configs/hpc-presets.yaml`

```yaml
presets:
  # Navy DSRC systems
  navy-hpc:
    scheduler: pbs
    container_source: bucket
    container_bucket: "gs://navy-containers/activate"
    offline_mode: true
    defaults:
      gpu_type: "nvidia_a100"
      max_model_len: 32768
  
  # AFRL systems  
  afrl-hpc:
    scheduler: slurm
    container_source: local
    offline_mode: true
    defaults:
      partition: "gpu"
      qos: "normal"
  
  # AWS cloud
  aws-cloud:
    scheduler: slurm
    container_source: pull
    offline_mode: false
    defaults:
      instance_type: "p4d.24xlarge"
  
  # Local development
  local-dev:
    scheduler: ssh
    container_source: build
    offline_mode: false
    defaults:
      gpu_type: "auto-detect"
```

### 3.3 Pre-flight Validation Script

**Priority**: Medium | **Effort**: Low

**New File**: `lib/preflight.sh`

```bash
#!/bin/bash
# lib/preflight.sh - Pre-flight checks for HPC deployment

preflight_checks() {
    local errors=0
    
    info "Running pre-flight checks..."
    
    # Check Singularity
    if ! command -v singularity &>/dev/null; then
        error "Singularity not found in PATH"
        ((errors++))
    else
        local version=$(singularity --version 2>/dev/null)
        info "Singularity: $version"
    fi
    
    # Check GPU access
    if ! nvidia-smi &>/dev/null; then
        warn "nvidia-smi not available - GPU may not be accessible"
    else
        local gpu_count=$(nvidia-smi -L | wc -l)
        info "GPUs detected: $gpu_count"
    fi
    
    # Check model path
    if [[ "$MODEL_SOURCE" == "local" ]]; then
        if [[ ! -d "$MODEL_PATH" ]]; then
            error "Model path not found: $MODEL_PATH"
            ((errors++))
        fi
    fi
    
    # Check disk space for cache
    local cache_dir="${MODEL_CACHE_BASE:-$HOME/.cache}"
    local free_gb=$(df -BG "$cache_dir" | awk 'NR==2 {print $4}' | tr -d 'G')
    if (( free_gb < 50 )); then
        warn "Low disk space for model cache: ${free_gb}GB free"
    fi
    
    # Check network (if HF download needed)
    if [[ "$MODEL_SOURCE" == "huggingface" ]]; then
        if ! curl -s --connect-timeout 5 https://huggingface.co &>/dev/null; then
            error "Cannot reach HuggingFace Hub - check network/proxy"
            ((errors++))
        fi
    fi
    
    if (( errors > 0 )); then
        error "Pre-flight checks failed with $errors error(s)"
        return 1
    fi
    
    info "Pre-flight checks passed ✓"
    return 0
}
```

---

## Phase 4: Code Quality & Reliability

### 4.1 Error Handling Improvements

**Priority**: Medium | **Effort**: Low

**start_service.sh changes**:
```bash
#!/bin/bash
set -euo pipefail  # Add -e for exit on error

trap cleanup EXIT ERR

cleanup() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        error "Script failed with exit code: $exit_code"
        # Capture logs for debugging
        if [[ -d "./logs" ]]; then
            tar -czf "debug-logs-$(date +%Y%m%d-%H%M%S).tar.gz" ./logs/
        fi
    fi
}
```

### 4.2 Configuration Validation

**Priority**: Medium | **Effort**: Medium

**New File**: `lib/config_validator.py`

```python
#!/usr/bin/env python3
"""Validate configuration before service launch."""

import os
import sys
import json
from pathlib import Path


def validate_model_config(config: dict) -> list[str]:
    """Validate model configuration."""
    errors = []
    
    model_path = config.get("MODEL_PATH") or config.get("model_path")
    if not model_path:
        errors.append("MODEL_PATH not specified")
    elif not Path(model_path).exists():
        errors.append(f"Model path does not exist: {model_path}")
    else:
        # Check for required model files
        required = ["config.json"]
        for f in required:
            if not (Path(model_path) / f).exists():
                errors.append(f"Missing required file: {model_path}/{f}")
    
    return errors


def validate_port_config(config: dict) -> list[str]:
    """Validate port configuration."""
    errors = []
    
    ports = {
        "VLLM_PORT": config.get("VLLM_PORT", 8000),
        "RAG_PROXY_PORT": config.get("RAG_PROXY_PORT", 8081),
        "RAG_SERVER_PORT": config.get("RAG_SERVER_PORT", 8080),
        "CHROMA_PORT": config.get("CHROMA_PORT", 8001),
    }
    
    # Check for port conflicts
    used_ports = list(ports.values())
    if len(used_ports) != len(set(used_ports)):
        errors.append("Port conflict detected - duplicate port assignments")
    
    return errors


def main():
    """Run all validations."""
    config = dict(os.environ)
    
    # Also load from env.sh if present
    env_file = Path("env.sh")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                key = key.replace("export ", "").strip()
                config[key] = value.strip().strip('"').strip("'")
    
    all_errors = []
    all_errors.extend(validate_model_config(config))
    all_errors.extend(validate_port_config(config))
    
    if all_errors:
        print("Configuration validation failed:", file=sys.stderr)
        for error in all_errors:
            print(f"  ✗ {error}", file=sys.stderr)
        sys.exit(1)
    
    print("Configuration validation passed ✓")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

### 4.3 Logging Improvements

**Priority**: Low | **Effort**: Low

**Add to `lib/functions.sh`**:
```bash
# Logging functions with timestamps
LOG_FILE="${LOG_DIR:-./logs}/service-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

info()  { log "INFO" "$@"; }
warn()  { log "WARN" "$@" >&2; }
error() { log "ERROR" "$@" >&2; }
debug() { [[ "${DEBUG:-0}" == "1" ]] && log "DEBUG" "$@"; }
```

---

## Phase 5: User Experience Improvements

### 5.1 Quick Start Guide

**Priority**: Medium | **Effort**: Low

**Update README.md** with clear quickstart:

```markdown
## Quick Start

### Option 1: Local Model (Recommended for HPC)

1. **Ensure model weights are available**:
   ```bash
   ls /path/to/your/model/
   # Should contain: config.json, tokenizer.json, *.safetensors
   ```

2. **Deploy via ParallelWorks**:
   - Select "Local Path" as Model Source
   - Enter full path to model directory
   - Choose your scheduler (SLURM/PBS/SSH)
   - Submit workflow

### Option 2: HuggingFace Download

1. **Get HuggingFace token** (for gated models):
   - Visit https://huggingface.co/settings/tokens
   - Create token with "read" permissions

2. **Deploy via ParallelWorks**:
   - Select "HuggingFace Hub" as Model Source
   - Enter model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
   - Paste your HF token
   - Submit workflow
```

### 5.2 Interactive Configuration Wizard

**Priority**: Low | **Effort**: Medium

**New File**: `scripts/configure.sh`

```bash
#!/bin/bash
# Interactive configuration wizard for local development

echo "=== ACTIVATE RAG-vLLM Configuration Wizard ==="
echo

# Model source
echo "How will you provide the model?"
select model_source in "Local Path" "HuggingFace Download"; do
    case $model_source in
        "Local Path")
            read -p "Enter model path: " MODEL_PATH
            if [[ ! -d "$MODEL_PATH" ]]; then
                echo "Warning: Path does not exist"
            fi
            break
            ;;
        "HuggingFace Download")
            read -p "Enter HuggingFace model ID: " HF_MODEL_ID
            read -sp "Enter HuggingFace token (optional): " HF_TOKEN
            echo
            MODEL_PATH="$HOME/.cache/activate-models/$HF_MODEL_ID"
            break
            ;;
    esac
done

# Deployment mode
echo
echo "What do you want to deploy?"
select runtype in "vLLM + RAG (Full Stack)" "vLLM Only"; do
    case $runtype in
        "vLLM + RAG"*) RUNTYPE="all"; break ;;
        "vLLM Only") RUNTYPE="vllm"; break ;;
    esac
done

# Generate env.sh
cat > env.sh << EOF
# Generated by configure.sh on $(date)
export MODEL_PATH="$MODEL_PATH"
export RUNTYPE="$RUNTYPE"
export HF_TOKEN="${HF_TOKEN:-}"
export TRANSFORMERS_OFFLINE=1
EOF

echo
echo "Configuration saved to env.sh"
echo "Run: ./start_service.sh"
```

---

## Phase 6: Testing & CI/CD

### 6.1 Basic Test Suite

**Priority**: Low | **Effort**: High

**New Directory**: `tests/`

```
tests/
├── conftest.py
├── test_rag_server.py
├── test_rag_proxy.py
├── test_indexer.py
└── integration/
    └── test_e2e.py
```

### 6.2 GitHub Actions Workflow

**Priority**: Low | **Effort**: Medium

**New File**: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check .

  shellcheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: shellcheck *.sh lib/*.sh

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/ -v
```

---

## Implementation Timeline

| Phase | Tasks | Duration | Dependencies |
|-------|-------|----------|--------------|
| **1** | Branch merge, YAML consolidation, script unification | 1 week | None |
| **2** | Model management, conditional forms | 1 week | Phase 1 |
| **3** | Singularity optimization, HPC presets | 1 week | Phase 1-2 |
| **4** | Error handling, validation, logging | 3 days | Phase 1 |
| **5** | Documentation, wizard | 2 days | Phase 1-3 |
| **6** | Testing, CI/CD | 1 week | Phase 1-4 |

**Total Estimated Time**: 4-5 weeks

---

## File Structure After Implementation

```
activate-rag-vllm/
├── workflow.yaml              # Unified workflow (replaces 4 files)
├── start_service.sh           # Main entrypoint
├── indexer.py
├── rag_proxy.py
├── rag_server.py
├── indexer_config.yaml
├── README.md                  # Updated with quickstart
├── lib/                       # NEW: Shared functions
│   ├── functions.sh
│   ├── model_manager.sh
│   ├── preflight.sh
│   └── config_validator.py
├── configs/                   # NEW: Configuration presets
│   ├── hpc-presets.yaml
│   └── defaults.yaml
├── singularity/
│   ├── singularity-compose.yml  # Updated
│   ├── Singularity.rag
│   ├── Singularity.vllm
│   └── env.sh.example
├── docker/                    # Retained for local dev
│   └── ...
├── scripts/                   # NEW: Utility scripts
│   └── configure.sh
├── tests/                     # NEW: Test suite
│   └── ...
├── docs/
│   ├── IMPLEMENTATION_PLAN.md # This document
│   ├── ARCHITECTURE.md        # NEW: Architecture docs
│   └── HPC_GUIDE.md           # NEW: HPC deployment guide
└── .github/
    └── workflows/
        └── ci.yml             # NEW: CI/CD
```

---

## Success Criteria

1. ✅ Single `workflow.yaml` handles all deployment modes
2. ✅ Users can specify local model path OR HuggingFace model ID
3. ✅ Git-lfs based HuggingFace downloads work on HPC systems
4. ✅ Pre-flight checks validate configuration before deployment
5. ✅ Clear error messages guide users to resolution
6. ✅ Documentation enables self-service onboarding
7. ✅ Singularity deployment works reliably on HPC clusters

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing workflows | Maintain backward compatibility, gradual rollout |
| HPC network restrictions | Default to offline mode, pre-pull containers |
| Model download failures | Implement retry logic, resume capability |
| GPU detection issues | Explicit `CUDA_VISIBLE_DEVICES` configuration |
| Port conflicts | Dynamic port allocation with conflict detection |

---

## Next Steps

1. **Immediate**: Create feature branch for Phase 1
2. **Week 1**: Complete branch merge and YAML consolidation
3. **Week 2**: Implement model management system
4. **Week 3**: Optimize Singularity deployment
5. **Week 4**: Documentation and testing
6. **Week 5**: User acceptance testing and rollout

---

*Document maintained by: ACTIVATE Team*  
*Last updated: January 16, 2026*
