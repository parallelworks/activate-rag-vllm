# Local Development Guide

This guide covers running the ACTIVATE RAG-vLLM stack locally for development, debugging, and testing.

## Prerequisites

- **Container Runtime**: Apptainer/Singularity or Docker
- **GPU**: NVIDIA GPU with CUDA support
- **Model Weights**: Local path or HuggingFace access

## Quick Start

### Using `run_local.sh`

The `run_local.sh` script provides a streamlined way to run the workflow locally:

```bash
# Clone the repository
git clone https://github.com/parallelworks/activate-rag-vllm.git
cd activate-rag-vllm

# Run with a local model
./run_local.sh --model /path/to/model/weights

# Run with HuggingFace model (downloads if not cached)
./run_local.sh --hf-model meta-llama/Llama-3.1-8B-Instruct --hf-token $HF_TOKEN
```

### Command-Line Options

```bash
./run_local.sh [OPTIONS]

Options:
  --config FILE       Load configuration from file
  --singularity       Force Apptainer/Singularity runtime
  --docker            Force Docker runtime
  --vllm-only         Run vLLM only (no RAG)
  --build             Build containers from source
  --model PATH        Path to local model weights
  --hf-model ID       HuggingFace model ID
  --hf-token TOKEN    HuggingFace API token
  --docs-dir DIR      Directory for RAG documents
  --api-key KEY       API key for vLLM server
  --debug             Enable debug output
  --dry-run           Show configuration without executing
  --help              Show help message
```

### Examples

```bash
# Force Docker runtime
./run_local.sh --docker --model /models/Llama-3.1-8B-Instruct

# vLLM only (no RAG services)
./run_local.sh --vllm-only --model /models/Llama-3.1-8B-Instruct

# Use a configuration file
./run_local.sh --config my-config.env

# Dry run to see what would happen
./run_local.sh --dry-run --model /models/Llama-3.1-8B-Instruct

# Debug mode with verbose output
./run_local.sh --debug --model /models/Llama-3.1-8B-Instruct
```

## Configuration File

Create a configuration file for repeatable setups:

```bash
cp local.env.example my-config.env
```

Edit `my-config.env`:

```bash
# Runtime: auto, singularity, docker
RUNMODE=singularity

# Deployment: all (vLLM + RAG), vllm (inference only)
RUNTYPE=all

# Model configuration
MODEL_SOURCE=local
MODEL_PATH=/models/Llama-3.1-8B-Instruct

# For HuggingFace models:
#MODEL_SOURCE=huggingface
#HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
#HF_TOKEN=hf_xxxxxxxxxxxxx

# Service ports (0 = auto-assign)
VLLM_SERVER_PORT=0
PROXY_PORT=0

# vLLM arguments
VLLM_EXTRA_ARGS="--dtype bfloat16 --trust_remote_code"

# RAG documents directory
DOCS_DIR=./docs
```

## Container Setup

### Apptainer/Singularity

#### Pre-built Containers

```bash
# Download from ParallelWorks bucket
pw bucket cp pw://mshaxted/codeassist/vllm.sif ./
pw bucket cp pw://mshaxted/codeassist/rag.sif ./
```

#### Build Locally

```bash
# Requires sudo or fakeroot
cd singularity
sudo apptainer build ../vllm.sif Singularity.vllm
sudo apptainer build ../rag.sif Singularity.rag

# Or with singularity-compose
singularity-compose build
```

### Docker

```bash
# Pull pre-built images
docker pull parallelworks/activate-rag-vllm:latest

# Or build locally
cd docker
docker compose build
```

## Model Setup

### Local Models

Download models using git-lfs (recommended for HPC compatibility):

```bash
# Install git-lfs
git lfs install

# Clone model repository
cd /path/to/models
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# For gated models, include token in URL
git clone https://user:${HF_TOKEN}@huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

### HuggingFace Hub

When using `--hf-model`, the script will:
1. Check if model is already cached in `~/pw/models`
2. If not, download using git-lfs (preferred) or huggingface-cli
3. Set `MODEL_PATH` to the cached location

## Running the Services

### What Gets Started

| Mode | Services |
|------|----------|
| `all` | vLLM, RAG Proxy, RAG Server, ChromaDB, Indexer |
| `vllm` | vLLM only |

### Endpoints After Startup

```
vLLM API:   http://localhost:{VLLM_SERVER_PORT}/v1
RAG Proxy:  http://localhost:{PROXY_PORT}/v1
RAG Server: http://localhost:{RAG_PORT}
ChromaDB:   http://localhost:{CHROMA_PORT}
```

### Testing

```bash
# Check models
curl http://localhost:8081/v1/models | jq

# Chat completion
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }' | jq

# Streaming
curl -N http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain RAG"}],
    "stream": true
  }'
```

## Debugging

### Logs

```bash
# All logs
tail -F logs/*

# Specific service
tail -f logs/vllm.out
tail -f logs/rag.out
tail -f logs/indexer.out
```

### Common Issues

#### CUDA Out of Memory

```bash
# Reduce GPU memory usage
VLLM_EXTRA_ARGS="--dtype float16 --gpu-memory-utilization 0.7 --max-model-len 4096"
```

#### Model Not Found

```bash
# Verify model directory
ls -la /path/to/model/
# Should contain: config.json, model*.safetensors, tokenizer*, etc.
```

#### Port Already in Use

```bash
# Check what's using the port
lsof -i :8081

# Stop existing services
./cancel.sh  # Generated by run_local.sh

# Or manually
singularity-compose down
docker compose down
```

#### Apptainer/Singularity Compose Not Found

```bash
# Install singularity-compose (works with Apptainer)
python3 -m venv ~/pw/software/singularity-compose
source ~/pw/software/singularity-compose/bin/activate
pip install singularity-compose
```

### Debug Mode

Enable verbose output:

```bash
DEBUG=1 ./run_local.sh --model /path/to/model
```

### Manual Startup

For maximum control during debugging:

```bash
# 1. Create environment file
cat > .run.env << EOF
export RUNMODE=singularity
export RUNTYPE=all
export MODEL_NAME=/models/Llama-3.1-8B-Instruct
export MODEL_PATH=/models/Llama-3.1-8B-Instruct
export VLLM_EXTRA_ARGS="--dtype bfloat16"
export TRANSFORMERS_OFFLINE=1
EOF

# 2. Source and run
source .run.env
./start_service.sh
```

## Interactive Configuration

Use the configuration wizard for guided setup:

```bash
./scripts/configure.sh
```

This will interactively prompt for:
- Model source (local/HuggingFace)
- Model path or ID
- Deployment mode
- vLLM settings
- Port configuration

## Stopping Services

```bash
# Using generated script
./cancel.sh

# Or manually for Apptainer/Singularity
singularity-compose down

# Or for Docker
docker compose down
```

## Development Workflow

1. **Make code changes** to `rag_proxy.py`, `rag_server.py`, or `indexer.py`
2. **Rebuild containers** if needed: `./run_local.sh --build --model ...`
3. **Test endpoints** with curl or your application
4. **Check logs** for errors: `tail -F logs/*`
5. **Iterate** until working

## Next Steps

- [Workflow Configuration](WORKFLOW_CONFIGURATION.md) - Customize the ACTIVATE workflow
- [Architecture](ARCHITECTURE.md) - Understand the system design
