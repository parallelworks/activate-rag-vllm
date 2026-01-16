# ACTIVATE — vLLM + RAG

Deploy GPU-accelerated language model inference with optional RAG (Retrieval-Augmented Generation) capabilities. Optimized for HPC environments using Singularity, with Docker support for local development.

[![Demo Video](https://www.dropbox.com/scl/fi/xyjf75inw6pa5uk2kyv1p/vllmragthumb.png?rlkey=498wwpesf90nfdon3xj5vyhwy&raw=1)](https://www.youtube.com/watch?v=6LiwXEOkuUc)

## Features

- **vLLM** - High-performance OpenAI-compatible inference server
- **RAG** - Retrieval-augmented generation with ChromaDB
- **Auto-Indexer** - Automatic document indexing with file watching
- **Enhanced Proxy** - OpenAI-compatible API with RAG integration
- **Multi-Scheduler** - Support for SLURM, PBS, and SSH execution
- **Flexible Model Sourcing** - Local models or HuggingFace downloads

## Quick Start

### Option 1: ParallelWorks Workflow (Recommended for HPC)

1. Deploy the workflow from the ParallelWorks marketplace
2. Select your compute cluster and scheduler (SLURM/PBS/SSH)
3. Choose model source:
   - **Local Path**: Point to pre-downloaded model weights
   - **HuggingFace**: Download automatically (requires network access)
4. Configure vLLM options and submit

### Option 2: Local Development

```bash
# Clone the repository
git clone -b refactor https://github.com/parallelworks/activate-rag-vllm.git
cd activate-rag-vllm

# Run the interactive configuration wizard
./scripts/configure.sh

# Start the service
./start_service.sh
```

### Option 3: Manual Configuration

```bash
# Set environment variables
export RUNMODE=singularity  # or docker
export RUNTYPE=all          # or vllm (inference only)
export MODEL_NAME=/path/to/your/model
export HF_TOKEN=hf_xxx      # if using gated models

# Start the service
./start_service.sh
```

## Model Setup

### Using Pre-Downloaded Models (Recommended for HPC)

Models should be pre-staged to a known location. Use git-lfs for reliable downloads:

```bash
# Create model directory
mkdir -p /path/to/models && cd /path/to/models

# Install git-lfs if needed
git lfs install

# Clone model (example: Llama-3.1-8B)
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# For gated models, authenticate first
git clone https://user:${HF_TOKEN}@huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

### Using HuggingFace Hub

When using the "HuggingFace Hub" model source:
1. Provide your HuggingFace token (for gated models like Llama)
2. Specify the model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
3. Set a cache directory with sufficient disk space

The workflow will automatically download the model on first run.

## Container Setup

### Singularity (HPC)

Pre-built containers can be pulled from a bucket:

```bash
# Using ParallelWorks CLI
pw bucket cp pw://mshaxted/codeassist/vllm.sif ./
pw bucket cp pw://mshaxted/codeassist/rag.sif ./
```

Or build locally (requires sudo/fakeroot):

```bash
cd singularity
singularity-compose build
```

### Docker (Local Development)

```bash
# Pull pre-built images
docker pull parallelworks/activate-rag-vllm:latest

# Or build locally
cd docker
docker compose build
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNMODE` | `singularity` | Container runtime (`singularity` or `docker`) |
| `RUNTYPE` | `all` | Deployment type (`all` for vLLM+RAG, `vllm` for inference only) |
| `MODEL_NAME` | - | Model path or HuggingFace ID |
| `VLLM_EXTRA_ARGS` | - | Additional vLLM server arguments |
| `HF_TOKEN` | - | HuggingFace token for gated models |
| `API_KEY` | - | Optional API key for vLLM server |
| `DOCS_DIR` | `./docs` | Directory for RAG documents |

### vLLM Configuration

Common `VLLM_EXTRA_ARGS` options:

```bash
# For 4-GPU setup with bfloat16
--dtype bfloat16 --tensor-parallel-size 4 --gpu-memory-utilization 0.85

# For single GPU with reduced memory
--dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.8

# Full example
export VLLM_EXTRA_ARGS="--dtype bfloat16 --tensor-parallel-size 4 --async-scheduling --max-model-len 16384 --gpu-memory-utilization 0.85 --trust_remote_code"
```

## API Endpoints

Once running, the service exposes OpenAI-compatible endpoints:

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (with RAG if enabled) |
| `POST /v1/completions` | Text completions |
| `GET /v1/models` | List available models |
| `POST /v1/embeddings` | Generate embeddings |
| `GET /health` | Health check |

### Testing the API

```bash
# Health check
curl http://localhost:8081/health | jq

# Chat completion
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }' | jq

# Streaming chat
curl -N http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Explain RAG"}],
    "stream": true
  }'
```

## Project Structure

```
activate-rag-vllm/
├── workflow.yaml              # ParallelWorks workflow definition
├── start_service.sh           # Main service entrypoint
├── rag_proxy.py               # OpenAI-compatible proxy with RAG
├── rag_server.py              # RAG search API server
├── indexer.py                 # Document auto-indexer
├── lib/                       # Shared bash/python utilities
│   ├── functions.sh           # Common bash functions
│   ├── model_manager.sh       # Model download/validation
│   ├── preflight.sh           # Pre-flight checks
│   └── config_validator.py    # Configuration validation
├── configs/                   # Configuration presets
│   └── hpc-presets.yaml       # HPC environment configs
├── scripts/                   # Utility scripts
│   └── configure.sh           # Interactive setup wizard
├── singularity/               # Singularity deployment
│   ├── singularity-compose.yml
│   ├── Singularity.vllm
│   └── Singularity.rag
├── docker/                    # Docker deployment
│   ├── docker-compose.yml
│   └── Dockerfile.rag
└── docs/                      # RAG documents directory
```

## HPC-Specific Notes

### SLURM

```bash
# Example SLURM directives
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --constraint=mla  # For Navy/AFRL DSRC systems
```

### PBS

```bash
# Example PBS directives
#PBS -q gpu
#PBS -l select=1:ncpus=92:mpiprocs=1:ngpus=4
#PBS -l walltime=04:00:00
```

### Offline Mode

For air-gapped HPC environments:
1. Pre-download model weights to a shared location
2. Pre-pull Singularity containers
3. Set `TRANSFORMERS_OFFLINE=1` in your environment
4. Download tiktoken encodings if using GPT-based tokenizers

## Troubleshooting

### Common Issues

1. **"CUDA out of memory"**
   - Reduce `--gpu-memory-utilization` (e.g., 0.7)
   - Reduce `--max-model-len`
   - Use `--tensor-parallel-size` to distribute across GPUs

2. **"Model not found"**
   - Verify the model path exists and contains `config.json`
   - Check if model download completed successfully
   - For gated models, ensure HF_TOKEN is set

3. **"Singularity not found"**
   - Load the module: `module load singularity` or `module load apptainer`

4. **Port already in use**
   - The service will automatically find available ports
   - Check for existing instances: `singularity-compose down`

### Logs

```bash
# View service logs
tail -f logs/vllm.out
tail -f logs/rag.out

# View all logs
tail -F logs/*
```

## Contributing

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for development roadmap and architecture details.

## License

See [LICENSE.md](LICENSE.md)

