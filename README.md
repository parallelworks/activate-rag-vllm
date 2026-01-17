# ACTIVATE — vLLM + RAG

Deploy GPU-accelerated language model inference with optional RAG (Retrieval-Augmented Generation) capabilities on the ACTIVATE platform. Optimized for HPC environments using Singularity.

## Overview

This workflow deploys an OpenAI-compatible inference server powered by [vLLM](https://github.com/vllm-project/vllm), with optional RAG capabilities for context-aware responses using your own documents.

```
┌──────────────────────────────────────────────────────────────┐
│                    ACTIVATE Platform                          │
│  ┌────────────────┐   ┌────────────────┐   ┌──────────────┐  │
│  │  Your Browser  │──▶│  RAG Proxy     │──▶│  vLLM Server │  │
│  │  (OpenWebUI,   │   │  (context      │   │  (LLM        │  │
│  │   Cline, etc)  │   │   injection)   │   │   inference) │  │
│  └────────────────┘   └───────┬────────┘   └──────────────┘  │
│                               │                               │
│                       ┌───────▼────────┐                      │
│                       │  RAG Server    │                      │
│                       │  + ChromaDB    │                      │
│                       │  + Indexer     │                      │
│                       └────────────────┘                      │
└──────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| **vLLM Server** | High-performance LLM inference with PagedAttention |
| **RAG Proxy** | OpenAI-compatible API with automatic context injection |
| **RAG Server** | Semantic search over indexed documents |
| **ChromaDB** | Vector database for document embeddings |
| **Auto-Indexer** | Watches document directory and indexes new files |

## Quick Start (ACTIVATE Platform)

### 1. Deploy from Marketplace

1. Navigate to the ACTIVATE workflow marketplace
2. Select **vLLM + RAG** workflow
3. Choose your compute cluster and scheduler (SLURM/PBS/SSH)

### 2. Configure Model Source

Choose how to provide model weights:

| Option | When to Use |
|--------|-------------|
| **📁 Local Path** | Model weights pre-staged on cluster (recommended for HPC) |
| **🤗 HuggingFace** | Download automatically (requires network + token for gated models) |

### 3. Set vLLM Parameters

Common configurations:

```bash
# 4-GPU with bfloat16 (recommended for large models)
--dtype bfloat16 --tensor-parallel-size 4 --gpu-memory-utilization 0.85

# Single GPU with memory constraints
--dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.8
```

### 4. Submit and Connect

- Submit the workflow
- Click the **Open WebUI** link in the job output, or
- Connect your IDE (Cline, Continue, etc.) to the provided endpoint

## Deployment Modes

| Mode | Description |
|------|-------------|
| **vLLM + RAG** | Full stack with document retrieval |
| **vLLM Only** | Inference server without RAG |

## API Endpoints

Once running, the service exposes OpenAI-compatible endpoints:

```bash
# List models
curl http://localhost:8081/v1/models

# Chat completion
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your-model", "messages": [{"role": "user", "content": "Hello!"}]}'

# Health check
curl http://localhost:8081/health
```

## Project Structure

```
activate-rag-vllm/
├── workflow.yaml          # ACTIVATE workflow definition
├── start_service.sh       # Service entrypoint
├── rag_proxy.py           # OpenAI-compatible proxy
├── rag_server.py          # RAG search server
├── indexer.py             # Document indexer
├── run_local.sh           # Local development runner
├── singularity/           # Singularity container configs
├── docker/                # Docker configs (local dev)
├── lib/                   # Shared utilities
├── configs/               # HPC preset configurations
└── docs/                  # Additional documentation
```

## Documentation

| Document | Description |
|----------|-------------|
| [Local Development Guide](docs/LOCAL_DEVELOPMENT.md) | Running locally for debugging |
| [Workflow Configuration](docs/WORKFLOW_CONFIGURATION.md) | YAML workflow customization |
| [Architecture](docs/ARCHITECTURE.md) | System design details |
| [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) | Development roadmap |

## Demo

[![Demo Video](https://www.dropbox.com/scl/fi/xyjf75inw6pa5uk2kyv1p/vllmragthumb.png?rlkey=498wwpesf90nfdon3xj5vyhwy&raw=1)](https://www.youtube.com/watch?v=6LiwXEOkuUc)

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--gpu-memory-utilization` or `--max-model-len` |
| Model not found | Verify path exists with `config.json`; check HF_TOKEN for gated models |
| Singularity not found | Load module: `module load singularity` |
| Port in use | Service auto-finds available ports; check for existing instances |

### Logs

```bash
tail -f logs/vllm.out   # vLLM server
tail -f logs/rag.out    # RAG services
```

## License

See [LICENSE.md](LICENSE.md)

