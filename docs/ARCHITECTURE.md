# ACTIVATE RAG-vLLM Architecture

This document describes the architecture and design of the ACTIVATE RAG-vLLM deployment system.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interfaces                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   Open WebUI    │  │   Cline/IDE     │  │   Custom Applications       │  │
│  │   (Optional)    │  │   Extensions    │  │   (API Clients)             │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────┬───────────────┘  │
└───────────┼────────────────────┼────────────────────────┼───────────────────┘
            │                    │                        │
            └────────────────────┼────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAG Proxy (Port 8081)                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  OpenAI-Compatible API                                                  │ │
│  │  • POST /v1/chat/completions  (with automatic RAG injection)           │ │
│  │  • POST /v1/completions                                                 │ │
│  │  • GET  /v1/models                                                      │ │
│  │  • POST /v1/embeddings                                                  │ │
│  │  • GET  /health                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Features:                                                                   │
│  • Request preprocessing (RAG context injection)                            │
│  • Streaming response handling                                               │
│  • Citation tracking and formatting                                          │
│  • API key validation (optional)                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐
│     RAG Server (Port 8080)      │   │        vLLM Server (Port 8000)       │
│  ┌───────────────────────────┐  │   │  ┌───────────────────────────────┐  │
│  │  FastAPI Application      │  │   │  │  vLLM OpenAI API Server       │  │
│  │  • POST /search           │  │   │  │  • High-performance inference │  │
│  │  • GET  /health           │  │   │  │  • Tensor parallelism         │  │
│  └─────────────┬─────────────┘  │   │  │  • Continuous batching        │  │
│                │                 │   │  │  • PagedAttention             │  │
│                ▼                 │   │  └───────────────────────────────┘  │
│  ┌───────────────────────────┐  │   │                                      │
│  │  SentenceTransformers     │  │   │  Model: Loaded from MODEL_PATH       │
│  │  (Embedding Generation)   │  │   │  GPU: Distributed via tensor-parallel│
│  └───────────────────────────┘  │   └─────────────────────────────────────┘
│                                 │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐
│   ChromaDB (Port 8001)          │◄──│       Indexer (Background)          │
│  ┌───────────────────────────┐  │   │  ┌───────────────────────────────┐  │
│  │  Vector Database          │  │   │  │  File Watcher (watchdog)      │  │
│  │  • Embeddings storage     │  │   │  │  • Monitors DOCS_DIR          │  │
│  │  • Similarity search      │  │   │  │  • Auto-indexes new files     │  │
│  │  • FTS5 (SQLite)          │  │   │  │  • Handles updates/deletes    │  │
│  └───────────────────────────┘  │   │  │  • Chunk & embed documents    │  │
│                                 │   │  └───────────────────────────────┘  │
│  Persisted to: ./cache/chroma   │   │                                      │
└─────────────────────────────────┘   └─────────────────────────────────────┘
```

## Component Details

### 1. RAG Proxy (`rag_proxy.py`)

The RAG Proxy is the main entry point for all API requests. It provides:

**Request Flow:**
1. Receive OpenAI-compatible request
2. If RAG is enabled and it's a chat completion:
   - Extract the user's query
   - Query RAG Server for relevant context
   - Inject context into the system prompt
3. Forward request to vLLM
4. Process response (add citations if streaming)
5. Return response to client

**Key Features:**
- Full OpenAI API compatibility
- Automatic context injection for RAG
- Streaming support with citation tracking
- System prompt customization
- API key authentication (optional)

### 2. RAG Server (`rag_server.py`)

The RAG Server provides semantic search over indexed documents:

**Endpoints:**
- `POST /search` - Search for relevant documents
- `GET /health` - Health check

**Components:**
- SentenceTransformers for embedding generation
- ChromaDB client for vector search
- Configurable TOP_K results
- Score-based filtering

### 3. vLLM Server

The vLLM server provides high-performance LLM inference:

**Features:**
- OpenAI-compatible API
- PagedAttention for efficient memory use
- Continuous batching for throughput
- Tensor parallelism for large models
- Supports various attention backends

**Configuration via `VLLM_EXTRA_ARGS`:**
- `--dtype` - Data type (float16, bfloat16)
- `--tensor-parallel-size` - Number of GPUs
- `--max-model-len` - Maximum context length
- `--gpu-memory-utilization` - GPU memory fraction

### 4. ChromaDB

Vector database for document embeddings:

- Persistent storage in `./cache/chroma`
- SQLite-based with FTS5 extension
- Efficient similarity search
- Automatic deduplication

### 5. Indexer (`indexer.py`)

Background service for document indexing:

**Process:**
1. Watch `DOCS_DIR` for changes
2. On new/modified file:
   - Load document (PDF, TXT, MD, etc.)
   - Split into chunks
   - Generate embeddings
   - Store in ChromaDB
3. On delete: Remove from ChromaDB

**Supported Formats:**
- PDF, TXT, MD, RST
- Python, JavaScript, YAML, JSON
- HTML, XML

## Deployment Modes

### Apptainer/Singularity (HPC)

```
┌─────────────────────────────────────────────────────────────────┐
│                     HPC Compute Node                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  vllm.sif (Apptainer Container)                          │    │
│  │  • vLLM Server                                          │    │
│  │  • GPU access via --nv                                  │    │
│  │  • Model mounted from host                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  rag.sif (Apptainer Container)                           │    │
│  │  • RAG Proxy                                            │    │
│  │  • RAG Server                                           │    │
│  │  • ChromaDB                                             │    │
│  │  • Indexer                                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Volumes:                                                        │
│  • ./logs:/logs                                                  │
│  • ./cache:/root/.cache                                          │
│  • ./docs:/docs                                                  │
│  • /path/to/model:/__model__                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- No root required to run
- Reproducible environments
- Works in air-gapped environments
- Compatible with job schedulers

### Docker (Development)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  vllm (Container)                                       │    │
│  │  • Official vLLM image                                  │    │
│  │  • GPU via nvidia-docker                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  rag (Container)                                        │    │
│  │  • Built from Dockerfile.rag                            │    │
│  │  • Includes all RAG components                          │    │
│  │  • depends_on: vllm                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Network: ragnet (internal Docker network)                       │
│  Ports exposed to host: PROXY_PORT only                          │
└─────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Easy local development
- Built-in dependency management
- Health checks
- Automatic restart

## Workflow Integration

### ParallelWorks Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                   ParallelWorks Platform                         │
│                                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌──────────────┐   │
│  │  workflow.yaml │───▶│  Job Scheduler │───▶│  Compute     │   │
│  │  (UI Form)     │    │  (SLURM/PBS)   │    │  Node        │   │
│  └────────────────┘    └────────────────┘    └──────┬───────┘   │
│                                                      │           │
│  ┌────────────────┐                                 │           │
│  │  Session       │◀────────────────────────────────┘           │
│  │  Management    │                                             │
│  │  (Port Tunnel) │                                             │
│  └────────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Workflow Steps:**
1. **prepare_job_directory** - Clone repo, create config
2. **[slurm|pbs|ssh]_job** - Submit/run job
3. **create_session** - Wait for service, expose port

## Data Flow

### RAG-Augmented Chat Request

```
User Query: "What does the documentation say about X?"
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  RAG Proxy                                                       │
│  1. Extract query from messages                                  │
│  2. Call RAG Server: POST /search                                │
│     └─▶ Returns relevant chunks with metadata                    │
│  3. Format context blocks: [1], [2], [3]...                      │
│  4. Inject into system prompt:                                   │
│     "Context:\n[1] source.md: <chunk>\n[2] doc.pdf: <chunk>"    │
│  5. Forward modified request to vLLM                             │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  vLLM Server                                                     │
│  • Process augmented prompt                                      │
│  • Generate response with citations                              │
│  • Stream tokens back                                            │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Response to User                                                │
│  "Based on the documentation [1], X is... [2] Additionally..."  │
│                                                                  │
│  References:                                                     │
│  [1] source.md (chunk 3)                                         │
│  [2] doc.pdf (chunk 7)                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Document Indexing Flow

```
New File: research_paper.pdf
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Indexer (File Watcher)                                          │
│  1. Detect file creation event                                   │
│  2. Load PDF with document loader                                │
│  3. Extract text content                                         │
│  4. Split into chunks (default: 512 tokens, 50 overlap)         │
│  5. Generate embeddings for each chunk                           │
│  6. Store in ChromaDB with metadata:                             │
│     - source: "research_paper.pdf"                               │
│     - chunk_index: 0, 1, 2...                                    │
│     - timestamp: "2026-01-16T..."                                │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  ChromaDB                                                        │
│  Collection: documents                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ID        │ Embedding        │ Metadata                     ││
│  │───────────│──────────────────│────────────────────────────────│
│  │ doc_0_0   │ [0.1, 0.2, ...]  │ {source: "research...", ...} ││
│  │ doc_0_1   │ [0.3, 0.1, ...]  │ {source: "research...", ...} ││
│  │ ...       │ ...              │ ...                          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Reference

### Environment Variables

| Variable | Component | Description |
|----------|-----------|-------------|
| `MODEL_NAME` | vLLM | Model path or HuggingFace ID |
| `VLLM_EXTRA_ARGS` | vLLM | Additional server arguments |
| `VLLM_SERVER_PORT` | vLLM | Server port (default: 8000) |
| `PROXY_PORT` | RAG Proxy | Proxy port (default: 8081) |
| `RAG_PORT` | RAG Server | RAG API port (default: 8080) |
| `CHROMA_PORT` | ChromaDB | Database port (default: 8001) |
| `DOCS_DIR` | Indexer | Documents directory |
| `TOP_K` | RAG Server | Number of context chunks |
| `EMBEDDING_MODEL` | RAG Server | Embedding model name |
| `HF_TOKEN` | All | HuggingFace authentication |
| `API_KEY` | vLLM | Optional API authentication |

### File Locations

| Path | Purpose |
|------|---------|
| `./logs/` | Service logs |
| `./cache/` | HuggingFace cache, model cache |
| `./cache/chroma/` | ChromaDB persistence |
| `./docs/` | RAG documents |
| `.run.env` | Runtime environment |
| `env.sh` | Apptainer/Singularity environment |
| `.env` | Docker environment |

## Security Considerations

### API Authentication

- Optional API key via `API_KEY` environment variable
- When set, all requests require `Authorization: Bearer <key>` header
- Recommended for production deployments

### Network Security

- Default: All ports bound to localhost (127.0.0.1)
- ParallelWorks manages port tunneling securely
- No direct external access to vLLM or RAG servers

### Model Access

- Gated models require HuggingFace token
- Token should be stored securely (org secrets in ParallelWorks)
- Use `TRANSFORMERS_OFFLINE=1` in air-gapped environments

## Performance Tuning

### vLLM Optimization

```bash
# Multi-GPU setup
--tensor-parallel-size 4

# Memory optimization
--gpu-memory-utilization 0.9
--max-model-len 8192

# Throughput optimization
--async-scheduling
```

### RAG Optimization

- Adjust `TOP_K` based on context window
- Use smaller embedding models for faster indexing
- Enable `CHROMA_AUTO_RESET` for development

### Resource Requirements

| Model Size | Min GPUs | Min VRAM | Recommended |
|------------|----------|----------|-------------|
| 7-8B | 1 | 16GB | A10, RTX 3090 |
| 13B | 1-2 | 24GB | A10, RTX 4090 |
| 70B | 4 | 80GB | A100-80GB |
| 405B | 8+ | 320GB+ | H100 cluster |
