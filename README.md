# ACTIVATE — vLLM + RAG + Proxy + Optional OpenWebUI

This Compose stack runs from the [github repo here](https://github.com/parallelworks/activate-rag-vllm-compose) and executes the below services in Docker or Singularity modes:

- **vLLM** model server (OpenAI-compatible)
- **RAG** retrieval API (Chroma)
- **Indexer** (filesystem → Chroma, auto-updates)
- **Enhanced Proxy** exposing **/v1/chat/completions**, **/v1/completions**, **/v1/embeddings**, **/v1/models**
- **Open WebUI** (optional) pointing to the Proxy

## Quickstart
```bash
# 1) Review and copy env
cp env.example env/env.sh
source env/env.sh

pip3 install singularity-compose
singularity-compose build

mkdir -p cache cache/chroma docs

singularity instance start 


# 2) Build RAG/Proxy image (vLLM & WebUI pulled automatically)
if [ "$RUN_OPENWEBUI" = "1" ]; then
    echo "building vllm, rag and openwebui"
    docker compose -f docker-compose.yml -f docker-compose.openwebui.yml build
else
    echo "building vllm and rag"
    docker compose build
fi

# 3) Launch
if [ "$RUN_OPENWEBUI" = "1" ]; then
    echo "running vllm, rag and openwebui"
    docker compose -f docker-compose.yml -f docker-compose.openwebui.yml up -d
else
    echo "running vllm and rag"
    docker compose up -d
fi

# 4) URLs
# Proxy:     http://localhost:${PROXY_PORT}/health
# OpenWebUI: http://localhost:${OPENWEBUI_PORT}
```

## Files you might care about
- `docker-compose.yml` — stack definition
- `Dockerfile.rag` — builds the RAG + Indexer + Proxy image
- `rag_proxy.py` — enhanced OpenAI-compatible proxy with streaming + extra endpoints
- `rag_server.py` — RAG search API
- `indexer.py`, `indexer_config.yaml` — auto indexer for filesystem changes
- `docs/` — mount point for your documents
- `cache/` — workload specific data storage

## Smoke tests
```bash
# Health
curl http://localhost:${PROXY_PORT}/health | jq

# Chat (non-stream)
curl -sS http://localhost:${PROXY_PORT}/v1/chat/completions  -H 'content-type: application/json'  -d '{"model":"'"${MODEL_NAME}"'","messages":[{"role":"user","content":"Summarize the docs."}], "max_tokens":200}' | jq

# Chat (stream)
curl -N http://localhost:${PROXY_PORT}/v1/chat/completions  -H 'content-type: application/json'  -d '{"model":"'"${MODEL_NAME}"'","messages":[{"role":"user","content":"Hello"}], "stream": true}'
```
