# ACTIVATE — vLLM + RAG

This Compose stack runs from the [github repo here](https://github.com/parallelworks/activate-rag-vllm) and executes the below services in Docker or Singularity modes:

- **vLLM** model server (OpenAI-compatible)
- **RAG** retrieval API (Chroma)
- **Indexer** (filesystem → Chroma, auto-updates)
- **Enhanced Proxy** exposing **/v1/chat/completions**, **/v1/completions**, **/v1/embeddings**, **/v1/models**
- **Open WebUI** (optional) pointing to the Proxy

See a turnkey demonstration of the workflow running on ACTIVATE at the link below:

<a href="https://www.youtube.com/watch?v=6LiwXEOkuUc">
<img target="_blank" src="https://www.dropbox.com/scl/fi/xyjf75inw6pa5uk2kyv1p/vllmragthumb.png?rlkey=498wwpesf90nfdon3xj5vyhwy&raw=1" width="350">
</a>

## Workflow Instructions

Pull down the weights of your choice into a known directory. For example we recommend using git lfs to pull down weights as this is more widely open to firewalls and is relatively fast at pulls:

```
cd /mymodeldir/
git lfs install
git clone https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
```

The workflow will provide a field to also pull down a prebuilt vllm singularity container if running in this mode, but you can also pull this down manually for example using the authenticated pw cli:

```
cd ~/pw/activate-rag-vllm
pw buckets cp pw://mshaxted/codeassist/vllm.sif ./
```

## Manual Quickstart
```bash
export HF_TOKEN=hf_xyz
export RUNMODE=docker # or singularity
export BUILD=true
export RUNTYPE=all # or vllm only

# run the service
./run.sh
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
