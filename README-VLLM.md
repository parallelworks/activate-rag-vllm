# ACTIVATE — vLLM

This Compose stack runs from the [github repo here](https://github.com/parallelworks/activate-rag-vllm) and executes the below services in Docker or Singularity modes:

- **vLLM** model server (OpenAI-compatible)
- **Open WebUI** (optional) pointing to the Proxy

## Quickstart
```bash
export HF_TOKEN=hf_xyz
export RUNMODE=docker # or singularity
export BUILD=true
export RUNTYPE=vllm

# run the service
./run.sh
```

## Files you might care about
- `docker-compose.yml` — stack definition
- `cache/` — workload specific data storage

## Smoke tests
```bash
# Chat (non-stream)
curl -sS http://localhost:${VLLM_PORT}/v1/chat/completions  -H 'content-type: application/json'  -d '{"model":"'"${MODEL_NAME}"'","messages":[{"role":"user","content":"Summarize the docs."}], "max_tokens":200}' | jq

# Chat (stream)
curl -N http://localhost:${VLLM_PORT}/v1/chat/completions  -H 'content-type: application/json'  -d '{"model":"'"${MODEL_NAME}"'","messages":[{"role":"user","content":"Hello"}], "stream": true}'
```
