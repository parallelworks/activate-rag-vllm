
import os
import json
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
import httpx
from fastapi import FastAPI, Body, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

RAG_URL = os.getenv("RAG_URL", "http://rag:8080")
VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "8192"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "2"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "10"))

_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    return _tokenizer

def token_len(messages: List[Dict[str, Any]]) -> int:
    tok = get_tokenizer()
    try:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return len(ids)
    except Exception:
        text = ""
        for m in messages:
            text += f"[{m.get('role','').upper()}]\n{m.get('content','')}\n"
        return len(tok.encode(text))

def pack_messages(system_prompt: str, user_query: str, chunks: List[str],
                  max_context: int, max_completion_tokens: int = 256,
                  per_chunk_header: str = "\n\n[CONTEXT]\n") -> Tuple[List[Dict[str, str]], int]:
    base = [{"role":"system","content":system_prompt},
            {"role":"user","content":user_query}]
    base_len = token_len(base)
    reserve = max_completion_tokens + 64
    budget = max_context - reserve
    remaining = budget - base_len
    used = 0
    for ch in chunks:
        candidate_user = user_query + per_chunk_header + ch
        cand = [{"role":"system","content":system_prompt},
                {"role":"user","content":candidate_user}]
        cand_len = token_len(cand)
        delta = cand_len - base_len
        if delta <= remaining:
            user_query = candidate_user
            base_len = cand_len
            remaining -= delta
            used += 1
        else:
            break
    return [{"role":"system","content":system_prompt},
            {"role":"user","content":user_query}], used

app = FastAPI(title="RAG→vLLM OpenAI Proxy (Plus v2)", version="1.3")
async_client = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT, connect=CONNECT_TIMEOUT))

@app.on_event("shutdown")
async def _shutdown():
    await async_client.aclose()

def _headers_with_auth(extra: Optional[Dict[str,str]] = None) -> Dict[str,str]:
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    if extra: headers.update(extra)
    return headers

async def _rag_search(query: str, k: int, file_contains: Optional[str]=None) -> List[Dict[str, Any]]:
    params = {"query": query, "top_k": k}
    if file_contains:
        params["file_contains"] = file_contains
    try:
        r = await async_client.get(f"{RAG_URL}/search", params=params)
        r.raise_for_status()
        js = r.json()
        return js.get("results", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail={"error":"RAG fetch failed","detail":str(e),"rag_url":RAG_URL})

def _extract_user_query(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content","")
    return ""

def _rag_cfg(req_json: Dict[str, Any], headers: Dict[str,str]) -> Dict[str, Any]:
    cfg = {"enabled": True, "top_k": TOP_K_DEFAULT, "system_prompt":"Answer using the provided context when relevant, and please cite sources with the file_path. If unsure, say so.", "file_contains": None}
    #cfg = {"enabled": True, "top_k": TOP_K_DEFAULT, "system_prompt":"Use only the provided [CONTEXT] snippets. Cite file_path:chunk_index for any claim.", "file_contains": None}
    if isinstance(req_json.get("rag"), dict):
        for k in cfg.keys():
            if k in req_json["rag"]:
                cfg[k] = req_json["rag"][k]
    hv = headers.get("x-rag-enabled")
    if hv is not None:
        cfg["enabled"] = hv.strip() not in ("0","false","False","no","off")
    hv = headers.get("x-rag-top-k")
    if hv:
        try: cfg["top_k"] = max(1, min(50, int(hv)))
        except: pass
    hv = headers.get("x-rag-system-prompt")
    if hv: cfg["system_prompt"] = hv
    hv = headers.get("x-rag-file-contains")
    if hv: cfg["file_contains"] = hv
    return cfg

def _passthrough(src: Dict[str, Any], allowed: List[str]) -> Dict[str, Any]:
    return {k:v for k,v in src.items() if k in allowed and v is not None}

@app.get("/v1/models")
async def list_models():
    try:
        r = await async_client.get(f"{VLLM_URL}/models", headers=_headers_with_auth())
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type","application/json"))
    except Exception as e:
        raise HTTPException(502, {"error": str(e), "vllm_url": VLLM_URL})

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    try:
        body = await request.body()
        r = await async_client.post(f"{VLLM_URL}/embeddings", content=body, headers=_headers_with_auth())
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type","application/json"))
    except Exception as e:
        raise HTTPException(502, {"error": str(e), "vllm_url": VLLM_URL})

ALLOWED_COMPLETION_FIELDS = [
    "model","prompt","suffix","max_tokens","temperature","top_p","n","stream","logprobs",
    "echo","stop","presence_penalty","frequency_penalty","best_of","logit_bias","user","seed","response_format"
]

@app.post("/v1/completions")
async def completions(request: Request):
    req_json = await request.json()
    headers = dict(request.headers)
    rag_cfg = _rag_cfg(req_json, headers)
    stream = bool(req_json.get("stream", False))

    prompt = req_json.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[-1] if prompt else ""

    used = 0
    results: List[Dict[str, Any]] = []
    if rag_cfg["enabled"]:
        results = await _rag_search(prompt, rag_cfg["top_k"], rag_cfg.get("file_contains"))
        chunks = [x["chunk_text"] for x in results]
        messages, used = pack_messages(rag_cfg["system_prompt"], prompt, chunks, MAX_CONTEXT, req_json.get("max_tokens") or DEFAULT_MAX_TOKENS)
        flat = ""
        for m in messages:
            flat += f"[{m['role'].upper()}]\n{m['content']}\n"
        prompt = flat

    payload = _passthrough(req_json, ALLOWED_COMPLETION_FIELDS)
    payload["prompt"] = prompt
    payload.setdefault("model", MODEL_NAME)
    url = f"{VLLM_URL}/completions"

    if stream:
        async def gen() -> AsyncGenerator[bytes, None]:
            try:
                async with async_client.stream("POST", url, json=payload, headers=_headers_with_auth()) as resp:
                    async for chunk in resp.aiter_raw():
                        yield chunk
            except Exception as e:
                err = json.dumps({"error": str(e), "vllm_url": VLLM_URL}).encode("utf-8")
                yield err
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        r = await async_client.post(url, json=payload, headers=_headers_with_auth())
    except Exception as e:
        raise HTTPException(502, {"error": str(e), "vllm_url": VLLM_URL})
    data = None
    try:
        data = r.json()
    except Exception:
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type","application/json"))
    if isinstance(data, dict):
        citations = [{"file_path":x["metadata"]["file_path"], "chunk_index":x["metadata"]["chunk_index"]} for x in results[:used]]
        data["_rag"] = {"used_chunks": used, "citations": citations, "enabled": bool(rag_cfg["enabled"])}
    return JSONResponse(status_code=r.status_code, content=data)

ALLOWED_CHAT_FIELDS = [
    "model","messages","max_tokens","temperature","top_p","n","stream","stop","presence_penalty",
    "frequency_penalty","logit_bias","user","tools","tool_choice","seed","response_format","logprobs"
]

class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[Any] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[Any] = None
    tool_choice: Optional[Any] = None
    seed: Optional[int] = None
    response_format: Optional[Any] = None
    logprobs: Optional[Any] = None
    rag: Optional[Dict[str, Any]] = None

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatReq = Body(...), request: Request = None):
    headers = dict(request.headers) if request else {}
    rag_cfg = _rag_cfg(req.dict(exclude_none=True), headers)
    stream = bool(req.stream)

    messages = req.messages or []
    user_query = _extract_user_query(messages)

    used = 0
    results: List[Dict[str, Any]] = []
    final_messages = messages

    if rag_cfg["enabled"]:
        results = await _rag_search(user_query, rag_cfg["top_k"], rag_cfg.get("file_contains"))
        chunks = [x["chunk_text"] for x in results]
        max_tok = req.max_tokens or DEFAULT_MAX_TOKENS
        final_messages, used = pack_messages(rag_cfg["system_prompt"], user_query, chunks, MAX_CONTEXT, max_tok)

    payload = _passthrough(req.dict(exclude_none=True), ALLOWED_CHAT_FIELDS)
    payload["messages"] = final_messages
    payload.setdefault("model", req.model or MODEL_NAME)
    url = f"{VLLM_URL}/chat/completions"

    if stream:
        async def gen() -> AsyncGenerator[bytes, None]:
            try:
                async with async_client.stream("POST", url, json=payload, headers=_headers_with_auth()) as resp:
                    # Use upstream content-type if available
                    ctype = resp.headers.get("content-type", "text/event-stream")
                    # We ignore ctype here because StreamingResponse is created outside; chunks are forwarded raw
                    async for chunk in resp.aiter_raw():
                        yield chunk
            except Exception as e:
                err = json.dumps({"error": str(e), "vllm_url": VLLM_URL}).encode("utf-8")
                yield err
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        r = await async_client.post(url, json=payload, headers=_headers_with_auth())
    except Exception as e:
        raise HTTPException(502, {"error": str(e), "vllm_url": VLLM_URL})
    data = None
    try:
        data = r.json()
    except Exception:
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type","application/json"))
    if isinstance(data, dict):
        citations = [{"file_path":x["metadata"]["file_path"], "chunk_index":x["metadata"]["chunk_index"]} for x in results[:used]]
        data["_rag"] = {"used_chunks": used, "citations": citations, "enabled": bool(rag_cfg["enabled"])}
    return JSONResponse(status_code=r.status_code, content=data)

@app.get("/health")
async def health():
    out = {"model": MODEL_NAME, "vllm_url": VLLM_URL, "rag_url": RAG_URL,
           "max_context": MAX_CONTEXT, "default_max_tokens": DEFAULT_MAX_TOKENS,
           "temperature": DEFAULT_TEMPERATURE}
    try:
        vr = await async_client.get(f"{VLLM_URL}/models", headers=_headers_with_auth())
        out["vllm_ok"] = (vr.status_code == 200)
        try:
            out["models"] = vr.json()
        except Exception:
            out["models"] = {"status_code": vr.status_code}
    except Exception as e:
        out["vllm_ok"] = False
        out["vllm_error"] = str(e)
    try:
        rr = await async_client.get(f"{RAG_URL}/health")
        out["rag_ok"] = (rr.status_code == 200)
    except Exception as e:
        out["rag_ok"] = False
        out["rag_error"] = str(e)
    return out

@app.get("/debug/probe")
async def debug_probe():
    res = {}
    try:
        r = await async_client.get(f"{VLLM_URL}/models", headers=_headers_with_auth())
        res["vllm_models_status"] = r.status_code
        try:
            res["vllm_models"] = r.json()
        except Exception:
            res["vllm_models"] = r.text
    except Exception as e:
        res["vllm_error"] = str(e)
    try:
        r = await async_client.get(f"{RAG_URL}/search", params={"query":"ping","top_k":1})
        res["rag_search_status"] = r.status_code
        try:
            res["rag_search"] = r.json()
        except Exception:
            res["rag_search"] = r.text
    except Exception as e:
        res["rag_error"] = str(e)
    return res

@app.get("/")
def root():
    return {"service": "RAG→vLLM OpenAI Proxy (Plus v2)", "version": "1.3"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PROXY_PORT","8081")))
