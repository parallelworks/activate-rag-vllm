import os
import re
import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
import httpx
from fastapi import FastAPI, Body, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# =========================
# Core config
# =========================
RAG_URL = os.getenv("RAG_URL", "http://127.0.0.1:8080")
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "8192"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "4"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "10"))
RAG_FAIL_OPEN = os.getenv("RAG_FAIL_OPEN", "true").lower() in ("1", "true", "yes", "on")

LOG = logging.getLogger("rag_proxy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =========================
# Revised system directive
# =========================
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT","You are a careful assistant. Use ONLY the provided context blocks to answer. Each block is numbered [1], [2], … and includes source metadata. When you use information from a block, you MUST cite it inline with [n]. At the end of your response, include a 'References:' section with one reference per line formatted as: [n] file_path (chunk index). Do not invent citations or sources. If the context does not contain the answer, say so briefly.")

# =========================
# Tokenizer helpers
# =========================
_tokenizer = None
_tokenizer_loaded = False

def get_tokenizer():
    """
    Try to load tokenizer with multiple fallbacks:
    1. Fast tokenizer (use_fast=True)
    2. Slow tokenizer (use_fast=False)
    3. None (will use char-based estimation in token_len)
    """
    global _tokenizer, _tokenizer_loaded
    if _tokenizer_loaded:
        return _tokenizer

    _tokenizer_loaded = True

    # Try fast tokenizer first
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        LOG.info("Loaded fast tokenizer for %s", MODEL_NAME)
        return _tokenizer
    except Exception as e:
        LOG.warning("Fast tokenizer failed for %s: %s", MODEL_NAME, e)

    # Try slow tokenizer
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        LOG.info("Loaded slow tokenizer for %s", MODEL_NAME)
        return _tokenizer
    except Exception as e:
        LOG.warning("Slow tokenizer also failed for %s: %s", MODEL_NAME, e)

    # All tokenizer loading failed - will use char-based estimation
    LOG.warning("All tokenizer loading attempts failed for %s. Using character-based estimation.", MODEL_NAME)
    _tokenizer = None
    return None

def token_len(messages: List[Dict[str, Any]]) -> int:
    """
    Estimate token count for messages.
    Uses tokenizer if available, otherwise falls back to character-based estimation.
    """
    tok = get_tokenizer()

    # Build text representation
    text = ""
    for m in messages:
        text += f"[{m.get('role','').upper()}]\n{m.get('content','')}\n"

    if tok is None:
        # Fallback: estimate ~4 chars per token (common approximation for English)
        return len(text) // 4

    try:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return len(ids)
    except Exception:
        try:
            return len(tok.encode(text))
        except Exception:
            # Final fallback to char-based estimation
            return len(text) // 4

# =========================
# Legacy packer (kept)
# =========================
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

# =========================
# NEW: Numbered context & citations
# =========================
CITE_RE = re.compile(r"\[(\d{1,3})\]")  # [1], [23], etc.

def build_context_and_map(results: List[Dict[str, Any]], max_chars: int = 8000
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Produce a numbered context block and a parallel citation map.
    Each result is expected to carry metadata with at least file_path & chunk_index.
    """
    blocks: List[str] = []
    citations: List[Dict[str, Any]] = []
    total = 0

    for i, r in enumerate(results, start=1):
        meta = r.get("metadata") or {}
        fp = meta.get("file_path") or "unknown"
        idx = meta.get("chunk_index")
        title = meta.get("title") or fp.split("/")[-1]
        sha = meta.get("doc_sha256")
        sim = r.get("similarity")
        head = f"[{i}] {title} — {fp}"
        if idx is not None:
            head += f" (chunk {idx})"
        if sim is not None:
            try:
                head += f"  • sim={float(sim):.3f}"
            except Exception:
                pass
        text = (r.get("chunk_text") or "").strip()
        block = f"{head}\n<<<\n{text}\n>>>"
        if total + len(block) > max_chars:
            break
        total += len(block)
        blocks.append(block)
        citations.append({
            "n": i,
            "file_path": fp,
            "chunk_index": idx,
            "title": title,
            "doc_sha256": sha,
            "similarity": sim
        })

    return "\n\n".join(blocks), citations

def extract_used_citations(text: str, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = {c["n"]: c for c in citations}
    used_order: List[Dict[str, Any]] = []
    seen = set()
    for m in CITE_RE.finditer(text or ""):
        n = int(m.group(1))
        if n in index and n not in seen:
            used_order.append(index[n])
            seen.add(n)
    return used_order

def _pack_numbered_chat(user_query: str, context_block: str,
                        max_context: int, max_completion_tokens: int) -> List[Dict[str, str]]:
    """System + user (with numbered context). Trims context if needed."""
    system_prompt = SYSTEM_PROMPT
    user_preface = (
        "Context blocks (cite with [n] when used):\n\n"
        f"{context_block}\n\n"
        "Now answer the user's question using ONLY the context above. "
        "Cite sources inline as [n].\n\n"
        f"User question: {user_query}"
    )
    msgs = [{"role":"system","content":system_prompt},
            {"role":"user","content":user_preface}]
    # crude trim loop if needed
    while token_len(msgs) > (max_context - max_completion_tokens - 64) and len(context_block) > 200:
        # drop last ~20% of the context
        trim_to = int(len(context_block) * 0.8)
        context_block = context_block[:trim_to]
        msgs[1]["content"] = (
            "Context blocks (cite with [n] when used):\n\n"
            f"{context_block}\n\n"
            "Now answer the user's question using ONLY the context above. "
            "Cite sources inline as [n].\n\n"
            f"User question: {user_query}"
        )
    return msgs

def _pack_numbered_prompt(user_query: str, context_block: str,
                          max_context: int, max_completion_tokens: int) -> str:
    """
    Build a flat prompt for /v1/completions that still encodes numbered context and the directive.
    """
    system = SYSTEM_PROMPT
    prompt = (
        f"[SYSTEM]\n{system}\n\n"
        "Context blocks (cite with [n] when used):\n\n"
        f"{context_block}\n\n"
        f"[USER]\n{user_query}\n\n"
        "Answer using ONLY the context above and cite sources inline as [n]."
    )
    # simple trimming loop (by chars) if tokenizer is unavailable
    msgs = [{"role":"system","content":system},{"role":"user","content":prompt}]
    while token_len(msgs) > (max_context - max_completion_tokens - 64) and len(context_block) > 200:
        context_block = context_block[:int(len(context_block) * 0.8)]
        prompt = (
            f"[SYSTEM]\n{system}\n\n"
            "Context blocks (cite with [n] when used):\n\n"
            f"{context_block}\n\n"
            f"[USER]\n{user_query}\n\n"
            "Answer using ONLY the context above and cite sources inline as [n]."
        )
        msgs = [{"role":"system","content":system},{"role":"user","content":prompt}]
    return prompt

# =========================
# FastAPI app
# =========================
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

def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)

# =========================
# RAG fetch
# =========================
async def _rag_search(query: str, k: int, file_contains: Optional[str]=None) -> List[Dict[str, Any]]:
    params = {"query": query, "top_k": k}
    if file_contains:
        params["file_contains"] = file_contains
    try:
        r = await async_client.get(f"{RAG_URL}/search", params=params)
        r.raise_for_status()
        js = r.json()
        return js.get("results", [])
    except httpx.HTTPStatusError as e:
        detail = {"error": "RAG fetch failed", "detail": str(e), "status": e.response.status_code,
                  "body": e.response.text, "rag_url": RAG_URL}
        LOG.warning("RAG fetch HTTP error: %s", detail)
        raise HTTPException(status_code=502, detail=detail)
    except Exception as e:
        detail = {"error": "RAG fetch failed", "detail": str(e), "rag_url": RAG_URL}
        LOG.warning("RAG fetch error: %s", detail)
        raise HTTPException(status_code=502, detail=detail)

def _extract_user_query(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content","")
    return ""

def _rag_cfg(req_json: Dict[str, Any], headers: Dict[str,str]) -> Dict[str, Any]:
    # default config; system prompt remains available but the proxy will inject SYSTEM_PROMPT too
    cfg = {"enabled": True, "top_k": TOP_K_DEFAULT,
           "system_prompt": "Use only the provided context; cite sources with [n]. If absent, say so.",
           "file_contains": None}
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

# =========================
# Model list / embeddings
# =========================
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

# =========================
# /v1/completions (kept)
# =========================
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
    citations_all: List[Dict[str, Any]] = []
    context_block = ""
    user_query = prompt  # completions => prompt is the user question

    if rag_cfg["enabled"] and user_query.strip():
        try:
            results = await _rag_search(user_query, rag_cfg["top_k"], rag_cfg.get("file_contains"))
            # Build numbered context/citation map
            context_block, citations_all = build_context_and_map(results)
            # Build a numbered prompt for completions
            prompt = _pack_numbered_prompt(
                user_query=user_query,
                context_block=context_block,
                max_context=MAX_CONTEXT,
                max_completion_tokens=req_json.get("max_tokens") or DEFAULT_MAX_TOKENS
            )
        except HTTPException as e:
            if RAG_FAIL_OPEN:
                LOG.warning("RAG disabled for this request due to error: %s", e.detail)
                rag_cfg["enabled"] = False
            else:
                raise

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
        detail = {"error": str(e), "vllm_url": VLLM_URL}
        LOG.warning("vLLM request error: %s", detail)
        raise HTTPException(502, detail)

    # Try to attach citations metadata
    try:
        data = r.json()
    except Exception:
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type","application/json"))

    if isinstance(data, dict):
        out_text = ""
        try:
            out_text = data["choices"][0].get("text","") or data["choices"][0].get("message",{}).get("content","")
        except Exception:
            pass

        citations_used = extract_used_citations(out_text, citations_all) if citations_all else []
        data["_rag"] = {
            "enabled": bool(rag_cfg["enabled"]),
            "query": user_query,
            "citations_all": citations_all,
            "citations_used": citations_used,
            "context_blocks": context_block
        }

        # Prefer used citations; fallback to all
        refs_text = _format_references(citations_used or citations_all)
        _append_refs_to_openai_response(data, refs_text)
    return JSONResponse(status_code=r.status_code, content=data)

# =========================
# /v1/chat/completions (kept)
# =========================
ALLOWED_CHAT_FIELDS = [
    "model","messages","max_tokens","temperature","top_p","n","stream","stop","presence_penalty",
    "frequency_penalty","logit_bias","user","tools","tool_choice","seed","response_format","logprobs"
]

def _format_references(citations: List[Dict[str, Any]]) -> str:
    """
    Build a multi-line 'References:' block.
    Ensures one reference per line, with stable numbering.
    """
    if not citations:
        return ""
    lines = ["References:"]
    for i, c in enumerate(citations, start=1):
        n = c.get("n") or i
        fp = c.get("file_path") or "?"
        idx = c.get("chunk_index")
        if idx is None:
            lines.append(f"[{n}] {fp}")
        else:
            lines.append(f"[{n}] {fp} (chunk {idx})")
    # Return without leading/trailing blank lines; caller adds spacing
    return "\n".join(lines)

def _append_refs_to_openai_response(data: Dict[str, Any], refs_text: str) -> None:
    """
    Append references to OpenAI-compatible responses in-place.
    Adds two newlines before refs and a trailing newline for cleanliness.
    Handles both /v1/completions and /v1/chat/completions shapes.
    """
    if not refs_text or "choices" not in data or not data["choices"]:
        return
    block = "\n\n" + refs_text + "\n"
    ch0 = data["choices"][0]
    # /v1/completions
    if isinstance(ch0.get("text"), str):
        ch0["text"] += block
        return
    # /v1/chat/completions
    msg = ch0.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        msg["content"] += block

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
    req_json = _model_to_dict(req)
    rag_cfg = _rag_cfg(req_json, headers)
    stream = bool(req.stream)

    messages = req.messages or []
    user_query = _extract_user_query(messages)

    results: List[Dict[str, Any]] = []
    citations_all: List[Dict[str, Any]] = []
    context_block = ""
    final_messages = messages

    if rag_cfg["enabled"] and user_query.strip():
        try:
            results = await _rag_search(user_query, rag_cfg["top_k"], rag_cfg.get("file_contains"))
            context_block, citations_all = build_context_and_map(results)
            # Build numbered chat messages with trimming if needed
            final_messages = _pack_numbered_chat(
                user_query=user_query,
                context_block=context_block,
                max_context=MAX_CONTEXT,
                max_completion_tokens=(req.max_tokens or DEFAULT_MAX_TOKENS)
            )
        except HTTPException as e:
            if RAG_FAIL_OPEN:
                LOG.warning("RAG disabled for this request due to error: %s", e.detail)
                rag_cfg["enabled"] = False
            else:
                raise

    payload = _passthrough(req_json, ALLOWED_CHAT_FIELDS)
    payload["messages"] = final_messages
    payload.setdefault("model", req.model or MODEL_NAME)
    url = f"{VLLM_URL}/chat/completions"

    if stream:
        async def gen() -> AsyncGenerator[bytes, None]:
            try:
                async with async_client.stream("POST", url, json=payload, headers=_headers_with_auth()) as resp:
                    # forward raw SSE; citations are only attached in non-stream mode
                    async for chunk in resp.aiter_raw():
                        yield chunk
            except Exception as e:
                err = json.dumps({"error": str(e), "vllm_url": VLLM_URL}).encode("utf-8")
                yield err
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        r = await async_client.post(url, json=payload, headers=_headers_with_auth())
    except Exception as e:
        detail = {"error": str(e), "vllm_url": VLLM_URL}
        LOG.warning("vLLM request error: %s", detail)
        raise HTTPException(502, detail)

    # Attach citations metadata
    try:
        data = r.json()
    except Exception:
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type","application/json"))

    if isinstance(data, dict):
        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            pass

        citations_used = extract_used_citations(content, citations_all) if citations_all else []
        data["_rag"] = {
            "enabled": bool(rag_cfg["enabled"]),
            "query": user_query,
            "citations_all": citations_all,
            "citations_used": citations_used,
            "context_blocks": context_block
        }

        # Prefer used citations; fallback to all
        refs_text = _format_references(citations_used or citations_all)
        _append_refs_to_openai_response(data, refs_text)
    return JSONResponse(status_code=r.status_code, content=data)

# =========================
# Health & debug
# =========================
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

@app.get("/debug/peek")
async def debug_peek(query: str, top_k: int = 4, file_contains: Optional[str] = None):
    """
    Quick end-to-end RAG probe:
    - Calls the RAG /search
    - Builds the numbered context using your build_context_and_map()
    - Returns exactly what will feed vLLM (context + citation map)
    """
    try:
        results = await _rag_search(query, top_k, file_contains)
    except HTTPException as e:
        return {"error": "rag_fetch_failed", "detail": e.detail}

    context_block, citations_all = build_context_and_map(results)
    return {
        "query": query,
        "top_k": top_k,
        "file_contains": file_contains,
        "results_len": len(results),
        "first_result_keys": (list(results[0].keys()) if results else []),
        "context_chars": len(context_block),
        "context_blocks": context_block,
        "citations_all": citations_all,
    }

@app.get("/")
def root():
    return {"service": "RAG→vLLM OpenAI Proxy for Singularity", "version": "1.4"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PROXY_PORT","8081")))
