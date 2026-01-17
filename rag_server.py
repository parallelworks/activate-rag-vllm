import argparse, os, time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query
import uvicorn

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def http_chroma_client(host: str, port: int):
    return chromadb.HttpClient(host=host, port=port, settings=Settings(allow_reset=True, anonymized_telemetry=False))

def build_where(file_path_eq: Optional[str]=None, file_path_in: Optional[List[str]]=None):
    where=None
    if file_path_eq:
        where={"file_path":{"$eq":file_path_eq}}
    elif file_path_in:
        if isinstance(file_path_in,str): file_path_in=[file_path_in]
        where={"file_path":{"$in":file_path_in}}
    return where

class Retriever:
    def __init__(self, collection: str, embedding_model: str, host: str, port: int):
        self.collection_name=collection
        self.host, self.port = host, port
        self.client=http_chroma_client(host, port)
        device=os.environ.get("EMBEDDING_DEVICE","cpu")
        self.embed=SentenceTransformer(embedding_model, device=device)
        self.refresh_secs=int(os.environ.get("CHROMA_CLIENT_REFRESH_SECS","120"))
        self._last=time.time()

    def _maybe_refresh(self):
        if self.refresh_secs>0 and (time.time()-self._last)>=self.refresh_secs:
            self.client=http_chroma_client(self.host, self.port)
            self._last=time.time()

    def get_collection(self):
        self._maybe_refresh()
        return self.client.get_or_create_collection(self.collection_name)

    def search(self, query: str, k: int, where: Optional[Dict[str,Any]]=None):
        q=self.embed.encode(query)
        col=self.get_collection()
        res=col.query(query_embeddings=[q.tolist()], n_results=max(1,k), where=where)
        ids=(res.get("ids") or [[]])[0]
        docs=(res.get("documents") or [[]])[0]
        metas=(res.get("metadatas") or [[]])[0]
        dists=(res.get("distances") or [[]])[0]
        out=[]
        for i in range(len(ids)):
            dist=dists[i] if i<len(dists) else None
            out.append({
                "id": ids[i],
                "chunk_text": docs[i],
                "metadata": metas[i],
                "distance": dist,
                "similarity": (1.0 - dist) if isinstance(dist,(float,int)) else None
            })
        return out

    def peek(self, limit:int=5):
        col=self.get_collection()
        got=col.get(limit=limit, include=["documents","metadatas","ids"])
        out=[]
        if got and got.get("ids"):
            ids=got["ids"]; docs=got.get("documents") or []; metas=got.get("metadatas") or []
            for i in range(min(limit,len(ids))):
                text=docs[i] if isinstance(docs,list) else docs.get("documents",[ ""])[i]
                meta=metas[i] if isinstance(metas,list) else metas.get("metadatas",[{}])[i]
                out.append({
                    "id": ids[i],
                    "file_path": (meta or {}).get("file_path"),
                    "chunk_index": (meta or {}).get("chunk_index"),
                    "doc_preview": (text or "")[:180]
                })
        try:
            count=self.get_collection().count()
        except Exception:
            count=None
        return {"count": count, "sample": out}

app=FastAPI(title="RAG Search (Chroma HTTP)", version="1.0")
retriever: Retriever

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/debug/peek")
def debug_peek(limit:int=5):
    return retriever.peek(limit=limit)

@app.get("/search")
def search(
    query: str,
    top_k: int = 4,
    file_contains: Optional[str]=None,
    file_path_eq: Optional[str]=None,
    file_path_in: Optional[List[str]] = Query(None),
):
    where=build_where(file_path_eq=file_path_eq, file_path_in=file_path_in)
    results=retriever.search(query, top_k, where=where)
    if file_contains:
        needle=file_contains.lower()
        filt=[r for r in results if needle in ((r.get("metadata") or {}).get("file_path","").lower())]
        if len(filt)<top_k:
            widened=retriever.search(query, top_k*4, where=where)
            seen=set(x["id"] for x in filt)
            for r in widened:
                fp=((r.get("metadata") or {}).get("file_path") or "").lower()
                if needle in fp and r["id"] not in seen:
                    filt.append(r); seen.add(r["id"])
                    if len(filt)>=top_k: break
        results=filt
    return {"query": query, "results": results[:top_k]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--collection", default="activate_rag")
    ap.add_argument("--embedding_model", default=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--port", type=int, default=8080)
    args=ap.parse_args()

    host=os.environ.get("CHROMA_HOST","127.0.0.1")
    port=int(os.environ.get("CHROMA_PORT","8000"))
    global retriever
    retriever=Retriever(args.collection, args.embedding_model, host, port)
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__=="__main__":
    main()
