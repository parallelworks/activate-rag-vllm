#!/usr/bin/env python3
"""
Indexer (HTTP) for Chroma 0.5.x — citation-enabled
- Watches one or more directories for files, chunks & embeds them with Sentence-Transformers.
- Writes to a Chroma HTTP server (no embedded cache issues).
- Polling + periodic rescan + GC (prune deleted files).
- SQLite FTS5 side index (per-op short-lived connections to avoid thread errors).
- ADDS rich metadata per chunk for citations: file_path, chunk_index, doc_sha256, span, title.

Env:
  CHROMA_HOST=chroma
  CHROMA_PORT=8000
  EMBEDDING_DEVICE=cpu|cuda
  INDEXER_LOGLEVEL=INFO|DEBUG
  INDEXER_RESCAN_SECONDS=20

Requires:
  chromadb==0.5.x, sentence-transformers, watchdog, pypdf, pyyaml
"""

import argparse, os, time, fnmatch, threading, csv, logging, sqlite3, hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import yaml

import math, time as _time
from collections import defaultdict

LOG = logging.getLogger("indexer")
logging.basicConfig(
    level=os.environ.get("INDEXER_LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------- Config / IO ----------

def http_chroma_client(host: str, port: int):
    return chromadb.HttpClient(
        host=host, port=port,
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )

def sha256_file(path: str, bufsize: int = 1024 * 1024) -> str:
    """Compute a stable document hash for reliable citation & de-dup."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_text(path: str) -> str:
    p = path.lower()
    if p.endswith(('.txt', '.md', '.log')):
        with open(path, 'r', errors='ignore') as f:
            return f.read()
    if p.endswith('.pdf'):
        text = []
        try:
            r = PdfReader(path)
            for pg in r.pages:
                t = pg.extract_text() or ""
                if t:
                    text.append(t)
        except Exception as e:
            LOG.warning("PDF read failed %s: %s", path, e)
        return "\n".join(text)
    if p.endswith('.csv'):
        rows = []
        with open(path, newline='', errors='ignore') as f:
            rdr = csv.reader(f)
            for row in rdr:
                rows.append(" ".join(row))
        return "\n".join(rows)
    return ""

def chunk_text_with_spans(text: str, size: int, overlap: int) -> List[Tuple[str, Tuple[int,int]]]:
    """
    Return list of (chunk_text, (start,end)) character spans so the proxy/UI can cite exact ranges.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        out.append((text[i:j], (i, j)))
        if j >= n:
            break
        i = max(0, j - overlap)
    return out

def matches_any(patterns, name):
    return any(fnmatch.fnmatch(name, p) for p in patterns)

# ---------- FTS helpers (per-operation connection; thread-safe) ----------

FTS_ROOT = "/root/.cache/fts"
FTS_DB = os.path.join(FTS_ROOT, "chunks.db")

def fts_exec(sql: str, params: tuple = (), many: Optional[List[tuple]] = None):
    os.makedirs(FTS_ROOT, exist_ok=True)
    conn = sqlite3.connect(FTS_DB, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                id, file_path, chunk_index UNINDEXED, text, tokenize='porter'
            );
        """)
        if many is not None:
            conn.executemany(sql, many)
        else:
            conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()

# ---------- Indexer ----------

# Per-file locks to avoid concurrent upserts on the same path
_FILE_LOCKS = defaultdict(threading.Lock)

def _jsonable_meta(meta: dict) -> dict:
    # ensure all metadata is JSON-serializable and small
    out = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out

def _chroma_add_with_retries(col, ids, docs, embs, metas, file_path: str,
                             initial_batch: int = None, max_retries: int = 3):
    """Add in batches with exponential backoff; shrink batch on failure."""
    BATCH = int(os.environ.get("CHROMA_BATCH_SIZE", "64")) if initial_batch is None else initial_batch
    n = len(ids)
    i = 0
    added = 0

    while i < n:
        j = min(i + BATCH, n)
        slice_ids   = ids[i:j]
        slice_docs  = docs[i:j]
        slice_embs  = embs[i:j]
        slice_metas = metas[i:j]

        attempt = 0
        while True:
            try:
                col.add(ids=slice_ids, documents=slice_docs,
                        embeddings=slice_embs, metadatas=slice_metas)
                added += (j - i)
                break
            except Exception as e:
                attempt += 1
                # If Chroma/HTTP rejects payload, reduce batch size and retry
                if attempt <= max_retries:
                    wait = min(2 ** attempt, 8)
                    LOG.warning("[UPSERT-RETRY] file=%s batch=%d attempt=%d err=%s; backing off %ss",
                                file_path, BATCH, attempt, e, wait)
                    _time.sleep(wait)
                    # halve batch for next try of this slice
                    if BATCH > 8:
                        BATCH = max(8, BATCH // 2)
                    continue
                LOG.error("[UPSERT-FAIL] file=%s slice=%d:%d err=%s", file_path, i, j, e)
                raise
        i = j
    return added

class Indexer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        host = os.environ.get("CHROMA_HOST", "127.0.0.1")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        self.client = http_chroma_client(host, port)
        self.col = self.client.get_or_create_collection(cfg['collection'])

        device = os.environ.get("EMBEDDING_DEVICE", "cpu")
        self.embed = SentenceTransformer(cfg['embedding_model'], device=device)

        self.pool = ThreadPoolExecutor(max_workers=max(4, (os.cpu_count() or 8)))
        self.seen: Dict[str, float] = {}
        self.lock = threading.Lock()

    def should_index(self, path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name
        if matches_any(self.cfg.get('exclude_globs', []), name):
            return False
        inc = self.cfg.get("include_ext") or []
        if inc and not any(name.lower().endswith(e) for e in inc):
            return False
        return True

    def stable(self, path: Path) -> bool:
        try:
            now = time.time()
            mtime = path.stat().st_mtime
            return (now - mtime) >= int(self.cfg.get("stabilize_seconds", 10))
        except FileNotFoundError:
            return False

    def upsert_file(self, path: str):
        p = Path(path)
        if not self.should_index(p): return
        if not self.stable(p):
            LOG.debug("Deferring (not stable yet): %s", path)
            return

        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            return

        # serialize per-file upserts (avoid concurrent delete/add on same file)
        lock = _FILE_LOCKS[path]
        with lock:
            with self.lock:
                prev = self.seen.get(path)
                self.seen[path] = mtime
            if prev is not None and prev == mtime:
                LOG.debug("No change: %s", path)
                return

            text = load_text(path)
            if not text.strip():
                self.col.delete(where={"file_path": path})
                fts_exec("DELETE FROM chunks WHERE file_path = ?", (path,))
                LOG.info("[DELETE-EMPTY] %s", path)
                return

            # Build chunks WITH spans (for citations)
            chunk_size = int(self.cfg["chunk_chars"])
            chunk_overlap = int(self.cfg["chunk_overlap"])
            chunks_spans = chunk_text_with_spans(text, chunk_size, chunk_overlap)
            chunks = [c for c, _ in chunks_spans]
            spans = [s for _, s in chunks_spans]

            LOG.info("[EMBED] %s -> %d chunks", path, len(chunks))
            vecs = self.embed.encode(chunks, show_progress_bar=False)

            # Compute file hash once (doc-level)
            file_hash = sha256_file(path)

            # Replace any existing entries for this file
            try:
                self.col.delete(where={"file_path": path})
            except Exception as e:
                LOG.warning("[DELETE-PREVIOUS] failed for %s: %s", path, e)

            ids = [f"{path}::{i}" for i in range(len(chunks))]
            base_metas = [{
                "file_path": path,
                "chunk_index": int(i),
                "doc_sha256": file_hash,
                "span_start": int(spans[i][0]),
                "span_end": int(spans[i][1]),
                "title": os.path.basename(path),
            } for i in range(len(chunks))]
            # No need to stringify; everything is primitive already
            metas = base_metas

            # Add in batches with retries/backoff
            added = _chroma_add_with_retries(self.col, ids, chunks, vecs, metas, file_path=path)

            # Verify write by reading back this file's entries
            try:
                # Chroma 0.5.x: get() with where+limit large to count
                probe = self.col.get(where={"file_path": path}, include=["ids"], limit=100000)
                count = len(probe.get("ids", []) or [])
            except Exception as e:
                LOG.warning("[VERIFY] read-back failed for %s: %s", path, e)
                count = added

            # FTS (thread-safe: per-op connection)
            try:
                fts_exec("DELETE FROM chunks WHERE file_path = ?", (path,))
                fts_exec(
                    "INSERT INTO chunks(id,file_path,chunk_index,text) VALUES (?,?,?,?)",
                    many=[(ids[i], path, i, chunks[i]) for i in range(len(chunks))]
                )
            except Exception as e:
                LOG.warning("[FTS] failed to update for %s: %s", path, e)

            LOG.info("[UPSERT] %s -> added=%d, verified=%d", path, added, count)

    def delete_file(self, path: str):
        self.col.delete(where={"file_path": path})
        with self.lock:
            self.seen.pop(path, None)
        fts_exec("DELETE FROM chunks WHERE file_path = ?", (path,))
        LOG.info("[DELETE] %s", path)

    def initial_scan(self):
        roots = self.cfg["watch_paths"]
        LOG.info("Initial scan: %s", roots)
        for root in roots:
            for dirpath, _, files in os.walk(root):
                for name in files:
                    p = os.path.join(dirpath, name)
                    if self.should_index(Path(p)) and self.stable(Path(p)):
                        self.pool.submit(self.upsert_file, p)

    def periodic_rescan(self, interval: int, stop_event: threading.Event):
        if interval <= 0:
            return
        LOG.info("Periodic rescan enabled: every %ss", interval)
        while not stop_event.is_set():
            roots = self.cfg["watch_paths"]
            for root in roots:
                for dirpath, _, files in os.walk(root):
                    for name in files:
                        p = os.path.join(dirpath, name)
                        if self.should_index(Path(p)) and self.stable(Path(p)):
                            self.pool.submit(self.upsert_file, p)

            # GC: prune missing files
            with self.lock:
                known = list(self.seen.keys())
            for path in known:
                if not os.path.exists(path):
                    self.delete_file(path)

            stop_event.wait(interval)

# ---------- Watchdog handlers ----------

class Handler(FileSystemEventHandler):
    def __init__(self, idx: Indexer):
        super().__init__()
        self.idx = idx

    def on_created(self, e):
        if not e.is_directory:
            self.idx.pool.submit(self.idx.upsert_file, e.src_path)

    def on_modified(self, e):
        if not e.is_directory:
            self.idx.pool.submit(self.idx.upsert_file, e.src_path)

    def on_moved(self, e):
        if not e.is_directory:
            # remove old, index new
            self.idx.delete_file(e.src_path)
            self.idx.pool.submit(self.idx.upsert_file, e.dest_path)

    def on_deleted(self, e):
        if not e.is_directory:
            self.idx.delete_file(e.src_path)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="indexer_config.yaml")
    ap.add_argument("--poll", action="store_true", help="Use polling observer (best for NFS/bind mounts)")
    ap.add_argument("--rescan-seconds", type=int, default=int(os.environ.get("INDEXER_RESCAN_SECONDS", "20")))
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # sanity defaults
    cfg.setdefault("watch_paths", ["/docs"])
    cfg.setdefault("collection", "activate_rag")
    cfg.setdefault("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    env_embedding = os.environ.get("EMBEDDING_MODEL")
    if env_embedding:
        cfg["embedding_model"] = env_embedding
    cfg.setdefault("include_ext", [".txt", ".pdf", ".md", ".csv", ".log"])
    cfg.setdefault("exclude_globs", [".DS_Store", ".*", "*.part", "*.tmp"])
    cfg.setdefault("chunk_chars", 1200)
    cfg.setdefault("chunk_overlap", 200)
    cfg.setdefault("stabilize_seconds", 10)

    idx = Indexer(cfg)
    idx.initial_scan()

    stop = threading.Event()
    t = threading.Thread(target=idx.periodic_rescan, args=(args.rescan_seconds, stop), daemon=True)
    t.start()

    # Observer selection
    try:
        ObserverClass = PollingObserver if args.poll else Observer
        obs = ObserverClass()
    except Exception as e:
        LOG.warning("Falling back to PollingObserver due to: %s", e)
        obs = PollingObserver()

    handler = Handler(idx)
    for p in cfg["watch_paths"]:
        LOG.info("Watching: %s (recursive)", p)
        obs.schedule(handler, path=p, recursive=True)
    obs.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        obs.stop()
        obs.join()

if __name__ == "__main__":
    main()
