#!/usr/bin/env python3
"""
KernelClaw Dashboard Backend
Run from ~/kernelclaw/dashboard/:
    pip install fastapi uvicorn python-multipart
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# ── Paths (relative to repo root ~/kernelclaw) ────────────────────────────────
REPO = Path(__file__).parent.parent          # ~/kernelclaw
RUNS_FILE      = REPO / "runs.jsonl"
MEMORY_FILE    = REPO / "memory.jsonl"
POLICY_FILE    = REPO / "policy_log.txt"
KERNELS_DIR    = REPO / "kernels"            # original .cu files
CANDIDATES_DIR = REPO / "candidates"         # rewritten .cu files
WATCHED_DIR    = REPO / "watched"            # drop zone target

WATCHED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="KernelClaw Dashboard API")

# Allow the HTML page served on any origin (Tailscale IP, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── /runs ─────────────────────────────────────────────────────────────────────
@app.get("/runs")
def get_runs():
    """
    Return all agent runs from runs.jsonl, newest first.
    Each object contains: timestamp, kernel_name, metrics, diagnosis,
    status, speedup, baseline_us, candidate_us.
    """
    runs = []
    if RUNS_FILE.exists():
        for line in RUNS_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip malformed lines
    # newest first
    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return runs


# ── /policy_log ───────────────────────────────────────────────────────────────
@app.get("/policy_log")
def get_policy_log(offset: int = Query(0, ge=0)):
    """
    Return new lines from policy_log.txt starting at `offset`.
    Response: { lines: [...], total: N }
    Supports incremental polling — client sends offset = lines it already has.
    """
    if not POLICY_FILE.exists():
        return {"lines": [], "total": 0}

    all_lines = POLICY_FILE.read_text(errors="replace").splitlines()
    total = len(all_lines)
    new_lines = all_lines[offset:]
    return {"lines": new_lines, "total": total}


# ── /memory ───────────────────────────────────────────────────────────────────
@app.get("/memory")
def get_memory():
    """Return persistent memory entries from memory.jsonl."""
    entries = []
    if MEMORY_FILE.exists():
        for line in MEMORY_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


# ── /source/kernel/{name} ─────────────────────────────────────────────────────
@app.get("/source/kernel/{kernel_name}", response_class=PlainTextResponse)
def get_baseline_source(kernel_name: str):
    """Return original kernel source (read-only)."""
    path = (KERNELS_DIR / kernel_name).with_suffix(".cu")
    if not path.exists():
        raise HTTPException(404, f"Kernel not found: {kernel_name}.cu")
    return path.read_text(errors="replace")


# ── /source/candidate/{name} ─────────────────────────────────────────────────
@app.get("/source/candidate/{kernel_name}", response_class=PlainTextResponse)
def get_candidate_source(kernel_name: str):
    """Return optimized candidate source (read-only)."""
    path = (CANDIDATES_DIR / kernel_name).with_suffix(".cu")
    if not path.exists():
        raise HTTPException(404, f"Candidate not found: {kernel_name}.cu")
    return path.read_text(errors="replace")


# ── /upload ───────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_kernel(file: UploadFile = File(...)):
    """
    Drag-and-drop endpoint. Writes the uploaded .cu (or .py) file to
    ~/kernelclaw/watched/ which the agent's file watcher monitors.
    Read-only w.r.t. agent directories — writes ONLY to watched/.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".cu", ".py"}:
        raise HTTPException(400, "Only .cu or .py files are accepted")

    # Sanitise filename (no path traversal)
    safe_name = Path(file.filename).name
    dest = WATCHED_DIR / safe_name

    contents = await file.read()
    dest.write_bytes(contents)
    return {"status": "ok", "written": str(dest)}


from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "runs_file":   str(RUNS_FILE),
        "policy_file": str(POLICY_FILE),
        "watched_dir": str(WATCHED_DIR),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
