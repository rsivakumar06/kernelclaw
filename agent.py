"""agent.py — KernelClaw orchestrator.

CLI:
    python3 agent.py --watch          # daemon: watches ~/kernelclaw/watched/
    python3 agent.py --once file.cu   # batch: run pipeline once
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI

# ── paths ─────────────────────────────────────────────────────────────────────
REPO        = Path(__file__).parent.resolve()
WATCHED_DIR = REPO / "watched"
RUNS_LOG    = REPO / "runs.jsonl"
MEMORY_LOG  = REPO / "memory.jsonl"
NCU_BIN     = "/usr/local/cuda/bin/ncu"

MODEL = "nvidia/nemotron-3-nano-30b-a3b"

SYSTEM_PROMPT = """You are a CUDA performance expert embedded in an autonomous optimization agent.

You receive a CUDA source file and Nsight Compute profiler output.
Identify the most impactful bottleneck and produce a rewritten kernel.

RULES (non-negotiable):
1. The rewrite must be functionally equivalent — same outputs for same inputs.
2. Every kernel binary MUST emit to stdout:
   TIME_US=<float>    (wall time in microseconds via CUDA events)
   CHECKSUM=<float>   (sum of all output elements for correctness check)
3. Respond in exactly two parts:
   (1) A 3-5 sentence diagnosis of the bottleneck.
   (2) A single fenced ```cuda code block with the complete rewritten source.
   Nothing else."""

# ── API client ────────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip().lstrip("export ")
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )

# ── ncu metrics parser ────────────────────────────────────────────────────────
def parse_ncu_metrics(ncu_text: str) -> dict:
    def grab(pattern):
        m = re.search(pattern, ncu_text)
        return float(m.group(1).replace(",", "")) if m else 0.0
    return {
        "occupancy":             grab(r"Achieved Occupancy\s+%\s+([\d.,]+)"),
        "mem_throughput_pct":    grab(r"Memory Throughput\s+%\s+([\d.,]+)"),
        "compute_throughput_pct":grab(r"Compute \(SM\) Throughput\s+%\s+([\d.,]+)"),
    }

# ── memory + cosine similarity ────────────────────────────────────────────────
def load_memory() -> list:
    if not MEMORY_LOG.exists():
        return []
    entries = []
    for line in MEMORY_LOG.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries

def vec(m: dict) -> list:
    return [m.get("occupancy", 0.0),
            m.get("mem_throughput_pct", 0.0),
            m.get("compute_throughput_pct", 0.0)]

def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)

def top_similar(metrics: dict, k: int = 2) -> list:
    memory = load_memory()
    if not memory:
        return []
    v = vec(metrics)
    scored = [(cosine(v, vec(e["metrics"])), e) for e in memory]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]

def save_memory(metrics: dict, code: str, speedup: float):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics":   metrics,
        "rewrite_code": code,
        "speedup":   speedup,
    }
    with open(MEMORY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    total = len(load_memory())
    print(f"[memory] saved  (total entries: {total})")

# ── compile ───────────────────────────────────────────────────────────────────
def compile_cu(src: Path, out: Path) -> tuple[bool, str]:
    r = subprocess.run(
        ["nvcc", "-O2", str(src), "-o", str(out)],
        capture_output=True, text=True
    )
    return r.returncode == 0, r.stderr

# ── ncu profiler ──────────────────────────────────────────────────────────────
def run_ncu(binary: Path) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["sudo", "-n", NCU_BIN, "--set", "full", str(binary)],
            capture_output=True, text=True, timeout=120
        )
        return True, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return False, "ncu timed out"

# ── Nemotron ──────────────────────────────────────────────────────────────────
def call_nemotron(client: OpenAI, kernel_src: str,
                  ncu_report: str, few_shots: list) -> str:
    few_shot_text = ""
    if few_shots:
        few_shot_text = "\n\nPAST SUCCESSFUL REWRITES (similar bottleneck profile):\n"
        for i, fs in enumerate(few_shots):
            few_shot_text += (f"\n--- Example {i+1}  speedup={fs['speedup']:.2f}x ---\n"
                              f"```cuda\n{fs['rewrite_code'][:600]}\n```\n")

    user = (
        f"KERNEL SOURCE:\n```cuda\n{kernel_src}\n```\n\n"
        f"NSIGHT COMPUTE OUTPUT:\n```\n{ncu_report}\n```"
        f"{few_shot_text}\n\n"
        "Diagnose and rewrite. "
        "The rewritten kernel MUST print TIME_US=<float> and CHECKSUM=<float> to stdout."
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
        max_tokens=8000,
    )
    return resp.choices[0].message.content

def extract_code(text: str) -> str | None:
    m = re.search(r"```(?:cuda|cpp|c\+\+)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None

def extract_diagnosis(text: str) -> str:
    parts = text.split("```")
    return parts[0].strip()[:800] if parts else text[:500]

# ── logging ───────────────────────────────────────────────────────────────────
def log_run(kernel_name: str, metrics: dict, diagnosis: str,
            status: str, speedup: float, baseline_us: float, candidate_us: float):
    entry = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "kernel_name":  kernel_name,
        "metrics":      metrics,
        "diagnosis":    diagnosis,
        "status":       status,
        "speedup":      round(speedup, 4),
        "baseline_us":  round(baseline_us, 2),
        "candidate_us": round(candidate_us, 2),
    }
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[log] {status:12s} | {kernel_name} | speedup={speedup:.2f}x")

# ── pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(cu_file: Path, client: OpenAI):
    kernel_name = cu_file.stem
    print(f"\n{'='*60}")
    print(f"[kernelclaw] {kernel_name}  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    work = cu_file.parent
    baseline_bin  = work / f"{kernel_name}_baseline.bin"
    candidate_src = work / "candidate.cu"
    candidate_bin = work / "candidate.bin"
    response_file = work / "response.txt"
    ncu_out_file  = work / "ncu_report.txt"

    # 1 — compile baseline
    print("[1/6] Compiling baseline...")
    ok, err = compile_cu(cu_file, baseline_bin)
    if not ok:
        print(f"[FAIL] compile error:\n{err}")
        log_run(kernel_name, {}, "", "compile_fail", 0, 0, 0)
        return

    # 2 — ncu profile
    print("[2/6] Profiling with Nsight Compute...")
    ok, ncu_text = run_ncu(baseline_bin)
    if not ok:
        print(f"[WARN] ncu failed — continuing without full profile")
        ncu_text = "ncu unavailable"
    else:
        ncu_out_file.write_text(ncu_text)

    metrics = parse_ncu_metrics(ncu_text)
    print(f"       occupancy={metrics['occupancy']:.1f}%  "
          f"mem={metrics['mem_throughput_pct']:.1f}%  "
          f"compute={metrics['compute_throughput_pct']:.1f}%")

    # 3 — memory lookup
    print("[3/6] Searching memory for similar kernels...")
    few_shots = top_similar(metrics, k=2)
    print(f"       found {len(few_shots)} similar past optimization(s)")

    # 4 — call Nemotron
    print("[4/6] Calling Nemotron...")
    response = call_nemotron(client, cu_file.read_text(), ncu_text, few_shots)
    if not response:
        print("[FAIL] Nemotron returned empty response")
        log_run(kernel_name, metrics, "", "compile_fail", 0, 0, 0)
        return
    response_file.write_text(response)
    diagnosis = extract_diagnosis(response)
    print(f"       {diagnosis[:120]}...")

    # 5 — extract candidate
    candidate_code = extract_code(response)
    if not candidate_code:
        print("[FAIL] No cuda block in Nemotron response — logged as rejection")
        log_run(kernel_name, metrics, diagnosis, "compile_fail", 0, 0, 0)
        return
    candidate_src.write_text(candidate_code)

    # 6 — verify + benchmark
    print("[5/6] Running verify.py...")
    try:
        vr = subprocess.run(
            [sys.executable, str(REPO / "verify.py"),
             str(cu_file), str(response_file)],
            capture_output=True, text=True, timeout=120,
            cwd=str(work)
        )
    except subprocess.TimeoutExpired:
        print("[FAIL] verify.py timed out")
        log_run(kernel_name, metrics, diagnosis, "timeout", 0, 0, 0)
        return

    print(vr.stdout[-600:] if vr.stdout else "[no stdout]")
    if vr.stderr:
        print("[stderr]", vr.stderr[:200])

    # parse VERDICT
    verdict = {}
    for line in (vr.stdout + vr.stderr).splitlines():
        if line.startswith("VERDICT="):
            try:
                verdict = json.loads(line[8:])
            except json.JSONDecodeError:
                pass
            break

    status       = verdict.get("status",       "compile_fail")
    speedup      = float(verdict.get("speedup",      0.0))
    baseline_us  = float(verdict.get("baseline_us",  0.0))
    candidate_us = float(verdict.get("candidate_us", 0.0))

    print("[6/6] Logging...")
    log_run(kernel_name, metrics, diagnosis, status, speedup, baseline_us, candidate_us)

    if status == "verified" and speedup > 1.0:
        save_memory(metrics, candidate_code, speedup)
        print(f"\n✅  VERIFIED  {speedup:.2f}x faster — saved to memory")
    elif status == "verified":
        print(f"\n~  verified but no speedup ({speedup:.2f}x)")
    else:
        print(f"\n✗  REJECTED ({status}) — logged as demo asset")

# ── watchdog ──────────────────────────────────────────────────────────────────
class CUHandler(FileSystemEventHandler):
    def __init__(self, client):
        self.client = client
        self._seen: set[str] = set()

    def on_created(self, event):  self._handle(event)
    def on_modified(self, event): self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix != ".cu":
            return
        key = f"{p}:{p.stat().st_mtime if p.exists() else 0}"
        if key in self._seen:
            return
        self._seen.add(key)
        print(f"\n[watchdog] detected {p.name}")
        time.sleep(0.3)
        run_pipeline(p, self.client)

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="KernelClaw — autonomous CUDA optimizer")
    g  = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--watch", action="store_true",
                   help="Daemon: watch ~/kernelclaw/watched/ for .cu files")
    g.add_argument("--once", metavar="FILE.cu",
                   help="Batch: run pipeline once on FILE.cu")
    args = ap.parse_args()

    client = get_client()

    if args.once:
        f = Path(args.once).resolve()
        if not f.exists():
            sys.exit(f"[error] {f} not found")
        run_pipeline(f, client)

    else:
        WATCHED_DIR.mkdir(exist_ok=True)
        print(f"[watchdog] monitoring {WATCHED_DIR}")
        print("[watchdog] drop a .cu file in there to trigger optimization\n")
        handler  = CUHandler(client)
        observer = Observer()
        observer.schedule(handler, str(WATCHED_DIR), recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ == "__main__":
    main()