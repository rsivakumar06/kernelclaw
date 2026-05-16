"""reason_only.py — KernelClaw reasoning component, runs inside NemoClaw sandbox.

The host-side agent.py does compile/profile/verify (privileged operations).
This script handles the *untrusted* part: the LLM-generated code rewrite.
NemoClaw's policy restricts it to: read kernel source, read ncu report,
call the Nemotron API, write candidate code. No compile, no exec, no network elsewhere.
"""
import os
import re
from pathlib import Path
from openai import OpenAI

REPO       = Path(__file__).parent.resolve()
KERNEL_SRC = REPO / "kernels" / "naive_matmul.cu"
NCU_REPORT = REPO / "kernels" / "naive_matmul.ncu.txt"
OUTPUT_DIR = REPO / "candidates"

MODEL = "nvidia/nemotron-3-super-120b-a12b"

SYSTEM = """You are a CUDA performance expert.

You receive a CUDA source file and Nsight Compute profiler output.
Identify the most impactful bottleneck and produce a rewritten kernel.

Rules:
1. The rewrite must be functionally equivalent.
2. Respond in exactly two parts: (1) a 3-5 sentence diagnosis, (2) a single
   fenced cuda code block with the complete rewritten source. Nothing else."""

def load_env():
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip().lstrip("export ")
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))

def main():
    print("=" * 60)
    print("KernelClaw reasoning agent — running inside NemoClaw sandbox")
    print("=" * 60)
    print()
    load_env()

    print(f"[1/4] Reading kernel source: {KERNEL_SRC.name}")
    src = KERNEL_SRC.read_text()
    print(f"      ({len(src)} bytes)")

    print(f"[2/4] Reading ncu profile:   {NCU_REPORT.name}")
    if not NCU_REPORT.exists():
        print(f"      ERROR: {NCU_REPORT} not found — capture it on the host first")
        return
    ncu = NCU_REPORT.read_text()
    print(f"      ({len(ncu)} bytes)")

    print(f"[3/4] Calling Nemotron ({MODEL})...")
    client = OpenAI(
        base_url="https://inference.local/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )
    user = (
        f"KERNEL SOURCE ({KERNEL_SRC.name}):\n```cuda\n{src}\n```\n\n"
        f"NSIGHT COMPUTE OUTPUT:\n```\n{ncu}\n```\n\n"
        "Diagnose and rewrite."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
        temperature=0.2,
        max_tokens=8000,
    )
    response = resp.choices[0].message.content
    print(f"      received {len(response)} chars")
    print()

    diagnosis = response.split("```")[0].strip()
    print("[4/4] DIAGNOSIS:")
    for line in diagnosis.splitlines():
        print(f"      {line}")
    print()

    m = re.search(r"```(?:cuda|cpp|c\+\+)?\s*\n(.*?)```", response, re.DOTALL)
    if not m:
        print("      NO CUDA CODE BLOCK in response — Nemotron returned only prose")
        return
    candidate = m.group(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "naive_matmul_candidate.cu"
    out.write_text(candidate)
    print(f"      Rewritten kernel saved → {out.relative_to(REPO)} ({len(candidate)} bytes)")
    print()
    print("Sandbox boundary respected: no compile, no profile, no exec.")
    print("Hand off to host-side verify.py for benchmarking.")

if __name__ == "__main__":
    main()
