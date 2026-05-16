"""verify.py — extract candidate kernel from Nemotron response, check correctness, benchmark.

Usage:
    python3 test_nemotron.py > response.txt
    python3 verify.py vadd.cu response.txt
"""
import re
import subprocess
import sys
import time
from pathlib import Path

RUNS = 11   # 1 warmup + 10 timed

def extract_code(response_text):
    m = re.search(r"```(?:cuda|cpp|c\+\+)?\n(.*?)```", response_text, re.DOTALL)
    return m.group(1) if m else None

def compile_cu(src, out):
    r = subprocess.run(["nvcc", "-O2", str(src), "-o", str(out)],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr

def run_capture(binary):
    r = subprocess.run([f"./{binary.name}"], capture_output=True, text=True,
                       cwd=binary.parent)
    return r.returncode == 0, r.stdout.strip()

def time_runs(binary, n=RUNS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        subprocess.run([f"./{binary.name}"], capture_output=True, text=True,
                       cwd=binary.parent, check=True)
        times.append(time.perf_counter() - t0)
    return sum(times[1:]) / (n - 1)  # drop warmup

def main():
    if len(sys.argv) != 3:
        print("Usage: verify.py <baseline.cu> <response.txt>")
        sys.exit(2)
    baseline_src = Path(sys.argv[1]).resolve()
    response_file = Path(sys.argv[2]).resolve()
    work = baseline_src.parent

    code = extract_code(response_file.read_text())
    if not code:
        print("FAIL: no cuda code block in response"); sys.exit(1)
    candidate_src = work / "candidate.cu"
    candidate_src.write_text(code)
    print(f"[ok] extracted {len(code)} bytes to {candidate_src.name}")

    baseline_bin = work / "baseline.bin"
    candidate_bin = work / "candidate.bin"

    ok, err = compile_cu(baseline_src, baseline_bin)
    if not ok:
        print(f"FAIL: baseline did not compile\n{err}"); sys.exit(1)
    print("[ok] baseline compiled")

    ok, err = compile_cu(candidate_src, candidate_bin)
    if not ok:
        print(f"REJECT: candidate did not compile\n{err}"); sys.exit(1)
    print("[ok] candidate compiled")

    ok_b, out_b = run_capture(baseline_bin)
    ok_c, out_c = run_capture(candidate_bin)
    if not (ok_b and ok_c) or out_b != out_c:
        print(f"REJECT: correctness failure\nbaseline:  {out_b}\ncandidate: {out_c}")
        sys.exit(1)
    print(f"[ok] outputs match: {out_b}")

    t_b = time_runs(baseline_bin)
    t_c = time_runs(candidate_bin)
    speedup = t_b / t_c if t_c > 0 else float("inf")
    print(f"\n=== BENCHMARK ===")
    print(f"baseline:  {t_b*1000:7.2f} ms/run  (avg of {RUNS-1})")
    print(f"candidate: {t_c*1000:7.2f} ms/run  (avg of {RUNS-1})")
    print(f"speedup:   {speedup:.2f}x  ({'faster' if speedup>1 else 'slower / noise'})")

if __name__ == "__main__":
    main()
